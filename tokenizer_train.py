import numpy as np
import os
import math
import json
import argparse
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from tokenizer_model import VectorQuantizerEMA, VQVAE_Transformer
def load_config(config_path: str):
    """
    从 JSON 配置文件读取超参数。
    """
    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)

parser = argparse.ArgumentParser()
parser.add_argument(
    "--config",
    type=str,
    default="vqvae_config.json",
    help="配置文件路径（JSON）",
)
args = parser.parse_args()

cfg = load_config(args.config)

# ---- 时间窗口 & 标签窗口（只用于路径和模型命名）----
SEQ_LEN = cfg["seq_len"]               # 时间窗口长度
LABEL_HORIZON = cfg["label_horizon"]   # 仅用于命名，不影响 VQ-VAE 本身

# ---- 路径配置 ----
DATA_ROOT = cfg["paths"]["data_root"]
MODEL_SAVE_DIR = cfg["paths"]["model_save_dir"]
MODEL_NAME_SUFFIX = cfg["paths"]["model_name_suffix"]

DATA_DIR = os.path.join(DATA_ROOT, f"sp_seq{SEQ_LEN}_n{LABEL_HORIZON}")
TRAIN_NPZ_PATH = os.path.join(DATA_DIR, "train_labeled.npz")
VAL_NPZ_PATH = os.path.join(DATA_DIR, "val_labeled.npz")   # 新增：val 集路径

os.makedirs(MODEL_SAVE_DIR, exist_ok=True)

# === 模型基础名称 & 路径 ===
MODEL_NAME_BASE = f"seq{SEQ_LEN}_n{LABEL_HORIZON}_{MODEL_NAME_SUFFIX}"
BEST_MODEL_PATH = os.path.join(MODEL_SAVE_DIR, MODEL_NAME_BASE + "_best.pth")
LAST_MODEL_PATH = os.path.join(MODEL_SAVE_DIR, MODEL_NAME_BASE + "_last.pth")
LOSS_FIG_PATH = os.path.join(MODEL_SAVE_DIR, MODEL_NAME_BASE + "_loss_curve.png")

# ---- 数据和模型规格 ----
INPUT_DIM = cfg["input_dim"]
D_MODEL = cfg["model"]["d_model"]
N_HEAD = cfg["model"]["n_head"]
NUM_EMBEDDINGS = cfg["model"]["num_embeddings"]
EMBEDDING_DIM = cfg["model"]["embedding_dim"]

NUM_ENCODER_LAYERS = cfg["model"]["num_encoder_layers"]
NUM_DECODER_LAYERS = cfg["model"]["num_decoder_layers"]
DIM_FEEDFORWARD = cfg["model"]["dim_feedforward"]
COMMITMENT_COST = cfg["vq"]["commitment_cost"]  # beta
EMA_DECAY = cfg["vq"]["ema_decay"]
EMA_EPSILON = cfg["vq"]["ema_eps"]
# ---- 训练参数 ----
BATCH_SIZE = cfg["train"]["batch_size"]
EPOCHS = cfg["train"]["epochs"]
LR = cfg["train"]["lr"]
ROTARY_DIM = cfg["model"].get("rotary_dim", None)
# ---- VQ 参数 ----

def set_seed(seed=42):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        # 极致确定性需要牺牲一点性能（可选）
        torch.backends.cudnn.deterministic = True 
        torch.backends.cudnn.benchmark = False
# ---- 设备 ----
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class TimeSeriesDataset(Dataset):
    """用于 (N, S, F) 时序数据的 Dataset"""
    def __init__(self, data_array: np.ndarray):
        # data_array: [N, SEQ_LEN, INPUT_DIM]
        assert data_array.ndim == 3, "data_array 必须是 [N, S, F]"
        self.data = data_array.astype(np.float32)
        
    def __len__(self):
        return len(self.data)
        
    def __getitem__(self, idx):
        return self.data[idx]

def main():
    set_seed(96)
    print("\n" + "=" * 60)
    print(f"  开始训练 VQ-VAE Stage 1 (seq_len={SEQ_LEN}, n={LABEL_HORIZON})")
    print("=" * 60)
    print(f"使用配置文件: {args.config}")
    
    # ---- 1. 加载 train 数据 ----
    try:
        print(f"正在从 {TRAIN_NPZ_PATH} 加载训练数据 (train.npz)...")
        train_npz = np.load(TRAIN_NPZ_PATH)
        X_train = train_npz["X"]  # [N_train, SEQ_LEN, INPUT_DIM]
        print(f"X_train 形状: {X_train.shape}")
        assert X_train.shape[1] == SEQ_LEN, "train.npz 中的 SEQ_LEN 不匹配"
        assert X_train.shape[2] == INPUT_DIM, "train.npz 中的 INPUT_DIM 不匹配"
    except FileNotFoundError:
        print(f"错误: 未找到训练数据文件: {TRAIN_NPZ_PATH}")
        print("请先运行数据处理脚本生成 train.npz (X, y, codes, times)。")
        return
    except Exception as e:
        print(f"加载 train.npz 时出错: {e}")
        return

    # ---- 2. 加载 val 数据 ----
    try:
        print(f"正在从 {VAL_NPZ_PATH} 加载验证数据 (val.npz)...")
        val_npz = np.load(VAL_NPZ_PATH)
        X_val = val_npz["X"]  # [N_val, SEQ_LEN, INPUT_DIM]
        print(f"X_val 形状: {X_val.shape}")
        assert X_val.shape[1] == SEQ_LEN, "val.npz 中的 SEQ_LEN 不匹配"
        assert X_val.shape[2] == INPUT_DIM, "val.npz 中的 INPUT_DIM 不匹配"
    except FileNotFoundError:
        print(f"错误: 未找到验证数据文件: {VAL_NPZ_PATH}")
        print("请先准备好 val.npz。")
        return
    except Exception as e:
        print(f"加载 val.npz 时出错: {e}")
        return

    train_dataset = TimeSeriesDataset(X_train)
    val_dataset = TimeSeriesDataset(X_val)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, drop_last=False)
    
    # ---- 3. 实例化模型 ----
    model = VQVAE_Transformer(
        input_dim=INPUT_DIM,
        seq_len=SEQ_LEN,
        d_model=D_MODEL,
        nhead=N_HEAD,
        num_encoder_layers=NUM_ENCODER_LAYERS,
        num_decoder_layers=NUM_DECODER_LAYERS,
        dim_feedforward=DIM_FEEDFORWARD,
        num_embeddings=NUM_EMBEDDINGS,
        embedding_dim=EMBEDDING_DIM,
        commitment_cost=COMMITMENT_COST,
        ema_decay=EMA_DECAY,
        rotary_dim=ROTARY_DIM,
        ema_epsilon=EMA_EPSILON
    ).to(DEVICE)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    
    print(f"开始在 {DEVICE} 上训练 ({EPOCHS} epochs)...")
    
    best_val_total_loss = float("inf")

    train_total_history = []
    val_total_history = []
    
    # ---- 4. 训练循环 ----
    for epoch in range(EPOCHS):
        # ======== Train ========
        model.train()
        total_recon_loss = 0.0
        total_vq_loss = 0.0
        
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Train]"):
            batch = batch.to(DEVICE)  # [B, S, F]
            
            optimizer.zero_grad()
            
            vq_loss, x_recon, _ = model(batch)
            recon_loss = F.mse_loss(x_recon, batch)
            loss = recon_loss + vq_loss
            
            loss.backward()
            optimizer.step()
            
            total_recon_loss += recon_loss.item()
            total_vq_loss += vq_loss.item()
        
        avg_recon = total_recon_loss / len(train_loader)
        avg_vq = total_vq_loss / len(train_loader)
        avg_total = avg_recon + avg_vq     

        train_total_history.append(avg_total)
        
        print(f"[Epoch {epoch+1}] Train Recon Loss: {avg_recon:.6f} | "
              f"Train VQ Loss: {avg_vq:.6f} | Train Total: {avg_total:.6f}")
        
        # ======== Val ========
        model.eval()
        val_recon_sum = 0.0
        val_vq_sum = 0.0

        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Val]"):
                batch = batch.to(DEVICE)
                vq_loss, x_recon, _ = model(batch)
                recon_loss = F.mse_loss(x_recon, batch)

                val_recon_sum += recon_loss.item()
                val_vq_sum += vq_loss.item()

        val_avg_recon = val_recon_sum / len(val_loader)
        val_avg_vq = val_vq_sum / len(val_loader)
        val_avg_total = val_avg_recon + val_avg_vq

        val_total_history.append(val_avg_total)

        print(f"[Epoch {epoch+1}] Val   Recon Loss: {val_avg_recon:.6f} | "
              f"Val   VQ Loss: {val_avg_vq:.6f} | Val   Total: {val_avg_total:.6f}")
        
        # ---- 用 val total loss 选择 best ----
        if val_avg_total < best_val_total_loss:
            best_val_total_loss = val_avg_total
            torch.save(model.state_dict(), BEST_MODEL_PATH)
            print(f"  -> 新 best 模型已保存到: {BEST_MODEL_PATH} (val total={best_val_total_loss:.6f})")
    
    # ---- 5. 保存最后一轮模型 ----
    torch.save(model.state_dict(), LAST_MODEL_PATH)
    print(f"\n最后一轮模型已保存到: {LAST_MODEL_PATH}")
    print(f"最佳模型已保存到: {BEST_MODEL_PATH}")

    # ---- 6. 绘制 train / val loss 曲线 ----
    epochs = np.arange(1, EPOCHS + 1)

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, train_total_history, label="Train Total Loss")
    plt.plot(epochs, val_total_history, label="Val Total Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"VQ-VAE Train/Val Loss (seq{SEQ_LEN}_n{LABEL_HORIZON})")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(LOSS_FIG_PATH)
    plt.close()

    print(f"Train / Val Loss 曲线已保存到: {LOSS_FIG_PATH}")
    print("训练结束。")


if __name__ == "__main__":
    main()