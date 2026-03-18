import os
import json
import argparse
import numpy as np
from tqdm import tqdm
from typing import List
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Sampler
# python pretrain_cde_fineturn.py --config ./config/20_casual.json --stage1_ckpt ./models/seq20_n1_sp_20_stage_v1_best.pth --stage2_ckpt ./models/seq20_n1_sp_20_stage_v1_tokenpred_codebook_td0p150_mp0p250_klk1_best.pth --save_path ./models/nocde.pth --time_decay 0.15
# ====== 你已有的 Stage1/Stage2：只 import，不重定义 ======
from tokenizer_model import VQVAE_Transformer,CDEBlock
# TODO: 把下面这一行改成你真实的 Stage2 token predictor 定义所在文件
from pretrain import TokenPredictorUsingCodebook   # <- 你需要改这里

# ================= 配置区 =================
TARGET_KEY = "t_value_7d"
DATE_KEY = "times"
LABEL_CLIP = 5.0  # 仍保持原来的裁剪范围：[-5, 5]
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

# 在 main() 开头调用
set_seed(42)


def load_config(config_path: str):
    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)


def pick_existing_path(data_dir: str, candidates: List[str]) -> str:
    """从候选文件名里选第一个存在的"""
    for name in candidates:
        p = os.path.join(data_dir, name)
        if os.path.exists(p):
            return p
    raise FileNotFoundError(f"None of these files exist under {data_dir}: {candidates}")


# ================= 你的 Dataset / Sampler（原样保留 + 支持 close_ratio） =================
class RegressionDataset(Dataset):
    def __init__(
        self,
        data_path,
        target_key: str = TARGET_KEY,
        date_key: str = DATE_KEY,
        limit=None,
        label_clip: float = LABEL_CLIP,
        target_mode: str = "close_ratio",   # "key" | "close_ratio"（默认直接用收益率）
        close_today_key: str = "close_today",
        close_future_key: str = "close_future",
        eps: float = 1e-12,
    ):
        pack = np.load(data_path, allow_pickle=True)
        self.X = pack["X"].astype(np.float32)

        if date_key not in pack:
            raise ValueError(f"Key '{date_key}' not found in {data_path}. Available: {list(pack.keys())}")
        raw_date = pack[date_key]

        # -------- 生成 raw_y + valid_mask --------
        if target_mode == "close_ratio":
            if close_today_key not in pack or close_future_key not in pack:
                raise ValueError(
                    f"target_mode=close_ratio 需要 '{close_today_key}' 和 '{close_future_key}'，"
                    f"但在 {data_path} 里没找到。Available: {list(pack.keys())}"
                )
            close_today = pack[close_today_key].astype(np.float32)
            close_future = pack[close_future_key].astype(np.float32)

            # y = close_future / close_today - 1
            raw_y = close_future / (close_today + eps) - 1.0

            # 过滤无效：nan/inf、以及 close_today 近似为 0 的情况
            valid_mask = (
                np.isfinite(raw_y)
                & np.isfinite(close_today)
                & np.isfinite(close_future)
                & (np.abs(close_today) > eps)
            )
        elif target_mode == "key":
            if target_key not in pack:
                raise ValueError(f"Key '{target_key}' not found in {data_path}. Available: {list(pack.keys())}")
            raw_y = pack[target_key].astype(np.float32)
            valid_mask = np.isfinite(raw_y)
        else:
            raise ValueError(f"Unknown target_mode: {target_mode}. Use 'key' or 'close_ratio'.")

        # -------- 应用过滤 --------
        self.X = self.X[valid_mask]
        raw_y = raw_y[valid_mask]
        raw_date = raw_date[valid_mask]

        # -------- 裁剪 label（保持你原来的逻辑：默认 [-5,5]）--------
        self.y = np.clip(raw_y, -float(label_clip), float(label_clip)).astype(np.float32)

        # -------- date 转 int64（支持 YYYYMMDD int 或字符串）--------
        if raw_date.dtype.kind in ("U", "S", "O"):
            def to_int(d):
                s = str(d)
                digits = "".join([c for c in s if c.isdigit()])
                return int(digits[:8])  # YYYYMMDD
            self.date = np.array([to_int(d) for d in raw_date], dtype=np.int64)
        else:
            self.date = raw_date.astype(np.int64)

        if limit:
            self.X = self.X[:limit]
            self.y = self.y[:limit]
            self.date = self.date[:limit]

        print(f"Loaded {len(self.y)} samples from {os.path.basename(data_path)}")
        print(f"Target mode: {target_mode} | clip=[{-label_clip}, {label_clip}]")
        print(f"Label stats: Mean={self.y.mean():.6f}, Std={self.y.std():.6f}, Min={self.y.min():.6f}, Max={self.y.max():.6f}")
        print(f"Date unique count: {len(np.unique(self.date))}")

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx], self.date[idx]


class TimepointBatchSampler(Sampler):
    """
    一个 batch = 同一个 date 的全部样本（所有股票）
    - 如果一天股票太多显存不够：可以 max_stocks 下采样
    """
    def __init__(self, date_ids, shuffle_time=True, min_stocks=2, max_stocks=None):
        self.date_ids = np.asarray(date_ids)
        self.shuffle_time = shuffle_time
        self.min_stocks = min_stocks
        self.max_stocks = max_stocks

        self.d2idx = {}
        for d in np.unique(self.date_ids):
            idx = np.where(self.date_ids == d)[0]
            if len(idx) >= self.min_stocks:
                self.d2idx[int(d)] = idx

        self.dates = list(self.d2idx.keys())

    def __iter__(self):
        dates = self.dates.copy()
        if self.shuffle_time:
            np.random.shuffle(dates)

        for d in dates:
            idx = self.d2idx[d].copy()
            if self.max_stocks is not None and len(idx) > self.max_stocks:
                idx = np.random.choice(idx, size=self.max_stocks, replace=False)
            yield idx.tolist()

    def __len__(self):
        return len(self.dates)


# ================= 指标/损失：Daily RankIC + 可选 Pairwise Rank Loss（保留） =================
def rank_normalize(x: torch.Tensor) -> torch.Tensor:
    order = torch.argsort(x)
    ranks = torch.empty_like(order, dtype=torch.float32)
    ranks[order] = torch.arange(len(x), device=x.device, dtype=torch.float32)
    ranks = (ranks - ranks.mean()) / (ranks.std(unbiased=False) + 1e-8)
    return ranks


@torch.no_grad()
def evaluate_daily_rank_ic(model: nn.Module, loader: DataLoader, device: torch.device):
    model.eval()

    daily_ic = []
    all_preds = []
    all_targets = []

    for x, y, d in loader:
        x = x.to(device)
        y = y.to(device)

        pred = model(x)

        all_preds.append(pred.detach().cpu())
        all_targets.append(y.detach().cpu())

        if pred.numel() >= 2 and pred.std() > 1e-8 and y.std() > 1e-8:
            ic = torch.corrcoef(torch.stack([pred, y]))[0, 1].item()
        else:
            ic = 0.0
        daily_ic.append(ic)

    all_preds = torch.cat(all_preds)
    all_targets = torch.cat(all_targets)
    mse = F.mse_loss(all_preds, all_targets).item()
    mean_ic = float(np.mean(daily_ic)) if len(daily_ic) else 0.0
    return mse, mean_ic, float(all_preds.mean().item()), float(all_preds.std().item())

class DownstreamRegModel(nn.Module):
    def __init__(
        self,
        stage1_tokenizer,
        stage2_model,
        d_model,
        cde_input_dim,
        cde_hidden_dim=256,
        cde_out_dim=128,
        pooling="mean",
        head_hidden=0,
        head_dropout=0.1,
        freeze_stage1=True,
        freeze_stage2=True,
    ):
        super().__init__()

        self.stage1 = stage1_tokenizer
        self.stage2 = stage2_model
        self.pooling = pooling

        if freeze_stage1:
            for p in self.stage1.parameters():
                p.requires_grad = False
            self.stage1.eval()

        if freeze_stage2:
            for p in self.stage2.parameters():
                p.requires_grad = False
            self.stage2.eval()

        # ===== Neural CDE 分支 =====x  
        self.cde_block = CDEBlock(
            input_dim=cde_input_dim,
            hidden_dim=cde_hidden_dim,
            out_dim=cde_out_dim,
        )

        # ===== Head =====
        self.head = SmallRegHead(
            d_model=d_model + cde_out_dim,
            hidden=head_hidden,
            dropout=head_dropout,
        )

    def extract_tokens(self, x):
        z_e = self.stage1.encode(x)
        _, _, idx = self.stage1.quantize(z_e)
        return idx

    def forward(self, x):
        # ===== 预训练分支 =====
        with torch.no_grad():
            tokens = self.extract_tokens(x)
            hidden = stage2_forward_hidden(self.stage2, tokens)

        if self.pooling == "last":
            h_pre = hidden[:, -1, :]
        else:
            h_pre = hidden.mean(dim=1)

        h_pre = F.layer_norm(h_pre, h_pre.shape[-1:])

        # ===== CDE 分支 =====
        h_cde = self.cde_block(x)   # [B, cde_out_dim]
 
        # ===== 拼接 =====
        feat = torch.cat([h_pre, h_cde], dim=-1)
        
        return self.head(feat)

# ================= 小回归头 + 包装模型（只定义 head/训练，不重定义 stage1/2） =================
class SmallRegHead(nn.Module):
    """小回归头：feat[B,D] -> pred[B]"""
    def __init__(self, d_model: int, hidden: int = 0, dropout: float = 0.1):
        super().__init__()
        if hidden and hidden > 0:
            self.net = nn.Sequential(
                nn.LayerNorm(d_model),
                nn.Linear(d_model, hidden),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden, 1),
            )
        else:
            self.net = nn.Sequential(
                nn.LayerNorm(d_model),
                nn.Dropout(dropout),
                nn.Linear(d_model, 1),
            )

    def forward(self, feat):
        return self.net(feat).squeeze(-1)


def infer_num_layers_from_state_dict(sd: dict) -> int:
    idxs = []
    for k in sd.keys():
        if k.startswith("layers."):
            try:
                idxs.append(int(k.split(".")[1]))
            except Exception:
                pass
    return (max(idxs) + 1) if idxs else 0


@torch.no_grad()
def stage2_forward_hidden(stage2_model, tokens_inp: torch.Tensor) -> torch.Tensor:
    """
    不重定义 Stage2 的情况下，从它内部拿 hidden。
    兼容两种：
    1) Stage2 自带 forward_hidden() 方法
    2) 否则按你 TokenPredictorUsingCodebook 的成员名计算：token_emb/layers/norm/causal_mask
    """
    if hasattr(stage2_model, "forward_hidden"):
        return stage2_model.forward_hidden(tokens_inp, mask_prob=0.0, keep_last_k=0)

    required = ["token_emb", "layers", "norm", "causal_mask"]
    for name in required:
        if not hasattr(stage2_model, name):
            raise RuntimeError(f"Stage2 model 缺少属性 {name}，请在 Stage2 类里实现 forward_hidden() 以供下游使用。")

    x = stage2_model.token_emb(tokens_inp)  # [B,T,D]
    T = tokens_inp.size(1)
    attn_mask = stage2_model.causal_mask(T, tokens_inp.device)
    for layer in stage2_model.layers:
        x = layer(x, mask=attn_mask)
    x = stage2_model.norm(x)               # [B,T,D]
    return x


# ================= 主训练流程 =================
def main():
    set_seed(421)
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)

    parser.add_argument("--stage1_ckpt", type=str, required=True)
    parser.add_argument("--stage2_ckpt", type=str, required=True)

    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--weight_decay", type=float, default=0.2)
    parser.add_argument("--grad_clip", type=float, default=1.0)

    parser.add_argument("--pooling", type=str, default="mean", choices=["last", "mean"])
    parser.add_argument("--head_hidden", type=int, default=0, help="0=linear; >0=MLP hidden dim")
    parser.add_argument("--head_dropout", type=float, default=0.2)

    parser.add_argument("--alpha", type=float, default=0.1, help="loss=alpha*MSE+(1-alpha)*rankloss; alpha=1 -> pure MSE")
    parser.add_argument("--tau", type=float, default=1.0)
    parser.add_argument("--max_pairs", type=int, default=20000)

    parser.add_argument("--max_stocks", type=int, default=None)
    parser.add_argument("--limit", type=int, default=None)

    # 关键：Stage2 初始化需要 time_decay/num_layers 等（因为 stage2_ckpt 通常只存 state_dict）
    parser.add_argument("--time_decay", type=float, default=0.1, help="必须与预训练 Stage2 一致")
    parser.add_argument("--num_layers", type=int, default=0, help="0=自动从 ckpt 推断；否则手动指定")
    parser.add_argument("--dropout", type=float, default=0.1)

    parser.add_argument("--save_path", type=str, default="./models/downstream_reg_head_best050.pth")

    # ===== 新增：目标定义（收益率 close_ratio）=====
    parser.add_argument("--target_mode", type=str, default="key", choices=["key", "close_ratio"])
    parser.add_argument("--target_key", type=str, default="y", help="target_mode=key 时使用")
    parser.add_argument("--close_today_key", type=str, default="close_today")
    parser.add_argument("--close_future_key", type=str, default="close_future")
    parser.add_argument("--label_clip", type=float, default=LABEL_CLIP, help="label 裁剪范围，默认仍是 ±5")

    args = parser.parse_args()
    cfg = load_config(args.config)

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    SEQ_LEN = cfg["seq_len"]
    DATA_DIR = os.path.join(cfg["paths"]["data_root"], f"sp_seq{SEQ_LEN}_n{cfg['label_horizon']}")

    # 兼容你说的 train_label.npz
    TRAIN_PATH = pick_existing_path(DATA_DIR, ["train_label.npz", "train_labeled.npz", "train.npz"])
    VAL_PATH = pick_existing_path(DATA_DIR, ["val_label.npz", "val_labeled.npz", "val.npz"])

    # -------- 数据 --------
    train_ds = RegressionDataset(
        TRAIN_PATH,
        target_key=args.target_key,
        date_key=DATE_KEY,
        limit=args.limit,
        label_clip=args.label_clip,
        target_mode=args.target_mode,
        close_today_key=args.close_today_key,
        close_future_key=args.close_future_key,
    )
    val_ds = RegressionDataset(
        VAL_PATH,
        target_key=args.target_key,
        date_key=DATE_KEY,
        limit=args.limit,
        label_clip=args.label_clip,
        target_mode=args.target_mode,
        close_today_key=args.close_today_key,
        close_future_key=args.close_future_key,
    )

    train_sampler = TimepointBatchSampler(train_ds.date, shuffle_time=True, min_stocks=2, max_stocks=args.max_stocks)
    val_sampler = TimepointBatchSampler(val_ds.date, shuffle_time=False, min_stocks=2, max_stocks=args.max_stocks)

    train_loader = DataLoader(train_ds, batch_sampler=train_sampler, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_sampler=val_sampler, num_workers=0, pin_memory=True)

    # -------- Stage1 --------
    vqvae = VQVAE_Transformer(
        input_dim=cfg["input_dim"],
        seq_len=SEQ_LEN,
        d_model=cfg["model"]["d_model"],
        nhead=cfg["model"]["n_head"],
        num_encoder_layers=cfg["model"]["num_encoder_layers"],
        num_decoder_layers=cfg["model"]["num_decoder_layers"],
        dim_feedforward=cfg["model"]["dim_feedforward"],
        num_embeddings=cfg["model"]["num_embeddings"],
        embedding_dim=cfg["model"]["embedding_dim"],
        commitment_cost=cfg["vq"]["commitment_cost"],
        ema_decay=cfg["vq"]["ema_decay"],
        ema_epsilon=cfg["vq"]["ema_eps"],
        rotary_dim=cfg["model"].get("rotary_dim", None),
    ).to(DEVICE)
    vqvae.load_state_dict(torch.load(args.stage1_ckpt, map_location=DEVICE), strict=True)
    vqvae.eval()

    # codebook 给 Stage2 用
    codebook = vqvae.vq_layer.embedding.weight.detach().clone().to("cpu")
    K, D = int(codebook.shape[0]), int(codebook.shape[1])

    # -------- Stage2 --------
    stage2_sd = torch.load(args.stage2_ckpt, map_location="cpu")
    if isinstance(stage2_sd, dict) and "state_dict" in stage2_sd:
        stage2_sd = stage2_sd["state_dict"]

    num_layers = args.num_layers if args.num_layers > 0 else infer_num_layers_from_state_dict(stage2_sd)
    if num_layers <= 0:
        raise RuntimeError("无法从 stage2_ckpt 推断 num_layers，请用 --num_layers 手动指定。")

    stage2 = TokenPredictorUsingCodebook(
        codebook_weight=codebook,
        nhead=cfg["model"]["n_head"],
        num_layers=num_layers,
        dim_feedforward=cfg["model"]["dim_feedforward"],
        dropout=args.dropout,
        rotary_dim=cfg["model"].get("rotary_dim", None),
        time_decay=args.time_decay,
    ).to(DEVICE)
    stage2.load_state_dict(stage2_sd, strict=True)
    stage2.eval()
    cde_hidden_dim =256
    cde_out_dim =64
    # -------- Downstream Model: 冻结 stage1+stage2，只训 head --------
    model = DownstreamRegModel(
        stage1_tokenizer=vqvae,
        stage2_model=stage2,
        d_model=D,                       # Stage2 hidden dim
        cde_input_dim=cfg["input_dim"],  # = X 的 feature 维度 F
        cde_hidden_dim=cde_hidden_dim,              # CDE 内部状态（建议 ≥ d_model）
        cde_out_dim= cde_out_dim,                 # compress 后维度（64 or 128 推荐）
        pooling=args.pooling,
        head_hidden=args.head_hidden,
        head_dropout=args.head_dropout,
        freeze_stage1=True,
        freeze_stage2=True,
    ).to(DEVICE)
    optimizer = torch.optim.AdamW(
        list(model.cde_block.parameters()) + list(model.head.parameters()), 
        lr=args.lr, weight_decay=args.weight_decay)
    criterion = nn.SmoothL1Loss(beta=1.0)

    best_val_ic = -1e9

    print("=" * 80)
    print("Downstream small head training")
    print(f"Train: {TRAIN_PATH}")
    print(f"Val:   {VAL_PATH}")
    print(f"Target: mode={args.target_mode} | clip=[{-args.label_clip}, {args.label_clip}]")
    if args.target_mode == "key":
        print(f"Target key: {args.target_key}")
    else:
        print(f"close_today_key={args.close_today_key} close_future_key={args.close_future_key} -> y=close_future/close_today-1")
    print(f"Stage1: {args.stage1_ckpt}")
    print(f"Stage2: {args.stage2_ckpt}")
    print(f"Stage2 num_layers={num_layers} time_decay={args.time_decay}")
    print(f"Head: pooling={args.pooling} head_hidden={args.head_hidden} dropout={args.head_dropout}")
    print("=" * 80)

    for epoch in range(1, args.epochs + 1):
        model.train()
        # 保持 stage1/stage2 eval（冻结）
        model.stage1.eval()
        model.stage2.eval()

        train_loss = 0.0
        train_ic_sum = 0.0
        steps = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}")
        for x, y, d in pbar:
            x = x.to(DEVICE)
            y = y.to(DEVICE)

            pred = model(x)
            loss = criterion(pred, y)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if args.grad_clip and args.grad_clip > 0:
                nn.utils.clip_grad_norm_(model.head.parameters(), args.grad_clip)
            optimizer.step()

            train_loss += float(loss.item())
            steps += 1

            with torch.no_grad():
                if pred.numel() >= 2 and pred.std() > 1e-8 and y.std() > 1e-8:
                    p = rank_normalize(pred)
                    yy = rank_normalize(y)
                    batch_ic = torch.corrcoef(torch.stack([p, yy]))[0, 1].item()
                else:
                    batch_ic = 0.0
                train_ic_sum += batch_ic

            pbar.set_postfix(loss=f"{loss.item():.4f}", ic=f"{batch_ic:.4f}", n=int(pred.numel()))

        avg_train_loss = train_loss / max(steps, 1)
        avg_train_ic = train_ic_sum / max(steps, 1)

        val_mse, val_ic, val_pred_mean, val_pred_std = evaluate_daily_rank_ic(model, val_loader, DEVICE)

        print(f"\nEpoch {epoch} Result:")
        print(f"  Train Loss: {avg_train_loss:.6f} | Train Daily RankIC(mean): {avg_train_ic:.6f}")
        print(f"  Val MSE:    {val_mse:.6f}      | Val Daily RankIC(mean):   {val_ic:.6f}")
        print(f"  Val pred mean/std: {val_pred_mean:.6f} / {val_pred_std:.6f}")

        if val_ic > best_val_ic:
            best_val_ic = val_ic
            torch.save({
        
                "head": model.head.state_dict(),
                "cde": model.cde_block.state_dict(),

                # ===== 结构超参数（必须）=====
                "cde_hidden_dim": cde_hidden_dim,
                "cde_out_dim": cde_out_dim,
                "pooling": args.pooling,

                # ===== 预训练依赖（引用而不是拷贝）=====
                "stage1_ckpt": args.stage1_ckpt,
                "stage2_ckpt": args.stage2_ckpt,
                "stage2_num_layers": num_layers,
                "time_decay": args.time_decay,

                # ===== 训练语义 =====
                "target_mode": args.target_mode,
                "label_clip": args.label_clip,
            }, args.save_path)
            print(f"  [New Best] saved to {args.save_path} | best_val_ic={best_val_ic:.6f}")

        print("-" * 80)

    print("Done.")


if __name__ == "__main__":
    main()
