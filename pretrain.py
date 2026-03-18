import os
import json
import argparse
import logging
from datetime import datetime

import numpy as np
from tqdm import tqdm

import torch 
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from tokenizer_model import VQVAE_Transformer, TransformerEncoderLayerRoPE

# python pretrain.py --config ./config/20_casual.json --time_decay 0.15 --mask_prob 0.n2   --keep_last_k 1
def load_config(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


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


def fmt_float(x, nd=3):
    # 文件名安全：0.1 -> 0p100
    return f"{float(x):.{nd}f}".replace(".", "p")


def setup_logger(log_path: str):
    os.makedirs(os.path.dirname(log_path), exist_ok=True)

    logger = logging.getLogger("train_stage2_tokenpred")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    fmt = logging.Formatter("[%(asctime)s] %(levelname)s - %(message)s")

    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setLevel(logging.INFO)
    fh.setFormatter(fmt)

    sh = logging.StreamHandler()
    sh.setLevel(logging.INFO)
    sh.setFormatter(fmt)

    logger.addHandler(fh)
    logger.addHandler(sh)

    return logger


class TimeSeriesDataset(Dataset):
    def __init__(self, x_array):
        self.x = x_array  # [N,S,F]

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        return torch.from_numpy(self.x[idx]).float()

@torch.no_grad()
def topk_acc_from_logits(logits, tgt, k=10):
    # logits: [B,T,K], tgt: [B,T]
    topk = torch.topk(logits, k=k, dim=-1).indices  # [B,T,k]
    hit = (topk == tgt.unsqueeze(-1)).any(dim=-1).float().mean().item()
    return hit

class TokenDataset(Dataset):
    def __init__(self, tokens):
        self.tokens = tokens  # [N,S]

    def __len__(self):
        return self.tokens.shape[0]

    def __getitem__(self, idx):
        return torch.from_numpy(self.tokens[idx]).long()
    
def unigram_baselines(train_tokens, val_tokens, K):
    # next-token 任务：tgt 是 val_tokens[:,1:]
    tgt = val_tokens[:, 1:].reshape(-1)

    # 训练集 unigram 分布（用作 baseline 概率）
    train_flat = train_tokens.reshape(-1)
    counts = np.bincount(train_flat, minlength=K).astype(np.float64)
    p = counts / (counts.sum() + 1e-12)

    # most-frequent baseline acc
    most_id = int(p.argmax())
    mf_acc = float((tgt == most_id).mean())

    # unigram cross-entropy baseline (nats)
    eps = 1e-12
    uni_ce = float((-np.log(p[tgt] + eps)).mean())
    uni_ppl = float(np.exp(uni_ce))

    # 统计分布偏斜
    top1_freq = float(p[most_id])
    top10_freq = float(np.sort(p)[-10:].sum())
    eff_vocab_ppl = float(np.exp(-np.sum(p * np.log(p + eps))))  # exp(entropy)

    return {
        "mf_acc": mf_acc,
        "unigram_ce": uni_ce,
        "unigram_ppl": uni_ppl,
        "top1_freq": top1_freq,
        "top10_mass": top10_freq,
        "effective_vocab(exp_entropy)": eff_vocab_ppl
    }

@torch.no_grad()
def tokenize_by_stage1(tokenizer, x_array, batch_size, device):
    tokenizer.eval()

    ds = TimeSeriesDataset(x_array)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, drop_last=False, num_workers=0)

    chunks = []
    for xb in tqdm(loader, desc="Tokenizing", leave=False):
        xb = xb.to(device)
        z_e = tokenizer.encode(xb)
        vq_loss, z_q, idx = tokenizer.quantize(z_e)  # idx: [B,S]
        chunks.append(idx.cpu().numpy().astype(np.int32))

    tokens = np.concatenate(chunks, axis=0)
    return tokens


class TokenPredictorUsingCodebook(nn.Module):
    """
    Stage2: token language model
    - 输入：历史 token 序列
    - 随机掩码：把部分 token 的 embedding 替换为 learnable mask_emb
    - 输出：预测下一天 token 的分布
    - token embedding：直接使用 Stage1 的 codebook 向量（冻结）
    """
    def __init__(self, codebook_weight, nhead, num_layers, dim_feedforward, dropout,
                 rotary_dim=None, time_decay=0.0):
        super().__init__()
        self.time_decay = float(time_decay)

        # codebook_weight: [K, D]
        self.K = int(codebook_weight.shape[0])
        self.D = int(codebook_weight.shape[1])

        # 用 Stage1 codebook 作为 embedding（冻结）
        self.token_emb = nn.Embedding(self.K, self.D)
        self.token_emb.weight.data.copy_(codebook_weight)
        self.token_emb.weight.requires_grad = False

        # learnable mask embedding
        self.mask_emb = nn.Parameter(torch.zeros(1, 1, self.D))

        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            layer = TransformerEncoderLayerRoPE(
                d_model=self.D,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                rotary_dim=rotary_dim,
                activation="relu",
            )
            self.layers.append(layer)

        self.norm = nn.LayerNorm(self.D)
        self.out = nn.Linear(self.D, self.K)

    def causal_mask(self, T, device):
        # 1) 因果 mask：禁止看未来
        mask = torch.full((T, T), float("-inf"), device=device)
        mask = torch.triu(mask, diagonal=1)

 
        if self.time_decay > 0:
            idx = torch.arange(T, device=device)
            dist = idx[:, None] - idx[None, :]   # i - j
            dist = torch.clamp(dist, min=0).float()
            bias = - self.time_decay * dist
            mask = mask + bias

        return mask

        

    def forward(self, tokens_inp, mask_prob, keep_last_k=0):
        """
        tokens_inp: [B, T]
        keep_last_k: 最后 k 个 token 不掩码（更贴近“只掩更早历史”）
        """
        B = tokens_inp.size(0)
        T = tokens_inp.size(1)

        x = self.token_emb(tokens_inp)  # [B,T,D]

        if self.training and mask_prob > 0:
            rand = torch.rand(B, T, device=tokens_inp.device)
            drop = rand < mask_prob

            # 只允许掩“更早历史”：最后 keep_last_k 个位置不掩
            if keep_last_k > 0 and T > keep_last_k:
                drop[:, T - keep_last_k:] = False
            else:
                # 序列太短就不掩（保持你原来的“稳”）
                drop[:] = False

            # （可选）第一个位置也不掩
            if T > 0:
                drop[:, 0] = False

            x = x.clone()
            x[drop] = self.mask_emb[0, 0]

        attn_mask = self.causal_mask(T, tokens_inp.device)
        for layer in self.layers:
            x = layer(x, mask=attn_mask)

        x = self.norm(x)
        logits = self.out(x)  # [B,T,K]
        return logits


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="vqvae_config.json")
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-4)

    # 掩码相关
    parser.add_argument("--mask_prob", type=float, default=0.2)
    parser.add_argument("--keep_last_k", type=int, default=1, help="last k positions will NOT be masked")

    # 模型结构相关
    parser.add_argument("--num_layers", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--grad_clip", type=float, default=1.0)

    # 时间衰减相关（原来写死0.1）
    parser.add_argument("--time_decay", type=float, default=0.1)

    # token cache   
    parser.add_argument("--rebuild_cache", action="store_true")
    args = parser.parse_args()
    print(args.time_decay)
    cfg = load_config(args.config)
    set_seed(119)

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- 路径（与你 Stage1 tokenizer_train.py 完全一致）---
    SEQ_LEN = cfg["seq_len"]
    LABEL_HORIZON = cfg["label_horizon"]

    DATA_ROOT = cfg["paths"]["data_root"]
    MODEL_SAVE_DIR = cfg["paths"]["model_save_dir"]
    MODEL_NAME_SUFFIX = cfg["paths"]["model_name_suffix"]

    DATA_DIR = os.path.join(DATA_ROOT, f"sp_seq{SEQ_LEN}_n{LABEL_HORIZON}")
    TRAIN_NPZ_PATH = os.path.join(DATA_DIR, "train.npz")
    VAL_NPZ_PATH = os.path.join(DATA_DIR, "val.npz")

    os.makedirs(MODEL_SAVE_DIR, exist_ok=True)

    MODEL_NAME_BASE = f"seq{SEQ_LEN}_n{LABEL_HORIZON}_{MODEL_NAME_SUFFIX}"
    STAGE1_BEST_PATH = os.path.join(MODEL_SAVE_DIR, MODEL_NAME_BASE + "_best.pth")

    # ---- 保存命名包含 time_decay / mask_prob / keep_last_k ----
    exp_tag = f"td{fmt_float(args.time_decay,3)}_mp{fmt_float(args.mask_prob,3)}_klk{int(args.keep_last_k)}"
    STAGE2_BEST_PATH = os.path.join(MODEL_SAVE_DIR, f"{MODEL_NAME_BASE}_tokenpred_codebook_{exp_tag}_best.pth")
    STAGE2_LAST_PATH = os.path.join(MODEL_SAVE_DIR, f"{MODEL_NAME_BASE}_tokenpred_codebook_{exp_tag}_last.pth")

    TOKEN_CACHE_TRAIN = os.path.join("./data", f"{SEQ_LEN}train_tokens_{MODEL_NAME_BASE}.npz")
    TOKEN_CACHE_VAL = os.path.join("./data", f"{SEQ_LEN}val_tokens_{MODEL_NAME_BASE}.npz")

    # ---- 日志文件 ----
    log_path = os.path.join(MODEL_SAVE_DIR, f"{MODEL_NAME_BASE}_tokenpred_codebook_{exp_tag}.log")
    logger = setup_logger(log_path)

    logger.info("=" * 80)
    logger.info("Stage2 Token Predictor (Codebook Embedding) - Training Start")
    logger.info(f"RunTime: {datetime.now().isoformat()}")
    logger.info(f"Config: {args.config}")
    logger.info(f"Seed: {args.seed}")
    logger.info(f"DEVICE: {DEVICE}")
    logger.info(f"SEQ_LEN={SEQ_LEN}, LABEL_HORIZON={LABEL_HORIZON}, MODEL_NAME_BASE={MODEL_NAME_BASE}")
    logger.info(f"time_decay={args.time_decay}, mask_prob={args.mask_prob}, keep_last_k={args.keep_last_k}")
    logger.info(f"Save BEST: {STAGE2_BEST_PATH}")
    logger.info(f"Save LAST: {STAGE2_LAST_PATH}")
    logger.info(f"Log file: {log_path}")
    logger.info("=" * 80)

    if not os.path.exists(STAGE1_BEST_PATH):
        raise FileNotFoundError("找不到 Stage1 best: " + STAGE1_BEST_PATH)

    # --- 读取数据 ---
    train_npz = np.load(TRAIN_NPZ_PATH)
    val_npz = np.load(VAL_NPZ_PATH)
    X_train = train_npz["X"]
    X_val = val_npz["X"]

    INPUT_DIM = cfg["input_dim"]
    D_MODEL = cfg["model"]["d_model"]
    N_HEAD = cfg["model"]["n_head"]
    DIM_FEEDFORWARD = cfg["model"]["dim_feedforward"]
    ROTARY_DIM = cfg["model"].get("rotary_dim", None)

    NUM_EMBEDDINGS = cfg["model"]["num_embeddings"]
    EMBEDDING_DIM = cfg["model"]["embedding_dim"]

    # --- 加载 Stage1 tokenizer（只为了取 codebook 和做 tokenization）---
    tokenizer = VQVAE_Transformer(
        input_dim=INPUT_DIM,
        seq_len=SEQ_LEN,
        d_model=D_MODEL,
        nhead=N_HEAD,
        num_encoder_layers=cfg["model"]["num_encoder_layers"],
        num_decoder_layers=cfg["model"]["num_decoder_layers"],
        dim_feedforward=DIM_FEEDFORWARD,
        num_embeddings=NUM_EMBEDDINGS,
        embedding_dim=EMBEDDING_DIM,
        commitment_cost=cfg["vq"]["commitment_cost"],
        ema_decay=cfg["vq"]["ema_decay"],
        ema_epsilon=cfg["vq"]["ema_eps"],
        dropout=float(cfg["model"].get("dropout", 0.1)),
        rotary_dim=ROTARY_DIM,
        activation="relu",
    ).to(DEVICE)

    state = torch.load(STAGE1_BEST_PATH, map_location=DEVICE)
    tokenizer.load_state_dict(state, strict=True)
    tokenizer.eval()
    for p in tokenizer.parameters():
        p.requires_grad = False

    # --- 取 Stage1 codebook 向量 ---
    codebook = tokenizer.vq_layer.embedding.weight.detach().clone().to("cpu")  # [K,D]
    K = int(codebook.shape[0])
    D = int(codebook.shape[1])

    logger.info("Stage1 best: %s", STAGE1_BEST_PATH)
    logger.info("Codebook shape: (%d, %d)", K, D)
    logger.info("epochs=%d batch_size=%d lr=%g weight_decay=%g dropout=%g num_layers=%d",
                args.epochs, args.batch_size, args.lr, args.weight_decay, args.dropout, args.num_layers)

    # 检查 head_dim 是否可用（RoPE 要 head_dim 为偶数）
    if D % N_HEAD != 0:
        raise ValueError(f"codebook_dim 不能被 n_head 整除。codebook_dim={D} n_head={N_HEAD}")
    head_dim = D // N_HEAD
    if head_dim % 2 != 0:
        raise ValueError(f"head_dim 必须为偶数。head_dim={head_dim}")

    # --- token 缓存 ---
    def load_or_make(cache_path, X):
        if (not args.rebuild_cache) and os.path.exists(cache_path):
            obj = np.load(cache_path)
            return obj["tokens"]
        toks = tokenize_by_stage1(tokenizer, X, batch_size=max(128, args.batch_size), device=DEVICE)
        np.savez_compressed(cache_path, tokens=toks.astype(np.int32))
        return toks

    logger.info("Prepare token cache ... rebuild_cache=%s", args.rebuild_cache)
    train_tokens = load_or_make(TOKEN_CACHE_TRAIN, X_train)
    val_tokens = load_or_make(TOKEN_CACHE_VAL, X_val)
    K = codebook.shape[0]  # 你的总码本大小
    print(unigram_baselines(train_tokens, val_tokens, K))
    
    logger.info("train_tokens: %s | val_tokens: %s", str(train_tokens.shape), str(val_tokens.shape))

    # --- Stage2 predictor（用 codebook 当 embedding）---
    predictor = TokenPredictorUsingCodebook(
        codebook_weight=codebook,
        nhead=N_HEAD,
        num_layers=args.num_layers,
        dim_feedforward=DIM_FEEDFORWARD,
        dropout=args.dropout,
        rotary_dim=ROTARY_DIM,
        time_decay=args.time_decay
    ).to(DEVICE)

    optimizer = torch.optim.AdamW(predictor.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    train_loader = DataLoader(TokenDataset(train_tokens), batch_size=args.batch_size,
                              shuffle=True, drop_last=True, num_workers=0)
    val_loader = DataLoader(TokenDataset(val_tokens), batch_size=args.batch_size,
                            shuffle=False, drop_last=False, num_workers=0)

    best_val = 1e18

    def run_one_epoch(loader, is_train):
        predictor.train() if is_train else predictor.eval()

        total_loss = 0.0
        total_acc = 0.0
        count = 0

        for tok in tqdm(loader, desc=("Train" if is_train else "Val"), leave=False):
            tok = tok.to(DEVICE)  # [B,S]

            inp = tok[:, :-1]  # [B,S-1]
            tgt = tok[:, 1:]   # [B,S-1]

            if is_train:
                logits = predictor(inp, mask_prob=args.mask_prob, keep_last_k=args.keep_last_k)
            else:
                logits = predictor(inp, mask_prob=0.0, keep_last_k=args.keep_last_k)

            B = logits.size(0)
            T = logits.size(1)

            logits2 = logits.reshape(B * T, K)
            tgt2 = tgt.reshape(B * T)

            loss = F.cross_entropy(logits2, tgt2)

            with torch.no_grad():
                pred = torch.argmax(logits, dim=-1)
                acc = (pred == tgt).float().mean().item()

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                if args.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(predictor.parameters(), args.grad_clip)
                optimizer.step()

            total_loss += float(loss.item())
            total_acc += float(acc)
            count += 1

        if count == 0:
            return 0.0, 0.0
        return total_loss / count, total_acc / count

    logger.info("Start training ...")
    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = run_one_epoch(train_loader, True)
        val_loss, val_acc = run_one_epoch(val_loader, False)

        msg = (f"Epoch {epoch:03d}/{args.epochs:03d} | "
               f"train_loss={train_loss:.6f} train_acc={train_acc:.6f} | "
               f"val_loss={val_loss:.6f} val_acc={val_acc:.6f}")
        logger.info(msg)

        # 每轮都保存 last（包含 exp_tag）
        torch.save(predictor.state_dict(), STAGE2_LAST_PATH)

        if val_loss < best_val:
            best_val = val_loss
            torch.save(predictor.state_dict(), STAGE2_BEST_PATH)
            logger.info("New BEST saved: %s | best_val_loss=%.6f", STAGE2_BEST_PATH, best_val)

    logger.info("Done.")
    logger.info("Best: %s", STAGE2_BEST_PATH)
    logger.info("Last: %s", STAGE2_LAST_PATH)
    logger.info("Token cache: %s | %s", TOKEN_CACHE_TRAIN, TOKEN_CACHE_VAL)
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
