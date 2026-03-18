# pre_fune_backtest_npzonly.py
# ------------------------------------------------------------
# 1) 从归一化后的测试集 npz（test_labeled.npz）读取 X / times / codes
# 2) 用 Stage1(VQVAE) + Stage2(TokenPredictor) + 训练好的 head 推理得到 score
# 3) 直接用 npz 内的 close_future/close_today-1 作为次日收益 ret
# 4) 每天选 TopK（默认50），等权收益 -> 累计收益/Sharpe
# 5) 指标：
#    - RankIC（score vs 次日收益，Spearman，按天）
#    - 方向准确率（sign(pred) vs sign(true_label)），输出：全截面 & TopK
#
# 兼容 Python 3.8+
# ------------------------------------------------------------

import os
import json
import argparse
from typing import Optional, Dict, List
import torch.nn.functional as F
import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Sampler
from pretrain_cde_fineturn import SmallRegHead,DownstreamRegModel
# ====== Stage1/Stage2：只 import，不重定义 ======
from tokenizer_model import VQVAE_Transformer,CDEBlock
# TODO: 改成你真实 Stage2 定义文件
from pretrain import TokenPredictorUsingCodebook  # <- 你需要改这里

# ---------- npz keys ----------
DEFAULT_TIME_KEY = "times"
DEFAULT_CODE_KEY = "codes"
DEFAULT_TVALUE_KEY = "t_value_7d"
DEFAULT_CLOSE_TODAY_KEY = "close_today"
DEFAULT_CLOSE_FUTURE_KEY = "close_future"


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def to_yyyymmdd_int(x) -> int:
    """支持 int(YYYYMMDD) / 'YYYY-MM-DD' / 'YYYY/MM/DD' / 'YYYYMMDD' 等，取前8位数字"""
    s = str(x)
    digits = "".join(ch for ch in s if ch.isdigit())
    if len(digits) < 8:
        raise ValueError(f"Cannot parse date: {x}")
    return int(digits[:8])


def normalize_symbol(x) -> str:
    """统一 codes/symbol 格式"""
    return str(x).strip().upper()


def strip_prefix(sd: dict, prefix: str = "module.") -> dict:
    """处理 DDP 保存的 module.xxx"""
    if not any(k.startswith(prefix) for k in sd.keys()):
        return sd
    return {k[len(prefix):]: v for k, v in sd.items()}


# ================= Dataset / Sampler（按天batch） =================
class TestDatasetNPZ(Dataset):
    """
    从 test_labeled.npz 读取：
      X: [N,S,F]
      times: [N]
      codes: [N]
      close_today: [N]
      close_future: [N]
      (可选) t_value_7d: [N] 仅用于额外分析/对比

    返回固定 tuple：
      (X, y_label, ret, tvalue, code, day)

    - y_label: 你用于“方向准确率”的标签，默认用 ret（收益率）
    - ret:     次日收益率 close_future/close_today-1（用于回测与RankIC）
    - tvalue:  若npz里有就提供，否则为nan
    """
    def __init__(
        self,
        npz_path: str,
        time_key: str = DEFAULT_TIME_KEY,
        code_key: str = DEFAULT_CODE_KEY,
        tvalue_key: str = DEFAULT_TVALUE_KEY,
        close_today_key: str = DEFAULT_CLOSE_TODAY_KEY,
        close_future_key: str = DEFAULT_CLOSE_FUTURE_KEY,
        label_mode: str = "ret",  # "ret" 或 "tvalue"
        limit: Optional[int] = None,
        eps: float = 1e-12,
    ):
        pack = np.load(npz_path, allow_pickle=True)

        required = ["X", time_key, code_key, close_today_key, close_future_key]
        for k in required:
            if k not in pack:
                raise ValueError(f"'{k}' not found in {npz_path}. Available keys: {list(pack.keys())}")

        X = pack["X"].astype(np.float32)
        raw_time = pack[time_key]
        raw_code = pack[code_key]

        close_today = pack[close_today_key].astype(np.float32)
        close_future = pack[close_future_key].astype(np.float32)

        # ret = close_future / close_today - 1
        ret = close_future / (close_today + eps) - 1.0

        # tvalue 可选
        if tvalue_key in pack:
            tvalue = pack[tvalue_key].astype(np.float32)
        else:
            tvalue = np.full((len(X),), np.nan, dtype=np.float32)

        # 过滤：ret 必须有限，且 close_today 非0
        valid = (
            np.isfinite(ret)
            & np.isfinite(close_today)
            & np.isfinite(close_future)
            & (np.abs(close_today) > eps)
        )

        X = X[valid]
        raw_time = raw_time[valid]
        raw_code = raw_code[valid]
        ret = ret[valid]
        tvalue = tvalue[valid]

        time_int = np.array([to_yyyymmdd_int(t) for t in raw_time], dtype=np.int64)
        code_str = np.array([normalize_symbol(c) for c in raw_code], dtype=object)

        if limit is not None:
            X = X[:limit]
            time_int = time_int[:limit]
            code_str = code_str[:limit]
            ret = ret[:limit]
            tvalue = tvalue[:limit]

        self.X = X
        self.time = time_int
        self.code = code_str
        self.ret = ret.astype(np.float32)
        self.tvalue = tvalue.astype(np.float32)

        if label_mode not in ("ret", "tvalue"):
            raise ValueError("label_mode must be 'ret' or 'tvalue'")
        self.label_mode = label_mode

        if label_mode == "ret":
            self.y_label = self.ret
        else:
            # 如果你坚持用 tvalue 做方向准确率，但数据里没有 tvalue，会得到全 nan -> 评估会变成0
            self.y_label = self.tvalue

        print(f"[DATA] Loaded test npz: {os.path.basename(npz_path)}")
        print(f"[DATA] Samples: {len(self.X)} | Unique days: {len(np.unique(self.time))}")
        print(f"[DATA] ret stats: mean={float(np.nanmean(self.ret)):.6f}, std={float(np.nanstd(self.ret)):.6f}")
        if np.isfinite(self.tvalue).any():
            print(f"[DATA] tvalue stats: mean={float(np.nanmean(self.tvalue)):.6f}, std={float(np.nanstd(self.tvalue)):.6f}")
        print(f"[DATA] label_mode={self.label_mode}")

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return (
            self.X[idx],
            self.y_label[idx],
            self.ret[idx],
            self.tvalue[idx],
            self.code[idx],
            int(self.time[idx]),
        )


class TimepointBatchSampler(Sampler):
    """一个 batch = 同一天全部股票"""
    def __init__(self, time_ids, shuffle_time=False, min_stocks=2, max_stocks=None):
        self.time_ids = np.asarray(time_ids)
        self.shuffle_time = shuffle_time
        self.min_stocks = min_stocks
        self.max_stocks = max_stocks

        self.t2idx = {}
        for t in np.unique(self.time_ids):
            idx = np.where(self.time_ids == t)[0]
            if len(idx) >= self.min_stocks:
                self.t2idx[int(t)] = idx

        self.times = list(self.t2idx.keys())

    def __iter__(self):
        times = self.times.copy()
        if self.shuffle_time:
            np.random.shuffle(times)
        for t in times:
            idx = self.t2idx[t].copy()
            if self.max_stocks is not None and len(idx) > self.max_stocks:
                idx = np.random.choice(idx, size=self.max_stocks, replace=False)
            yield idx.tolist()

    def __len__(self):
        return len(self.times)


def batch_day_to_int(day) -> int:
    """DataLoader 会把 day(int) collate 成 tensor，这里统一取第一个"""
    if torch.is_tensor(day):
        return int(day.flatten()[0].item())
    if isinstance(day, np.ndarray):
        return int(day.flatten()[0])
    if isinstance(day, (list, tuple)):
        return int(day[0])
    return int(day)


# ================= 模型封装（stage1+stage2+head） =================


def infer_num_layers_from_state_dict(sd: dict) -> int:
    idxs = []
    for k in sd.keys():
        if k.startswith("layers."):
            try:
                idxs.append(int(k.split(".")[1]))
            except Exception:
                pass
    return (max(idxs) + 1) if idxs else 0


def infer_head_hidden(head_sd: dict, d_model: int) -> int:
    """从 head state_dict 推断 hidden dim（若是线性头则返回0）"""
    # 兼容 SmallRegHead 的两种结构：
    # - Linear head: net.2.weight shape [1, d_model]
    # - MLP head:    net.1.weight shape [hidden, d_model]
    for k, v in head_sd.items():
        if not (k.endswith("weight") and isinstance(v, torch.Tensor) and v.ndim == 2):
            continue
        out_dim, in_dim = int(v.shape[0]), int(v.shape[1])
        if in_dim == d_model and out_dim != 1:
            return out_dim
    return 0


@torch.no_grad()
def stage2_forward_hidden(stage2_model, tokens_inp: torch.Tensor) -> torch.Tensor:
    """
    从 Stage2 内部拿 hidden:
    1) 如果 stage2 有 forward_hidden() 方法就用
    2) 否则尝试用 token_emb/layers/norm/causal_mask
    """
    if hasattr(stage2_model, "forward_hidden"):
        return stage2_model.forward_hidden(tokens_inp, mask_prob=0.0, keep_last_k=0)

    required = ["token_emb", "layers", "norm", "causal_mask"]
    for name in required:
        if not hasattr(stage2_model, name):
            raise RuntimeError(
                f"Stage2 model 缺少属性 {name}。请在 Stage2 类里实现 forward_hidden() 以供下游使用。"
            )

    x = stage2_model.token_emb(tokens_inp)  # [B,T,D]
    T = tokens_inp.size(1)
    attn_mask = stage2_model.causal_mask(T, tokens_inp.device)
    for layer in stage2_model.layers:
        x = layer(x, mask=attn_mask)
    x = stage2_model.norm(x)
    return x





# ================= 指标：RankIC / 方向准确率 / 回测 =================
def spearman_rank_ic(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 2:
        return 0.0
    ra = a.argsort().argsort().astype(np.float64)
    rb = b.argsort().argsort().astype(np.float64)
    ra = (ra - ra.mean()) / (ra.std(ddof=0) + 1e-12)
    rb = (rb - rb.mean()) / (rb.std(ddof=0) + 1e-12)
    return float((ra * rb).mean())


@torch.no_grad()
def evaluate_direction_accuracy(
    model: nn.Module,
    test_loader: DataLoader,
    device: torch.device,
    eps: float = 0.0,
    on_topk: Optional[int] = None,  # None=全截面；例如 50=只看Top50
) -> Dict[str, float]:
    """
    方向准确率：sign(pred) 与 sign(true_label) 是否一致
    - eps: |y|<eps 的样本剔除（eps=0 时剔除 y==0）
    - on_topk: 只在当日 topk 样本上统计（可选）
    """
    model.eval()

    total = 0
    correct = 0
    daily_acc = []

    total_pos = total_neg = 0
    correct_pos = correct_neg = 0

    for x, y_label, ret, tvalue, codes, day in tqdm(
        test_loader, desc=f"DirAcc({'ALL' if on_topk is None else 'Top'+str(on_topk)})"
    ):
        x = x.to(device)
        y_true = torch.as_tensor(y_label, device=device, dtype=torch.float32)
        pred = model(x)

        if on_topk is not None:
            k = min(int(on_topk), pred.numel())
            idx = torch.topk(pred, k=k, largest=True).indices
            pred = pred[idx]
            y_true = y_true[idx]

        if eps > 0:
            mask = torch.abs(y_true) >= eps
        else:
            mask = y_true != 0

        if mask.sum().item() < 1:
            continue

        p = pred[mask]
        t = y_true[mask]

        sp = torch.sign(p)
        st = torch.sign(t)

        ok = (sp == st)
        c = int(ok.sum().item())
        n = int(ok.numel())

        daily_acc.append(c / n)
        total += n
        correct += c

        pos_mask = st > 0
        neg_mask = st < 0
        if pos_mask.any():
            total_pos += int(pos_mask.sum().item())
            correct_pos += int((sp[pos_mask] == st[pos_mask]).sum().item())
        if neg_mask.any():
            total_neg += int(neg_mask.sum().item())
            correct_neg += int((sp[neg_mask] == st[neg_mask]).sum().item())

    overall_acc = correct / total if total > 0 else 0.0
    mean_daily_acc = float(np.mean(daily_acc)) if daily_acc else 0.0
    median_daily_acc = float(np.median(daily_acc)) if daily_acc else 0.0
    pos_acc = correct_pos / total_pos if total_pos > 0 else 0.0
    neg_acc = correct_neg / total_neg if total_neg > 0 else 0.0

    return {
        "overall_acc": float(overall_acc),
        "mean_daily_acc": float(mean_daily_acc),
        "median_daily_acc": float(median_daily_acc),
        "pos_acc": float(pos_acc),
        "neg_acc": float(neg_acc),
        "n_eval": int(total),
        "n_days": int(len(daily_acc)),
    }

class DownstreamRegModelWithCDE(nn.Module):
    def __init__(
        self,
        stage1_tokenizer,
        stage2_model,
        d_model,
        cde_input_dim,
        cde_hidden_dim,
        cde_out_dim,
        pooling="mean",
        head_hidden=0,
        head_dropout=0.0,
    ):
        super().__init__()

        self.stage1 = stage1_tokenizer
        self.stage2 = stage2_model
        self.pooling = pooling

        # 冻结预训练模型
        for p in self.stage1.parameters():
            p.requires_grad = False
        for p in self.stage2.parameters():
            p.requires_grad = False

        self.stage1.eval()
        self.stage2.eval()

        # ===== CDE 分支 =====
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

    @torch.no_grad()
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

        # ===== 拼接 + head =====
        feat = torch.cat([h_pre, h_cde], dim=-1)
        return self.head(feat)

def yyyymmdd_to_str(x: int) -> str:
    s = str(int(x))
    return f"{s[:4]}-{s[4:6]}-{s[6:8]}"

@torch.no_grad()
def backtest_topk_from_npz_ret(
    model: nn.Module,
    test_loader: DataLoader,
    device: torch.device,
    topk: int = 50,
    annual_days: int = 252,
    out_curve_csv: Optional[str] = None,
) -> Dict[str, float]:
    """
    回测（不读取原始行情）：
    - 每天截面：score
    - 次日收益：直接用 npz 的 ret（close_future/close_today-1）
    - 组合：TopK 等权
    - IC：全截面 score vs ret（Spearman）
    """
    model.eval()
    detail_rows = []
    daily_rows = []
    missing_topk_total = 0
    topk_total = 0

    for x, y_label, ret, tvalue, codes, day in tqdm(test_loader, desc="Backtest(NPZ-ret)"):
        day = batch_day_to_int(day)

        x = x.to(device)
        scores = model(x).detach().cpu().numpy()
        ret_np = np.asarray(ret, dtype=np.float64)
        codes_np = np.asarray(codes)
        valid = np.isfinite(scores) & np.isfinite(ret_np)
        if valid.sum() < 2:
            continue

        ic = spearman_rank_ic(scores[valid], ret_np[valid])

        order = np.argsort(-scores)
        k = min(int(topk), len(order))
        top_idx = order[:k]
        for i in top_idx:
            detail_rows.append({
                "date": day,
                "code": str(codes_np[i]),
                "score": float(scores[i]),
                "ret": float(ret_np[i]),
        })
        top_ret = ret_np[top_idx]
        top_valid = np.isfinite(top_ret)
        missing_topk_total += int((~top_valid).sum())
        topk_total += k

        if top_valid.sum() == 0:
            continue

        port_ret = float(np.mean(top_ret[top_valid]))
        daily_rows.append((day, port_ret, ic))

    if not daily_rows:
        raise RuntimeError(
            "没有产生任何可用回测日。请检查 test_labeled.npz 是否包含有效的 close_today/close_future，"
            "以及是否存在大量 nan/inf/0 close_today 导致被过滤。"
        )

    daily_rows.sort(key=lambda z: z[0])
    days = np.array([r[0] for r in daily_rows], dtype=np.int64)
    rets = np.array([r[1] for r in daily_rows], dtype=np.float64)
    ics = np.array([r[2] for r in daily_rows], dtype=np.float64)

    cum_curve = np.cumprod(1.0 + rets) - 1.0
    cum_ret = float(cum_curve[-1])

    mu = float(rets.mean())
    sd = float(rets.std(ddof=1)) if len(rets) > 1 else 0.0
    sharpe = (mu / (sd + 1e-12)) * np.sqrt(annual_days) if sd > 0 else 0.0

    ic_mean = float(ics.mean())
    ic_sd = float(ics.std(ddof=1)) if len(ics) > 1 else 0.0
    icir = (ic_mean / (ic_sd + 1e-12)) * np.sqrt(annual_days) if ic_sd > 0 else 0.0

    missing_rate = float(missing_topk_total / max(topk_total, 1))

    if out_curve_csv:
        # ===== 按天聚合表 =====
        daily_df = pd.DataFrame({
            "date": days,
            "daily_ret": rets,
            "cum_ret": cum_curve,
            "rank_ic": ics,
        })

        # ===== 明细表 =====
        detail_df = pd.DataFrame(detail_rows)
        daily_df["date"] = daily_df["date"].apply(yyyymmdd_to_str)
        detail_df["date"] = detail_df["date"].apply(yyyymmdd_to_str)
        # merge：给每只股票补上当天的组合收益 & IC
        out_df = detail_df.merge(daily_df, on="date", how="left")

        os.makedirs(os.path.dirname(out_curve_csv) or ".", exist_ok=True)
        out_df.to_csv(out_curve_csv, index=False, encoding="utf-8")

        print(f"[OK] Saved curve with details -> {out_curve_csv}")

    return {
        "n_days": int(len(days)),
        "cum_return": cum_ret,
        "sharpe": sharpe,
        "daily_ret_mean": mu,
        "daily_ret_std": sd,
        "rankic_mean": ic_mean,
        "rankic_std": ic_sd,
        "icir": icir,
        "topk_missing_rate": missing_rate,
    }


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--stage1_ckpt", type=str, required=True)
    parser.add_argument("--stage2_ckpt", type=str, required=True)
    parser.add_argument("--head_ckpt", type=str, required=True)

    # npz keys
    parser.add_argument("--time_key", type=str, default=DEFAULT_TIME_KEY)
    parser.add_argument("--code_key", type=str, default=DEFAULT_CODE_KEY)
    parser.add_argument("--tvalue_key", type=str, default=DEFAULT_TVALUE_KEY)
    parser.add_argument("--close_today_key", type=str, default=DEFAULT_CLOSE_TODAY_KEY)
    parser.add_argument("--close_future_key", type=str, default=DEFAULT_CLOSE_FUTURE_KEY)

    # label mode for direction accuracy
    parser.add_argument("--label_mode", type=str, default="ret", choices=["ret", "tvalue"],
                        help="方向准确率用哪个标签：ret=收益率(close_future/close_today-1)，tvalue=npz里的t_value_7d")

    # stage2 init
    parser.add_argument("--pooling", type=str, default="mean", choices=["last", "mean"])
    parser.add_argument("--time_decay", type=float, default=0.1)
    parser.add_argument("--num_layers", type=int, default=0)
    parser.add_argument("--dropout", type=float, default=0.1)

    # eval/backtest
    parser.add_argument("--topk", type=int, default=50)
    parser.add_argument("--annual_days", type=int, default=252)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--max_stocks", type=int, default=None)
    parser.add_argument("--out_curve_csv", type=str, default="./curve.csv")
    parser.add_argument("--diracc_eps", type=float, default=0.0, help="|label|<eps 视为无方向剔除；eps=0则剔除label==0")

    args = parser.parse_args()
    cfg = load_config(args.config)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ====== test_labeled.npz 路径（按你要求）======
    seq_len = cfg["seq_len"]
    data_dir = os.path.join(cfg["paths"]["data_root"], f"sp_seq{seq_len}_n{cfg['label_horizon']}")
    test_path = os.path.join(data_dir, "test_labeled.npz")
    if not os.path.exists(test_path):
        raise FileNotFoundError(f"Test npz not found: {test_path}")

    # ====== Test loader（按天 batch）======
    test_ds = TestDatasetNPZ(
        test_path,
        time_key=args.time_key,
        code_key=args.code_key,
        tvalue_key=args.tvalue_key,
        close_today_key=args.close_today_key,
        close_future_key=args.close_future_key,
        label_mode=args.label_mode,
        limit=args.limit,
    )
    test_sampler = TimepointBatchSampler(test_ds.time, shuffle_time=False, min_stocks=2, max_stocks=args.max_stocks)
    test_loader = DataLoader(test_ds, batch_sampler=test_sampler, num_workers=0, pin_memory=True)

    # ====== Load Stage1 ======
    vqvae_sd = torch.load(args.stage1_ckpt, map_location="cpu")
    if isinstance(vqvae_sd, dict) and "state_dict" in vqvae_sd:
        vqvae_sd = vqvae_sd["state_dict"]
    vqvae_sd = strip_prefix(vqvae_sd)

    vqvae = VQVAE_Transformer(
        input_dim=cfg["input_dim"],
        seq_len=seq_len,
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
    ).to(device)
    vqvae.load_state_dict(vqvae_sd, strict=True)
    vqvae.eval()

    codebook = vqvae.vq_layer.embedding.weight.detach().clone().to("cpu")
    d_model_codebook = int(codebook.shape[1])

    # ====== Load Stage2 ======
    stage2_sd = torch.load(args.stage2_ckpt, map_location="cpu")
    if isinstance(stage2_sd, dict) and "state_dict" in stage2_sd:
        stage2_sd = stage2_sd["state_dict"]
    stage2_sd = strip_prefix(stage2_sd)

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
    ).to(device)
    stage2.load_state_dict(stage2_sd, strict=True)
    stage2.eval()

    # ====== Load Head ======
    head_pack = torch.load(args.head_ckpt, map_location="cpu")

    cde_hidden_dim = head_pack["cde_hidden_dim"]
    cde_out_dim = head_pack["cde_out_dim"]
    pooling = head_pack.get("pooling", args.pooling)

    model = DownstreamRegModelWithCDE(
        stage1_tokenizer=vqvae,
        stage2_model=stage2,
        d_model=d_model_codebook,
        cde_input_dim=cfg["input_dim"],
        cde_hidden_dim=cde_hidden_dim,
        cde_out_dim=cde_out_dim,
        pooling=pooling,
        head_hidden=0,
        head_dropout=0.2,
    ).to(device)

    # 加载权重
    model.head.load_state_dict(head_pack["head"], strict=True)
    model.cde_block.load_state_dict(head_pack["cde"], strict=True)

    model.eval()

    print("=" * 80)
    print("Test metrics (NPZ only, no raw csv)")
    print(f"Test npz: {test_path}")
    print(f"close_today_key={args.close_today_key}, close_future_key={args.close_future_key} -> ret=close_future/close_today-1")
    print(f"TopK={args.topk} | pooling={pooling} | Stage2(num_layers={num_layers}, time_decay={args.time_decay})")
    print(f"DirAcc label_mode={args.label_mode} eps={args.diracc_eps}")
    print("=" * 80)

    # ====== 1) 方向准确率（基于 label_mode）======
    # acc_all = evaluate_direction_accuracy(model, test_loader, device, eps=args.diracc_eps, on_topk=None)
    # acc_topk = evaluate_direction_accuracy(model, test_loader, device, eps=args.diracc_eps, on_topk=args.topk)

    # print("\n===== Direction Accuracy =====")
    # print(f"[ALL]  overall={acc_all['overall_acc']:.4f} | mean_daily={acc_all['mean_daily_acc']:.4f} | "
    #       f"pos={acc_all['pos_acc']:.4f} | neg={acc_all['neg_acc']:.4f} | n={acc_all['n_eval']} | days={acc_all['n_days']}")
    # print(f"[Top{args.topk}] overall={acc_topk['overall_acc']:.4f} | mean_daily={acc_topk['mean_daily_acc']:.4f} | "
    #       f"pos={acc_topk['pos_acc']:.4f} | neg={acc_topk['neg_acc']:.4f} | n={acc_topk['n_eval']} | days={acc_topk['n_days']}")
    # print("==============================\n")

    # ====== 2) 回测收益 + RankIC（始终用 ret）======
    metrics = backtest_topk_from_npz_ret(
        model=model,
        test_loader=test_loader,
        device=device,
        topk=args.topk,
        annual_days=args.annual_days,
        out_curve_csv=args.out_curve_csv,
    )

    print("\n===== Backtest Result (ret from npz) =====")
    print(f"Days:              {metrics['n_days']}")
    print(f"Cumulative Return: {metrics['cum_return']:.6f}  (复利累乘)")
    print(f"Sharpe(ann):       {metrics['sharpe']:.6f}")
    print(f"Daily mean/std:    {metrics['daily_ret_mean']:.8f} / {metrics['daily_ret_std']:.8f}")
    print(f"RankIC mean/std:   {metrics['rankic_mean']:.6f} / {metrics['rankic_std']:.6f}")
    print(f"ICIR(ann):         {metrics['icir']:.6f}")
    print(f"TopK missing rate: {metrics['topk_missing_rate']:.4%}")
    print("=========================================")

    print(f"\n[OK] Curve saved to: {args.out_curve_csv}")


if __name__ == "__main__":
    main()
