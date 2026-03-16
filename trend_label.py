import os
import glob
import numpy as np
import pandas as pd

# ====== 你需要改的配置 ======
DATA_DIR = "/home/wuyuzhang/litianyuan/stock/sp500_origin"     # 原始csv目录
NPZ_DIR  = "/home/wuyuzhang/litianyuan/stock/sp_seq30_n1"      # train/val/test.npz 所在目录
HORIZON  = 7                            # 固定窗口：t..t+7（共8个点）
OVERWRITE = False                        # True=覆盖原npz；False=生成 *_labeled.npz
USE_LOG_PRICE = True                     # Trend Scanning 常用 log(price)

# 你的原始列名
COL_CODE = "symbol"
COL_TIME = "date"
COL_CLOSE = "close"


def to_str_array(arr: np.ndarray) -> np.ndarray:
    """npz里 times/codes 可能是 bytes，这里统一转成 str"""
    arr = np.asarray(arr)
    if arr.dtype.kind == "S":  # bytes
        return np.char.decode(arr, "utf-8")
    return arr.astype(str)


def load_all_csv(data_dir: str) -> pd.DataFrame:
    files = glob.glob(os.path.join(data_dir, "*.csv"))
    if not files:
        raise FileNotFoundError(f"No csv found in {data_dir}")

    dfs = []
    for f in files:
        df = pd.read_csv(f)
        dfs.append(df)
    return pd.concat(dfs, ignore_index=True)


def build_lookup(df: pd.DataFrame):
    """
    为每个 symbol 建：
      - dates: [T] 形式 'YYYYMMDD'
      - closes: [T] float32
      - date_to_idx: dict(date_str -> idx)
    """
    df = df[[COL_CODE, COL_TIME, COL_CLOSE]].copy()
    df[COL_TIME] = pd.to_datetime(df[COL_TIME], errors="coerce")
    df[COL_CLOSE] = pd.to_numeric(df[COL_CLOSE], errors="coerce")
    df = df.dropna(subset=[COL_CODE, COL_TIME, COL_CLOSE])
    df = df.sort_values([COL_CODE, COL_TIME]).reset_index(drop  =True)

    lookup = {}
    for sym, g in df.groupby(COL_CODE, sort=False):
        dates = g[COL_TIME].dt.strftime("%Y%m%d").to_numpy()
        closes = g[COL_CLOSE].to_numpy(dtype=np.float32)
        date_to_idx = {d: i for i, d in enumerate(dates)}
        lookup[str(sym)] = (closes, date_to_idx)
    return lookup


def slope_tvalue(y: np.ndarray) -> float:
    """
    对 y[0..L] 做 OLS: y = b0 + b1*x，返回 b1 的 t-value
    y: shape [n], n>=3
    """
    y = y.astype(np.float64, copy=False)
    n = y.shape[0]
    if n < 3:
        return 0.0

    x = np.arange(n, dtype=np.float64)
    x_mean = x.mean()
    y_mean = y.mean()

    xx = x - x_mean
    yy = y - y_mean

    ssx = np.sum(xx * xx)
    if ssx <= 0:
        return 0.0

    b1 = np.sum(xx * yy) / ssx
    b0 = y_mean - b1 * x_mean

    resid = y - (b0 + b1 * x)
    s2 = np.sum(resid * resid) / (n - 2)  # 自由度 n-2
    se_b1 = np.sqrt(s2 / ssx) + 1e-12
    tval = b1 / se_b1
    return float(tval)


def add_label_to_one_npz(npz_path: str, lookup: dict, horizon: int, overwrite: bool):
    with np.load(npz_path, allow_pickle=True) as pack:
        codes = to_str_array(pack["codes"])
        times = to_str_array(pack["times"])
        N = len(times)

        # ====== 新标签：Trend Scanning 固定 horizon 的 t-value ======
        t_value_7d = np.full((N,), np.nan, dtype=np.float32)
        # 可选：方向标签（方便分类）
        label_dir_7d = np.full((N,), -1, dtype=np.int8)

        valid_7d = np.zeros((N,), dtype=bool)
        close_today = np.full((N,), np.nan, dtype=np.float32)
        close_future = np.full((N,), np.nan, dtype=np.float32)

        missing_symbol = 0
        missing_date = 0
        no_future = 0

        win = horizon + 1  # t..t+horizon 共 win 个点

        for i in range(N):
            sym = codes[i]
            t = times[i]

            if sym not in lookup:
                missing_symbol += 1
                continue

            closes, date_to_idx = lookup[sym]
            if t not in date_to_idx:
                missing_date += 1
                continue

            idx = date_to_idx[t]
            idx_f = idx + horizon
            if idx_f >= len(closes):
                no_future += 1
                continue

            seg = closes[idx: idx + win].astype(np.float64)
            if np.any(~np.isfinite(seg)) or seg.shape[0] != win:
                continue

            # 保存原始 close 便于检查
            c0 = float(seg[0])
            cf = float(seg[-1])
            close_today[i] = c0
            close_future[i] = cf

            # Trend Scanning 常用 log price；确保 >0
            if USE_LOG_PRICE:
                seg = np.log(np.clip(seg, 1e-12, None))

            tval = slope_tvalue(seg)
            t_value_7d[i] = np.float32(tval)

            # 方向标签（你如果只要 t-value，可忽略这个字段）
            label_dir_7d[i] = 1 if tval > 0 else 0
            valid_7d[i] = True

        # ===== 保存（覆盖或生成新文件）=====
        out_path = npz_path if overwrite else npz_path.replace(".npz", "_labeled.npz")
        os.makedirs(os.path.dirname(out_path), exist_ok=True)

        # 临时文件必须以 .npz 结尾，避免 numpy 自动补后缀
        tmp_path = out_path + ".tmp.npz"

        out_dict = {k: pack[k] for k in pack.files}
        out_dict["t_value_7d"] = t_value_7d       
        out_dict["label_dir_7d"] = label_dir_7d  
        out_dict["valid_7d"] = valid_7d
        out_dict["close_today"] = close_today
        out_dict["close_future"] = close_future

        np.savez_compressed(tmp_path, **out_dict)
        os.replace(tmp_path, out_path)

    print(f"[OK] {os.path.basename(npz_path)} -> {os.path.basename(out_path)}")
    print(f"  N={N}, valid={int(valid_7d.sum())}, invalid={int(N - valid_7d.sum())}")
    print(f"  missing_symbol={missing_symbol}, missing_date={missing_date}, no_future={no_future}")


def main():
    print("Loading price data...")
    df = load_all_csv(DATA_DIR)
    lookup = build_lookup(df)
    print(f"Lookup symbols: {len(lookup)}")

    for name in ["train.npz", "val.npz", "test.npz"]:
        p = os.path.join(NPZ_DIR, name)
        if not os.path.exists(p):
            print(f"[SKIP] not found: {p}")
            continue
        add_label_to_one_npz(p, lookup, HORIZON, overwrite=OVERWRITE)


if __name__ == "__main__":
    main()
