import os
import glob
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import joblib
from typing import Tuple
from dataclasses import dataclass

# ======================
# 超参数配置
# ======================
DATA_DIR = "../../data/sp500_origin"

SEQ_LEN = 60
PRED_N = 1

TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
TEST_RATIO = 0.1

OUTPUT_DIR = "../../data/sp_seq60_n1"
os.makedirs(OUTPUT_DIR, exist_ok=True)

COL_CODE = "symbol"
COL_TIME = "date"

FEATURE_COLS = ["open", "high", "low", "close", "volume"]
COL_CLOSE = "close"

CLIP_QUANTILE = 0.01  # 只做 winsorize/clip


# ======================
# 数据结构：保存 clip 参数（用于复现/推理）
# ======================
@dataclass
class ClipParams:
    clip_quantile: float
    train_end_time: pd.Timestamp
    low_clip: pd.Series
    high_clip: pd.Series


# ======================
# 工具函数
# ======================
def load_all_csv(data_dir: str) -> pd.DataFrame:
    pattern = os.path.join(data_dir, "*.csv")
    files = glob.glob(pattern)
    if not files:
        raise FileNotFoundError(f"在目录 {data_dir} 下没有找到 csv 文件")

    dfs = []
    for f in files:
        print(f"读取文件: {f}")
        dfs.append(pd.read_csv(f))

    df_all = pd.concat(dfs, ignore_index=True)
    print(f"合并后数据形状: {df_all.shape}")
    return df_all


def preprocess_raw_df_basic(df: pd.DataFrame) -> pd.DataFrame:
    """
    只做：选列、时间/数值转换、去缺失、按 (symbol, date) 排序
    不做异常值处理
    """
    needed_cols = [COL_CODE, COL_TIME] + FEATURE_COLS
    df = df[needed_cols].copy()

    df[COL_TIME] = pd.to_datetime(df[COL_TIME], errors="coerce")

    for c in FEATURE_COLS:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df.dropna(subset=needed_cols)
    df = df.sort_values(by=[COL_CODE, COL_TIME]).reset_index(drop=True)

    print(f"基础预处理后数据形状(去缺失+排序): {df.shape}")
    return df


def compute_split_end_times(df: pd.DataFrame) -> Tuple[pd.Timestamp, pd.Timestamp]:
    """
    ✅ 一次性计算 train_end_time / val_end_time
    后面 clip/scaler 用 train_end_time；最终 split 用同一对边界
    """
    all_times = np.sort(df[COL_TIME].unique())
    if len(all_times) < 3:
        raise ValueError("可用交易日太少，无法划分 Train/Val/Test。")

    n = len(all_times)

    train_end_idx = int(n * TRAIN_RATIO) - 1
    train_end_idx = max(train_end_idx, 0)

    val_end_idx = int(n * (TRAIN_RATIO + VAL_RATIO)) - 1
    val_end_idx = max(val_end_idx, train_end_idx)

    train_end_time = pd.to_datetime(all_times[train_end_idx])
    val_end_time = pd.to_datetime(all_times[val_end_idx])

    print(f"统一边界：train_end_time={train_end_time.strftime('%Y-%m-%d')}, "
          f"val_end_time={val_end_time.strftime('%Y-%m-%d')}")
    return train_end_time, val_end_time


def fit_clip_params_on_train_period(
    df: pd.DataFrame,
    train_end_time: pd.Timestamp,
    feature_cols,
    clip_quantile: float
) -> ClipParams:
    """
    ✅ 无泄漏：clip 阈值只在训练期（<= train_end_time）估计
    """
    train_df = df.loc[df[COL_TIME] <= train_end_time, feature_cols].copy()
    if len(train_df) == 0:
        raise ValueError("训练期没有数据，无法估计 clip 阈值。")

    low_clip = train_df.quantile(clip_quantile)
    high_clip = train_df.quantile(1 - clip_quantile)

    print(
        f"clip 阈值仅用训练期估计：clip_q={clip_quantile}, 训练期样本数={len(train_df)}"
    )

    return ClipParams(
        clip_quantile=clip_quantile,
        train_end_time=train_end_time,
        low_clip=low_clip,
        high_clip=high_clip
    )


def apply_clip_with_params(df: pd.DataFrame, params: ClipParams, feature_cols) -> pd.DataFrame:
    """
    ✅ 全量（Train/Val/Test）只做 clip，不做 drop
    """
    df = df.copy()
    for col in feature_cols:
        df[col] = df[col].clip(lower=params.low_clip[col], upper=params.high_clip[col])
    print("全量已用训练期阈值进行分位数裁剪(clip)，不执行 drop")
    print(f"clip 后数据形状: {df.shape}")
    return df


def fit_scaler_on_train_period(df: pd.DataFrame, train_end_time: pd.Timestamp):
    """
    ✅ scaler 仅用训练期拟合
    """
    train_mask = df[COL_TIME] <= train_end_time
    X_fit = df.loc[train_mask, FEATURE_COLS].to_numpy(dtype=np.float32)
    if X_fit.shape[0] == 0:
        raise ValueError("训练期无可用于 scaler 拟合的数据。")

    scaler = StandardScaler()
    scaler.fit(X_fit)
    print(f"StandardScaler 仅用训练期拟合：<= {train_end_time.strftime('%Y%m%d')}, 行数={X_fit.shape[0]}")
    return scaler


def build_windows_and_labels_prescaled(df: pd.DataFrame):
    """
    df[FEATURE_COLS] 已缩放后的特征；
    df['close_raw'] 真正原始 close，用于打标签：
      r = future_close / last_close - 1

    ✅ times 用 label 日期（label_idx 的日期），便于严格按日期切分
    """
    X_list, y_list, code_list, time_list = [], [], [], []

    for code, df_code in df.groupby(COL_CODE):
        df_code = df_code.sort_values(COL_TIME)

        feats_scaled = df_code[FEATURE_COLS].to_numpy(dtype=np.float32)
        closes_raw = df_code["close_raw"].to_numpy(dtype=np.float32)
        times_int = df_code[COL_TIME].dt.strftime("%Y%m%d").astype(np.int32).to_numpy()


        T = len(df_code)
        max_start = T - SEQ_LEN - PRED_N
        if max_start < 0:
            continue

        for start in range(max_start + 1):
            end = start + SEQ_LEN - 1
            label_idx = end + PRED_N

            window = feats_scaled[start:start + SEQ_LEN]

            last_close = closes_raw[end]
            future_close = closes_raw[label_idx]
            if last_close == 0 or np.isnan(last_close) or np.isnan(future_close):
                continue

            r_simple = future_close / last_close - 1.0

            X_list.append(window)
            y_list.append(r_simple)
            code_list.append(code)
            time_list.append(times_int[label_idx]) # ✅ label 日期

    if not X_list:
        raise RuntimeError("没有生成任何窗口样本，请检查数据量、SEQ_LEN 和 PRED_N 设置。")

    X = np.stack(X_list, axis=0).astype(np.float32)
    y = np.array(y_list, dtype=np.float32)
    codes = np.array(code_list)
    times = np.array(time_list)

    print(f"总共生成窗口样本数: {X.shape[0]}")
    print(f"X 形状: {X.shape}, y(简单收益) 形状: {y.shape}")
    return X, y, codes, times


def split_datasets_time_based_by_date(
    X, y, codes, times,
    train_end_date: int,
    val_end_date: int
):
    """
    ✅ 按固定日期边界切分：同一天的样本全部归到同一个集合
    times: int YYYYMMDD（这里是 label 日期）
    train_end_date / val_end_date: int YYYYMMDD，与 clip/scaler 的边界一致
    """
    train_mask = times <= train_end_date
    val_mask = (times > train_end_date) & (times <= val_end_date)
    test_mask = times > val_end_date

    def take(mask):
        return X[mask], y[mask], codes[mask], times[mask]

    X_train, y_train, codes_train, times_train = take(train_mask)
    X_val, y_val, codes_val, times_val = take(val_mask)
    X_test, y_test, codes_test, times_test = take(test_mask)

    print("按固定日期边界切分后：")
    print(f"  Train 样本数: {X_train.shape[0]}")
    print(f"  Val   样本数: {X_val.shape[0]}")
    print(f"  Test  样本数: {X_test.shape[0]}")
    if len(times_train) > 0:
        print(f"  Train 时间范围: {times_train.min()} ~ {times_train.max()}")
    if len(times_val) > 0:
        print(f"  Val   时间范围: {times_val.min()} ~ {times_val.max()}")
    if len(times_test) > 0:
        print(f"  Test  时间范围: {times_test.min()} ~ {times_test.max()}")

    return (X_train, y_train, codes_train, times_train,
            X_val, y_val, codes_val, times_val,
            X_test, y_test, codes_test, times_test)



def save_splits(
    X_train, y_train, codes_train, times_train,
    X_val, y_val, codes_val, times_val,
    X_test, y_test, codes_test, times_test,
):
    np.savez_compressed(
        os.path.join(OUTPUT_DIR, "train.npz"),
        X=X_train, y=y_train, codes=codes_train, times=times_train
    )
    np.savez_compressed(
        os.path.join(OUTPUT_DIR, "val.npz"),
        X=X_val, y=y_val, codes=codes_val, times=times_val
    )
    np.savez_compressed(
        os.path.join(OUTPUT_DIR, "test.npz"),
        X=X_test, y=y_test, codes=codes_test, times=times_test
    )
    print(f"Train/Val/Test 数据已保存到目录: {OUTPUT_DIR}")


# ======================
# 主流程
# ======================
def main():
    df_raw = load_all_csv(DATA_DIR)

    # A) 基础预处理
    df = preprocess_raw_df_basic(df_raw)

    # ✅ B) 先备份真正原始 close（任何 clip/scaler 之前）
    df["close_raw"] = df[COL_CLOSE].astype(np.float32)

    # C) 确定训练期截止日（用于 clip/scaler）
    # C) ✅ 一次性确定 Train/Val 截止日（用于 clip/scaler + 最终 split）
    train_end_time, val_end_time = compute_split_end_times(df)


    # D) ✅ clip 参数只用训练期估计
    clip_params = fit_clip_params_on_train_period(
        df=df,
        train_end_time=train_end_time,
        feature_cols=FEATURE_COLS,
        clip_quantile=CLIP_QUANTILE
    )

    # E) ✅ 全量 clip（训练/验证/测试一致），不 drop
    df = apply_clip_with_params(df, clip_params, FEATURE_COLS)

    # 保存 clip 参数
    clip_path_out = os.path.join(OUTPUT_DIR, "clip_params.pkl")
    joblib.dump(clip_params, clip_path_out)
    print(f"clip 参数已保存: {clip_path_out}")

    # F) scaler 仅用训练期拟合（clip 后）
    scaler = fit_scaler_on_train_period(df, train_end_time)

    # G) 用训练期 scaler 对全量数据缩放（不动 close_raw）
    df[FEATURE_COLS] = scaler.transform(df[FEATURE_COLS].to_numpy(dtype=np.float32))

    scaler_path_out = os.path.join(OUTPUT_DIR, "scaler.pkl")
    joblib.dump(scaler, scaler_path_out)
    print(f"scaler 已保存: {scaler_path_out}")

    # H) 构建窗口 + 用 close_raw 打标签（times=label 日期）
    X, y, codes, times = build_windows_and_labels_prescaled(df)

    # I) ✅ 按日期切分
    train_end_date = int(train_end_time.strftime("%Y%m%d"))
    val_end_date = int(val_end_time.strftime("%Y%m%d"))

    (X_train, y_train, codes_train, times_train,
    X_val, y_val, codes_val, times_val,
    X_test, y_test, codes_test, times_test) = split_datasets_time_based_by_date(
        X, y, codes, times,
        train_end_date=train_end_date,
        val_end_date=val_end_date
    )


    # J) 保存
    save_splits(
        X_train, y_train, codes_train, times_train,
        X_val, y_val, codes_val, times_val,
        X_test, y_test, codes_test, times_test,
    )


if __name__ == "__main__":
    main()
