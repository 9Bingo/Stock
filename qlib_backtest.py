# -*- coding: utf-8 -*-
"""
Qlib回测脚本
将模型预测结果转换为Qlib信号格式并进行回测

所有参数在文件内配置，无需命令行参数
"""

import os
import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from typing import Optional, List

import qlib
from qlib.config import REG_CN, REG_US
from qlib.backtest import backtest
from qlib.backtest.executor import SimulatorExecutor
from qlib.contrib.evaluate import risk_analysis
from qlib.contrib.strategy import TopkDropoutStrategy
from qlib.utils.time import Freq


# ============================================================================
#                              配置区域
# ============================================================================

@dataclass
class BacktestConfig:
    """
    Qlib回测配置

    所有回测参数在此处配置
    """

    # ==================== 基础配置 ====================

    # 预测结果文件路径
    pred_path: str = "curve.csv"

    # 市场类型: "cn" (A股) 或 "us" (美股)
    market: str = "us"

    # Qlib数据路径 (None表示使用默认路径)
    # A股默认: ~/.qlib/qlib_data/cn_data
    # 美股默认: ~/.qlib/qlib_data/us_data
    qlib_data_path: Optional[str] = None

    # 回测结果保存目录
    save_dir: str = "outputs"

    # ==================== 回测时间配置 ====================

    # 回测开始时间 (None表示使用预测数据的开始时间)
    start_time: Optional[str] = None

    # 回测结束时间 (None表示使用预测数据的结束时y间)
    end_time: Optional[str] = None

    # ==================== 策略参数 (TopkDropoutStrategy) ====================

    # 持仓股票数量
    topk: int = 50

    # 每次调仓卖出数量
    n_drop: int = 1

    # 卖出方式: "bottom" (得分最低) 或 "random" (随机)
    method_sell: str = "bottom"

    # 买入方式: "top" (得分最高) 或 "random" (随机)
    method_buy: str = "top"

    # 最小持仓天数
    hold_thresh: int = 1

    # 是否只考虑可交易股票
    only_tradable: bool = True

    # 涨跌停时是否禁止所有交易
    forbid_all_trade_at_limit: bool = True

    # 仓位比例 (0-1)
    risk_degree: float = 0.95

    # ==================== 账户配置 ====================

    # 初始资金
    account: float = 100_000_000  # 1亿

    # 基准指数代码 (None表示使用默认值)
    # A股默认: SH000300 (沪深300)
    # 美股默认: SPY (SP500 ETF)
    benchmark: Optional[str] = None

    # ==================== 执行器配置 ====================

    # 交易频率: "day", "1min", "30min" 等
    freq: str = "day"

    # 是否生成组合指标
    generate_portfolio_metrics: bool = True

    # ==================== 交易所配置 (A股) ====================

    # A股涨跌停限制 (0.095 = 9.5%, None = 无限制)
    cn_limit_threshold: float = 0.095

    # A股成交价格: "close", "open", "vwap"
    cn_deal_price: str = "close"

    # A股买入手续费率
    cn_open_cost: float = 0.0005  # 0.05%

    # A股卖出手续费率 (含印花税0.1%)
    cn_close_cost: float = 0.0015  # 0.15%

    # A股最低手续费 (元)
    cn_min_cost: float = 5.0

    # A股交易单位 (手), None = 无限制
    cn_trade_unit: Optional[int] = 100

    # A股冲击成本/滑点
    cn_impact_cost: float = 0.0

    # ==================== 交易所配置 (美股) ====================

    # 美股涨跌停限制 (None = 无限制)
    us_limit_threshold: Optional[float] = None

    # 美股成交价格
    us_deal_price: str = "close"

    # 美股买入手续费率
    us_open_cost: float = 0.0001  # 0.01%

    # 美股卖出手续费率
    us_close_cost: float = 0.0001  # 0.01%

    # 美股最低手续费 (美元)
    us_min_cost: float = 1.0

    # 美股交易单位, None = 无限制
    us_trade_unit: Optional[int] = None

    # 美股冲击成本/滑点
    us_impact_cost: float = 0.0

    # ==================== 其他配置 ====================

    # 保存Qlib信号的路径 (None = 不保存)
    save_signal_path: Optional[str] = None

    # 仅转换信号格式，不执行回测
    convert_only: bool = False

    def get_qlib_data_path(self) -> str:
        """获取Qlib数据路径"""
        if self.qlib_data_path is not None:
            return os.path.expanduser(self.qlib_data_path)
        if self.market == "us":
            return os.path.expanduser("~/.qlib/qlib_data/us_data")
        return os.path.expanduser("~/.qlib/qlib_data/cn_data")

    def get_benchmark(self) -> str:
        """获取基准指数代码"""
        if self.benchmark is not None:
            return self.benchmark
        if self.market == "us":
            return "^GSPC"
        return "SH000300"

    def get_exchange_kwargs(self) -> dict:
        """获取交易所配置"""
        if self.market == "us":
            return {
                "freq": self.freq,
                "limit_threshold": self.us_limit_threshold,
                "deal_price": self.us_deal_price,
                "open_cost": self.us_open_cost,
                "close_cost": self.us_close_cost,
                "min_cost": self.us_min_cost,
                "trade_unit": self.us_trade_unit,
                "impact_cost": self.us_impact_cost,
            }
        return {
            "freq": self.freq,
            "limit_threshold": self.cn_limit_threshold,
            "deal_price": self.cn_deal_price,
            "open_cost": self.cn_open_cost,
            "close_cost": self.cn_close_cost,
            "min_cost": self.cn_min_cost,
            "trade_unit": self.cn_trade_unit,
            "impact_cost": self.cn_impact_cost,
        }


# 创建默认配置实例
# 修改此处的配置来调整回测参数
CONFIG = BacktestConfig(
    # ===== 修改这里的参数 =====
    pred_path="curve.csv",
    market="us",
    topk=50,
    n_drop=5,
    # ==========================
)


# ============================================================================
#                              功能函数
# ============================================================================

def load_predictions(pred_path: str) -> pd.DataFrame:
    """
    加载预测结果CSV文件

    参数:
        pred_path: predictions.csv路径
q
    返回:
        DataFrame with columns: date, symbol, pred, label, label_raw
    """
    df = pd.read_csv(pred_path)
    df = df.rename(columns={"code": "symbol", "score": "pred"})
    df['date'] = pd.to_datetime(df['date'])
    print(f"加载预测结果: {len(df)} 条记录")
    print(f"日期范围: {df['date'].min()} ~ {df['date'].max()}")
    print(f"股票数量: {df['symbol'].nunique()}")
    return df


def convert_to_qlib_signal(df: pd.DataFrame, market: str = "cn") -> pd.Series:
    """
    将预测DataFrame转换为Qlib需要的信号格式

    Qlib要求的格式:
        pd.Series with MultiIndex (instrument, datetime)
        值: float (信号分数，越大越看好)

    参数:
        df: 包含 date, symbol, pred 列的DataFrame
        market: 市场类型 'cn' 或 'us'

    返回:
        pd.Series with MultiIndex (instrument, datetime)
    """
    df = df.copy()

    if market == "cn":
        # A股: 000001.SZ -> SZ000001, 600000.SS -> SH600000
        def convert_symbol(symbol):
            code, exchange = symbol.split('.')
            exchange_map = {
                'SS': 'SH',  # Shanghai Stock Exchange
                'SZ': 'SZ',  # Shenzhen Stock Exchange
            }
            qlib_exchange = exchange_map.get(exchange, exchange)
            return f"{qlib_exchange}{code}"
        df['instrument'] = df['symbol'].apply(convert_symbol)
    else:
        # 美股: AAPL -> AAPL (保持不变)
        df['instrument'] = df['symbol']

    # 创建MultiIndex
    df = df.set_index(['instrument', 'date'])
    df.index.names = ['instrument', 'datetime']

    # 提取信号（pred列）
    signal = df['pred'].sort_index()

    print(f"\n信号格式转换完成:")
    print(f"  索引层级: {signal.index.names}")
    print(f"  信号数量: {len(signal)}")
    print(f"\n信号示例:")
    print(signal.head(10))

    return signal


def save_qlib_signal(signal: pd.Series, save_path: str):
    """
    保存Qlib格式的信号

    参数:
        signal: Qlib格式的信号序列
        save_path: 保存路径
    """
    os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else ".", exist_ok=True)
    signal.to_pickle(save_path)
    print(f"信号已保存至: {save_path}")


def run_qlib_backtest(signal: pd.Series, config: BacktestConfig) -> dict:
    """
    使用Qlib进行回测

    参数:
        signal: Qlib格式的信号序列
        config: 回测配置

    返回:
        回测结果字典
    """
    # 初始化Qlib
    print(f"\n初始化Qlib...")
    qlib_data_path = config.get_qlib_data_path()
    print(f"  数据路径: {qlib_data_path}")
    print(f"  市场: {config.market.upper()}")

    region = REG_US if config.market == "us" else REG_CN
    qlib.init(provider_uri=qlib_data_path, region=region)

    # 创建策略
    strategy = TopkDropoutStrategy(
        signal=signal,
        topk=config.topk,
        n_drop=config.n_drop,
        method_sell=config.method_sell,
        method_buy=config.method_buy,
        hold_thresh=config.hold_thresh,
        only_tradable=config.only_tradable,
        forbid_all_trade_at_limit=config.forbid_all_trade_at_limit,
        risk_degree=config.risk_degree,
    )

    # 执行器配置
    executor_config = {
        "time_per_step": config.freq,
        "generate_portfolio_metrics": config.generate_portfolio_metrics,
    }

    # 获取交易所配置
    exchange_kwargs = config.get_exchange_kwargs()

    # 回测配置
    benchmark = config.get_benchmark()
    backtest_config = {
        "start_time": config.start_time,
        "end_time": config.end_time,
        "account": config.account,
        "benchmark": benchmark,
        "exchange_kwargs": exchange_kwargs,
    }

    print(f"\n开始回测...")
    print(f"  时间范围: {config.start_time} ~ {config.end_time}")
    print(f"  TopK: {config.topk}, n_drop: {config.n_drop}")
    print(f"  基准: {benchmark}")
    print(f"  仓位比例: {config.risk_degree:.1%}")
    print(f"  交易所配置:")
    for k, v in exchange_kwargs.items():
        print(f"    {k}: {v}")

    # 执行回测
    portfolio_metric_dict, indicator_dict = backtest(
        strategy=strategy,
        executor=SimulatorExecutor(**executor_config),
        **backtest_config
    )

    # 解析结果
    analysis_freq = "{0}{1}".format(*Freq.parse(config.freq))
    report_normal, positions = portfolio_metric_dict.get(analysis_freq)

    # 风险分析
    analysis_result = {
        "portfolio": risk_analysis(report_normal["return"] - report_normal["cost"], freq=analysis_freq),
        "benchmark": risk_analysis(report_normal["bench"], freq=analysis_freq),
        "excess_return_without_cost": risk_analysis(
            report_normal["return"] - report_normal["bench"], freq=analysis_freq
        ),
        "excess_return_with_cost": risk_analysis(
            report_normal["return"] - report_normal["bench"] - report_normal["cost"], freq=analysis_freq
        ),
    }

    # 打印结果
    print("\n" + "=" * 60)
    print("Qlib 回测结果")
    print("=" * 60)

    def print_analysis(title: str, result):
        print(f"\n【{title}】")
        if isinstance(result, pd.DataFrame):
            for idx in result.index:
                value = result.loc[idx, 'risk'] if 'risk' in result.columns else result.loc[idx].iloc[0]
                if isinstance(value, (int, float, np.floating, np.integer)):
                    print(f"  {idx}: {value:.4f}")
                else:
                    print(f"  {idx}: {value}")
        elif isinstance(result, pd.Series):
            for key, value in result.items():
                if isinstance(value, (int, float, np.floating, np.integer)):
                    print(f"  {key}: {value:.4f}")
                else:
                    print(f"  {key}: {value}")
        else:
            print(result)

    print_analysis("组合表现", analysis_result["portfolio"])
    print_analysis("基准表现", analysis_result["benchmark"])
    print_analysis("超额收益（不扣成本）", analysis_result["excess_return_without_cost"])
    print_analysis("超额收益（扣除成本）", analysis_result["excess_return_with_cost"])

    print("=" * 60)

    # 保存结果到文件
    os.makedirs(config.save_dir, exist_ok=True)

    # 1. 保存每日收益报告
    report_path = os.path.join(config.save_dir, "qlib_backtest_report.csv")
    report_normal.to_csv(report_path)
    print(f"\n每日收益报告已保存至: {report_path}")

    # 2. 保存综合指标
    metrics_path = os.path.join(config.save_dir, "qlib_backtest_metrics.csv")
    metrics_data = {
        "指标类别": [],
        "指标名称": [],
        "数值": [],
    }
    for category, result in analysis_result.items():
        if isinstance(result, pd.DataFrame):
            for idx in result.index:
                for col in result.columns:
                    value = result.loc[idx, col]
                    if isinstance(value, (int, float, np.floating, np.integer)):
                        metrics_data["指标类别"].append(category)
                        metrics_data["指标名称"].append(str(idx))
                        metrics_data["数值"].append(float(value))
        elif isinstance(result, pd.Series):
            for idx, value in result.items():
                if isinstance(value, (int, float, np.floating, np.integer)):
                    metrics_data["指标类别"].append(category)
                    metrics_data["指标名称"].append(str(idx))
                    metrics_data["数值"].append(float(value))
    metrics_df = pd.DataFrame(metrics_data)
    metrics_df.to_csv(metrics_path, index=False)
    print(f"综合指标已保存至: {metrics_path}")

    # 3. 保存累计收益曲线数据
    cum_return_path = os.path.join(config.save_dir, "qlib_cumulative_returns.csv")
    cum_return_df = pd.DataFrame({
        "portfolio_return": (report_normal["return"] - report_normal["cost"]).cumsum(),
        "benchmark_return": report_normal["bench"].cumsum(),
        "excess_return": (report_normal["return"] - report_normal["bench"] - report_normal["cost"]).cumsum(),
    })
    cum_return_df.to_csv(cum_return_path)
    print(f"累计收益曲线已保存至: {cum_return_path}")

    return {
        "report": report_normal,
        "positions": positions,
        "analysis": analysis_result,
    }


def main(config: BacktestConfig = None):
    """
    主函数

    参数:
        config: 回测配置，None表示使用全局CONFIG
    """
    if config is None:
        config = CONFIG

    # 1. 加载预测结果
    print("=" * 60)
    print("加载预测结果")
    print("=" * 60)
    pred_df = load_predictions(config.pred_path)

    # 自动检测市场类型（根据股票代码格式）
    sample_symbol = pred_df['symbol'].iloc[0]
    if '.' in sample_symbol:
        detected_market = "cn"
    else:
        detected_market = "us"

    # 如果检测到的市场类型与配置不符，给出警告
    if config.market != detected_market:
        print(f"\n警告: 配置的市场类型({config.market})与检测到的代码格式({detected_market})不符")
        print(f"自动切换到: {detected_market}")
        config.market = detected_market

    print(f"\n配置信息:")
    print(f"  市场: {config.market.upper()}")
    print(f"  基准: {config.get_benchmark()}")
    print(f"  TopK: {config.topk}")
    print(f"  n_drop: {config.n_drop}")

    # 2. 转换为Qlib信号格式
    print("\n" + "=" * 60)
    print("转换信号格式")
    print("=" * 60)
    signal = convert_to_qlib_signal(pred_df, market=config.market)

    # 3. 保存信号（可选）
    if config.save_signal_path:
        save_qlib_signal(signal, config.save_signal_path)

    # 4. 仅转换模式
    if config.convert_only:
        print("\n仅转换模式，跳过回测")
        return signal

    # 5. 确定回测时间范围
    if config.start_time is None:
        config.start_time = pred_df['date'].min().strftime('%Y-%m-%d')
    if config.end_time is None:
        config.end_time = pred_df['date'].max().strftime('%Y-%m-%d')

    # 6. 执行Qlib回测
    print("\n" + "=" * 60)
    print("执行Qlib回测")
    print("=" * 60)

    try:
        results = run_qlib_backtest(signal=signal, config=config)
        return results
    except Exception as e:
        print(f"\n回测失败: {e}")
        print("\n请确保:")
        print("  1. 已安装qlib: pip install pyqlib")
        if config.market == "us":
            print("  2. 已下载Qlib美股数据:")
            print("     python -m qlib.run.get_data qlib_data --target_dir ~/.qlib/qlib_data/us_data --region us")
        else:
            print("  2. 已下载Qlib中国A股数据:")
            print("     python -m qlib.run.get_data qlib_data --target_dir ~/.qlib/qlib_data/cn_data --region cn")
        print(f"  3. 数据路径正确: {config.get_qlib_data_path()}")
        raise


# ============================================================================
#                              使用示例
# ============================================================================

def example_cn_backtest():
    """A股回测示例"""
    config = BacktestConfig(
        pred_path="outputs/LSTM_csi300_predictions.csv",
        market="cn",
        topk=50,
        n_drop=5,
        benchmark="SH000300",
        cn_open_cost=0.0005,
        cn_close_cost=0.0015,
    )
    return main(config)


def example_us_backtest():
    """美股回测示例"""
    config = BacktestConfig(
        pred_path="outputs/LSTM_sp500_predictions.csv",
        market="us",
        topk=30,
        n_drop=3,
        benchmark="SPY",
        us_open_cost=0.0001,
        us_close_cost=0.0001,
    )
    return main(config)


if __name__ == "__main__":
    # 使用全局CONFIG配置运行
    main()

    # 或者使用自定义配置:
    # example_cn_backtest()
    # example_us_backtest()
