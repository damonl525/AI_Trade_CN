"""动态择时 vs 二元择时 vs 无择时 — 三路对比回测

核心问题: 熊市里一直躲 vs 动态调整仓位，哪个更好?
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime

from config import DATA_MARKET
from data.fetcher import load
from engine.backtest import run_backtest
from engine.timing import timing_filter_series
from engine.dynamic_timing import dynamic_position_series

# ── 标的池 ──
POOL = {
    "510050": "上证50",
    "510300": "沪深300",
    "510500": "中证500",
    "159915": "创业板",
    "588000": "科创50",
    "512880": "证券",
    "512010": "医药",
    "512690": "酒/消费",
    "510880": "红利",
    "511010": "国债",
}

OUTPUT = Path(__file__).parent / "output"


def load_pool_data() -> dict[str, pd.DataFrame]:
    """加载池内所有行情, date 列 → DatetimeIndex"""
    data = {}
    for code in POOL:
        path = DATA_MARKET / f"etf_{code}.parquet"
        if path.exists():
            df = load(path)
            if "date" in df.columns:
                df["date"] = pd.to_datetime(df["date"])
                df = df.set_index("date").sort_index()
            data[code] = df
    return data


def compute_rotation_returns(
    data: dict[str, pd.DataFrame],
    lookback: int = 10,
    top_k: int = 3,
    rebalance_days: int = 40,
) -> pd.Series:
    """动量轮动的日收益序列 (无择时)"""
    # 对齐所有标的日期
    closes = {}
    for code, df in data.items():
        closes[code] = df["close"]

    prices = pd.DataFrame(closes).dropna()
    if prices.empty:
        return pd.Series(dtype=float)

    # 动量计算
    momentum = prices.pct_change(lookback)
    # 注意: pct_change(lookback) 得到的是 lookback 日总收益, 非年化

    # 持仓权重
    weights = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)

    rebalance_dates = prices.index[::rebalance_days]

    for i, date in enumerate(rebalance_dates):
        if i == 0:
            continue
        mom_slice = momentum.loc[: date].iloc[-1]
        top = mom_slice.nlargest(top_k)
        # 排除国债
        top = top[top.index != "511010"]
        if top.empty or top.sum() == 0:
            continue

        end_date = (
            rebalance_dates[i + 1] if i + 1 < len(rebalance_dates) else prices.index[-1]
        )
        w = pd.Series(0.0, index=prices.columns)
        for code in top.index:
            w[code] = 1.0 / len(top)
        weights.loc[date:end_date] = w.values

    # 日收益
    daily_ret = prices.pct_change().fillna(0)
    port_ret = (daily_ret * weights.shift(1)).sum(axis=1).fillna(0)
    port_ret.name = "rotation"
    return port_ret


def compute_dynamic_rotation_returns(
    data: dict[str, pd.DataFrame],
    lookback: int = 10,
    top_k: int = 3,
    rebalance_days: int = 40,
    index_code: str = "000300",
) -> tuple[pd.Series, pd.DataFrame]:
    """动量轮动 + 动态仓位缩放"""
    base_ret = compute_rotation_returns(data, lookback, top_k, rebalance_days)
    dpos = dynamic_position_series(index_code)

    # 对齐日期
    common_idx = base_ret.index.intersection(dpos.index)
    base_ret = base_ret.loc[common_idx]
    dpos = dpos.loc[common_idx]

    # 仓位缩放: 不是 binary, 是连续
    scaled_ret = base_ret * dpos["position"]
    # 剩余仓位买国债
    bond_ret = pd.Series(0.0, index=common_idx)
    if "511010" in data:
        bond_ret = data["511010"]["close"].pct_change().fillna(0)
        bond_ret = bond_ret.loc[common_idx]
    cash_ret = bond_ret * (1 - dpos["position"])
    total_ret = scaled_ret + cash_ret
    total_ret.name = "dynamic"
    return total_ret, dpos


def compute_timing_rotation_returns(
    data: dict[str, pd.DataFrame],
    lookback: int = 10,
    top_k: int = 3,
    rebalance_days: int = 40,
) -> pd.Series:
    """动量轮动 + 二元择时 (0/50/100)"""
    base_ret = compute_rotation_returns(data, lookback, top_k, rebalance_days)
    timing = timing_filter_series()

    common_idx = base_ret.index.intersection(timing.index)
    base_ret = base_ret.loc[common_idx]
    timing = timing.loc[common_idx]

    scaled_ret = base_ret * timing
    bond_ret = pd.Series(0.0, index=common_idx)
    if "511010" in data:
        bond_ret = data["511010"]["close"].pct_change().fillna(0)
        bond_ret = bond_ret.loc[common_idx]
    cash_ret = bond_ret * (1 - timing)
    total_ret = scaled_ret + cash_ret
    total_ret.name = "binary_timing"
    return total_ret


def stats(series: pd.Series, name: str) -> dict:
    """收益统计"""
    cum = (1 + series).prod() - 1
    ann = (1 + cum) ** (252 / len(series)) - 1 if len(series) > 0 else 0
    # 最大回撤
    cum_series = (1 + series).cumprod()
    peak = cum_series.cummax()
    dd = (cum_series - peak) / peak
    max_dd = dd.min()
    # 夏普
    sharpe = (
        series.mean() / series.std() * np.sqrt(252)
        if series.std() > 0
        else 0
    )
    return {
        "name": name,
        "总收益": f"{cum:.1%}",
        "年化": f"{ann:.1%}",
        "最大回撤": f"{max_dd:.1%}",
        "夏普": f"{sharpe:.2f}",
        "波动率": f"{series.std() * np.sqrt(252):.1%}",
    }


def main():
    import argparse

    parser = argparse.ArgumentParser(description="动态择时对比回测")
    parser.add_argument("--years", type=float, default=5.0, help="回测窗口(年)")
    parser.add_argument("--lookback", type=int, default=10, help="动量窗口(天)")
    parser.add_argument("--top_k", type=int, default=3, help="持仓数")
    parser.add_argument("--rebalance", type=int, default=40, help="调仓周期(天)")
    args = parser.parse_args()

    print("📊 加载数据...")
    data = load_pool_data()
    if len(data) < 3:
        print("❌ 数据不足, 请先运行 uv run python main.py fetch")
        return

    print(f"🔬 回测窗口: {args.years}年")
    print(f"   参数: 动量{args.lookback}天, Top{args.top_k}, {args.rebalance}天调仓")

    # 限制窗口
    all_dates = pd.DatetimeIndex([])
    for df in data.values():
        if isinstance(df.index, pd.DatetimeIndex):
            all_dates = all_dates.union(df.index)
        else:
            all_dates = all_dates.union(pd.to_datetime(df.index))
    if len(all_dates) == 0:
        print("❌ 日期数据为空")
        return
    cutoff = all_dates.max() - pd.Timedelta(days=int(args.years * 365))
    if cutoff < all_dates.min():
        cutoff = all_dates.min()

    for code in list(data.keys()):
        data[code] = data[code].loc[data[code].index >= cutoff]

    # 1. 无择时
    print("\n⏳ 回测中 (1/3) 无择时...")
    ret_raw = compute_rotation_returns(
        data, args.lookback, args.top_k, args.rebalance
    )

    # 2. 二元择时
    print("⏳ 回测中 (2/3) 二元择时 (0/50/100)...")
    ret_binary = compute_timing_rotation_returns(
        data, args.lookback, args.top_k, args.rebalance
    )

    # 3. 动态择时
    print("⏳ 回测中 (3/3) 动态择时 (连续0-100%)...")
    ret_dynamic, dpos_df = compute_dynamic_rotation_returns(
        data, args.lookback, args.top_k, args.rebalance
    )

    # ── 输出 ──
    print("\n" + "=" * 72)
    print("  📊 动态择时 vs 二元择时 vs 无择时")
    print("=" * 72)

    results = []
    for s, name in [
        (ret_raw, "无择时(永远满仓)"),
        (ret_binary, "二元择时(0/50/100)"),
        (ret_dynamic, "🏆动态择时(0-100%)"),
    ]:
        if s is not None and len(s) > 0:
            r = stats(s, name)
            results.append(r)
            tag = "🏆" if "动态" in name else "  "
            print(
                f"{tag} {name:22s} | 总收益{r['总收益']:>8s} | 年化{r['年化']:>8s}"
            )
            print(
                f"   {'':22s} | 回撤{r['最大回撤']:>8s} | 夏普{r['夏普']:>6s} | 波动{r['波动率']:>8s}"
            )

    # 额外: regime 分布
    if not dpos_df.empty:
        regime_counts = dpos_df["regime"].value_counts()
        total = len(dpos_df)
        print(f"\n  📈 动态择时 regime 分布:")
        for r in ["强牛", "弱牛", "震荡", "弱熊", "强熊"]:
            cnt = regime_counts.get(r, 0)
            print(f"     {r}: {cnt}天 ({cnt/total*100:.0f}%)")

    # 分年 — 用共同索引
    print(f"\n  📅 分年表现:")
    common_idx = ret_dynamic.index if ret_dynamic is not None else pd.DatetimeIndex([])
    if len(common_idx) > 0:
        for yr in sorted(set(common_idx.year)):
            mask = common_idx.year == yr
            if mask.sum() > 100:
                idx_yr = common_idx[mask]
                r_dyn = (1 + ret_dynamic.loc[idx_yr]).prod() - 1
                r_raw = (1 + ret_raw.loc[idx_yr]).prod() - 1
                r_bin = (1 + ret_binary.loc[idx_yr]).prod() - 1
                print(
                    f"     {yr}: 动态{r_dyn:+.1%} | 二元{r_bin:+.1%} | 不择时{r_raw:+.1%}"
                )

    # 保存
    df_out = pd.DataFrame(results)
    report_path = OUTPUT / "dynamic_timing_backtest.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    df_out.to_markdown(report_path, index=False)
    print(f"\n📄 报告 → {report_path}")

    # 简短结论
    if len(results) == 3:
        dyn_dd = float(results[2]["最大回撤"].rstrip("%")) / 100
        raw_dd = float(results[0]["最大回撤"].rstrip("%")) / 100
        dd_reduction = (abs(raw_dd) - abs(dyn_dd)) / abs(raw_dd) * 100 if raw_dd != 0 else 0
        print(f"\n  💡 动态择时回撤比无择时改善: {dd_reduction:.0f}%")


if __name__ == "__main__":
    main()
