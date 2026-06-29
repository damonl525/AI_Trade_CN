"""全动量公式同数据对比 — 均叠加动态择时
用法: uv run python backtest_momentum_formulas.py
"""

import pandas as pd
import numpy as np
from pathlib import Path
from config import DATA_MARKET
from data.fetcher import load
from engine.dynamic_timing import dynamic_position_series
from live_signal import FULL_POOL


def load_pool_data(pool: dict) -> dict[str, pd.DataFrame]:
    data = {}
    for code in pool:
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
    formula: str = "base",
    lookback: int = 10,
    top_k: int = 3,
    rebalance_days: int = 40,
) -> pd.Series:
    """动量轮动日收益 — 支持多种动量公式"""
    closes = {}
    for code, df in data.items():
        closes[code] = df["close"]
    prices = pd.DataFrame(closes).dropna()
    if prices.empty:
        return pd.Series(dtype=float)

    # ── 动量计算公式 ──
    if formula == "base":
        momentum = prices.pct_change(lookback)
    elif formula == "vol_weighted":
        raw_mom = prices.pct_change(lookback)
        vol = prices.pct_change().rolling(20).std() * np.sqrt(252)
        vol = vol.replace(0, np.nan)
        momentum = raw_mom.div(vol.clip(lower=0.01))
    elif formula == "frog":
        # Frog-in-the-Pan: 连续小涨 > 跳涨
        raw_mom = prices.pct_change(lookback)
        daily_ret = prices.pct_change()
        # 计算正收益天数占比(涨的交易日比例)
        pos_days = (daily_ret > 0).rolling(lookback).sum()
        # Frog 得分 = 动量 × 正收益比例
        momentum = raw_mom * (pos_days / lookback)
    elif formula == "dual":
        # 绝对动量门槛: 跌的不买
        raw_mom = prices.pct_change(lookback)
        momentum = raw_mom.where(raw_mom > 0, -999)  # 负动量打到底
    elif formula == "multi_window":
        mom_5 = prices.pct_change(5)
        mom_20 = prices.pct_change(20)
        mom_60 = prices.pct_change(60)
        momentum = (mom_5 + mom_20 + mom_60) / 3.0
    else:
        momentum = prices.pct_change(lookback)

    # ── 权重生成 ──
    weights = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
    rebalance_dates = prices.index[::rebalance_days]

    for i, date in enumerate(rebalance_dates):
        if i == 0:
            continue
        mom_slice = momentum.loc[:date].iloc[-1]
        top = mom_slice.nlargest(top_k)
        top = top[top.index != "511010"]  # 排除国债
        if top.empty or top.sum() == 0:
            continue
        end_date = (
            rebalance_dates[i + 1] if i + 1 < len(rebalance_dates) else prices.index[-1]
        )
        w = pd.Series(0.0, index=prices.columns)
        for code in top.index:
            w[code] = 1.0 / len(top)
        weights.loc[date:end_date] = w.values

    daily_ret = prices.pct_change().fillna(0)
    port_ret = (daily_ret * weights.shift(1)).sum(axis=1).fillna(0)
    port_ret.name = formula
    return port_ret


def compute_dynamic(port_ret: pd.Series, data, dpos) -> pd.Series:
    """给轮动收益叠加动态仓位缩放 + 国债配置"""
    common_idx = port_ret.index.intersection(dpos.index)
    port_ret = port_ret.loc[common_idx]
    dpos = dpos.loc[common_idx]

    scaled = port_ret * dpos["position"]
    bond_ret = pd.Series(0.0, index=common_idx)
    if "511010" in data:
        b = data["511010"]["close"].pct_change().fillna(0)
        bond_ret = b.loc[common_idx]
    cash_ret = bond_ret * (1 - dpos["position"])
    total = scaled + cash_ret
    total.name = port_ret.name
    return total


def apply_drawdown_guard(
    port_ret: pd.Series,
    dpos: pd.DataFrame,
    threshold: float = -0.08,
    recovery: float = -0.03,
    guard_pos: float = 0.30,
) -> pd.Series:
    """回撤保护: 累计回撤超过阈值时强制降仓

    逻辑:
      - 跟踪滚动的累计净值峰值
      - 当前回撤超过 threshold (默认 -8%) → 仓位上限锁在 guard_pos (默认 30%)
      - 回撤恢复到 recovery (-3%) 以内 → 解除保护
    """
    cum = (1 + port_ret).cumprod()
    peak = cum.expanding().max()
    drawdown = (cum - peak) / peak

    # 保护状态: True = 当前处于回撤保护中
    guarded = pd.Series(False, index=port_ret.index)
    in_guard = False
    for i in range(len(drawdown)):
        dd = drawdown.iloc[i]
        if not in_guard and dd < threshold:
            in_guard = True
        elif in_guard and dd > recovery:
            in_guard = False
        guarded.iloc[i] = in_guard

    # 对齐 dpos 到 port_ret 的索引
    common_idx = port_ret.index.intersection(dpos.index)
    guarded = guarded.loc[common_idx]
    pos = dpos.loc[common_idx, "position"].copy()
    pos[guarded] = pos[guarded].clip(upper=guard_pos)

    scaled = port_ret.loc[common_idx] * pos
    # 国债配置
    bond_ret = pd.Series(0.0, index=common_idx)
    cash_ret = bond_ret * (1 - pos)
    total = scaled + cash_ret
    total.name = f"{port_ret.name}+DD"
    return total


def stats(returns: pd.Series) -> dict:
    if returns.empty or returns.std() == 0:
        return {"total_return": 0.0, "cagr": 0.0, "max_dd": 0.0, "sharpe": 0.0, "volatility": 0.0, "win_rate": 0.0}
    total = (1 + returns).prod() - 1
    n_years = len(returns) / 252
    cagr = (1 + total) ** (1 / n_years) - 1 if n_years > 0 else 0
    cum = (1 + returns).cumprod()
    peak = cum.expanding().max()
    dd = (cum - peak) / peak
    max_dd = dd.min()
    sharpe = (returns.mean() / returns.std()) * np.sqrt(252) if returns.std() > 0 else 0
    vol = returns.std() * np.sqrt(252)
    wr = (returns > 0).mean()
    return {"total_return": total, "cagr": cagr, "max_dd": max_dd, "sharpe": sharpe, "volatility": vol, "win_rate": wr}


# ── 主逻辑 ──
if __name__ == "__main__":
    pool = FULL_POOL
    print(f"\n[LOAD] {len(pool)} ETFs ...")
    data = load_pool_data(pool)
    symbols_found = list(data.keys())
    print(f"  实际有数据: {symbols_found}")

    if len(symbols_found) < 3:
        print("❌ 数据不足")
        exit(1)

    dpos = dynamic_position_series("000300")
    if dpos.empty:
        print("❌ 动态择时数据缺失")
        exit(1)

    formulas = ["base", "vol_weighted", "frog", "dual", "multi_window"]
    labels = {
        "base": "基准20日", "vol_weighted": "波动率加权",
        "frog": "Frog-in-Pan", "dual": "双动量",
        "multi_window": "多窗口融合",
    }

    print(f"\n{'='*75}")
    print(f"  同数据 · 同池子 · 同动态择时 — 全部动量公式对比")
    print(f"{'='*75}")
    print(f"  数据窗口: {dpos.index[0].strftime('%Y-%m-%d')} ~ {dpos.index[-1].strftime('%Y-%m-%d')}")
    print(f"  ETF 数: {len(symbols_found)}  参数: lookback=10 top_k=3 rebal=40d")
    print()

    results = []
    for f in formulas:
        port = compute_rotation_returns(data, formula=f, lookback=10, top_k=3, rebalance_days=40)
        dyn = compute_dynamic(port, data, dpos)
        s = stats(dyn)
        results.append((labels[f], f, s))
        print(f"  {labels[f]:<8} 总收益{s['total_return']:>+8.2%}  年化{s['cagr']:>7.2%}  "
              f"回撤{s['max_dd']:>8.2%}  夏普{s['sharpe']:>5.2f}  波动{s['volatility']:>5.1%}  "
              f"胜率{s['win_rate']:>4.0%}")

    # ── 回撤保护对比 ──
    print(f"\n  {'─'*65}")
    print(f"  🛡️ 回撤保护对比 (阈值 −8%, 保护期仓位 ≤ 30%):")
    port_d = compute_rotation_returns(data, formula="dual", lookback=10, top_k=3, rebalance_days=40)
    dyn_dd = apply_drawdown_guard(port_d, dpos)
    s_dd = stats(dyn_dd)
    port_d_base = compute_rotation_returns(data, formula="dual", lookback=10, top_k=3, rebalance_days=40)
    dyn_base = compute_dynamic(port_d_base, data, dpos)
    s_base = stats(dyn_base)
    print(f"  原版双动量     总收益{s_base['total_return']:>+8.2%}  年化{s_base['cagr']:>7.2%}  "
          f"回撤{s_base['max_dd']:>8.2%}  夏普{s_base['sharpe']:>5.2f}")
    print(f"  双动量+回撤保护  总收益{s_dd['total_return']:>+8.2%}  年化{s_dd['cagr']:>7.2%}  "
          f"回撤{s_dd['max_dd']:>8.2%}  夏普{s_dd['sharpe']:>5.2f}")
    improvement = s_dd['max_dd'] - s_base['max_dd']
    print(f"  回撤改善: {improvement:+.2%}  |  收益差: {s_dd['total_return']-s_base['total_return']:+.2%}")
    results.append(("双动量+DD", "dual+DD", s_dd))

    # 排名
    results.sort(key=lambda x: x[2]["total_return"], reverse=True)
    print(f"\n{'='*75}")
    print(f"  🏆 最终排名 (按总收益):")
    for i, (label, _, s) in enumerate(results, 1):
        print(f"  {i}. {label:<8}  +{s['total_return']:+.2%}  年化{s['cagr']:+.2%}  "
              f"回撤{s['max_dd']:+.2%}  夏普{s['sharpe']:+.2f}")

    # 分年对比
    print(f"\n{'='*75}")
    print(f"  📅 分年对比:")
    # 重新跑全部并记录
    yearly = {}
    for f in formulas:
        port = compute_rotation_returns(data, formula=f, lookback=10, top_k=3, rebalance_days=40)
        dyn = compute_dynamic(port, data, dpos)
        yearly[f] = dyn

    years = sorted(set(d.year for d in dpos.index))
    header = f"  {'年份':<6}"
    for f in formulas:
        header += f" {labels[f]:<10}"
    print(header)
    print(f"  {'─'*6} {'─'*60}")

    for yr in years:
        row = f"  {yr:<6}"
        for f in formulas:
            yr_ret = yearly[f][yearly[f].index.year == yr]
            if yr_ret.empty:
                row += f" {'N/A':>10}"
            else:
                total = (1 + yr_ret).prod() - 1
                row += f" {total:>+9.1%}"
        print(row)
