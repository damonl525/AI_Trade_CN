"""训练/测试分离回测 — 科学验证策略泛化能力

用法:
    PYTHONPATH= uv run python train_test_backtest.py

设计理念:
    全期回测（2020-2026）的 +76.84% 是用同一段数据调参+评估，
    有严重过拟合风险。本脚本实施标准量化验证流程：

    ┌─────────────────────────────────────────────────────┐
    │  训练期 (2016-2021)         测试期 (2022-2026)       │
    │  ┌──────────────┐          ┌──────────────┐         │
    │  │ 11只老ETF    │    →     │ 16只全ETF    │         │
    │  │ 全部公式×参数│  最优参   │ 检验泛化能力  │         │
    │  │ 找训练冠军   │   数     │ 看是否过拟合  │         │
    │  └──────────────┘          └──────────────┘         │
    └─────────────────────────────────────────────────────┘

训练期:
    - 时间段: 2016-01-01 ~ 2021-12-31
    - ETF: 仅11只在2016年已上市的（排除科创50/AI智能/机器人等）
    - 目的: 在全部5种公式×2种lookback×2种top_k里找最优

测试期:
    - 时间段: 2022-01-01 ~ 2026-06-26
    - ETF: 全部16只（含后上市的，按上市时间动态加入）
    - 目的: 用训练最优参数跑测试期，对比全期最优，判断过拟合程度

判断标准:
    - 训练冠军在测试期大幅衰减 → 过拟合
    - 训练非冠军在测试期表现更好 → 泛化能力强

2026-06-29 实测结论:
    - 训练最优 Frog/10/3: 训练 +166% → 测试 +55%（衰减111pp，过拟合）
    - 双动量 dual/10/3:   训练 +147% → 测试 +78%（衰减69pp，泛化胜出）
    - 建议: 继续用双动量/10/3，但实盘预期下调到年化5-10%
"""
import pandas as pd
import numpy as np
from pathlib import Path
from config import DATA_MARKET
from data.fetcher import load
from engine.dynamic_timing import dynamic_position_series
from live_signal import FULL_POOL

# ── 时间分割点 ──
TRAIN_END = "2021-12-31"
TEST_START = "2022-01-01"

# ── 数据加载 ──
def load_all_data() -> dict[str, pd.DataFrame]:
    data = {}
    for code in FULL_POOL:
        path = DATA_MARKET / f"etf_{code}.parquet"
        if path.exists():
            df = load(path)
            if "date" in df.columns:
                df["date"] = pd.to_datetime(df["date"])
                df = df.set_index("date").sort_index()
            data[code] = df
    return data


def build_price_matrix(data: dict, symbols: list[str]) -> pd.DataFrame:
    """从 data dict 构建 close 价格矩阵"""
    closes = {}
    for code in symbols:
        if code in data:
            closes[code] = data[code]["close"]
    return pd.DataFrame(closes).dropna()


# ── 动量公式 (复用 backtest_momentum_formulas.py 逻辑) ──
def compute_momentum(prices: pd.DataFrame, formula: str, lookback: int) -> pd.DataFrame:
    if formula == "base":
        return prices.pct_change(lookback)
    elif formula == "vol_weighted":
        raw = prices.pct_change(lookback)
        vol = prices.pct_change().rolling(20).std() * np.sqrt(252)
        vol = vol.replace(0, np.nan)
        return raw.div(vol.clip(lower=0.01))
    elif formula == "frog":
        raw = prices.pct_change(lookback)
        daily_ret = prices.pct_change()
        pos_days = (daily_ret > 0).rolling(lookback).sum()
        return raw * (pos_days / lookback)
    elif formula == "dual":
        raw = prices.pct_change(lookback)
        return raw.where(raw > 0, -999)
    elif formula == "multi_window":
        return (prices.pct_change(5) + prices.pct_change(20) + prices.pct_change(60)) / 3.0
    else:
        return prices.pct_change(lookback)


def compute_rotation_returns(
    prices: pd.DataFrame,
    formula: str = "base",
    lookback: int = 10,
    top_k: int = 3,
    rebalance_days: int = 40,
) -> pd.Series:
    """动量轮动日收益"""
    momentum = compute_momentum(prices, formula, lookback)
    weights = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
    rebalance_dates = prices.index[::rebalance_days]

    for i, date in enumerate(rebalance_dates):
        if i == 0:
            continue
        mom_slice = momentum.loc[:date].iloc[-1]
        top = mom_slice.nlargest(top_k)
        # 排除国债 511010（不应进入 Top K 被选为权益仓位）
        top = top[top.index != "511010"]
        if top.empty or top.sum() <= -900:  # dual 公式会把负动量打到 -999
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


def apply_dynamic_timing(
    port_ret: pd.Series, dpos: pd.DataFrame, bond_ret: pd.Series
) -> pd.Series:
    """叠加动态仓位缩放 + 剩余仓位配国债"""
    common = port_ret.index.intersection(dpos.index)
    ret = port_ret.loc[common]
    pos = dpos.loc[common, "position"]
    bond = bond_ret.loc[common] if len(bond_ret) > 0 else pd.Series(0.0, index=common)

    scaled = ret * pos
    cash_ret = bond * (1 - pos)
    total = scaled + cash_ret
    total.name = ret.name
    return total


def calc_stats(returns: pd.Series) -> dict:
    if returns.empty or returns.std() == 0:
        return {"total": 0, "cagr": 0, "max_dd": 0, "sharpe": 0}
    total = (1 + returns).prod() - 1
    years = len(returns) / 252
    cagr = (1 + total) ** (1 / years) - 1 if years > 0 else 0
    cum = (1 + returns).cumprod()
    peak = cum.expanding().max()
    dd = (cum - peak) / peak
    max_dd = dd.min()
    sharpe = (returns.mean() / returns.std()) * np.sqrt(252) if returns.std() > 0 else 0
    return {"total": total, "cagr": cagr, "max_dd": max_dd, "sharpe": sharpe,
            "years": years, "ann_vol": returns.std() * np.sqrt(252)}


def run_backtest_period(
    data: dict,
    dpos: pd.DataFrame,
    symbols: list[str],
    period_label: str,
    formulas: list[str],
    lookbacks: list[int],
    top_ks: list[int],
) -> pd.DataFrame:
    """在指定时间段内跑全部公式×参数组合"""
    prices = build_price_matrix(data, symbols)
    # 切割时间段
    prices = prices[(prices.index >= dpos.index[0]) & (prices.index <= dpos.index[-1])]

    # 国债收益序列
    bond_ret = pd.Series(0.0, index=prices.index)
    if "511010" in data:
        b = data["511010"]["close"].pct_change().fillna(0)
        common = prices.index.intersection(b.index)
        bond_ret.loc[common] = b.loc[common]

    results = []
    for f in formulas:
        for lb in lookbacks:
            for tk in top_ks:
                try:
                    port = compute_rotation_returns(prices, formula=f, lookback=lb, top_k=tk)
                    dyn = apply_dynamic_timing(port, dpos, bond_ret)
                    s = calc_stats(dyn)
                    results.append({
                        "period": period_label,
                        "formula": f, "lookback": lb, "top_k": tk,
                        "total_return": s["total"], "cagr": s["cagr"],
                        "max_dd": s["max_dd"], "sharpe": s["sharpe"],
                        "years": s["years"], "ann_vol": s.get("ann_vol", 0),
                    })
                except Exception as e:
                    continue
    return pd.DataFrame(results)


# ═══════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("=" * 80)
    print("  训练/测试分离回测 — 科学验证")
    print(f"  训练期: 2016-01-01 ~ {TRAIN_END}")
    print(f"  测试期: {TEST_START} ~ 2026-06-26")
    print("=" * 80)

    # 1. 加载全部数据
    print("\n[1/5] 加载数据...")
    all_data = load_all_data()
    print(f"  共 {len(all_data)} 只 ETF 有数据")

    # 2. 确定训练期可用的 ETF（2016 年已存在）
    train_symbols = []
    for code, df in all_data.items():
        if str(df.index[0])[:10] <= "2016-01-05":
            train_symbols.append(code)
    print(f"  训练期可用 ETF: {len(train_symbols)} 只")
    print(f"    {', '.join(sorted(train_symbols))}")

    test_symbols_all = sorted(all_data.keys())
    print(f"  测试期可用 ETF: {len(test_symbols_all)} 只 (含后上市的)")

    # 3. 动态择时（全期）
    print("\n[2/5] 计算动态择时仓位...")
    dpos_full = dynamic_position_series("000300")
    dpos_full = dpos_full[dpos_full.index >= "2016-01-01"]
    print(f"  全期仓位序列: {dpos_full.index[0].strftime('%Y-%m-%d')} ~ {dpos_full.index[-1].strftime('%Y-%m-%d')}")
    print(f"  平均仓位: {dpos_full['position'].mean():.1%}")

    # 4. 训练期：跑全部公式×参数组合
    print("\n[3/5] 训练期 (2016-2021) 优化...")
    formulas = ["base", "vol_weighted", "frog", "dual", "multi_window"]
    labels = {"base": "基准", "vol": "波动率加权", "frog": "Frog", "dual": "双动量", "multi": "多窗口"}

    dpos_train = dpos_full[dpos_full.index <= TRAIN_END]
    df_train = run_backtest_period(
        all_data, dpos_train, train_symbols, "TRAIN",
        formulas=formulas, lookbacks=[10, 20], top_ks=[2, 3],
    )

    if df_train.empty:
        print("  ❌ 训练期回测失败")
        exit(1)

    # 找最优
    best = df_train.loc[df_train["total_return"].idxmax()]
    print(f"\n  🏆 训练期最优:")
    print(f"     公式: {best['formula']}  lookback={int(best['lookback'])}  top_k={int(best['top_k'])}")
    print(f"     总收益: {best['total_return']:+.2%}  年化: {best['cagr']:+.2%}  "
          f"回撤: {best['max_dd']:+.2%}  夏普: {best['sharpe']:.2f}")

    # 打印训练期全部结果
    print(f"\n  📊 训练期全部组合 (按总收益排序):")
    top10 = df_train.nlargest(10, "total_return")
    print(f"  {'排名':<5} {'公式':<12} {'lb':>4} {'tk':>4} {'总收益':>9} {'年化':>8} {'回撤':>8} {'夏普':>6}")
    print(f"  {'-'*62}")
    for i, (_, r) in enumerate(top10.iterrows()):
        print(f"  {i+1:<5} {r['formula']:<12} {int(r['lookback']):>4} {int(r['top_k']):>4} "
              f"{r['total_return']:>+8.2%} {r['cagr']:>+7.2%} {r['max_dd']:>+7.2%} {r['sharpe']:>5.2f}")

    # 5. 测试期：用训练最优参数
    print(f"\n[4/5] 测试期 (2022-2026) 验证...")
    dpos_test = dpos_full[dpos_full.index >= TEST_START]

    # ── 训练最优 ──
    print(f"\n  🔬 方案A: 训练最优参数 ({best['formula']}, lb={int(best['lookback'])}, tk={int(best['top_k'])})")
    prices_test = build_price_matrix(all_data, test_symbols_all)
    prices_test = prices_test[prices_test.index >= TEST_START]
    bond_test = pd.Series(0.0, index=prices_test.index)
    if "511010" in all_data:
        b = all_data["511010"]["close"].pct_change().fillna(0)
        common = prices_test.index.intersection(b.index)
        bond_test.loc[common] = b.loc[common]

    port_a = compute_rotation_returns(
        prices_test, formula=best["formula"],
        lookback=int(best["lookback"]), top_k=int(best["top_k"])
    )
    dyn_a = apply_dynamic_timing(port_a, dpos_test, bond_test)
    s_a = calc_stats(dyn_a)
    print(f"     总收益: {s_a['total']:+.2%}  年化: {s_a['cagr']:+.2%}  "
          f"回撤: {s_a['max_dd']:+.2%}  夏普: {s_a['sharpe']:.2f}")

    # ── 全期最优（当前文档里的 +76.84%）──
    print(f"\n  🔬 方案B: 当前全期最优 (dual, lb=10, tk=3)")
    port_b = compute_rotation_returns(prices_test, formula="dual", lookback=10, top_k=3)
    dyn_b = apply_dynamic_timing(port_b, dpos_test, bond_test)
    s_b = calc_stats(dyn_b)
    print(f"     总收益: {s_b['total']:+.2%}  年化: {s_b['cagr']:+.2%}  "
          f"回撤: {s_b['max_dd']:+.2%}  夏普: {s_b['sharpe']:.2f}")

    # ── 测试期全公式排名 ──
    print(f"\n  📊 测试期全公式对比 (lb=10, tk=3):")
    for f in formulas:
        port = compute_rotation_returns(prices_test, formula=f, lookback=10, top_k=3)
        dyn = apply_dynamic_timing(port, dpos_test, bond_test)
        s = calc_stats(dyn)
        print(f"     {labels.get(f, f):<8}  {s['total']:>+8.2%}  年化{s['cagr']:>+7.2%}  "
              f"回撤{s['max_dd']:>+7.2%}  夏普{s['sharpe']:>5.2f}")

    # 6. 总结
    print(f"\n[5/5] 总结对比")
    print(f"  {'='*70}")
    print(f"  {'':<20} {'训练期 (2016-2021)':<25} {'测试期 (2022-2026)':<25}")
    print(f"  {'─'*70}")
    print(f"  {'训练最优':<20} {best['total_return']:>+24.2%}  "
          f"{s_a['total']:>+24.2%}")
    print(f"  {'全期最优 (dual/10/3)':<20} {'—':>25}  "
          f"{s_b['total']:>+24.2%}")
    print(f"  {'─'*70}")
    gap = s_a['total'] - best['total_return']
    print(f"  {'训练→测试衰减':<20} {gap:>+24.2%}")
    print()
    if gap < -0.10:
        print(f"  ⚠️ 严重过拟合: 测试期比训练期低了 {abs(gap):.0%}，策略泛化能力差")
    elif gap < -0.03:
        print(f"  ⚠️ 轻度过拟合: 测试期比训练期低了 {abs(gap):.0%}")
    else:
        print(f"  ✅ 策略稳定: 测试期表现与训练期一致或更好")
