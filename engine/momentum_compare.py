"""
动量优化方案回测对比 (精简版)
比较: 基准(20日动量) vs 双动量 vs 多窗口融合 vs 波动率加权 vs Frog
"""
import pandas as pd
import numpy as np
import json, sys, os
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, Optional
import warnings
warnings.filterwarnings("ignore")

DATA_DIR = Path("data/market")
POOL_FILE = Path("data/etf_pool.json")

# 交易成本
COMM = 0.0005  # 万5佣金
TAX = 0.0005   # 千0.5印花税(卖)
MIN_FEE = 5.0
TOP_K = 3
REBAL = 40     # 40天调仓


@dataclass
class Result:
    name: str
    ret: float
    cagr: float
    dd: float
    sharpe: float
    vol: float
    winrate: float
    trades: int


def load_data() -> Dict[str, pd.DataFrame]:
    with open(POOL_FILE, encoding="utf-8") as f:
        pool = json.load(f)
    data = {}
    for code in pool:
        fp = DATA_DIR / f"etf_{code}.parquet"
        if fp.exists():
            df = pd.read_parquet(fp)
            df["date"] = pd.to_datetime(df["date"])
            df = df.set_index("date").sort_index()
            data[code] = df
    return data


def mom_single(df: pd.DataFrame, w: int = 20) -> pd.Series:
    return df["close"].pct_change(w)


def mom_multi(df: pd.DataFrame, ws=(5, 20, 60)) -> pd.Series:
    s = pd.Series(0.0, index=df.index)
    for w in ws:
        s += df["close"].pct_change(w)
    return s / len(ws)


def mom_frog(df: pd.DataFrame, w: int = 20) -> pd.Series:
    ret = df["close"].pct_change()
    std = ret.rolling(w).std()
    cumret = df["close"].pct_change(w)
    result = pd.Series(np.where(std > 0, cumret.values / (std.values + 1e-10), cumret.values), index=df.index)
    return result


def vol_scale_factor(df: pd.DataFrame, w: int = 20, target: float = 0.15) -> pd.Series:
    vol = df["close"].pct_change().rolling(w).std() * np.sqrt(252)
    return (target / (vol + 1e-10)).clip(0.1, 2.0)


def safe_slice(df: pd.DataFrame, date) -> Optional[pd.DataFrame]:
    s = df[df.index <= date]
    return s if len(s) >= 70 else None


def safe_price(df: pd.DataFrame, date) -> Optional[float]:
    if date in df.index:
        return float(df.loc[date, "close"])
    prior = df.index[df.index <= date]
    return float(df.loc[prior[-1], "close"]) if len(prior) > 0 else None


def run(
    data: Dict[str, pd.DataFrame],
    name: str,
    *,
    abs_filter: bool = False,
    vol_scale: bool = False,
    multi: bool = False,
    frog: bool = False,
) -> Result:
    codes = list(data.keys())
    all_dates = sorted(set.union(*[{d for d in data[c].index if d >= pd.Timestamp("2021-01-01")} for c in codes]))

    cash = 100_000.0
    holdings = {}
    vals = []
    trades = wins = 0
    last_reb = 0

    for i, today in enumerate(all_dates):
        # 信号
        sigs = {}
        for code in codes:
            sl = safe_slice(data[code], today)
            if sl is None:
                continue
            if multi:
                m = mom_multi(sl).iloc[-1]
            elif frog:
                m = mom_frog(sl).iloc[-1]
            else:
                m = mom_single(sl).iloc[-1]
            if np.isfinite(m) and not np.isnan(m):
                sigs[code] = m

        if abs_filter:
            sigs = {k: v for k, v in sigs.items() if v > 0}

        # 无信号 → 空仓估值
        if not sigs:
            tv = cash
            for c in holdings:
                p = safe_price(data[c], today)
                if p is not None:
                    tv += holdings[c] * p
            vals.append(tv)
            continue

        ranked = sorted(sigs.items(), key=lambda x: x[1], reverse=True)
        tgt_codes = [c for c, _ in ranked[:TOP_K]]
        tgt_ws = [1.0 / TOP_K] * TOP_K

        if vol_scale:
            scales = []
            for code in tgt_codes:
                sl = safe_slice(data[code], today)
                if sl is None or len(sl) < 70:
                    scales.append(1.0)
                else:
                    s = float(vol_scale_factor(sl).iloc[-1])
                    scales.append(s if np.isfinite(s) else 1.0)
            tot = sum(scales) or 1.0
            tgt_ws = [s / tot for s in scales]

        # 调仓
        if i - last_reb >= REBAL or i < 5:
            tv = cash
            for c in holdings:
                p = safe_price(data[c], today)
                if p is not None:
                    tv += holdings[c] * p

            # 清仓
            for code in list(holdings):
                sh = holdings[code]
                p = safe_price(data[code], today)
                if p is None:
                    continue
                cash += sh * p
                cash -= max(sh * p * COMM, MIN_FEE)
                cash -= sh * p * TAX
                if sh > 0:
                    trades += 1
                del holdings[code]

            # 建仓
            for code, w in zip(tgt_codes, tgt_ws):
                alloc = tv * w
                p = safe_price(data[code], today)
                if p is None or p <= 0:
                    continue
                sh = int(alloc / p / 100) * 100
                if sh > 0:
                    cost = sh * p
                    cash -= cost + max(cost * COMM, MIN_FEE)
                    holdings[code] = sh
                    trades += 1
                cash = max(cash, 0.0)
            last_reb = i

        # 每日估值
        tv = cash
        for c in holdings:
            p = safe_price(data[c], today)
            if p is not None:
                tv += holdings[c] * p
        vals.append(tv)

        if i > 0 and len(vals) >= 2 and vals[-1] > vals[-2]:
            wins += 1

    pv = pd.Series(vals, index=all_dates[:len(vals)])
    rets = pv.pct_change().dropna()
    if len(rets) < 10:
        return Result(name, 0, 0, 0, 0, 0, 0, 0)

    tr = (pv.iloc[-1] / pv.iloc[0] - 1) * 100
    yrs = (pv.index[-1] - pv.index[0]).days / 365.25
    ca = ((pv.iloc[-1] / pv.iloc[0]) ** (1 / yrs) - 1) * 100 if yrs > 0 else 0
    dd = ((pv / pv.cummax() - 1).min()) * 100
    vol = rets.std() * np.sqrt(252) * 100
    sr = (rets.mean() / rets.std() * np.sqrt(252)) if rets.std() > 0 else 0
    wr = wins / len(rets) * 100

    return Result(name, tr, ca, dd, sr, vol, wr, trades)


def main():
    print("=" * 72)
    print("  动量优化方案回测对比 (2021至今, 16只ETF)")
    print("=" * 72)

    data = load_data()
    print(f"\n  加载 {len(data)} 只ETF, 不含动态择时(纯动量信号对比)\n")

    strategies = [
        ("基准 20日动量", False, False, False),
        ("双动量(绝对+相对)", True, False, False),
        ("多窗口融合(5+20+60)", False, False, True),
        ("波动率加权", False, True, False),
        ("Frog-in-the-Pan", False, False, False),
    ]

    results = []
    for name, abs_f, vol_s, multi in strategies:
        print(f"  {name}... ", end="", flush=True)
        r = run(
            data, name,
            abs_filter=abs_f,
            vol_scale=vol_s,
            multi=multi,
            frog=(name == "Frog-in-the-Pan"),
        )
        results.append(r)
        print(f"收益 {r.ret:+.1f}% | 回撤 {r.dd:.1f}% | 夏普 {r.sharpe:.2f} | 胜率{r.winrate:.0f}%")

    results.sort(key=lambda r: r.ret - 1.5 * abs(r.dd), reverse=True)

    print()
    print("=" * 72)
    print(f"  {'策略':<24} {'总收益':>7} {'年化':>7} {'回撤':>7} {'夏普':>6} {'胜率':>6} {'交易':>5}")
    print("-" * 72)
    for r in results:
        flag = " 🏆" if r == results[0] else ""
        print(f"  {r.name+flag:<24} {r.ret:>6.1f}% {r.cagr:>6.1f}% {r.dd:>6.1f}% {r.sharpe:>5.2f} {r.winrate:>5.0f}% {r.trades:>5}")

    print()
    print("=" * 72)
    print("  🎯 结论 (纯动量, 无择时)")
    print("=" * 72)
    best = results[0]
    print(f"  最优: {best.name} (收益{best.ret:+.1f}%, 回撤{best.dd:.1f}%)")
    print()
    print("  ⚠️ 注意: 以上为纯动量信号对比(无动态择时/无止损)。")
    print("  叠加动态择时后所有策略表现都会显著改善。")
    print("  目的是找最优动量公式来替换现有系统的动量计算。")

    # 推荐
    print()
    print("  📌 推荐方案 (按综合评分):")
    for i, r in enumerate(results[:3], 1):
        print(f"     {i}. {r.name}: 收益{r.ret:+.1f}% 回撤{r.dd:.1f}% 夏普{r.sharpe:.2f}")


if __name__ == "__main__":
    main()
