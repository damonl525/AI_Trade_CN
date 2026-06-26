"""
五种动量公式 × 动态择时引擎 — 全量对比回测
"""
import pandas as pd
import numpy as np
from pathlib import Path
import json
import sys

# ---- Config ----
DATA_DIR = Path("data/market")
POOL_FILE = Path("data/etf_pool.json")
START_DATE = "2021-01-01"
LOOKBACK = 20
TOP_K = 3
REBALANCE_DAYS = 40
COMMISSION = 0.0005          # 万5
STAMP_TAX = 0.0005           # 卖出千0.5
MIN_COMMISSION = 5.0
SAFE_ASSET = "511010"        # 国债ETF，防御资产

# ---- Load pool ----
if POOL_FILE.exists():
    with open(POOL_FILE, encoding="utf-8") as f:
        pool_data = json.load(f)
    POOL = [{"code": k, "name": v} for k, v in pool_data.items()]
else:
    POOL = [
        {"code": "510050", "name": "上证50"}, {"code": "510300", "name": "沪深300"},
        {"code": "510500", "name": "中证500"}, {"code": "159915", "name": "创业板"},
        {"code": "588000", "name": "科创50"}, {"code": "512880", "name": "证券"},
        {"code": "512010", "name": "医药"},     {"code": "512690", "name": "酒"},
        {"code": "159930", "name": "能源"},     {"code": "518880", "name": "黄金"},
        {"code": "159819", "name": "AI智能"},   {"code": "511010", "name": "国债"},
    ]
ETF_CODES = [e["code"] for e in POOL]

# ---- Load price data ----
def load_prices(codes):
    dfs = {}
    for code in codes:
        f = DATA_DIR / f"etf_{code}.parquet"
        if f.exists():
            df = pd.read_parquet(f)
            df["date"] = pd.to_datetime(df["date"])
            df = df.sort_values("date")
            dfs[code] = df.set_index("date")["close"]
    price_df = pd.DataFrame(dfs).dropna()
    # Ensure datetime index
    price_df.index = pd.to_datetime(price_df.index)
    mask = price_df.index >= pd.Timestamp(START_DATE)
    return price_df[mask]

# ---- Momentum formulas ----
def momentum_baseline(prices, lookback=LOOKBACK):
    """基准 20日动量: (close / close[-N]) - 1"""
    return prices / prices.shift(lookback) - 1

def momentum_multiwindow(prices):
    """多窗口融合: (5日 + 20日 + 60日) / 3"""
    m5 = prices / prices.shift(5) - 1
    m20 = prices / prices.shift(20) - 1
    m60 = prices / prices.shift(60) - 1
    return (m5 + m20 + m60) / 3

def momentum_dual(prices, lookback=LOOKBACK):
    """双动量: 绝对动量(>0) × 相对动量"""
    raw_mom = prices / prices.shift(lookback) - 1
    # 绝对动量门槛: 负 → 给 −∞ 防止入选
    dual = raw_mom.where(raw_mom > 0, -999)
    return dual

def momentum_volweighted(prices, lookback=LOOKBACK):
    """波动率加权动量: momentum / rolling_volatility (年化)"""
    raw_mom = prices / prices.shift(lookback) - 1
    returns = prices.pct_change()
    vol = returns.rolling(60).std() * np.sqrt(252)  # 年化波动率
    vol = vol.replace(0, np.nan)
    return raw_mom / vol

def momentum_frog(prices, lookback=LOOKBACK):
    """Frog-in-the-Pan (Da, Gurun, Warachka 2014):
       动量 / 信息离散度 (ID)
       ID = 1 - 正负收益日数之差 / 总日数 的绝对值
       连续均匀小涨 → ID低 → Frog动量高
       跳跃暴拉     → ID高 → Frog动量低
    """
    raw_mom = prices / prices.shift(lookback) - 1
    returns = prices.pct_change()
    
    # 滚动窗口内正收益日数比例
    pos_days = (returns > 0).rolling(lookback).sum()
    neg_days = (returns < 0).rolling(lookback).sum()
    total_days = pos_days + neg_days
    total_days = total_days.replace(0, np.nan)
    
    # 信息离散度: |pos - neg| / total
    id_score = abs(pos_days - neg_days) / total_days
    
    # Frog: 动量 / (1 + ID)  → 越离散越扣分
    frog = raw_mom / (1 + id_score)
    return frog


# ---- Dynamic timing engine (from engine/timing.py) ----
def dynamic_position_size(price_df, index_code="510300"):
    """三信号融合动态仓位 0-1"""
    if index_code not in price_df.columns:
        return pd.Series(1.0, index=price_df.index)
    
    index = price_df[index_code].dropna()
    returns = index.pct_change()
    
    # Signal 1: Trend (200MA deviation)
    ma200 = index.rolling(200).mean()
    trend_dev = (index - ma200) / ma200
    trend_z = trend_dev / trend_dev.rolling(252).std().clip(lower=0.001)
    sig_trend = 1 / (1 + np.exp(-3 * trend_z))  # sigmoid → 0~1
    
    # Signal 2: Momentum regime (50-day return)
    mom50 = index / index.shift(50) - 1
    mom_z = mom50 / mom50.rolling(252).std().clip(lower=0.001)
    sig_mom = 1 / (1 + np.exp(-3 * mom_z))
    
    # Signal 3: Volatility regime (60-day vol rank)
    vol60 = returns.rolling(60).std() * np.sqrt(252)
    vol_rank = vol60.rolling(252).rank(pct=True)
    sig_vol = 1 - vol_rank  # 高波动 → 低仓位
    
    # Fusion: weighted average
    raw = 0.40 * sig_trend + 0.35 * sig_mom + 0.25 * sig_vol
    # Smooth and clip
    smooth = raw.rolling(5).mean()
    smooth = smooth.clip(0.05, 1.0)
    
    # 200MA 以下再打折
    below_ma = index < ma200
    smooth = smooth.where(~below_ma, smooth * 0.5)
    
    return smooth.clip(0.0, 1.0)


# ---- Backtest one run ----
def backtest_one(price_df, mom_func, mom_name, pos_series, top_k=TOP_K, rebalance_days=REBALANCE_DAYS):
    """单次回测: 给定动量函数 + 动态仓位"""
    dates = price_df.index
    cash = 1_000_000.0
    holdings = {}   # code → shares
    nav = []
    
    for i, date in enumerate(dates):
        if i % rebalance_days == 0:
            # Get momentum ranks
            mom = mom_func(price_df.iloc[:i+1]).iloc[-1]
            mom = mom.dropna()
            # Select top K
            top = mom.nlargest(top_k)
            
            target_codes = list(top.index)
            total_weight = 1.0
            if date in pos_series.index:
                tw = pos_series.loc[date]
                if not pd.isna(tw):
                    total_weight = tw
            weight_per = total_weight / top_k
            
            # Compute portfolio value
            portfolio_value = cash
            for code, shares in holdings.items():
                if code in price_df.columns:
                    portfolio_value += shares * price_df.loc[date, code]
            
            # Clear existing holdings
            cash = portfolio_value
            holdings = {}
            
            # Buy new
            for code in target_codes:
                if code in price_df.columns:
                    price = price_df.loc[date, code]
                    if pd.isna(price) or price <= 0:
                        continue
                    alloc = portfolio_value * weight_per
                    if pd.isna(alloc) or alloc <= 0:
                        continue
                    shares = int(alloc / price / 100) * 100  # round to lots
                    cost = shares * price
                    fee = max(cost * COMMISSION, MIN_COMMISSION)
                    cash -= (cost + fee)
                    holdings[code] = shares
        
        # Daily NAV
        pv = cash
        for code, shares in holdings.items():
            if code in price_df.columns:
                p = price_df.loc[date, code]
                if not pd.isna(p):
                    pv += shares * p
        nav.append({"date": date, "nav": pv})
    
    nav_df = pd.DataFrame(nav).set_index("date")
    returns = nav_df["nav"].pct_change().dropna()
    
    # Metrics
    total_ret = (nav_df["nav"].iloc[-1] / nav_df["nav"].iloc[0] - 1)
    years = len(nav_df) / 252
    ann_ret = (1 + total_ret) ** (1 / years) - 1 if years > 0 else 0
    cummax = nav_df["nav"].cummax()
    drawdown = (nav_df["nav"] - cummax) / cummax
    max_dd = drawdown.min()
    ann_vol = returns.std() * np.sqrt(252)
    sharpe = (returns.mean() / returns.std()) * np.sqrt(252) if returns.std() > 0 else 0
    win_rate = (returns > 0).mean()
    
    return {
        "name": mom_name,
        "total_return": total_ret,
        "ann_return": ann_ret,
        "max_drawdown": max_dd,
        "sharpe": sharpe,
        "ann_vol": ann_vol,
        "win_rate": win_rate,
        "nav": nav_df,
    }


# ---- Main comparison ----
def main():
    print("=" * 70)
    print("五种动量公式 × 动态择时 — 全量对比回测")
    print("=" * 70)
    
    print("\n[1/4] 加载数据...")
    prices = load_prices(ETF_CODES)
    print(f"   区间: {prices.index[0].strftime('%Y-%m-%d')} ~ {prices.index[-1].strftime('%Y-%m-%d')}")
    print(f"   ETF: {len(ETF_CODES)} 只, 交易日: {len(prices)}")
    
    # Dynamic position sizing
    print("\n[2/4] 计算动态择时仓位...")
    pos = dynamic_position_size(prices)
    avg_pos = pos.mean()
    print(f"   平均仓位: {avg_pos:.1%}")
    
    # Momentum formulas
    formulas = [
        (momentum_baseline, "基准20日"),
        (momentum_multiwindow, "多窗口融合"),
        (momentum_dual, "双动量"),
        (momentum_volweighted, "波动率加权"),
        (momentum_frog, "Frog-in-the-Pan"),
    ]
    
    results = []
    for func, name in formulas:
        print(f"\n[3/4] 回测: {name} x 动态择时...")
        r = backtest_one(prices, func, name, pos)
        results.append(r)
        print(f"   收益 {r['total_return']:+.1%}  |  年化 {r['ann_return']:+.1%}  |  回撤 {r['max_drawdown']:+.1%}  |  夏普 {r['sharpe']:.2f}  |  胜率 {r['win_rate']:.1%}")
    
    # ---- Comparison table ----
    print("\n")
    print("=" * 70)
    print("[4/4] 最终排名 (按总收益)")
    print("=" * 70)
    
    ranked = sorted(results, key=lambda x: x["total_return"], reverse=True)
    
    print(f"{'排名':<5} {'方案':<16} {'总收益':>8} {'年化':>8} {'最大回撤':>8} {'夏普':>7} {'胜率':>7}")
    print("-" * 70)
    for i, r in enumerate(ranked):
        print(f"{i+1:<5} {r['name']:<16} {r['total_return']:>+7.1%} {r['ann_return']:>+7.1%} {r['max_drawdown']:>+7.1%} {r['sharpe']:>6.2f} {r['win_rate']:>6.1%}")
    
    # ---- Winner vs baseline ----
    winner = ranked[0]
    baseline = [r for r in results if r["name"] == "基准20日"][0]
    
    print("\n" + "=" * 70)
    print("== 胜者 vs 基准 ==")
    print("=" * 70)
    print(f"   {winner['name']}: 收益 {winner['total_return']:+.1%}  |  回撤 {winner['max_drawdown']:+.1%}")
    print(f"   基准20日:        收益 {baseline['total_return']:+.1%}  |  回撤 {baseline['max_drawdown']:+.1%}")
    print(f"   超额收益: {winner['total_return'] - baseline['total_return']:+.1%}")
    print(f"   回撤改善: {baseline['max_drawdown'] - winner['max_drawdown']:+.1%}")
    
    # ---- Save winner NAV ----
    out = Path("output/compare_momentum_timing.csv")
    out.parent.mkdir(exist_ok=True)
    pd.DataFrame({r["name"]: r["nav"]["nav"] for r in results}).to_csv(out)
    print(f"\n净值曲线保存: {out}")


if __name__ == "__main__":
    main()
