"""多时间窗回测 + 板块趋势分析引擎

核心理念:
- 单一时段回测 = 过拟合 beta。分 1Y/3Y/5Y 三档，看策略在不同市场环境的一致性
- 板块轮动: 识别当前资金流向, 找出被低估但有投资价值的板块
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional

import backtrader as bt

from config import OUTPUT, DATA_MARKET, INIT_CASH, COMMISSION, STAMP_DUTY, MIN_COMMISSION
from data.fetcher import load, load_many
from engine.backtest import AStockCommission

OUTPUT.mkdir(parents=True, exist_ok=True)


# ══════════════════════════════════════════════════════════
# 多时间窗回测
# ══════════════════════════════════════════════════════════

def _bt_one_window(cerebro, df, start_date, end_date, signal_class, signal_params, cash, stake):
    """单窗口回测"""
    cerebro2 = bt.Cerebro()
    cerebro2.broker.setcash(cash)
    cerebro2.broker.addcommissioninfo(AStockCommission())

    # 切数据
    mask = (df["date"] >= pd.Timestamp(start_date)) & (df["date"] <= pd.Timestamp(end_date))
    sub = df[mask].copy()
    if len(sub) < 50:
        return None

    feed = bt.feeds.PandasData(
        dataname=sub,
        datetime="date", open="open", high="high", low="low", close="close", volume="volume",
        openinterest=-1,
    )
    cerebro2.adddata(feed)
    cerebro2.addstrategy(signal_class, **(signal_params or {}))
    cerebro2.addsizer(bt.sizers.FixedSize, stake=stake)
    cerebro2.addanalyzer(bt.analyzers.TradeAnalyzer, _name="trades")
    cerebro2.addanalyzer(bt.analyzers.SharpeRatio, _name="sharpe", riskfreerate=0.025, annualize=True)
    cerebro2.addanalyzer(bt.analyzers.DrawDown, _name="dd")
    cerebro2.addanalyzer(bt.analyzers.Returns, _name="returns")

    start_val = cerebro2.broker.getvalue()
    r = cerebro2.run()
    strat = r[0]
    end_val = cerebro2.broker.getvalue()

    trades = strat.analyzers.trades.get_analysis()
    n_total = trades.get("total", {}).get("total", 0)
    n_win = trades.get("won", {}).get("total", 0)
    win_rate = n_win / n_total if n_total > 0 else 0.0

    dd_v = strat.analyzers.dd.get_analysis().get("max", {}).get("drawdown", 0)
    sharpe = strat.analyzers.sharpe.get_analysis().get("sharperatio", None)
    ret = strat.analyzers.returns.get_analysis()

    return {
        "start": start_date, "end": end_date,
        "days": len(sub),
        "总收益%": round((end_val / start_val - 1) * 100, 2),
        "年化%": round(ret.get("rnorm100", 0), 2),
        "最大回撤%": round(dd_v, 2),
        "夏普": round(sharpe, 2) if sharpe else None,
        "交易数": n_total,
        "胜率%": round(win_rate * 100, 1),
    }


def multi_timeframe_bt(data_path, signal_class, signal_params=None,
                       cash=INIT_CASH, stake=100,
                       windows: Optional[list] = None) -> pd.DataFrame:
    """多时间窗回测

    Args:
        data_path: 数据文件路径
        signal_class: backtrader 策略类
        signal_params: 策略参数字典
        cash: 初始资金
        stake: 每笔股数
        windows: 自定义时间窗列表，默认 [1Y, 3Y, 5Y, ALL]

    Returns:
        DataFrame: 每行一个窗口
    """
    df = load(data_path)
    latest = df["date"].max().date()

    if windows is None:
        windows = [
            ("1年", str(latest - timedelta(days=365))),
            ("3年", str(latest - timedelta(days=365 * 3))),
            ("5年", str(latest - timedelta(days=365 * 5))),
            ("全期", str(df["date"].min().date())),
        ]

    results = []
    for label, start in windows:
        row = _bt_one_window(None, df, start, str(latest), signal_class, signal_params, cash, stake)
        if row:
            row["窗口"] = label
            row["区间"] = f"{row['start'][:10]} ~ {row['end'][:10]}"
            results.append(row)

    return pd.DataFrame(results)


# ══════════════════════════════════════════════════════════
# 板块趋势分析
# ══════════════════════════════════════════════════════════

SECTOR_ETFS = {
    "红利":    "510880",
    "国债":    "511010",
    "证券":    "512880",
    "医药":    "512010",
    "酒/消费": "512690",
    "沪深300": "510300",
    "中证500": "510500",
    "科创50":  "588000",
    "创业板":  "159915",
    "上证50":  "510050",
}


def sector_analysis(periods: list = None):
    """板块趋势分析 — 识别资金流向 + 估值洼地

    Returns:
        DataFrame: 板块动量/波动率/相对强弱排名
    """
    if periods is None:
        periods = [20, 60, 120]  # 1月/1季/半年

    available = {}
    for name, code in SECTOR_ETFS.items():
        p = DATA_MARKET / f"etf_{code}.parquet"
        if p.exists():
            available[name] = load(p)

    if not available:
        return pd.DataFrame()

    # 对齐日期
    dfs = {k: df.set_index("date")["close"] for k, df in available.items()}
    prices = pd.DataFrame(dfs).dropna()

    rows = []
    latest = prices.index[-1]

    for name in prices.columns:
        close = prices[name]
        row = {"板块": name, "最新价": round(close.iloc[-1], 3)}

        for p in periods:
            if len(close) > p:
                mom = (close.iloc[-1] / close.iloc[-p] - 1) * 100
                row[f"{p}日动量%"] = round(mom, 1)

        # 60日波动率
        if len(close) > 60:
            row["60日波动%"] = round(close.pct_change().tail(60).std() * 100, 2)

        # 相对 300 的 alpha (120日)
        if "沪深300" in prices.columns and len(close) > 120:
            ret_self = close.pct_change().tail(120).sum() * 100
            ret_300 = prices["沪深300"].pct_change().tail(120).sum() * 100
            row["vs沪深300(%)"] = round(ret_self - ret_300, 1)

        # 现在 vs 120日高点回撤
        if len(close) > 120:
            hh = close.iloc[-120:].max()
            row["距120日高点%"] = round((close.iloc[-1] / hh - 1) * 100, 1)

        # 综合信号
        signals = []
        if row.get(f"{periods[0]}日动量%", 0) > 0:
            signals.append("短线强")
        else:
            signals.append("短线弱")

        if row.get("距120日高点%", -100) < -15:
            signals.append("超跌")
        elif row.get("距120日高点%", 0) > -5:
            signals.append("高位")

        if row.get("vs沪深300(%)", -100) > 5:
            signals.append("超额强")
        elif row.get("vs沪深300(%)", 100) < -5:
            signals.append("超额弱")

        row["信号"] = " | ".join(signals)
        rows.append(row)

    result = pd.DataFrame(rows)
    # 按20日动量排序
    col = f"{periods[0]}日动量%"
    if col in result.columns:
        result = result.sort_values(col, ascending=False)
    return result.reset_index(drop=True)


# ══════════════════════════════════════════════════════════
# ETF 轮动多时间窗
# ══════════════════════════════════════════════════════════

class S_Rotation_AnyPool(bt.Strategy):
    """通用 ETF 轮动 — 外部传入标的列表

    参数:
        symbols: list[str] 标的代码
        top_k (3): 持仓数
        lookback (20): 动量回看
        rebal_freq (20): 调仓频率
    """
    params = (("symbols", []), ("top_k", 3), ("lookback", 20), ("rebal_freq", 20))

    def __init__(self):
        self.day = 0
        self.roc = {}
        for i, d in enumerate(self.datas):
            self.roc[d._name] = bt.ind.ROC(d.close, period=self.p.lookback)

    def next(self):
        self.day += 1
        if self.day % self.p.rebal_freq != 0:
            return

        scores = {}
        for i, d in enumerate(self.datas):
            if len(d) > self.p.lookback:
                scores[d._name] = (self.roc[d._name][0], i)

        top = sorted(scores.items(), key=lambda x: x[1][0], reverse=True)[:self.p.top_k]
        top_names = {t[0] for t in top}

        for i, d in enumerate(self.datas):
            pos = self.getposition(d)
            if pos.size > 0 and d._name not in top_names:
                self.close(data=d)

        for i, d in enumerate(self.datas):
            if d._name in top_names:
                target = self.broker.getvalue() / self.p.top_k
                price = d.close[0]
                size = target / price
                pos = self.getposition(d)
                cur = pos.size if pos else 0
                diff = size - cur
                if diff > 0:
                    self.buy(data=d, size=int(diff // 100 * 100))
                elif diff < 0:
                    self.sell(data=d, size=int(-diff // 100 * 100))


def rotation_multi_tf(symbols: list, windows: list = None,
                      top_k: int = 3, lookback: int = 20, rebal_freq: int = 20,
                      cash: float = INIT_CASH, use_timing: bool = False):
    """ETF轮动多时间窗回测

    Args:
        symbols: ETF 代码列表
        windows: [(label, start_date), ...]
        use_timing: 是否启用择时(200MA过滤, 熊市空仓)
    """
    # 加载所有数据
    defs = {}
    available_syms = []
    for s in symbols:
        p = DATA_MARKET / f"etf_{s}.parquet"
        if p.exists():
            df = load(p)
            defs[s] = df
            available_syms.append(s)

    if len(available_syms) < 2:
        return pd.DataFrame()

    latest = max(df["date"].max() for df in defs.values())
    if windows is None:
        windows = [
            ("半年", str((latest - timedelta(days=182)).date())),
            ("1年", str((latest - timedelta(days=365)).date())),
            ("3年", str((latest - timedelta(days=365 * 3)).date())),
            ("5年", str((latest - timedelta(days=365 * 5)).date())),
        ]

    results = []
    for label, start_str in windows:
        start_dt = pd.Timestamp(start_str)

        cerebro = bt.Cerebro()
        cerebro.broker.setcash(cash)
        cerebro.broker.addcommissioninfo(AStockCommission())

        # 如果启用择时，先加载指数数据
        idx_added = False
        if use_timing:
            idx_path = DATA_MARKET / "index_000300.parquet"
            if idx_path.exists():
                idx_df = load(idx_path)
                idx_sub = idx_df[(idx_df["date"] >= start_dt) & (idx_df["date"] <= latest)].copy()
                # 至少需要200条数据才能算200MA
                if len(idx_sub) >= 200:
                    idx_feed = bt.feeds.PandasData(
                        dataname=idx_sub,
                        datetime="date", open="open", high="high", low="low",
                        close="close", volume="volume", openinterest=-1,
                    )
                    cerebro.adddata(idx_feed, name="idx_000300")
                    idx_added = True

        feed_count = 0
        for s in available_syms:
            df = defs[s]
            sub = df[(df["date"] >= start_dt) & (df["date"] <= latest)].copy()
            if len(sub) < 50:
                continue
            feed = bt.feeds.PandasData(
                dataname=sub,
                datetime="date", open="open", high="high", low="low",
                close="close", volume="volume", openinterest=-1,
            )
            cerebro.adddata(feed, name=s)
            feed_count += 1

        if feed_count < 2:
            continue

        if use_timing and idx_added:
            cerebro.addstrategy(S_Rotation_Timing_AnyPool, symbols=available_syms,
                                top_k=top_k, lookback=lookback, rebal_freq=rebal_freq)
        else:
            cerebro.addstrategy(S_Rotation_AnyPool, symbols=available_syms,
                                top_k=top_k, lookback=lookback, rebal_freq=rebal_freq)

        cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name="trades")
        cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name="sharpe", riskfreerate=0.025, annualize=True)
        cerebro.addanalyzer(bt.analyzers.DrawDown, _name="dd")
        cerebro.addanalyzer(bt.analyzers.Returns, _name="returns")

        start_val = cerebro.broker.getvalue()
        r = cerebro.run()
        strat = r[0]
        end_val = cerebro.broker.getvalue()

        trades = strat.analyzers.trades.get_analysis()
        n_total = trades.get("total", {}).get("total", 0)
        n_win = trades.get("won", {}).get("total", 0)
        win_rate = n_win / n_total if n_total > 0 else 0.0

        dd_v = strat.analyzers.dd.get_analysis().get("max", {}).get("drawdown", 0)
        sharpe = strat.analyzers.sharpe.get_analysis().get("sharperatio", None)
        ret = strat.analyzers.returns.get_analysis()

        results.append({
            "窗口": label,
            "区间": f"{start_str[:10]} ~ {str(latest.date())[:10]}",
            "总收益%": round((end_val / start_val - 1) * 100, 2),
            "年化%": round(ret.get("rnorm100", 0), 2),
            "最大回撤%": round(dd_v, 2),
            "夏普": round(sharpe, 2) if sharpe else None,
            "交易数": n_total,
            "胜率%": round(win_rate * 100, 1),
            "标的数": feed_count,
        })

    return pd.DataFrame(results)


# ════════════════════════════════════════════
# 带择时的通用ETF轮动策略
# ════════════════════════════════════════════

class S_Rotation_Timing_AnyPool(bt.Strategy):
    """ETF轮动 + 择时: 指数<200MA → 全空, 指数>200MA → 动量轮动

    参数:
        symbols: list[str]
        top_k (3): 持仓数
        lookback (20): 动量回看
        rebal_freq (20): 调仓频率
        timing_ma (200): 择时均线
    """
    params = (("symbols", []), ("top_k", 3), ("lookback", 20),
              ("rebal_freq", 20), ("timing_ma", 200))

    def __init__(self):
        self.day = 0
        self.roc = {}
        for d in self.datas:
            if not d._name.startswith("idx_"):
                self.roc[d._name] = bt.ind.ROC(d.close, period=self.p.lookback)

        # 找指数数据
        self.idx_close = None
        self.idx_ma = None
        for d in self.datas:
            if d._name.startswith("idx_"):
                self.idx_close = d.close
                self.idx_ma = bt.ind.SMA(d.close, period=self.p.timing_ma)
                break

    def _in_bear(self):
        """熊市判断: 价格 < 200MA"""
        if self.idx_ma is None or len(self.idx_ma) < self.p.timing_ma:
            return False
        return self.idx_close[0] < self.idx_ma[0]

    def next(self):
        self.day += 1
        if self.day % self.p.rebal_freq != 0:
            return

        # 择时: 熊市全空
        if self._in_bear():
            for d in self.datas:
                if not d._name.startswith("idx_"):
                    pos = self.getposition(d)
                    if pos.size > 0:
                        self.close(data=d)
            return

        # 动量排名
        scores = {}
        for d in self.datas:
            if d._name.startswith("idx_"):
                continue
            if d._name in self.roc and len(d) > self.p.lookback:
                scores[d._name] = self.roc[d._name][0]

        if not scores:
            return

        top = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:self.p.top_k]
        top_names = {t[0] for t in top}

        # 清仓不在 Top K 的
        for d in self.datas:
            if d._name.startswith("idx_"):
                continue
            pos = self.getposition(d)
            if pos.size > 0 and d._name not in top_names:
                self.close(data=d)

        # 等权买入 Top K
        for d in self.datas:
            if d._name in top_names:
                target = self.broker.getvalue() / self.p.top_k
                price = d.close[0]
                size = target / price
                pos = self.getposition(d)
                cur = pos.size if pos else 0
                diff = size - cur
                if diff > 0:
                    self.buy(data=d, size=int(diff // 100 * 100))
                elif diff < 0:
                    self.sell(data=d, size=int(-diff // 100 * 100))


# ══════════════════════════════════════════════════════════
# 综合报告
# ══════════════════════════════════════════════════════════

def rotation_multi_tf_timing_compare(symbols: list, top_k: int = 3,
                                         lookback: int = 20, rebal_freq: int = 20,
                                         cash: float = INIT_CASH):
    """择时 ON vs OFF 多时间窗对比 — 核心看回撤降了多少"""
    print("=" * 70)
    print("  🔄 择时效果对比 — 200MA过滤: 熊市空仓 vs 一直满仓")
    print("=" * 70)

    r_off = rotation_multi_tf(symbols, top_k=top_k, lookback=lookback,
                               rebal_freq=rebal_freq, cash=cash, use_timing=False)
    r_on = rotation_multi_tf(symbols, top_k=top_k, lookback=lookback,
                              rebal_freq=rebal_freq, cash=cash, use_timing=True)

    if r_off.empty or r_on.empty:
        print("  ❌ 数据不足")
        return pd.DataFrame()

    # 合并对比
    compare = r_off[["窗口", "总收益%", "最大回撤%", "夏普", "胜率%"]].copy()
    compare.columns = ["窗口", "收益(无择时)", "回撤(无择时)", "夏普(无择时)", "胜率(无择时)"]

    m = r_on.set_index("窗口")
    compare["收益(择时)"] = compare["窗口"].map(m["总收益%"])
    compare["回撤(择时)"] = compare["窗口"].map(m["最大回撤%"])
    compare["夏普(择时)"] = compare["窗口"].map(m["夏普"])
    compare["胜率(择时)"] = compare["窗口"].map(m["胜率%"])

    compare["回撤降幅"] = compare.apply(
        lambda r: f"{(1 - r['回撤(择时)']/r['回撤(无择时)'])*100:.0f}%"
        if r["回撤(无择时)"] > 0.1 else "—", axis=1
    )
    compare["收益影响"] = compare.apply(
        lambda r: f"{r['收益(择时)'] - r['收益(无择时)']:+.1f}%", axis=1
    )

    print("\n" + compare.to_string(index=False))

    # 结论
    dd_off = r_off["最大回撤%"].mean()
    dd_on = r_on["最大回撤%"].mean()
    ret_off = r_off["总收益%"].mean()
    ret_on = r_on["总收益%"].mean()

    print(f"\n  📊 平均效果:")
    print(f"     回撤: {dd_off:.1f}% → {dd_on:.1f}%  (降 {(1-dd_on/dd_off)*100:.0f}%)")
    print(f"     收益: {ret_off:.1f}% → {ret_on:.1f}%  ({ret_on-ret_off:+.1f}%)")

    if dd_on < dd_off * 0.7:
        print(f"\n  ✅ 择时有效! 回撤显著降低，建议启用择时。")
    else:
        print(f"\n  ⚠️ 择时效果不显著 — 近期可能没有大熊市，或者震荡市频繁假信号。")

    print("=" * 70)
    return compare


def full_report():
    """生成全量分析报告"""
    print("=" * 70)
    print("  AI量化综合分析报告")
    print(f"  生成时间: {datetime.now():%Y-%m-%d %H:%M}")
    print("=" * 70)

    # 1. 板块分析
    print("\n" + "─" * 70)
    print("  📊 一、板块趋势分析")
    print("─" * 70)
    sector = sector_analysis()
    if not sector.empty:
        print(sector.to_string(index=False))
    else:
        print("  (无板块数据)")

    # 2. 原轮动池
    print("\n" + "─" * 70)
    print("  📊 二、原池 (50/300/500/科创/创业) 多窗回测")
    print("─" * 70)
    orig = rotation_multi_tf(["510050", "510300", "510500", "159915", "588000"])
    if not orig.empty:
        print(orig.to_string(index=False))

    # 3. +防御池
    print("\n" + "─" * 70)
    print("  📊 三、+防御池 (+红利+国债) 多窗回测")
    print("─" * 70)
    defense = rotation_multi_tf(["510050", "510300", "510500", "159915", "588000", "510880", "511010"])
    if not defense.empty:
        print(defense.to_string(index=False))

    # 4. 全行业池
    print("\n" + "─" * 70)
    print("  📊 四、全行业池 (10只) 多窗回测")
    print("─" * 70)
    full = rotation_multi_tf(["510050", "510300", "510500", "159915", "588000",
                               "512880", "512010", "512690", "510880", "511010"])
    if not full.empty:
        print(full.to_string(index=False))

    # 5. 🆕 择时对比
    print("\n" + "─" * 70)
    print("  📊 五、择时过滤对比 — 200MA牛市才持仓")
    print("─" * 70)
    rotation_multi_tf_timing_compare(
        ["510050", "510300", "510500", "159915", "588000",
         "512880", "512010", "512690", "510880", "511010"])

    print("\n" + "=" * 70 + "\n")
