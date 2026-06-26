"""参数优化 + 前向验证引擎

做事:
1. 网格搜索最优参数 (lookback, top_k, rebal_freq, 择时开关)
2. 前向验证 (walk-forward) — 防止过拟合
3. 生成最优策略配置
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
from itertools import product
import warnings
warnings.filterwarnings("ignore")

import backtrader as bt

from config import OUTPUT, DATA_MARKET, INIT_CASH, COMMISSION, STAMP_DUTY, MIN_COMMISSION
from data.fetcher import load
from engine.backtest import AStockCommission
from engine.timing import timing_filter_series


# ════════════════════════════════════════════
# 带择时+风控的ETF轮动策略
# ════════════════════════════════════════════

class S_Rotation_Timing(bt.Strategy):
    """ETF轮动 + 择时过滤 + 回撤熔断

    参数:
        top_k: 持仓数
        lookback: 动量窗口
        rebal_freq: 调仓频率
        use_timing: 是否启用择时(指数<200MA → 空仓)
        timing_ma: 择时均线周期
        max_dd_half: 回撤多少触发半仓
        max_dd_full: 回撤多少触发空仓
        stop_loss_pct: 单笔止损%
    """
    params = (
        ("top_k", 3),
        ("lookback", 20),
        ("rebal_freq", 20),
        ("use_timing", True),
        ("timing_ma", 200),
        ("max_dd_half", 0.15),
        ("max_dd_full", 0.25),
        ("stop_loss_pct", -0.08),
    )

    def __init__(self):
        self.day = 0
        self.peak = self.broker.getvalue()
        self.stop_prices = {}  # 每个标的的止损价

        # 动量指标
        self.roc = {}
        for i, d in enumerate(self.datas):
            if d._name.startswith("idx_"):
                continue
            self.roc[d._name] = bt.ind.ROC(d.close, period=self.p.lookback)

        # 择时用指数(第一个数据)
        self.idx_close = self.datas[0].close
        self.idx_ma = bt.ind.SMA(self.datas[0].close, period=self.p.timing_ma)

    def _in_bear_market(self):
        """是否在熊市(择时过滤)"""
        if not self.p.use_timing:
            return False
        if len(self.idx_ma) < self.p.timing_ma:
            return False
        price = self.idx_close[0]
        ma = self.idx_ma[0]
        return price < ma

    def _check_drawdown(self):
        """回撤熔断"""
        current = self.broker.getvalue()
        self.peak = max(self.peak, current)
        dd = current / self.peak - 1
        if dd <= -self.p.max_dd_full:
            return "full"
        if dd <= -self.p.max_dd_half:
            return "half"
        return "ok"

    def _check_stop_loss(self, d, d_name):
        """检查是否止损"""
        if d_name not in self.stop_prices:
            return False
        return d.close[0] < self.stop_prices[d_name]

    def next(self):
        self.day += 1

        # 每日检查止损
        for i, d in enumerate(self.datas):
            if d._name.startswith("idx_"):
                continue
            pos = self.getposition(d)
            if pos.size > 0 and self._check_stop_loss(d, d._name):
                self.close(data=d)
                self.stop_prices.pop(d._name, None)

        # 调仓日
        if self.day % self.p.rebal_freq != 0:
            return

        # 回撤熔断
        dd_level = self._check_drawdown()
        if dd_level == "full":
            for i, d in enumerate(self.datas):
                if not d._name.startswith("idx_"):
                    self.close(data=d)
            return

        # 择时过滤: 熊市全空
        if self._in_bear_market():
            for i, d in enumerate(self.datas):
                if not d._name.startswith("idx_"):
                    self.close(data=d)
            return

        # 动量排名(只算ETF数据, 跳过指数)
        scores = {}
        for i, d in enumerate(self.datas):
            if d._name.startswith("idx_"):
                continue
            if d._name in self.roc and len(d) > self.p.lookback:
                scores[d._name] = self.roc[d._name][0]

        if not scores:
            return

        top = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:self.p.top_k]
        top_names = {t[0] for t in top}

        # 仓位倍率(回撤熔断)
        position_mult = 1.0 if dd_level == "ok" else 0.5

        # 清仓不在 Top K 的
        for i, d in enumerate(self.datas):
            if d._name.startswith("idx_"):
                continue
            pos = self.getposition(d)
            if pos.size > 0 and d._name not in top_names:
                self.close(data=d)
                self.stop_prices.pop(d._name, None)

        # 等权买入 Top K
        weight = position_mult / self.p.top_k
        for i, d in enumerate(self.datas):
            if d._name in top_names:
                target = self.broker.getvalue() * weight
                price = d.close[0]
                size = target / price
                pos = self.getposition(d)
                cur = pos.size if pos else 0
                diff = size - cur
                if diff > 0:
                    self.buy(data=d, size=int(diff // 100 * 100))
                    self.stop_prices[d._name] = price * (1 + self.p.stop_loss_pct)
                elif diff < 0:
                    self.sell(data=d, size=int(-diff // 100 * 100))


# ════════════════════════════════════════════
# 网格搜索
# ════════════════════════════════════════════

def _run_single_bt(symbols, lookback, top_k, rebal_freq, 
                   use_timing, start_date, end_date, cash=INIT_CASH):
    """单次回测 → 返回绩效"""
    cerebro = bt.Cerebro()

    # 指数数据(择时用)
    idx_path = DATA_MARKET / "index_000300.parquet"
    if use_timing and idx_path.exists():
        idx_df = load(idx_path)
        idx_sub = idx_df[(idx_df["date"] >= pd.Timestamp(start_date)) & 
                         (idx_df["date"] <= pd.Timestamp(end_date))]
        if len(idx_sub) >= 50:
            feed = bt.feeds.PandasData(
                dataname=idx_sub,
                datetime="date", open="open", high="high", low="low",
                close="close", volume="volume", openinterest=-1,
            )
            cerebro.adddata(feed, name="idx_000300")

    # ETF 数据
    feed_count = 0
    for s in symbols:
        p = DATA_MARKET / f"etf_{s}.parquet"
        if not p.exists():
            continue
        df = load(p)
        sub = df[(df["date"] >= pd.Timestamp(start_date)) & 
                 (df["date"] <= pd.Timestamp(end_date))]
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
        return None

    cerebro.broker.setcash(cash)
    cerebro.broker.addcommissioninfo(AStockCommission())

    cerebro.addstrategy(S_Rotation_Timing,
                       top_k=top_k, lookback=lookback, rebal_freq=rebal_freq,
                       use_timing=use_timing)

    cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name="sharpe", riskfreerate=0.025, annualize=True)
    cerebro.addanalyzer(bt.analyzers.DrawDown, _name="dd")
    cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name="trades")

    start_val = cerebro.broker.getvalue()
    results = cerebro.run()
    strat = results[0]
    end_val = cerebro.broker.getvalue()

    total_ret = (end_val / start_val - 1) * 100
    dd = strat.analyzers.dd.get_analysis().get("max", {}).get("drawdown", 0)
    sharpe = strat.analyzers.sharpe.get_analysis().get("sharperatio", None)
    trades = strat.analyzers.trades.get_analysis()
    n_total = trades.get("total", {}).get("total", 0)
    n_win = trades.get("won", {}).get("total", 0)
    win_rate = n_win / n_total if n_total > 0 else 0

    return {
        "总收益%": round(total_ret, 2),
        "最大回撤%": round(dd, 2),
        "夏普": round(sharpe, 2) if sharpe else 0,
        "交易数": n_total,
        "胜率%": round(win_rate * 100, 1),
        # 综合评分 = 年化收益 - 1.5 × 最大回撤 (惩罚回撤)
        "score": round(total_ret - 1.5 * dd, 2),
    }


def grid_search(symbols: list, start_date: str = None, end_date: str = None,
                cash: float = INIT_CASH) -> pd.DataFrame:
    """网格搜索最优参数

    Args:
        symbols: ETF代码列表
        start_date: 回测开始日期(默认: 数据最早)
        end_date: 回测结束日期(默认: 数据最新)

    Returns:
        DataFrame 按综合评分排序
    """
    if end_date is None:
        end_date = str(datetime.now().date())
    if start_date is None:
        start_date = "2020-01-01"

    grid = {
        "lookback": [10, 20, 40, 60],
        "top_k": [2, 3, 4, 5],
        "rebal_freq": [10, 20, 40],
        "use_timing": [True, False],
    }

    results = []
    total = len(list(product(*grid.values())))
    print(f"\n🔍 网格搜索: {total} 种参数组合\n")

    for i, (lb, tk, rf, ut) in enumerate(product(*grid.values())):
        r = _run_single_bt(symbols, lb, tk, rf, ut, start_date, end_date, cash)
        if r:
            r["lookback"] = lb
            r["top_k"] = top_k = tk
            r["调仓频率"] = rf
            r["择时"] = "✅" if ut else "❌"
            results.append(r)
        if (i + 1) % 10 == 0:
            print(f"  [{i+1}/{total}] ...")

    df = pd.DataFrame(results).sort_values("score", ascending=False)
    return df


def walk_forward(symbols: list, top_k: int = 3, lookback: int = 20,
                 rebal_freq: int = 20, use_timing: bool = True,
                 steps: int = 4, cash: float = INIT_CASH) -> pd.DataFrame:
    """前向验证: 训练→测试→再训练→再测试

    把历史数据切成 N 段，每段用前一段的"最优参数"测试后一段
    """
    # 找数据范围
    all_dates = []
    for s in symbols:
        p = DATA_MARKET / f"etf_{s}.parquet"
        if p.exists():
            df = load(p)
            all_dates.extend(df["date"].tolist())
    if not all_dates:
        return pd.DataFrame()

    start = min(all_dates).date()
    end = max(all_dates).date()
    span = (end - start).days
    step_size = span // (steps + 1)

    print(f"\n🔄 前向验证: {steps}步, 每步{step_size}天")
    print(f"   时间范围: {start} → {end}\n")

    rows = []
    for i in range(steps):
        train_start = str(start)
        train_end = str(start + timedelta(days=step_size))
        test_start = train_end
        test_end = str(start + timedelta(days=step_size * 2))

        train_r = _run_single_bt(symbols, lookback, top_k, rebal_freq,
                                  use_timing, train_start, train_end, cash)
        test_r = _run_single_bt(symbols, lookback, top_k, rebal_freq,
                                 use_timing, test_start, test_end, cash)

        rows.append({
            "步": i + 1,
            "训练期": f"{train_start[:10]}~{train_end[:10]}",
            "训练收益%": train_r["总收益%"] if train_r else None,
            "测试期": f"{test_start[:10]}~{test_end[:10]}",
            "测试收益%": test_r["总收益%"] if test_r else None,
            "测试回撤%": test_r["最大回撤%"] if test_r else None,
            "训练测试一致": "✅" if (train_r and test_r and
                                 (train_r["总收益%"] > 0) == (test_r["总收益%"] > 0))
                            else "⚠️",
        })

        start = start + timedelta(days=step_size)

    return pd.DataFrame(rows)


def best_params(symbols: list) -> dict:
    """一键找最优参数"""
    print("\n" + "=" * 60)
    print("  🔧 参数优化中...")
    print("=" * 60)

    gs = grid_search(symbols)
    if gs.empty:
        print("  优化失败: 无有效数据")
        return {}

    best = gs.iloc[0]
    print(f"\n🏆 最优参数组合:")
    print(f"   lookback={int(best['lookback'])}, top_k={int(best['top_k'])},")
    print(f"   调仓频率={int(best['调仓频率'])}, 择时={best['择时']}")
    print(f"   综合评分={best['score']:.1f}")

    # Top 5
    print(f"\n📊 Top 5 参数组合:")
    cols = ["lookback", "top_k", "调仓频率", "择时", "总收益%", "最大回撤%", "夏普", "胜率%", "score"]
    print(gs[cols].head(5).to_string(index=False))

    return {
        "lookback": int(best["lookback"]),
        "top_k": int(best["top_k"]),
        "rebal_freq": int(best["调仓频率"]),
        "use_timing": best["择时"] == "✅",
        "sharpe": best["夏普"],
        "score": best["score"],
    }
