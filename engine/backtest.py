"""A股回测引擎 — backtrader 封装

关键: A股成本模型 (万5 + 千0.5印花税 + 最低5元佣金)
"""

import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

import backtrader as bt
import pandas as pd
from pathlib import Path
from datetime import datetime

from config import COMMISSION, STAMP_DUTY, MIN_COMMISSION, INIT_CASH, OUTPUT
from data.fetcher import load


# ── A股佣金模型 ──
class AStockCommission(bt.CommInfoBase):
    params = (
        ("commission", COMMISSION),
        ("stamp_duty", STAMP_DUTY),       # 仅卖出收印花税
        ("min_commission", MIN_COMMISSION),
        ("stocklike", True),
    )

    def _getcommission(self, size, price, pseudoexec):
        """万5佣金 + 卖出千0.5印花税, 每笔最低5元"""
        value = abs(size) * price
        comm = max(value * self.p.commission, self.p.min_commission)
        if size < 0:                       # 卖出加印花税
            comm += value * self.p.stamp_duty
        return comm


# ── 回测主类 ──
def run_backtest(
    data_path: Path | str,
    signal_class,
    signal_params: dict | None = None,
    cash: float = INIT_CASH,
    stake: int = 100,            # 每笔 100 股
    plot: bool = True,
) -> dict:
    """一行回测: 给数据路径+信号 → 返回绩效指标

    Returns:
        dict: total_return, annual_return, max_drawdown, sharpe, win_rate, trades
    """
    cerebro = bt.Cerebro()

    # ── 资金 ──
    cerebro.broker.setcash(cash)
    cerebro.broker.addcommissioninfo(AStockCommission())

    # ── 数据 ──
    df = load(data_path)
    feed = bt.feeds.PandasData(
        dataname=df,
        datetime="date",
        open="open", high="high", low="low", close="close", volume="volume",
        openinterest=-1,
    )
    cerebro.adddata(feed)

    # ── 策略 ──
    if signal_params is None:
        signal_params = {}
    cerebro.addstrategy(signal_class, **signal_params)

    # ── 规模 ──
    cerebro.addsizer(bt.sizers.FixedSize, stake=stake)

    # ── 分析器 ──
    cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name="trades")
    cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name="sharpe", riskfreerate=0.025, annualize=True)
    cerebro.addanalyzer(bt.analyzers.DrawDown, _name="dd")
    cerebro.addanalyzer(bt.analyzers.Returns, _name="returns")
    cerebro.addanalyzer(bt.analyzers.VWR, _name="vwr")

    # ── 跑 ──
    start_val = cerebro.broker.getvalue()
    results = cerebro.run()
    strat = results[0]
    end_val = cerebro.broker.getvalue()

    # ── 提取指标 ──
    trades = strat.analyzers.trades.get_analysis()
    n_total = trades.get("total", {}).get("total", 0)
    n_win = trades.get("won", {}).get("total", 0)
    n_loss = trades.get("lost", {}).get("total", 0)
    win_rate = n_win / n_total if n_total > 0 else 0.0

    total_return = (end_val / start_val - 1) * 100

    dd = strat.analyzers.dd.get_analysis()
    max_dd = dd.get("max", {}).get("drawdown", 0)

    sharpe = strat.analyzers.sharpe.get_analysis().get("sharperatio", None)

    returns = strat.analyzers.returns.get_analysis()
    annual_return = returns.get("rnorm100", 0)

    # ── 终端打印 ──
    report = {
        "初始资金": f"{start_val:,.0f}",
        "最终资金": f"{end_val:,.0f}",
        "总收益率": f"{total_return:.2f}%",
        "年化收益": f"{annual_return:.2f}%",
        "最大回撤": f"{max_dd:.2f}%",
        "夏普比率": f"{sharpe:.2f}" if sharpe else "-",
        "总交易": f"{n_total}笔",
        "胜率": f"{win_rate:.1%}",
        "盈亏比": f"{n_win}:{n_loss}",
    }

    print("\n" + "=" * 50)
    print(f"  回测结束 — {datetime.now():%Y-%m-%d %H:%M}")
    print("=" * 50)
    for k, v in report.items():
        print(f"  {k:　<8} {v}")
    print("=" * 50 + "\n")

    if plot:
        try:
            img = OUTPUT / "backtest_result.png"
            cerebro.plot(style="candlestick", savefig=str(img), dpi=150)
            print(f"📊 图表 → {img}")
        except Exception:
            print("⚠️ 图表生成失败")

    return report


# ── 快速数据预览 ──
def preview(path: Path | str, tail: int = 5):
    """看最后几行 + 基础统计"""
    df = load(path)
    stats = {
        "行数": len(df),
        "起始": str(df["date"].iloc[0])[:10],
        "终止": str(df["date"].iloc[-1])[:10],
        "收盘均值": f"{df['close'].mean():.2f}",
        "日波动": f"{df['close'].pct_change().std() * 100:.2f}%",
    }
    for k, v in stats.items():
        print(f"  {k}: {v}")
    print(f"\n  最后{tail}行:\n{df.tail(tail).to_string(index=False)}")
    return df
