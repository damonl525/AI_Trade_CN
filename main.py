"""AI量化平台 CLI v3 (含 AI 日报 + 入场检查)

新增命令:
  uv run main optimize             参数优化(网格搜索)
  uv run main walkforward          前向验证
  uv run main signal               今日买卖信号
  uv run main signal --tk 4        4只持仓信号
"""

import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

import argparse
from pathlib import Path
import pandas as pd
from datetime import datetime

from config import ROOT, DATA_MARKET, OUTPUT
from data.fetcher import fetch_a_stock, fetch_etf, fetch_index, load
from engine.backtest import run_backtest, preview
from engine.signals import *   # S_* 策略全部导入

SIGNALS = {
    "MA_Cross":     S_MA_Cross,
    "MA_Buffer":    S_MA_Buffer,
    "Momentum_MA":  S_Momentum_MA,
    "Bollinger":    S_Bollinger,
    "Breakout_ATR": S_Breakout_ATR,
    "Rotation":     S_Rotation,
}

SIGNAL_HELP = {
    "MA_Cross":     "双均线交叉 (默认 5/20), 简单基准",
    "MA_Buffer":    "均线+缓冲, 减少假突破",
    "Momentum_MA":  "动量突破+均线过滤器, 趋势跟随",
    "Bollinger":    "布林线回归, 下轨买/中轨卖",
    "Breakout_ATR": "突破+ATR动态止损, 截断亏损让利润跑",
    "Rotation":     "ETF轮动, 动量排名/Top K/定频调仓",
}

# ── ETF 池 ──
FULL_SYMBOLS = ["510050", "510300", "510500", "159915", "588000",
                "512880", "512010", "512690", "510880", "511010"]

def cmd_fetch(kind: str, symbol: str, start: str = "2020-01-01", force: bool = False):
    fetchers = {
        "stock": lambda: fetch_a_stock(symbol, start=start, force=force),
        "etf":   lambda: fetch_etf(symbol, start=start, force=force),
        "index": lambda: fetch_index(symbol, start=start, force=force),
    }
    if kind not in fetchers:
        print(f"❌ 未知类型: {kind} (可选: stock / etf / index)")
        return
    path = fetchers[kind]()
    print(f"✅ 已保存 → {path}")
    preview(path)

def cmd_bt(symbol: str, signal_name: str):
    if signal_name not in SIGNALS:
        print(f"❌ 未知信号: {signal_name}")
        cmd_btlist()
        return
    path = DATA_MARKET / f"{symbol}.parquet"
    if not path.exists():
        path = DATA_MARKET / f"etf_{symbol}.parquet"
    if not path.exists():
        print(f"❌ 未找到数据: {symbol}.parquet 或 etf_{symbol}.parquet")
        return
    run_backtest(path, SIGNALS[signal_name])

def cmd_btlist():
    print("\n可用策略信号:")
    for name, desc in SIGNAL_HELP.items():
        print(f"  {name:<14} {desc}")

def cmd_sector():
    from engine.optimizer import sector_analysis
    df = sector_analysis()
    if df.empty:
        print("暂无板块数据，请先 fetch etf <代码>")
        return
    print("\n📊 板块趋势分析 (按动量排序)")
    print(df.to_string(index=False))

def cmd_report():
    from engine.optimizer import full_report
    full_report()

def cmd_optimize(symbols: list = None):
    """参数优化"""
    from engine.optimize import best_params, grid_search

    if symbols is None:
        symbols = FULL_SYMBOLS

    # 先确保数据
    _ensure_data(symbols)

    bp = best_params(symbols)

    if bp:
        print(f"\n{'='*60}")
        print(f"  ✅ 最优策略配置 (保存到 config_strategy.py)")
        print(f"{'='*60}")
        conf = f'''"""最优策略配置 — 自动生成于 {datetime.now():%Y-%m-%d %H:%M}"""

STRATEGY = {{
    "lookback": {bp["lookback"]},
    "top_k": {bp["top_k"]},
    "rebal_freq": {bp["rebal_freq"]},
    "use_timing": {bp["use_timing"]},
}}
'''
        with open(ROOT / "config_strategy.py", "w", encoding="utf-8") as f:
            f.write(conf)
        print(f"   配置已保存 → config_strategy.py")

def cmd_walkforward(symbols: list = None):
    """前向验证"""
    from engine.optimize import walk_forward

    if symbols is None:
        symbols = FULL_SYMBOLS

    _ensure_data(symbols)

    try:
        # 加载最优配置
        from config_strategy import STRATEGY
    except ImportError:
        STRATEGY = {"lookback": 20, "top_k": 3, "rebal_freq": 20, "use_timing": True}
        print("⚠️ 未找到 config_strategy.py, 使用默认参数")

    wf = walk_forward(symbols, **STRATEGY)
    if wf.empty:
        print("  前向验证失败")
        return
    print("\n📊 前向验证结果:")
    print(wf.to_string(index=False))

def cmd_signal(top_k: int = None, lookback: int = None,
               rebal_freq: int = None, use_timing: bool = None):
    """生成今日买卖信号"""
    from live_signal import generate_signals, print_signals, _momentum

    # 加载配置
    try:
        from config_strategy import STRATEGY
    except ImportError:
        STRATEGY = {"lookback": 20, "top_k": 3, "rebal_freq": 20, "use_timing": True}

    if top_k is None:
        top_k = STRATEGY.get("top_k", 3)
    if lookback is None:
        lookback = STRATEGY.get("lookback", 20)

    sig = generate_signals(top_k=top_k, lookback=lookback)
    print_signals(sig)

def cmd_live():
    """快速刷新所有拉数据 + 出信号"""
    print("🔄 刷新行情数据...")
    symbols = FULL_SYMBOLS
    for s in symbols:
        try:
            fetch_etf(s, force=True)
            print(f"  ✅ {s}")
        except Exception as e:
            print(f"  ⚠️ {s}: {e}")

    # 拉指数
    try:
        fetch_index("000300", force=True)
        print(f"  ✅ 沪深300指数")
    except:
        pass

    print("\n🔮 生成信号...")
    cmd_signal()

def _ensure_data(symbols: list):
    """确保有必要的数据"""
    for s in symbols:
        p = DATA_MARKET / f"etf_{s}.parquet"
        if not p.exists():
            try:
                fetch_etf(s)
                print(f"  ✅ 拉取 {s}")
            except:
                print(f"  ⚠️ 跳过 {s}")

    # 指数
    idx_path = DATA_MARKET / "index_000300.parquet"
    if not idx_path.exists():
        try:
            fetch_index("000300")
            print(f"  ✅ 拉取沪深300")
        except:
            pass

def main():
    parser = argparse.ArgumentParser(description="AI量化平台 v2")
    sub = parser.add_subparsers(dest="cmd")

    # ── fetch ──
    p_fetch = sub.add_parser("fetch", help="拉行情数据")
    p_fetch.add_argument("kind", choices=["stock", "etf", "index"])
    p_fetch.add_argument("symbol", help="代码: 600519 / 510050 / 000300")
    p_fetch.add_argument("--start", default="2020-01-01")
    p_fetch.add_argument("--force", action="store_true")

    # ── bt ──
    p_bt = sub.add_parser("bt", help="回测")
    p_bt.add_argument("symbol", help="代码")
    p_bt.add_argument("signal", help="信号名")

    # ── btlist ──
    sub.add_parser("btlist", help="列出信号")

    # ── sector ──
    sub.add_parser("sector", help="板块趋势分析")

    # ── report ──
    sub.add_parser("report", help="全量综合报告")

    # ── optimize 🆕 ──
    sub.add_parser("optimize", help="参数优化(网格搜索最优参数)")

    # ── walkforward 🆕 ──
    sub.add_parser("walkforward", help="前向验证")

    # ── signal 🆕 ──
    p_sig = sub.add_parser("signal", help="今日买卖信号")
    p_sig.add_argument("--tk", type=int, help="持仓数")
    p_sig.add_argument("--lb", type=int, help="动量窗口")

    # ── live 🆕 ──
    sub.add_parser("live", help="一键刷新数据+出信号")

    # ── daily 🆕 ──
    p_daily = sub.add_parser("daily", help="AI日报(调LLM解读)")
    p_daily.add_argument("--brief", action="store_true", help="简洁版(不调AI)")
    p_daily.add_argument("--tk", type=int, help="持仓数")
    p_daily.add_argument("--lb", type=int, help="动量窗口")

    # ── entry 🆕 ──
    p_entry = sub.add_parser("entry", help="入场检查(零持仓:现在能进场吗?)")
    p_entry.add_argument("--tk", type=int, help="持仓数")
    p_entry.add_argument("--lb", type=int, help="动量窗口")

    # ── timing 🆕 ──
    p_timing = sub.add_parser("timing", help="择时ON/OFF对比(看看择时到底有没有用)")
    p_timing.add_argument("--tk", type=int, default=3, help="持仓数")
    p_timing.add_argument("--lb", type=int, default=20, help="动量窗口")

    # ── dynbt 🆕 ── 动态择时 vs 二元择时 vs 无择时
    p_dyn = sub.add_parser("dynbt", help="动态择时对比回测(连续仓位vs二元vs无)")
    p_dyn.add_argument("--years", type=float, default=5.0, help="回测窗口(年)")
    p_dyn.add_argument("--lb", type=int, default=10, help="动量窗口")
    p_dyn.add_argument("--tk", type=int, default=3, help="持仓数")
    p_dyn.add_argument("--rebal", type=int, default=40, help="调仓周期(天)")

    # ── pos 🆕 ── 当前动态仓位
    sub.add_parser("pos", help="当前动态仓位建议(连续0-100%)")

    args = parser.parse_args()

    if args.cmd == "fetch":
        cmd_fetch(args.kind, args.symbol, getattr(args, "start", "2020-01-01"), 
                  getattr(args, "force", False))
    elif args.cmd == "bt":
        cmd_bt(args.symbol, args.signal)
    elif args.cmd == "btlist":
        cmd_btlist()
    elif args.cmd == "sector":
        cmd_sector()
    elif args.cmd == "report":
        cmd_report()
    elif args.cmd == "optimize":
        cmd_optimize()
    elif args.cmd == "walkforward":
        cmd_walkforward()
    elif args.cmd == "signal":
        cmd_signal(top_k=getattr(args, "tk", None),
                   lookback=getattr(args, "lb", None))
    elif args.cmd == "live":
        cmd_live()
    elif args.cmd == "daily":
        _cmd_daily(args)
    elif args.cmd == "entry":
        _cmd_entry(args)
    elif args.cmd == "timing":
        _cmd_timing(args)
    elif args.cmd == "dynbt":
        _cmd_dynbt(args)
    elif args.cmd == "pos":
        _cmd_pos()
    else:
        parser.print_help()

def _cmd_timing(args):
    """择时ON/OFF对比"""
    from engine.optimizer import rotation_multi_tf_timing_compare
    rotation_multi_tf_timing_compare(
        FULL_SYMBOLS,
        top_k=getattr(args, "tk", 3),
        lookback=getattr(args, "lb", 20))


def _cmd_daily(args):
    from ai_summary import generate_daily_report
    from live_signal import generate_signals
    tk = getattr(args, "tk", None) or 3
    lb = getattr(args, "lb", None) or 20
    sig = generate_signals(top_k=tk, lookback=lb)
    report = generate_daily_report(sig, brief=args.brief)
    print(report)


def _cmd_entry(args):
    from ai_summary import entry_check
    from live_signal import generate_signals
    from engine.dynamic_timing import dynamic_position_now
    tk = getattr(args, "tk", None) or 3
    lb = getattr(args, "lb", None) or 20
    sig = generate_signals(top_k=tk, lookback=lb)
    entry_check(sig, dynamic_position_now())


def _cmd_dynbt(args):
    """动态择时对比回测"""
    import subprocess, sys
    cmd = [
        sys.executable, str(Path(__file__).parent / "backtest_dynamic.py"),
        "--years", str(getattr(args, "years", 5)),
        "--lookback", str(getattr(args, "lb", 10)),
        "--top_k", str(getattr(args, "tk", 3)),
        "--rebalance", str(getattr(args, "rebal", 40)),
    ]
    subprocess.run(cmd)


def _cmd_pos():
    """当前动态仓位"""
    from engine.dynamic_timing import dynamic_position_now
    dp = dynamic_position_now()
    print(f"\n  📊 动态仓位: {dp['pct']} ({dp['regime']})")
    print(f"     trend_z={dp['trend_z']} | mom={dp['mom_val']:+.1f}% | vol_rank={dp['vol_rank']}")
    if dp['position'] >= 0.7:
        print(f"     🟢 可以积极进场")
    elif dp['position'] >= 0.4:
        print(f"     🟡 谨慎进场，半仓")
    else:
        print(f"     🔴 建议观望或微量参与")

if __name__ == "__main__":
    main()
