"""动态买卖信号生成器 — LIVE 操作指南

核心理念:
  每天收盘后运行 → 输出明天的操作清单
  "持有 A/B/C，减仓 D，加仓 E，清仓 F"
"""

import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta

from config import DATA_MARKET, OUTPUT
from data.fetcher import load
from engine.timing import market_regime
from engine.dynamic_timing import dynamic_position_now


# ════════════════════════════════════════════
# 全量 ETF 池
# ════════════════════════════════════════════

FULL_POOL = {
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
    "159941": "纳指ETF",
    "513100": "纳指100",
    "159915": "创业板ETF",
    "562500": "机器人",
}


def _momentum(symbol: str, days: int = 20) -> float:
    """计算 N 日动量"""
    p = DATA_MARKET / f"etf_{symbol}.parquet"
    if not p.exists():
        return -999
    df = load(p)
    if len(df) < days:
        return -999
    return (df["close"].iloc[-1] / df["close"].iloc[-days] - 1) * 100


def _volatility(symbol: str, days: int = 60) -> float:
    """计算 N 日波动率"""
    p = DATA_MARKET / f"etf_{symbol}.parquet"
    if not p.exists():
        return 99
    df = load(p)
    if len(df) < days:
        return 99
    return df["close"].pct_change().tail(days).std() * 100


def _last_close(symbol: str) -> float:
    p = DATA_MARKET / f"etf_{symbol}.parquet"
    if not p.exists():
        return 0
    df = load(p)
    return float(df["close"].iloc[-1])


def _last_date(symbol: str) -> str:
    p = DATA_MARKET / f"etf_{symbol}.parquet"
    if not p.exists():
        return "N/A"
    df = load(p)
    return str(df["date"].iloc[-1])[:10]


# ════════════════════════════════════════════
# 主信号生成
# ════════════════════════════════════════════

def generate_signals(
    top_k: int = 3,
    lookback: int = 20,
    min_momentum: float = -10,
) -> dict:
    """生成今天的买卖信号

    Returns:
        dict: {
            'regime': 市场状态,
            'date': 数据日期,
            'hold': [当前应持有的 ETF],
            'buy': [今天新进入 Top K 的],
            'sell': [已跌出 Top K 的],
            'scores': 所有 ETF 的动量排名,
            'action': 操作建议文本,
        }
    """
    # 1. 择时判断 — 双视角
    regime = market_regime()           # 二元: bull/bear/neutral
    dyn_pos = dynamic_position_now()   # 🆕 动态: 连续 0-100%

    # 2. 动量排名
    scores = {}
    for code, name in FULL_POOL.items():
        mom = _momentum(code, lookback)
        if mom > -900:  # 有数据
            scores[code] = {
                "name": name,
                "momentum": round(mom, 1),
                "price": _last_close(code),
                "volatility": round(_volatility(code, 60), 1),
                "date": _last_date(code),
            }

    # 3. 排序取 Top K
    ranked = sorted(scores.items(), key=lambda x: x[1]["momentum"], reverse=True)
    top_names = {r[0] for r in ranked[:top_k]}

    # 4. 仓位倍率 — 🆕 优先动态连续仓位，二元备查
    position_mult = dyn_pos["position"]  # 0-1 连续
    regime_binary = regime  # 保留二元供展示

    # 5. 构建结果
    hold = ranked[:top_k]
    filtered = ranked[top_k:]

    return {
        "regime": regime,
        "dyn_pos": dyn_pos,          # 🆕 动态连续仓位
        "date": datetime.now().strftime("%Y-%m-%d"),
        "data_date": ranked[0][1]["date"] if ranked else "N/A",
        "hold": hold,
        "out": filtered,
        "position_mult": position_mult,
        "top_k": top_k,
        "lookback": lookback,
    }


def print_signals(sig: dict, portfolio_value: float = 100_000):
    """打印人类可读的操作清单"""

    r = sig["regime"]
    pct = sig["position_mult"] * 100
    dp = sig.get("dyn_pos", {})
    
    print("\n" + "=" * 70)
    print(f"  🤖 AI量化动态信号 · {sig['date']}")
    print(f"  数据日期: {sig['data_date']}  |  策略: 动量轮动Top{sig['top_k']}")
    print("=" * 70)

    # 择时状态
    emoji = {"bull": "🟢 牛市", "bear": "🔴 熊市", "neutral": "🟡 震荡"}
    print(f"\n  📡 市场状态: {emoji.get(r['regime'], '❓')}")
    print(f"     指数 {r.get('price', '?')}  |  "
          f"200MA {r.get('ma200', '?')}  |  "
          f"20日动量 {r.get('mom_20', 0):+.1f}%")
    print(f"     信号: {' | '.join(r['signals'])}")
    
    # 🆕 动态仓位详情
    if dp:
        print(f"  🎚️  动态仓位: {pct:.0f}% ({dp.get('regime', '?')})")
        print(f"     trend_z={dp.get('trend_z', 0):+.1f} | "
              f"mom={dp.get('mom_val', 0):+.1f}% | "
              f"vol_rank={dp.get('vol_rank', 0):.2f}")
    else:
        print(f"     → 建议仓位: {pct:.0f}%")

    # 持仓清单
    print(f"\n  {'─' * 66}")
    print(f"  📌 当前应持有 (Top {sig['top_k']}, 各 {pct/sig['top_k']:.0f}%):")
    print(f"  {'─' * 66}")
    total_weight = pct
    for i, (code, info) in enumerate(sig["hold"]):
        details = (f"{i+1}. {code} {info['name']:<6}  "
                   f"¥{info['price']:.3f}  "
                   f"动量 {info['momentum']:+.1f}%  "
                   f"波动 {info['volatility']:.1f}%")
        print(f"  {details}")

    # 债券仓位
    if pct < 100:
        bond_pct = 100 - pct
        print(f"\n  🛡️  避险仓位: {bond_pct:.0f}% → 国债ETF(511010) 或 现金")

    # 其他排名
    if sig["out"]:
        print(f"\n  {'─' * 66}")
        print(f"  👀 观察列表 (未入Top {sig['top_k']}):")
        print(f"  {'─' * 66}")
        for i, (code, info) in enumerate(sig["out"][:10]):
            print(f"  {i+1}. {code} {info['name']:<6}  "
                  f"动量 {info['momentum']:+.1f}%")

    print(f"\n  {'─' * 66}")
    print(f"  ⚡ 明天操作:")
    print(f"  {'─' * 66}")

    if pct == 0:
        print(f"  ❌ 空仓 — 熊市避险，资金转入国债ETF或现金")
    else:
        print(f"  ✅ 持仓: " + " + ".join(f"{c}({i['name']})"
              for c, i in sig["hold"]))

    print("\n" + "=" * 70)

    # 保存到文件
    out_path = OUTPUT / "latest_signal.md"
    _save_markdown(sig, out_path, portfolio_value)
    print(f"\n📄 报告已保存 → {out_path}")
    print("=" * 70 + "\n")


def _save_markdown(sig: dict, path: Path, portfolio: float):
    """保存 Markdown 操作报告"""
    r = sig["regime"]
    pct = sig["position_mult"] * 100

    lines = [
        f"# AI量化动态信号 · {sig['date']}",
        f"",
        f"> 数据日期: **{sig['data_date']}** | 策略: 动量轮动 Top {sig['top_k']}",
        f"> 初始资金: ¥{portfolio:,.0f} | 回看窗口: {sig['lookback']}日",
        "",
        f"## 📡 市场状态",
        f"",
        f"- **指数点位**: {r.get('price', 'N/A')}",
        f"- **200日均线**: {r.get('ma200', 'N/A')}",
        f"- **20日动量**: {r.get('mom_20', 0):+.1f}%",
        f"- **信号**: {' | '.join(r['signals'])}",
        f"- **建议仓位**: **{pct:.0f}%**",
        "",
    ]

    if sig["hold"]:
        lines.append(f"## 📌 应持有 ({pct:.0f}% 仓位)")
        lines.append("")
        lines.append("| # | 代码 | 名称 | 价格 | 动量% | 波动% |")
        lines.append("|---|------|------|------|-------|-------|")
        for i, (code, info) in enumerate(sig["hold"]):
            lines.append(f"| {i+1} | {code} | {info['name']} | ¥{info['price']:.3f} | {info['momentum']:+.1f} | {info['volatility']:.1f} |")
        lines.append("")

    if pct < 100:
        lines.append(f"## 🛡️ 避险 ({100-pct:.0f}%)")
        lines.append("")
        lines.append("国债ETF(511010) 或 现金")
        lines.append("")

    if sig["out"]:
        lines.append(f"## 👀 观察列表")
        lines.append("")
        for i, (code, info) in enumerate(sig["out"][:10]):
            lines.append(f"{i+1}. **{code}** {info['name']} — 动量 {info['momentum']:+.1f}%")
        lines.append("")

    lines.append("## ⚡ 操作建议")
    lines.append("")

    if pct == 0:
        lines.append("❌ **空仓** — 熊市避险，全部资金转入国债ETF(511010)或现金。")
    else:
        lines.append(f"✅ 等权持有: " + " + ".join(
            f"**{c}**({i['name']})" for c, i in sig["hold"]))
        lines.append(f"✅ 总仓位: {pct:.0f}%")

    lines.append("")
    lines.append(f"---")
    lines.append(f"*⚠️ 本报告由 AI 量化平台自动生成，不构成投资建议。*")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")
