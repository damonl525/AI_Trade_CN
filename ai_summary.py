"""AI 摘要模块 — 调用 LLM 把量化信号翻译成人话日报

默认: 智谱 GLM-4V-Flash (免费)
自定义: 设置环境变量 AI_SUMMARY_PROVIDER + AI_SUMMARY_KEY

用法:
    uv run python ai_summary.py                    # 默认智谱，生成日报
    uv run python ai_summary.py --signal            # 只输出信号数据(不调 AI)
    uv run python ai_summary.py --brief             # 简洁版
"""

import json
import os
import sys
from datetime import datetime
from pathlib import Path
from urllib.request import Request, urlopen
from urllib.error import URLError
from typing import Optional

from config import OUTPUT
from live_signal import generate_signals, FULL_POOL
from engine.timing import market_regime
from data.fetcher import load

# ════════════════════════════════════════════
# AI Provider 配置 (可自定义)
# ════════════════════════════════════════════

# ═══ 默认使用 DeepSeek (用户配置) ═══
# 已内置 Key, 想换模型设置环境变量即可:
#   $env:AI_SUMMARY_KEY="sk-xxxx"          # 换 Key
#   $env:AI_SUMMARY_MODEL="deepseek-chat"  # 换模型
#   $env:AI_SUMMARY_PROVIDER="zhipu"       # 切回智谱
#   $env:AI_SUMMARY_BASE_URL="https://api.deepseek.com"

_BUILTIN_DS_KEY = os.environ.get("AI_SUMMARY_KEY", "")  # Set your key in env

PROVIDER = os.environ.get("AI_SUMMARY_PROVIDER", "deepseek")
MODEL = os.environ.get("AI_SUMMARY_MODEL", "deepseek-v4-flash")
API_KEY = os.environ.get("AI_SUMMARY_KEY") or _BUILTIN_DS_KEY
BASE_URL = os.environ.get("AI_SUMMARY_BASE_URL", "https://api.deepseek.com")


def _call_llm(system_prompt: str, user_prompt: str,
              max_tokens: int = 1024, temperature: float = 0.4) -> Optional[str]:
    """调用 LLM (默认智谱，可自定义)"""
    if not API_KEY:
        return None

    url = f"{BASE_URL.rstrip('/')}/chat/completions"
    
    payload = json.dumps({
        "model": MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "max_tokens": max_tokens,
        "temperature": temperature,
    }).encode("utf-8")

    req = Request(url, data=payload, headers={
        "Content-Type": "application/json",
        "Authorization": f"Bearer {API_KEY}",
    })

    try:
        with urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            return data["choices"][0]["message"]["content"]
    except URLError as e:
        print(f"  ⚠️ AI 调用失败: {e}")
        return None
    except (KeyError, json.JSONDecodeError) as e:
        print(f"  ⚠️ AI 响应解析失败: {e}")
        return None


def _build_context(sig: dict) -> str:
    """构建传给 LLM 的上下文数据"""
    r = sig["regime"]
    lines = []

    # 市场状态
    lines.append(f"## 市场状态")
    lines.append(f"- 指数点位: {r.get('price', 'N/A')}")
    lines.append(f"- 200日均线: {r.get('ma200', 'N/A')}")
    lines.append(f"- 20日动量: {r.get('mom_20', 0):+.1f}%")
    lines.append(f"- 信号: {' | '.join(r['signals'])}")
    regime_map = {"bull": "牛市", "bear": "熊市", "neutral": "震荡"}
    lines.append(f"- 市场判断: {regime_map.get(r['regime'], '未知')}")
    lines.append(f"- 建议仓位: {sig['position_mult']*100:.0f}%")
    lines.append("")

    # 应持有
    if sig["hold"]:
        lines.append(f"## 应持有 (Top {sig['top_k']}, 动量窗口 {sig['lookback']}日)")
        for i, (code, info) in enumerate(sig["hold"]):
            lines.append(f"{i+1}. {code} {info['name']} — ¥{info['price']:.3f} — "
                        f"动量 {info['momentum']:+.1f}% — 波动 {info['volatility']:.1f}%")
    else:
        lines.append("## 应持有: 无 (空仓)")

    lines.append("")

    # 观察列表
    if sig["out"]:
        lines.append("## 观察列表 (未入围)")
        for i, (code, info) in enumerate(sig["out"][:8]):
            lines.append(f"{i+1}. {code} {info['name']} — 动量 {info['momentum']:+.1f}%")

    # 板块温度
    lines.append("")
    lines.append("## 板块20日动量速览")
    from config import DATA_MARKET
    pool = {"510050": "上证50", "510300": "沪深300", "510500": "中证500",
            "159915": "创业板", "588000": "科创50", "512880": "证券",
            "512010": "医药", "512690": "酒/消费", "510880": "红利", "511010": "国债"}
    for code, name in pool.items():
        p = DATA_MARKET / f"etf_{code}.parquet"
        if p.exists():
            df = load(p)
            if len(df) >= 20:
                mom = (df["close"].iloc[-1] / df["close"].iloc[-20] - 1) * 100
                lines.append(f"- {code} {name}: {mom:+.1f}%")

    return "\n".join(lines)


SYSTEM_PROMPT = """你是一个 AI 量化分析助手，专门为 A 股 ETF 动量轮动策略提供每日简报。
用户是一个临床统计程序员，不懂股市，需要你用最简单的人话帮他理解今天的市场情况和操作建议。

要求:
1. 用中文，口语化但不随意
2. 先一句话总结今天该不该持有、持有什么
3. 解释为什么（市场状态 + 动量数据）
4. 如果仓位不满 100%，解释避险原因
5. 指出当前最强和最弱的板块
6. 如果发现任何异常（极端动量、极端波动），主动提醒风险
7. 最后给一句可操作的建议（买/持/减/空）
8. 不要写免责声明，不要写"不构成投资建议"
9. 控制在 300 字以内
10. 不要编造数据，只基于提供的上下文"""


def generate_daily_report(sig: dict = None, brief: bool = False,
                          use_ai: bool = True) -> str:
    """生成每日 AI 摘要报告

    Args:
        sig: 信号数据 (None = 自动生成)
        brief: 简洁版 (不调 AI)
        use_ai: 是否调 AI (False = 只出数据)
    
    Returns:
        str: Markdown 格式日报
    """
    if sig is None:
        sig = generate_signals()

    context = _build_context(sig) if use_ai else ""

    lines = []
    lines.append(f"# 🤖 AI量化日报 · {sig['date']}")
    lines.append(f"> 数据日期: **{sig['data_date']}** | 策略: 动量轮动 Top{sig['top_k']}")
    lines.append("")

    # ── AI 解读 ──
    if use_ai and not brief:
        ai_text = _call_llm(SYSTEM_PROMPT, context, max_tokens=512)
        if ai_text:
            lines.append("## 📝 AI 市场解读")
            lines.append("")
            lines.append(ai_text.strip())
            lines.append("")
            lines.append("---")
            lines.append("")
        else:
            lines.append("> ⚠️ AI 摘要生成失败（检查 API Key），以下为原始数据")
            lines.append("")

    # ── 原始信号数据 ──
    r = sig["regime"]
    emoji = {"bull": "🟢", "bear": "🔴", "neutral": "🟡"}
    lines.append("## 📡 市场状态")
    lines.append("")
    lines.append(f"| 指标 | 数值 |")
    lines.append(f"|------|------|")
    lines.append(f"| 状态 | {emoji.get(r['regime'], '❓')} {r['regime']} |")
    lines.append(f"| 指数点位 | {r.get('price', 'N/A')} |")
    lines.append(f"| 200日均线 | {r.get('ma200', 'N/A')} |")
    lines.append(f"| 20日动量 | {r.get('mom_20', 0):+.1f}% |")
    lines.append(f"| 建议仓位 | **{sig['position_mult']*100:.0f}%** |")
    lines.append("")

    if sig["hold"]:
        lines.append(f"## 📌 应持有")
        lines.append("")
        lines.append("| # | 代码 | 名称 | 价格 | 动量 | 波动 |")
        lines.append("|---|------|------|------|------|------|")
        for i, (code, info) in enumerate(sig["hold"]):
            lines.append(f"| {i+1} | {code} | {info['name']} | "
                        f"¥{info['price']:.3f} | {info['momentum']:+.1f}% | "
                        f"{info['volatility']:.1f}% |")
    else:
        lines.append("## 📌 应持有: 无 (空仓)")

    lines.append("")

    # 未入围
    if sig["out"]:
        lines.append("## 👀 观察列表")
        lines.append("")
        for i, (code, info) in enumerate(sig["out"][:8]):
            lines.append(f"{i+1}. **{code}** {info['name']} — 动量 {info['momentum']:+.1f}%")

    lines.append("")
    lines.append(f"*生成时间: {datetime.now():%Y-%m-%d %H:%M} · AI量化平台*")

    report = "\n".join(lines)

    # 保存
    out_path = OUTPUT / "daily_report.md"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(report, encoding="utf-8")

    return report


def entry_check(sig: dict = None, dyn_pos: dict = None) -> str:
    """入场检查: 现在能不能进场？

    专为零持仓用户设计，给出明确的 GO / NO-GO 判断。
    v3: 加入动态连续仓位 + 二元择时双视角
    
    Returns:
        str: 详细入场评估报告
    """
    if sig is None:
        sig = generate_signals(top_k=3, lookback=20)

    r = sig["regime"]

    # 🆕 获取动态仓位
    if dyn_pos is None:
        try:
            from engine.dynamic_timing import dynamic_position_now
            dyn_pos = dynamic_position_now()
        except Exception:
            dyn_pos = {"position": sig["position_mult"], "regime": r["regime"],
                       "trend_z": 0, "mom_val": 0, "vol_rank": 0.5}

    # 二元仓位 (原逻辑)
    pct_binary = sig["position_mult"] * 100
    # 动态仓位 (连续)
    pct_dynamic = dyn_pos["position"] * 100

    # ── 评分体系 (满分 10) ──
    score = 0
    checks = []

    # 1. 市场状态 (0-2分) — 二元
    if r["regime"] == "bull":
        score += 2
        checks.append(("✅", "市场状态(二元)", "牛市确认，指数>200MA且趋势向上", 2))
    elif r["regime"] == "neutral":
        score += 1
        checks.append(("🟡", "市场状态(二元)", "震荡市，方向不明", 1))
    else:
        checks.append(("❌", "市场状态(二元)", "熊市警告，指数<200MA且向下", 0))

    # 2. 动态仓位 (0-3分) — 🆕 连续
    dp = dyn_pos["position"]
    if dp >= 0.70:
        score += 3
        checks.append(("✅", f"动态仓位({dyn_pos['regime']})",
                       f"连续仓位 {dp*100:.0f}%，信号强烈看多", 3))
    elif dp >= 0.40:
        score += 1
        checks.append(("🟡", f"动态仓位({dyn_pos['regime']})",
                       f"连续仓位 {dp*100:.0f}%，偏谨慎", 1))
    else:
        checks.append(("❌", f"动态仓位({dyn_pos['regime']})",
                       f"连续仓位 {dp*100:.0f}%，信号看空", 0))

    # 3. 动量质量 (0-3分)
    if sig["hold"]:
        top_mom = sig["hold"][0][1]["momentum"]
        if top_mom > 5:
            score += 3
            checks.append(("✅", "动量质量", f"Top1 动量 {top_mom:+.1f}%，趋势强劲", 3))
        elif top_mom > 0:
            score += 1
            checks.append(("🟡", "动量质量", f"Top1 动量 {top_mom:+.1f}%，勉强为正", 1))
        else:
            checks.append(("❌", "动量质量", f"Top1 动量 {top_mom:+.1f}%，全板块下行", 0))
    else:
        checks.append(("❌", "动量质量", "无可用信号", 0))

    # 4. 波动风险 (0-2分)
    if sig["hold"]:
        avg_vol = sum(x[1]["volatility"] for x in sig["hold"]) / len(sig["hold"])
        if avg_vol < 1.5:
            score += 2
            checks.append(("✅", "波动风险", f"平均波动 {avg_vol:.1f}%，低风险环境", 2))
        elif avg_vol < 2.5:
            score += 1
            checks.append(("🟡", "波动风险", f"平均波动 {avg_vol:.1f}%，中等风险", 1))
        else:
            checks.append(("⚠️", "波动风险", f"平均波动 {avg_vol:.1f}%，高波动警告", 0))
    else:
        checks.append(("—", "波动风险", "无持仓数据", 0))

    # ── 入场决策 ──
    if score >= 8:
        decision = "🟢 建议入场"
        detail = "市场环境良好，动量+波动均处于健康区间。可分批建仓。"
    elif score >= 5:
        decision = "🟡 可以入场但要谨慎"
        detail = "部分指标不在最佳状态。建议半仓试探，确认趋势后再加仓。"
    elif score >= 3:
        decision = "🟠 暂不建议入场"
        detail = "市场信号较弱。可小仓位试探（不超过 1/3），严格止损。"
    else:
        decision = "🔴 不要入场"
        detail = "市场环境不适合建仓。建议观望，等信号转好再进。"

    # ── 输出 ──
    lines = []
    lines.append("╔══════════════════════════════════════════════════╗")
    lines.append("║          🎯 AI量化 · 入场检查报告                ║")
    lines.append(f"║          {sig['date']}                           ║")
    lines.append("╚══════════════════════════════════════════════════╝")
    lines.append("")
    lines.append(f"  📊 综合评分: {score}/10")
    lines.append(f"  🎯 入场决策: {decision}")
    lines.append(f"  💬 {detail}")
    lines.append("")
    lines.append("  ── 评分明细 ──")
    for icon, name, desc, pts in checks:
        lines.append(f"  {icon} {name} ({pts}分): {desc}")
    lines.append("")
    lines.append("  ── 如果入场，该买什么 ──")
    lines.append(f"  二元择时仓位: {pct_binary:.0f}% | 动态连续仓位: {pct_dynamic:.0f}%")
    lines.append(f"  建议参考: 动态仓位 {pct_dynamic:.0f}% (更细腻，熊市也能微投)")

    if sig["hold"] and pct_dynamic > 0:
        # 用动态仓位计算实际仓位
        n_hold = sig["top_k"]
        lines.append(f"  等权持有 {n_hold} 只:")
        for i, (code, info) in enumerate(sig["hold"]):
            weight = pct_dynamic / n_hold
            lines.append(f"  {i+1}. {code} {info['name']:<6}  ¥{info['price']:.3f}  "
                        f"仓位 {weight:.0f}%  动量 {info['momentum']:+.1f}%")
    elif sig["hold"] and pct_binary > 0:
        n_hold = sig["top_k"]
        lines.append(f"  等权持有 {n_hold} 只 (二元仓位):")
        for i, (code, info) in enumerate(sig["hold"]):
            weight = pct_binary / n_hold
            lines.append(f"  {i+1}. {code} {info['name']:<6}  ¥{info['price']:.3f}  "
                        f"仓位 {weight:.0f}%  动量 {info['momentum']:+.1f}%")
    else:
        lines.append("  当前信号建议空仓或极少仓位。")

    lines.append("")
    lines.append("  ── 风控提醒 ──")
    lines.append("  ⚡ 单笔止损: -8%")
    lines.append("  ⚡ 总回撤超 15% → 减半仓")
    lines.append("  ⚡ 总回撤超 25% → 全清")
    lines.append("")

    report = "\n".join(lines)
    print(report)

    # 保存
    out_path = OUTPUT / "entry_check.md"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    marker = f"# AI量化入场检查 · {sig['date']}\n\n综合评分: {score}/10 | 决策: {decision}\n\n{detail}"
    out_path.write_text(marker + "\n\n" + report, encoding="utf-8")

    return report


# ════════════════════════════════════════════
# CLI
# ════════════════════════════════════════════

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="AI量化摘要模块")
    p.add_argument("--signal", action="store_true", help="只输出信号数据(不调AI)")
    p.add_argument("--brief", action="store_true", help="简洁版")
    p.add_argument("--entry", action="store_true", help="入场检查(零持仓)")
    p.add_argument("--tk", type=int, default=3, help="持仓数")
    p.add_argument("--lb", type=int, default=20, help="动量窗口")
    args = p.parse_args()

    sig = generate_signals(top_k=args.tk, lookback=args.lb)

    if args.entry:
        entry_check(sig)
    elif args.signal or args.brief:
        from live_signal import print_signals
        print_signals(sig)
    else:
        report = generate_daily_report(sig, brief=args.brief)
        print(report)
