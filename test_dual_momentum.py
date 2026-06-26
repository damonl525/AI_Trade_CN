"""只读测试：基准 vs 双动量，不动任何文件"""
import json
from pathlib import Path
import pandas as pd

pool = json.loads(Path("data/etf_pool.json").read_text(encoding="utf-8"))
DATA_MARKET = Path("data/market")

results = []
for code, name in pool.items():
    p = DATA_MARKET / f"etf_{code}.parquet"
    if not p.exists():
        results.append((code, name, -999, 0))
        continue
    df = pd.read_parquet(p)
    if len(df) < 20:
        results.append((code, name, -999, 0))
        continue
    mom_20 = (float(df["close"].iloc[-1]) / float(df["close"].iloc[-20]) - 1) * 100
    price = float(df["close"].iloc[-1])
    results.append((code, name, mom_20, price))

# === 基准 20日动量 ===
print("=" * 70)
print("  基准 20日动量 (当前算法) — 排名")
print("=" * 70)
sorted_base = sorted([r for r in results if r[2] > -900], key=lambda x: x[2], reverse=True)
for i, (code, name, mom, price) in enumerate(sorted_base):
    flag = "Top3" if i < 3 else f"  #{i+1}"
    print(f"  {flag}  {code} {name:<8}  动量{mom:+.1f}%  {price:.3f}")

# === 双动量 ===
print()
print("=" * 70)
print("  双动量 (正动量过滤) — 排名")
print("=" * 70)
sorted_dual = sorted([r for r in results if r[2] > 0], key=lambda x: x[2], reverse=True)

kicked = [r for r in results if r[2] <= 0 and r[2] > -900]
print(f"  正动量 ETF: {len(sorted_dual)}/{len(results)} 只")
print()

for i, (code, name, mom, price) in enumerate(sorted_dual):
    flag = "Top3" if i < 3 else f"  #{i+1}"
    print(f"  {flag}  {code} {name:<8}  动量{mom:+.1f}%  {price:.3f}")

print(f"\n  双动量踢出 ({len(kicked)} 只):")
for code, name, mom, price in kicked:
    print(f"     {code} {name:<8}  动量{mom:+.1f}%")

# === 对比 ===
print()
print("=" * 70)
print("  对比总结")
print("=" * 70)

base_top3 = [r[0] for r in sorted_base[:3]]
dual_top3 = [r[0] for r in sorted_dual[:3]] if sorted_dual else []

print(f"  基准 Top3:  {' + '.join(base_top3)}")
print(f"  双动  Top3:  {' + '.join(dual_top3) if dual_top3 else '空仓！无正动量ETF'}")

same = set(base_top3) == set(dual_top3)
print(f"  持仓一致:   {'是' if same else '否 — 不同！'}")

added = set(dual_top3) - set(base_top3)
removed = set(base_top3) - set(dual_top3)
if added:
    print(f"  双动量新进: {added}")
if removed:
    print(f"  双动量踢出: {removed}")

# === 熊市模拟 ===
print()
print("=" * 70)
print("  极端场景模拟: 如果全池跌 -3% (模拟熊市)")
print("=" * 70)
sim_kicked = 0
sim_positive = 0
for code, name, mom, price in results:
    if mom > -900:
        sim = mom - 3  # all drop 3%
        if sim > 0:
            sim_positive += 1
        else:
            sim_kicked += 1

print(f"  正动量 ETF: {sim_positive}/{len(pool)}")
print(f"  双动量踢出: {sim_kicked}")
if sim_positive < 3:
    print(f"  后果: Top K=3 但只剩 {sim_positive} 只 -> 只能持有 {sim_positive} 只或空仓")
