"""AI量化平台全局配置"""
from pathlib import Path

# ── 路径 ──
ROOT = Path(__file__).parent
DATA_MARKET = ROOT / "data" / "market"      # 行情数据 Parquet
DATA_SIGNALS = ROOT / "data" / "signals"     # 信号输出
OUTPUT = ROOT / "output"                     # 回测报告/图表

for d in [DATA_MARKET, DATA_SIGNALS, OUTPUT]:
    d.mkdir(parents=True, exist_ok=True)

# ── A股交易成本 ──
COMMISSION = 0.0005      # 万5 佣金（双边）
STAMP_DUTY = 0.0005      # 千0.5 印花税（仅卖出）
MIN_COMMISSION = 5.0     # 最低佣金 5元/笔

# ── 回测默认 ──
INIT_CASH = 100_000      # 初始资金 10万
BENCHMARK = "000300"     # 基准 = 沪深300
RISK_FREE = 0.025        # 无风险利率 2.5%

# ── 数据 ──
DEFAULT_START = "2020-01-01"
DEFAULT_FREQ = "D"       # 日频

# Platform ready
