"""AI量化平台全局配置"""
from pathlib import Path
import json

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

# ════════════════════════════════════════════
# ETF 全量池 (可通过 pool add/remove 增删)
# ════════════════════════════════════════════

_FULL_POOL_FILE = ROOT / "data" / "etf_pool.json"

_DEFAULT_POOL = {
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
    "562500": "机器人",
    "518880": "黄金ETF",
    "159819": "AI智能",
    "159930": "能源ETF",
}


def load_pool() -> dict:
    """加载 ETF 池 (优先用户自定义, 否则默认)"""
    if _FULL_POOL_FILE.exists():
        try:
            data = json.loads(_FULL_POOL_FILE.read_text("utf-8"))
            if isinstance(data, dict) and len(data) > 0:
                return data
        except Exception:
            pass
    return dict(_DEFAULT_POOL)


def save_pool(pool: dict) -> None:
    """保存 ETF 池到文件"""
    _FULL_POOL_FILE.parent.mkdir(parents=True, exist_ok=True)
    _FULL_POOL_FILE.write_text(json.dumps(pool, ensure_ascii=False, indent=2), "utf-8")


def reset_pool() -> dict:
    """重置为默认池"""
    if _FULL_POOL_FILE.exists():
        _FULL_POOL_FILE.unlink()
    return dict(_DEFAULT_POOL)


# 运行时的池 = 当前有效的全部 ETF 代码
FULL_POOL = load_pool()
FULL_SYMBOLS = list(FULL_POOL.keys())

# Platform ready
