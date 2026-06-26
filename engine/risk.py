"""风控引擎 — 仓位管理 + 止损 + 回撤熔断

原则:
1. 不在单个标的上押超过 40%
2. 最大回撤 > 15% → 强制半仓
3. 最大回撤 > 25% → 强制空仓
4. 每笔买入设 -8% 止损
"""

import backtrader as bt
import numpy as np


# ════════════════════════════════════════════
# 风控参数
# ════════════════════════════════════════════
MAX_SINGLE_POSITION = 0.40    # 单标的最多 40%
DRAWDOWN_HALF = 0.15           # 回撤 15% → 半仓
DRAWDOWN_FULL = 0.25           # 回撤 25% → 空仓
STOP_LOSS = -0.08              # 单笔 -8% 止损
TAKE_PROFIT = 0.30             # 单笔 +30% 止盈


# ════════════════════════════════════════════
# 仓位计算
# ════════════════════════════════════════════

def calc_position_size(total_value: float, price: float, 
                       weight: float, min_lots: int = 100) -> int:
    """计算目标仓位股数(整手)
    
    Args:
        total_value: 总资金
        price: 当前价格
        weight: 目标权重 (0~1)
        min_lots: 最小单位(ETF是100)
    
    Returns:
        应持有的股数(整手取整)
    """
    raw = total_value * weight / price
    lots = int(raw / min_lots) * min_lots
    return max(0, lots)


def drawdown_breach(current_value: float, peak_value: float) -> dict:
    """回撤熔断检查
    
    Returns:
        dict: {breach: bool, level: 'none'|'half'|'full', dd_pct: float}
    """
    if peak_value <= 0:
        return {"breach": False, "level": "none", "dd_pct": 0}
    
    dd_pct = (current_value / peak_value) - 1
    
    if dd_pct <= -DRAWDOWN_FULL:
        return {"breach": True, "level": "full", "dd_pct": dd_pct}
    elif dd_pct <= -DRAWDOWN_HALF:
        return {"breach": True, "level": "half", "dd_pct": dd_pct}
    else:
        return {"breach": False, "level": "none", "dd_pct": dd_pct}


# ════════════════════════════════════════════
# 风控 Backtrader 观察者
# ════════════════════════════════════════════

class RiskMonitor(bt.Observer):
    """实时监控回撤，打印警告"""
    
    lines = ("drawdown", "peak",)
    plotinfo = dict(plot=True, subplot=True, plotname="回撤监控")
    
    def next(self):
        current = self._owner.broker.getvalue()
        peak = max(current, self.lines.peak[-1] if len(self) > 1 else current)
        self.lines.peak[0] = peak
        self.lines.drawdown[0] = (current / peak - 1) * 100
