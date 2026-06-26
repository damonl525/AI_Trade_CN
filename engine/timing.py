"""择时引擎 — 市场状态判断

核心理念:
- 牛市/震荡市: 动量轮动全仓
- 熊市: 空仓或全仓债基 511010
- 用多个过滤器避免单指标误判
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime

from config import DATA_MARKET
from data.fetcher import load


def market_regime(index_code: str = "000300", lookback_ma: int = 200) -> dict:
    """判断当前市场状态

    Returns:
        dict: {
            'regime': 'bull' | 'bear' | 'neutral',
            'signals': list[str],
            'action': str  # 'full' | 'half' | 'cash'
        }
    """
    path = DATA_MARKET / f"index_{index_code}.parquet"
    if not path.exists():
        return {"regime": "neutral", "signals": ["无指数数据"], "action": "half"}

    df = load(path)
    if "date" in df.columns:
        df = df.copy()
        df["date"] = pd.to_datetime(df["date"])
        df = df.set_index("date").sort_index()
    close = df["close"]

    # 1. 200MA 位置
    ma200 = close.rolling(lookback_ma).mean()
    price = close.iloc[-1]
    above_ma = price > ma200.iloc[-1]

    # 2. 斜率(趋势方向): 200MA 的 20日变化
    if len(ma200) > 20:
        slope = (ma200.iloc[-1] - ma200.iloc[-20]) / ma200.iloc[-20] * 100
    else:
        slope = 0

    # 3. 近期动量
    mom_20 = (close.iloc[-1] / close.iloc[-20] - 1) * 100 if len(close) > 20 else 0

    signals = []
    if above_ma:
        signals.append(f"价格>200MA")
    else:
        signals.append(f"价格<200MA")

    if slope > 0.5:
        signals.append("MA向上")
    elif slope < -0.5:
        signals.append("MA向下")
    else:
        signals.append("MA走平")

    signals.append(f"20日动量{mom_20:+.1f}%")

    # ── 决策 ──
    if above_ma and slope > -0.3 and mom_20 > -3:
        regime = "bull"
        action = "full"
    elif not above_ma and slope < -1 and mom_20 < -5:
        regime = "bear"
        action = "cash"
    else:
        regime = "neutral"
        action = "half"

    return {
        "regime": regime,
        "signals": signals,
        "action": action,
        "price": round(price, 2),
        "ma200": round(ma200.iloc[-1], 2),
        "mom_20": round(mom_20, 1),
    }


def should_rotate_to_bonds() -> bool:
    """是否应该全仓债基避险"""
    m = market_regime()
    return m["regime"] == "bear"


def timing_filter_series(index_code: str = "000300", lookback_ma: int = 200) -> pd.Series:
    """返回每日的仓位倍率: 1=满仓, 0.5=半仓, 0=空仓

    可直接在回测中乘到仓位计算上
    """
    path = DATA_MARKET / f"index_{index_code}.parquet"
    if not path.exists():
        return pd.Series(dtype=float)

    df = load(path)
    if "date" in df.columns:
        df = df.copy()
        df["date"] = pd.to_datetime(df["date"])
        df = df.set_index("date").sort_index()
    close = df["close"]
    ma200 = close.rolling(lookback_ma).mean()

    # 斜率
    slope = (ma200 - ma200.shift(20)) / ma200.shift(20) * 100

    # 20日动量
    mom_20 = close.pct_change(20) * 100

    # 仓位倍率
    above = close > ma200
    slope_ok = slope > -0.3
    mom_ok = mom_20 > -3

    # bull → 1, neutral → 0.5, bear → 0
    position = pd.Series(0.5, index=df.index)
    position[above & slope_ok & mom_ok] = 1.0
    position[~(above) & (slope < -1) & (mom_20 < -5)] = 0.0

    return position
