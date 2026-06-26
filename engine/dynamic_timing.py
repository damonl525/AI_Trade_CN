"""动态择时 — 连续仓位 0-100% 而非 binary on/off

三个信号投票:
  Trend Strength  (40%): 价格距 200MA 多远 → 站得越高仓位越重
  Momentum Regime (35%): 60日动量方向 → 跌势中逐步减仓  
  Volatility Regime(25%): 波动率升高 → 降仓避险

输出: 0-100% 连续仓位, 不是 0/50/100 三档
"""

import pandas as pd
import numpy as np
from pathlib import Path

from config import DATA_MARKET
from data.fetcher import load


def _safe_rank(series: pd.Series, lookback: int = 252) -> pd.Series:
    """百分位排名: 0=最低, 1=最高。安全处理 nan"""
    ranked = series.rolling(lookback, min_periods=60).rank(pct=True)
    return ranked.fillna(0.5)


def dynamic_position_series(
    index_code: str = "000300",
    ma_period: int = 200,
    mom_period: int = 60,
    vol_period: int = 20,
    vol_lookback: int = 252,
) -> pd.DataFrame:
    """返回连续仓位倍率 (0~1) 的时间序列 & 诊断列

    Returns DataFrame columns:
        position : 0-1 建议仓位
        regime   : 强牛/弱牛/震荡/弱熊/强熊
        trend_z  : 价格距200MA的标准差距离
        mom_val  : 60日动量(%)
        vol_pct   : 当前波动率百分位
    """
    path = DATA_MARKET / f"index_{index_code}.parquet"
    if not path.exists():
        return pd.DataFrame({"position": [0.5]})

    df = load(path)
    if "date" in df.columns:
        df = df.copy()
        df["date"] = pd.to_datetime(df["date"])
        df = df.set_index("date").sort_index()
    close = df["close"]

    # ── Signal 1: Trend Strength (价格 vs 200MA, z-score 化) ──
    ma = close.rolling(ma_period).mean()
    std = close.rolling(ma_period).std()
    trend_z = ((close - ma) / std).fillna(0)  # z-score

    # z-score → weight (sigmoid-like, 0-1)
    # z=0 (价格=MA) → 0.5, z=2 (强牛) → 1.0, z=-2 (强熊) → 0.0
    trend_weight = 1.0 / (1.0 + np.exp(-1.5 * trend_z))
    trend_weight = trend_weight.clip(0.0, 1.0)

    # ── Signal 2: Momentum Regime (60日动量) ──
    mom_val = close.pct_change(mom_period) * 100
    mom_z = (mom_val - mom_val.rolling(252, min_periods=60).mean()) / mom_val.rolling(
        252, min_periods=60
    ).std()
    mom_z = mom_z.fillna(0)
    mom_weight = 1.0 / (1.0 + np.exp(-1.2 * mom_z))
    mom_weight = mom_weight.clip(0.0, 1.0)

    # ── Signal 3: Volatility Regime (波动率百分位) ──
    ret_1d = close.pct_change()
    vol = ret_1d.rolling(vol_period).std() * np.sqrt(252)  # 年化波动
    vol_rank = _safe_rank(vol, vol_lookback)
    # 高波动 → 低仓位: weight = 1 - rank
    vol_weight = 1.0 - vol_rank
    vol_weight = vol_weight.clip(0.0, 1.0)

    # ── 加权融合 ──
    position = 0.40 * trend_weight + 0.35 * mom_weight + 0.25 * vol_weight

    # 硬地板: 任何信号极度恐慌时最低 15%
    min_floor = pd.Series(0.15, index=position.index)
    mask_panic = (trend_z < -2.5) & (mom_val < -25)
    position[mask_panic] = np.minimum(position[mask_panic], 0.15)

    # 硬天花板: 极度亢奋时锁在 65% (防追顶)
    mask_euphoria = (trend_z > 2.5) & (mom_val > 40)
    position[mask_euphoria] = np.minimum(position[mask_euphoria], 0.65)

    position = position.clip(0.0, 1.0).fillna(0.5)

    # ── Regime 标签 ──
    def _label(p):
        if p >= 0.80:
            return "强牛"
        if p >= 0.60:
            return "弱牛"
        if p >= 0.35:
            return "震荡"
        if p >= 0.15:
            return "弱熊"
        return "强熊"

    regime = position.apply(_label)

    return pd.DataFrame(
        {
            "position": position,
            "regime": regime,
            "trend_z": trend_z,
            "mom_val": mom_val,
            "vol_rank": vol_rank,
        },
        index=df.index,
    )


def dynamic_position_now(index_code: str = "000300") -> dict:
    """当前时点的动态仓位建议"""
    dpos = dynamic_position_series(index_code)
    if dpos.empty:
        return {"position": 0.5, "regime": "无数据"}

    last = dpos.iloc[-1]
    return {
        "position": round(float(last["position"]), 2),
        "pct": f"{last['position']*100:.0f}%",
        "regime": last["regime"],
        "trend_z": round(float(last["trend_z"]), 2),
        "mom_val": round(float(last["mom_val"]), 1),
        "vol_rank": round(float(last["vol_rank"]), 2),
    }
