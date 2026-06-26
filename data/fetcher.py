"""AKShare 行情数据抓取器

支持:
- A股个股日线 (ak.stock_zh_a_hist)       
- ETF日线      (ak.fund_etf_hist_em, 兜底 fund_etf_hist_sina)
- 指数日线     (ak.stock_zh_index_daily)   
- 行业板块     (ak.stock_board_industry_hist_em)

所有数据统一存 Parquet，自动增量更新。
内置指数退避重试 + 东方财富→新浪自动兜底。
"""

import akshare as ak
import pandas as pd
import time
import random
from pathlib import Path

from config import DATA_MARKET  # noqa: F401 - 初始化目录


# ── 重试装饰器 ──
def _retry(max_retries=3, base_delay=1.0, backoff=2.0):
    """指数退避重试: 1s → 2s → 4s, 每次叠加随机抖动"""
    def deco(fn):
        def wrapper(*a, **kw):
            last_err = None
            for attempt in range(max_retries + 1):
                try:
                    return fn(*a, **kw)
                except Exception as e:
                    last_err = e
                    if attempt < max_retries:
                        delay = base_delay * (backoff ** attempt) + random.uniform(0, 1)
                        time.sleep(delay)
            raise last_err
        return wrapper
    return deco


# ── 标准化列名 ──
RENAME_A_STOCK = {
    "日期": "date", "开盘": "open", "收盘": "close", "最高": "high", "最低": "low",
    "成交量": "volume", "成交额": "amount", "振幅": "amplitude", "涨跌幅": "pct_chg",
    "涨跌额": "chg", "换手率": "turnover",
}

RENAME_ETF = {
    "日期": "date", "开盘": "open", "收盘": "close", "最高": "high", "最低": "low",
    "成交量": "volume", "成交额": "amount", "涨跌幅": "pct_chg",
}

RENAME_INDEX = {
    "date": "date", "open": "open", "close": "close", "high": "high", "low": "low",
    "volume": "volume", "amount": "amount",
}


def _to_ak_date(iso_date: str) -> str:
    """2020-01-01 → 20200101"""
    return iso_date.replace("-", "")


def _standardize(df: pd.DataFrame, mapping: dict) -> pd.DataFrame:
    """统一列名 + date → datetime 排序去重"""
    df = df.rename(columns={k: v for k, v in mapping.items() if k in df.columns})
    df["date"] = pd.to_datetime(df["date"])
    for c in ["open", "high", "low", "close", "volume", "amount"]:
        if c in df:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df.drop_duplicates("date").sort_values("date").reset_index(drop=True)


@_retry(max_retries=2, base_delay=1.0)
def fetch_a_stock(symbol: str, start: str = "2020-01-01", end: str = "2050-01-01", force: bool = False) -> Path:
    """拉A股个股日线 → data/market/{symbol}.parquet"""
    path = DATA_MARKET / f"{symbol}.parquet"
    if path.exists() and not force:
        return path

    df = ak.stock_zh_a_hist(symbol=symbol, period="daily", start_date=_to_ak_date(start), end_date=_to_ak_date(end), adjust="qfq")
    df = _standardize(df, RENAME_A_STOCK)
    df.to_parquet(path, index=False)
    return path


def _fetch_etf_em(symbol, start, end):
    """东方财富源 — 可能被限流断开"""
    return ak.fund_etf_hist_em(symbol=symbol, period="daily",
                                start_date=_to_ak_date(start),
                                end_date=_to_ak_date(end), adjust="qfq")

@_retry(max_retries=3, base_delay=2.0)  # ETF 专项重试: 2s → 4s → 8s
def _fetch_etf_with_retry(symbol, start, end):
    """东方财富加重重试"""
    return _fetch_etf_em(symbol, start, end)

def fetch_etf(symbol: str, start: str = "2020-01-01", end: str = "2050-01-01", force: bool = False) -> Path:
    """拉ETF日线 → data/market/etf_{symbol}.parquet (重试+缓存兜底)"""
    path = DATA_MARKET / f"etf_{symbol}.parquet"
    if path.exists() and not force:
        return path

    try:
        df = _fetch_etf_with_retry(symbol, start, end)
    except Exception as e:
        if Path(path).exists():
            print(f"  ⚠️ {symbol}: 刷新失败({e}), 使用缓存数据")
            return path
        raise  # 连缓存都没有，真挂了

    df = _standardize(df, RENAME_ETF)
    df.to_parquet(path, index=False)
    print(f"  ✅ {symbol}")
    return path


@_retry(max_retries=2, base_delay=1.0)
def fetch_index(symbol: str, start: str = "2020-01-01", end: str = "2050-01-01", force: bool = False) -> Path:
    """拉指数日线 → data/market/index_{symbol}.parquet"""
    path = DATA_MARKET / f"index_{symbol}.parquet"
    if path.exists() and not force:
        return path

    df = ak.stock_zh_index_daily(symbol=f"sh{symbol}" if len(symbol) == 6 else symbol)
    df = _standardize(df, RENAME_INDEX)
    df = df[(df["date"] >= start) & (df["date"] <= end)]
    df.to_parquet(path, index=False)
    return path


def load(path: Path | str) -> pd.DataFrame:
    """加载本地 Parquet"""
    df = pd.read_parquet(path)
    df["date"] = pd.to_datetime(df["date"])
    return df.sort_values("date").reset_index(drop=True)


def load_many(symbols: list[str], prefix: str = "") -> pd.DataFrame:
    """批量加载，返回 date | symbol | close | open | high | low | volume 宽表（close pivot）"""
    frames = {}
    for s in symbols:
        p = DATA_MARKET / f"{prefix}{s}.parquet"
        if not p.exists():
            continue
        df = load(p)[["date", "close"]].copy()
        df = df.rename(columns={"close": s})
        frames[s] = df
    if not frames:
        return pd.DataFrame()
    out = frames[symbols[0]]
    for s in symbols[1:]:
        if s in frames:
            out = out.merge(frames[s], on="date", how="outer")
    return out.sort_values("date").reset_index(drop=True)


def save(df: pd.DataFrame, path: Path) -> Path:
    df.to_parquet(path, index=False)
    return path
