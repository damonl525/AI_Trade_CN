import pandas as pd
from pathlib import Path

DATA = Path("data/market")
for f in sorted(DATA.glob("etf_*.parquet")):
    df = pd.read_parquet(f)
    print(f"{f.stem}: {df['date'].min()} ~ {df['date'].max()}  ({len(df)} rows)")
