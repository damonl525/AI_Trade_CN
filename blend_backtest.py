"""回测：波动率加权 + 基准 + Frog + 三向融合，叠加动态择时"""
import pandas as pd, numpy as np, os, warnings
warnings.filterwarnings('ignore')

data_dir = 'data/market'
files = [f for f in os.listdir(data_dir) if f.endswith('.parquet') and not f.startswith('hs300')]
prices = {}
for f in files:
    code = f.replace('etf_', '').replace('.parquet', '')
    df = pd.read_parquet(os.path.join(data_dir, f))
    if 'close' in df.columns and 'date' in df.columns and len(df) > 60:
        df['date'] = pd.to_datetime(df['date'])
        df = df.set_index('date')
        prices[code] = df['close']

price_df = pd.DataFrame(prices).dropna()
print(f'Data: {price_df.index[0].strftime("%Y-%m-%d")} ~ {price_df.index[-1].strftime("%Y-%m-%d")}, {len(price_df)}d, {len(prices)} ETFs')
print(f'Columns (first 5): {list(price_df.columns[:5])}')

# ======== Momentum ========
def baseline_mom(s, lb=20):
    return s / s.shift(lb) - 1

def frog_mom(s, lb=20):
    r = s.pct_change()
    pos = r.clip(lower=0)
    den = r.abs().rolling(lb).sum()
    return pos.rolling(lb).sum() / den.replace(0, np.nan)

def volweight_mom(s, lb=20):
    r = s.pct_change()
    vol = r.rolling(60).std()
    raw = s / s.shift(lb) - 1
    return raw * (0.15 / vol.replace(0, np.nan))

# ======== Dynamic timing ========
def dyn_timing(idx_close, dates):
    w = pd.Series(0.0, index=dates)
    if idx_close is None or len(idx_close) < 200:
        return w
    aligned = idx_close.reindex(idx_close.index.union(dates)).sort_index().ffill()
    for i, d in enumerate(dates):
        if d not in aligned.index:
            continue
        loc = aligned.index.get_loc(d)
        if loc < 200:
            continue
        ma200 = aligned.iloc[max(0,loc-200):loc+1].mean()
        cur = aligned.iloc[loc]
        std60 = aligned.iloc[max(0,loc-60):loc+1].pct_change().std()
        tz = (cur/ma200 - 1) / std60 if std60 and std60 > 0 else 0
        m50 = cur / aligned.iloc[max(0,loc-50)] - 1 if loc >= 50 else 0
        m50 = max(-0.5, min(0.5, m50))
        vr = 0.5
        if loc >= 60:
            rv = aligned.iloc[max(0,loc-20):loc+1].pct_change().std()
            hv = aligned.iloc[max(0,loc-60):loc+1].pct_change().std()
            if hv and hv > 0:
                vr = 1.0 - min(1.0, max(0.0, (rv/hv - 0.5)/1.5))
        ts = 1/(1+np.exp(-3*tz))
        ms = 0.5 + m50
        w.iloc[i] = max(0.0, min(1.0, 0.40*ts + 0.35*ms + 0.25*vr))
    return w

# ======== Backtest ========
def bt(mom_df, name, idx_c, td, top_k=3, freq=40):
    tw = dyn_timing(idx_c, td)
    cash, holdings, nav = 1.0, {}, []
    last = -999
    for ti, d in enumerate(td):
        row = mom_df.loc[d].dropna() if d in mom_df.index else pd.Series(dtype=float)
        if len(row) < 2:
            pnow = price_df.loc[d] if d in price_df.index else pd.Series(dtype=float)
            hv = sum(holdings.get(c,0)*pnow[c] for c in holdings if c in pnow.index)
            nav.append(cash + hv)
            continue
        do_rebal = (ti - last >= freq)
        if do_rebal:
            top = row.sort_values(ascending=False).head(top_k).index.tolist()
            w_pos = tw.iloc[ti] if ti < len(tw) else 0.5
            pnow = price_df.loc[d]
            tv = cash + sum(holdings.get(c,0)*pnow[c] for c in holdings if c in pnow.index)
            holdings = {}
            if w_pos > 0.05 and len(top) > 0:
                per = tv * w_pos / len(top)
                for code in top:
                    if code in pnow.index and pnow[code] > 0:
                        holdings[code] = per / pnow[code]
                cash = tv * (1 - w_pos)
            else:
                cash = tv
            last = ti
        pnow = price_df.loc[d]
        hv = sum(holdings.get(c,0)*pnow[c] for c in holdings if c in pnow.index)
        nav.append(cash + hv)
    ns = pd.Series(nav, index=td)
    rets = ns.pct_change().dropna()
    tot = ns.iloc[-1] - 1
    yrs = max(0.01, (td[-1] - td[0]).days / 365.25)
    ann = (ns.iloc[-1])**(1/yrs) - 1
    dd = (ns / ns.cummax() - 1).min()
    sh = (rets.mean()/rets.std()*np.sqrt(252)) if rets.std() > 0 else 0
    wr = (rets > 0).sum()/len(rets) if len(rets) > 0 else 0
    return {'name': name, 'tot': tot, 'ann': ann, 'dd': dd, 'sh': sh, 'wr': wr, 'nav': ns}

# ======== Compute momentums ========
print('Computing signals...')
mom = {}
mom['baseline'] = price_df.apply(lambda c: baseline_mom(c, 20))
print('  OK baseline')
mom['frog'] = price_df.apply(lambda c: frog_mom(c, 20))
print('  OK frog')
mom['volw'] = price_df.apply(lambda c: volweight_mom(c, 20))
print('  OK volweighted')

# 3-way blend: z-score equal-weight
print('  Computing 3-way blend...')
blend = pd.DataFrame(index=price_df.index, columns=price_df.columns, dtype=float)
for col in price_df.columns:
    bz = (mom['baseline'][col] - mom['baseline'][col].mean()) / mom['baseline'][col].std()
    fz = (mom['frog'][col] - mom['frog'][col].mean()) / mom['frog'][col].std()
    vz = (mom['volw'][col] - mom['volw'][col].mean()) / mom['volw'][col].std()
    blend[col] = (bz + fz + vz) / 3.0
mom['blend3'] = blend
print('  OK blend3')

# Index for timing (use 510050 close as proxy)
idx_c = price_df.get('510050', price_df.iloc[:, 0])

# Common start date
mask = pd.Series(True, index=price_df.index)
for n in ['baseline','frog','volw','blend3']:
    mask &= mom[n].notna().all(axis=1)
td = price_df.index[mask]
print(f'Trade dates: {len(td)} ({td[0].strftime("%Y-%m-%d")} ~ {td[-1].strftime("%Y-%m-%d")})')

# ======== Run ========
print('\nBacktesting...')
results = []
for n in ['baseline','frog','volw','blend3']:
    r = bt(mom[n], n, idx_c, td)
    results.append(r)
    print(f'  OK {n}')

# ======== Print ========
labels = {'baseline':'1.Baseline 20d','frog':'2.Frog-in-Pan','volw':'3.Vol-Weighted','blend3':'BEST 3-Way Blend'}
print()
print('='*75)
print(f'{"Method":<22} {"Total":>8} {"Ann":>8} {"MaxDD":>8} {"Sharpe":>7} {"Win%":>7}')
print('-'*60)
for r in sorted(results, key=lambda x: x['ann'], reverse=True):
    print(f'{labels.get(r["name"],r["name"]):<22} {r["tot"]:>7.1%} {r["ann"]:>7.1%} {r["dd"]:>7.1%} {r["sh"]:>6.2f} {r["wr"]:>6.1%}')
print('='*75)

# Yearly breakdown
print('\nYearly returns:')
years = sorted(set(d.year for d in td))
print(f'{"Year":<8}', end='')
for r in results:
    print(f'{labels.get(r["name"],r["name"]):>18}', end='')
print()
for y in years:
    m = td.year == y
    print(f'{y:<8}', end='')
    for r in results:
        if m.sum() > 1:
            seg = r['nav'].loc[m]
            ret = seg.iloc[-1]/seg.iloc[0] - 1
            print(f'{ret:>17.1%} ', end='')
        else:
            print(f'{"n/a":>18}', end='')
    print()
