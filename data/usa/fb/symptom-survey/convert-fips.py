#!/usr/bin/env python3

import numpy as np
import pandas as pd
import sys

signal = sys.argv[1]
fips_map = pd.read_csv(
    "county_fips_master.csv", index_col=["fips"], encoding="ISO-8859-1"
)

df = pd.read_csv(
    f"county/{signal}.csv",
    parse_dates=["date"],
    usecols=["county", "date", signal, f"{signal}_sample_size"],
)
df = df.rename(columns={"county": "fips"})
df.dropna(axis=0, subset=["date"], inplace=True)

dates = np.unique(df["date"])
_fips = np.unique(df["fips"])
counties = []
fips = []
for f in _fips:
    try:
        cname = fips_map.loc[int(f)].county_name
        sname = fips_map.loc[int(f)].state_name
        counties.append(f"{cname}, {sname}")
        fips.append(f)
    except:
        print(f"Skipping FIPS {f}")
fips = np.array(fips)

cols = {
    pd.to_datetime(str(date)).strftime("%Y-%m-%d"): np.zeros(
        len(counties), dtype=np.float
    )
    for date in dates
}
cols["region"] = counties


df_agg = df.groupby("fips")
for _fips, group in df_agg:
    fix = np.where(fips == _fips)[0]
    if len(fix) < 1:
        continue
    fix = fix[0]
    group = group.sort_values(by="date")
    _dates = group["date"].to_numpy()
    _cases = group[signal].to_numpy()
    for i, _date in enumerate(_dates):
        dix = pd.to_datetime(_date).strftime("%Y-%m-%d")
        cols[dix][fix] = _cases[i]

df = pd.DataFrame(cols)
df.set_index("region", inplace=True)
df.to_csv(f"data-{signal}-fips.csv")
