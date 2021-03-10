#!/usr/bin/env python3

import numpy as np
import pandas as pd
import sys
import torch as th
from collections import defaultdict

dparser = lambda x: pd.to_datetime(x, format="%m/%d/%Y")
df = pd.read_csv(
    sys.stdin,
    date_parser=dparser,
    parse_dates=["Test Date"],
    usecols=["Test Date", "County", "Cumulative Number of Positives"],
)
df = df.dropna()
df.columns = ["date", "county", "cases"]

print(df.head())

dates = np.unique(df["date"])
counties = np.unique(df["county"])
index = []

cols = {
    pd.to_datetime(date).strftime("%Y-%m-%d"): np.zeros(len(counties), dtype=np.int)
    for date in dates
}
cols["region"] = counties

county_index = {c: i for i, c in enumerate(counties)}

df_agg = df.groupby(["county"])
for name, group in df_agg:
    group = group.sort_values(by="date")
    cix = county_index[name]
    print(name, cix)
    _dates = group["date"].to_numpy()
    _cases = group["cases"].to_numpy()
    for i, _date in enumerate(_dates):
        dix = pd.to_datetime(_date).strftime("%Y-%m-%d")
        cols[dix][cix] = _cases[i]

df = pd.DataFrame(cols)
df.set_index("region", inplace=True)
df.to_csv(sys.argv[1])
