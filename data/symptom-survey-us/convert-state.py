#!/usr/bin/env python3

import numpy as np
import pandas as pd
import sys

signal = sys.argv[1]

df = pd.read_csv(
    f"state/{signal}.csv",
    parse_dates=["date"],
    usecols=["state", "date", signal, f"{signal}_sample_size"],
)
df.dropna(axis=0, subset=["date"], inplace=True)

dates = np.unique(df["date"])
states = np.unique(df["state"])

cols = {
    pd.to_datetime(date).strftime("%Y-%m-%d"): np.zeros(len(states), dtype=np.float)
    for date in dates
}
cols["region"] = states


df_agg = df.groupby("state")
for _state, group in df_agg:
    six = np.where(states == _state)[0][0]
    group = group.sort_values(by="date")
    _dates = group["date"].to_numpy()
    _cases = group[signal].to_numpy()
    for i, _date in enumerate(_dates):
        dix = pd.to_datetime(_date).strftime("%Y-%m-%d")
        cols[dix][six] = _cases[i]

print(cols)
df = pd.DataFrame(cols)
df.set_index("region", inplace=True)
df.to_csv(f"data-{signal}-state.csv")
