#!/usr/bin/env python3

import h5py
import numpy as np
import pandas as pd

from itertools import count

fout = "timeseries.h5"

df = pd.read_csv("data.csv", parse_dates=["EventDate"])

df_sorted = df.sort_values(by="EventDate")
df_sorted["DaysSinceFirst"] = df_sorted["EventDate"] - df_sorted["EventDate"].iloc[0]
df_sorted["DaysSinceFirst"] = df_sorted["DaysSinceFirst"].dt.days.astype(np.float32)
n_events = df_sorted.shape[0]

str_dt = h5py.special_dtype(vlen=str)
ds_dt = h5py.special_dtype(vlen=np.dtype("int"))
ts_dt = h5py.special_dtype(vlen=np.dtype("float32"))

with h5py.File(fout, "w") as fout:
    dnames = fout.create_dataset("nodes", (n_events,), dtype=str_dt)
    dnames[:] = df_sorted["County"].values.astype(str)
    cnames = fout.create_dataset("cascades", (1,), dtype=str_dt)
    cnames[:] = ["covid19_florida"]
    node = fout.create_dataset("node", (n_events,), dtype=ds_dt)
    node[:] = np.arange(n_events)
    time = fout.create_dataset("time", (n_events,), dtype=ts_dt)
    time[:] = df_sorted["DaysSinceFirst"].values
    ags = fout.create_dataset("ags", (n_events,), dtype=ds_dt)
    ags[:] = np.ones(n_events)