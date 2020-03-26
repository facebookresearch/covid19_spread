#!/usr/bin/env python3

import h5py
import numpy as np
import pandas as pd

from collections import defaultdict as ddict
from itertools import count

fout = "timeseries.h5"

dparser = lambda x: pd.to_datetime(x, format="%Y-%m-%d")
df = pd.read_csv(
    "data.csv",
    date_parser=dparser,
    parse_dates=["DateVal"],
    usecols=["DateVal", "CMODateCount"],
)

# This is kinda simple because there's only one node (UK government policy)

nevents = df["CMODateCount"].sum()
print("Number of events", nevents)

ts=dates_in_seconds=df["DateVal"].values.astype(np.int64)//10**9
ws=counts=df["CMODateCount"].values.astype(np.int64)
DAY_SECONDS=86400

times=np.zeros(nevents, dtype="float64")

start=0
for date_in_seconds, count in zip(ts, ws):
    if count == 0:
        continue
    x=np.linspace(start=date_in_seconds,
                  stop=date_in_seconds+DAY_SECONDS,
                  num=count,
                  endpoint=False,
                  dtype=np.float64)
    times[start:(start + count)] = x
    start += count


times=times-times[0]
days=(times/DAY_SECONDS).astype(np.float32)

str_dt = h5py.special_dtype(vlen=str)
ds_dt = h5py.special_dtype(vlen=np.dtype("int"))
ts_dt = h5py.special_dtype(vlen=np.dtype("float32"))
with h5py.File(fout, "w") as fout:
    _nodes = fout.create_dataset("nodes", (1,), dtype=str_dt)
    _nodes[:] = ["united-kingdom"]
    _cascades = fout.create_dataset("cascades", (1,), dtype=str_dt)
    _cascades[:] = ["covid19_gb"]
    _node = fout.create_dataset("node", (1,), dtype=ds_dt)
    _node[0] = np.zeros(nevents, dtype=np.int)
    _agss = fout.create_dataset("ags", (1,), dtype=ds_dt)
    _agss[:] = np.ones(nevents, dtype=np.int)

    _time = fout.create_dataset("time", (1,), dtype=ts_dt)
    _time[0] = times
