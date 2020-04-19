#!/usr/bin/env python3

import h5py
import numpy as np
import pandas as pd
import os
import sys

from collections import defaultdict as ddict
from itertools import count

fout = sys.argv[1]
smooth = "SMOOTH" in os.environ and os.environ["SMOOTH"] == "1"
print("Smoothing data =", smooth)

if smooth:
    fout += "_smooth.h5"
else:
    fout += ".h5"


dparser = lambda x: pd.to_datetime(x, format="%m/%d/%Y")
df = pd.read_csv(
    sys.stdin,
    header=0,
    date_parser=dparser,
    parse_dates=["Test Date"],
    usecols=["Test Date", "County", "New Positives"],
)

df.columns = ["date", "county", "cases"]
print(df.head())
df = df.sort_values(by="date")
last_date = str(pd.to_datetime(df["date"]).max().date())

nevents = df["cases"].sum()
print(sorted(df.county.unique()))

ncount = count()
region_ids = ddict(ncount.__next__)
_ns = []
_ts = []


# convert timestamps to number of days since first outbreak:
df["date"] = df["date"].values.astype(np.int64) // 10 ** 9
df["date"] = (df["date"] - df["date"].min()) / (24 * 60 * 60) + 1
print(df.date.min(), df.date.max())

# convert counts to events
df_agg = df.groupby(["county"])
for county, group in df_agg:
    group = group.sort_values(by="date")
    ts = group["date"].values.astype(np.int)
    # ws = np.diff([0] + ws.tolist())
    if smooth:
        _ws = group["cases"]  # .values.astype(np.float)
        ws = _ws.rolling(window=3).mean().to_numpy()
        ws[ws != ws] = 0
        ws[0] = _ws.iloc[0]
    else:
        ws = group["cases"].values.astype(np.float)
    ncases = ws.sum()
    print(county, ncases, ws, ts)
    es = []
    # if len(ts) < 2:
    #    continue
    # tp = ts[0] - 1
    for i in range(len(ts)):
        w = int(ws[i])
        if w <= 0:
            continue
        tp = ts[i] - 1
        # print(tp, ts[i], w)
        _es = sorted(np.random.uniform(tp, ts[i], w))
        if len(es) > 0:
            # print(es[-1], _es[0], _es[-1])
            assert es[-1] < _es[0], (_es[0], es[-1])
        es += _es
        # tp = ts[i]
        # es += [ts[i]] * w
    # assert len(es) == ncases, (len(es), ncases)
    if len(es) > 0:
        kid = region_ids[county]
        _ts += es
        _ns += [kid] * len(es)


# assert len(_ts) == nevents, (len(_ts), nevents)
knames = [None for _ in range(len(region_ids))]
for kreis, i in region_ids.items():
    knames[i] = kreis


str_dt = h5py.special_dtype(vlen=str)
ds_dt = h5py.special_dtype(vlen=np.dtype("int"))
ts_dt = h5py.special_dtype(vlen=np.dtype("float32"))
with h5py.File(fout, "w") as fout:
    fout.attrs["basedate"] = last_date
    _dnames = fout.create_dataset("nodes", (len(knames),), dtype=str_dt)
    _dnames[:] = knames
    _cnames = fout.create_dataset("cascades", (1,), dtype=str_dt)
    _cnames[:] = ["covid19_nys"]
    ix = np.argsort(_ts)
    node = fout.create_dataset("node", (1,), dtype=ds_dt)
    node[0] = np.array(_ns, dtype=np.int)[ix]
    time = fout.create_dataset("time", (1,), dtype=ts_dt)
    time[0] = np.array(_ts, dtype=np.float)[ix]
