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
    parse_dates=["Datum"],
    usecols=["Datum", "Gemeentenaam", "Gemeentecode", "Aantal"],
)
df = df.dropna()
print(df)

nevents = df["Aantal"].sum()
# nevents = len(df)
print("Number of events", nevents)

ncount = count()
kreis_ids = ddict(ncount.__next__)
_ns = []
_ts = []
_ags = []


t0 = (df["Datum"].values.astype(np.int64) // 10 ** 9).min()
df_agg = df.groupby(["Gemeentenaam", "Gemeentecode"])
for (name, aid), group in df_agg:
    print(name, aid)
    group = group.sort_values(by="Datum")
    ts = group["Datum"].values.astype(np.int64) // 10 ** 9
    ws = group["Aantal"].values.astype(np.float)
    es = []
    for i in range(len(ts)):
        w = int(ws[i])
        if i == 0:
            tp = ts[0] - w * 10
        else:
            tp = ts[i - 1]
        _es = np.linspace(
            max(tp, int(ts[i] - w * 10)) + 1, int(ts[i]), w, endpoint=True, dtype=np.int
        )
        es += _es.tolist()
        # es += [ts[i]] * w
    if len(es) > 0:
        kid = kreis_ids[name]
        _ts += es
        _ns += [kid] * len(es)
        _ags += [aid] * len(es)


# _ns = [kreis_ids[k] for k in df["District"]]
# _ts = df["Date"].values.astype(np.int64) // 10 ** 9
# _ags = df["AGS"]
# _ws = df["Count"]

# convert timestamps to number of days since first outbreak:
min_ts = min(_ts)
_ts = [t - min_ts for t in _ts]
_ts = [t / (24 * 60 * 60.) for t in _ts]

assert len(_ts) == nevents, (len(_ts), nevents)
knames = [None for _ in range(len(kreis_ids))]
for kreis, i in kreis_ids.items():
    knames[i] = kreis

str_dt = h5py.special_dtype(vlen=str)
ds_dt = h5py.special_dtype(vlen=np.dtype("int"))
ts_dt = h5py.special_dtype(vlen=np.dtype("float32"))
with h5py.File(fout, "w") as fout:
    _dnames = fout.create_dataset("nodes", (len(knames),), dtype=str_dt)
    _dnames[:] = knames
    _cnames = fout.create_dataset("cascades", (1,), dtype=str_dt)
    _cnames[:] = ["covid19_nl"]
    ix = np.argsort(_ts)
    node = fout.create_dataset("node", (1,), dtype=ds_dt)
    node[0] = np.array(_ns, dtype=np.int)[ix]
    time = fout.create_dataset("time", (1,), dtype=ts_dt)
    time[0] = np.array(_ts, dtype=np.float)[ix]
    _agss = fout.create_dataset("ags", (len(_dnames),), dtype=ds_dt)
    _agss[:] = np.array(_ags, dtype=np.int)[ix]
    # mark = fout.create_dataset("mark", (1,), dtype=ts_dt)
    # mark[0] = np.array(_ws, dtype=np.float)[ix]
