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
    parse_dates=["date"],
    usecols=["date", "abbreviation_canton_and_fl", "ncumul_conf"],
)
df = df.dropna()

cantons = [
    "AG",
    "AI",
    "AR",
    "BE",
    "BL",
    "BS",
    "FL",
    "FR",
    "GE",
    "GL",
    "GR",
    "JU",
    "LU",
    "NE",
    "NW",
    "OW",
    "SG",
    "SH",
    "SO",
    "SZ",
    "TG",
    "TI",
    "UR",
    "VD",
    "VS",
    "ZG",
    "ZH",
]

canton_id_list = []
for row in df["abbreviation_canton_and_fl"]:
    canton_id_list.append(cantons.index(row))

df["canton_id"] = canton_id_list
df["ncumul_conf"] = df["ncumul_conf"].astype(int)
print(df)

nevents = df["ncumul_conf"].sum()
# nevents = len(df)
print("Number of events", nevents)

ncount = count()
kreis_ids = ddict(ncount.__next__)
_ns = []
_ts = []
_ags = []


t0 = (df["date"].values.astype(np.int64) // 10 ** 9).min()
df_agg = df.groupby(["abbreviation_canton_and_fl", "canton_id"])
for (name, aid), group in df_agg:
    # print(name, aid)
    group = group.sort_values(by="date")
    ts = group["date"].values.astype(np.int64) // 10 ** 9
    ws = group["ncumul_conf"].values.astype(np.float)
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
_ts = [t / (24 * 60 * 60.0) for t in _ts]

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
