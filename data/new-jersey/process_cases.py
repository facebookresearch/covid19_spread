#!/usr/bin/env python3

import h5py
import numpy as np
import pandas as pd
import sys

from collections import defaultdict as ddict
from datetime import datetime
from itertools import count

fout = "timeseries.h5"

df = pd.read_csv(
    sys.argv[1],
    header=0,
    usecols=[
        "Start day",
        "Atlantic",
        "Bergen",
        "Burlington",
        "Camden",
        "Cape May",
        "Cumberland",
        "Essex",
        "Gloucester",
        "Hudson",
        "Hunterdon",
        "Mercer",
        "Middlesex",
        "Monmouth",
        "Morris",
        "Ocean",
        "Passaic",
        "Salem",
        "Somerset",
        "Sussex",
        "Union",
        "Warren",
        "Unknown",
    ],
)
df = df.sort_values(by="Start day")
# df = df.dropna()

counties = df.columns[1:]
df["Start day"] = np.arange(len(df)) + 1
print(df[counties])

ncount = count()
kreis_ids = ddict(ncount.__next__)
_ns = []
_ts = []
_ags = []
ats = df["Start day"].to_numpy()
print(ats)

# cmap = {"Morristown": "Morris"}

for county in counties:
    print(county)
    ws = df[county].to_numpy()
    ix = np.where(ws > 0)[0]
    if len(ix) < 1:
        continue
    ts = ats[ix]
    ws = np.diff([0] + ws[ix].tolist())
    es = []
    for i in range(len(ts)):
        w = int(ws[i])
        if w <= 0:
            continue
        tp = ts[i] - 1
        # print(tp, ts[i], w)
        _es = sorted(np.random.uniform(tp, ts[i], w))
        if len(es) > 0:
            print(es[-1], _es[0], _es[-1])
            assert es[-1] < _es[0], (_es[0], es[-1])
        es += _es
        # es += [ts[i]] * w
    if len(es) > 0:
        # if county in cmap:
        #    county = cmap[county]
        kid = kreis_ids[county]
        _ts += es
        _ns += [kid] * len(es)

# assert len(_ts) == nevents, (len(_ts), nevents)
knames = [None for _ in range(len(kreis_ids))]
for kreis, i in kreis_ids.items():
    knames[i] = kreis

pop = pd.read_csv("new_jersey-population.csv", header=None)
pop.columns = ["county", "population"]


str_dt = h5py.special_dtype(vlen=str)
ds_dt = h5py.special_dtype(vlen=np.dtype("int"))
ts_dt = h5py.special_dtype(vlen=np.dtype("float32"))
with h5py.File(fout, "w") as fout:
    _dnames = fout.create_dataset("nodes", (len(knames),), dtype=str_dt)
    _dnames[:] = knames
    _cnames = fout.create_dataset("cascades", (1,), dtype=str_dt)
    _cnames[:] = ["covid19_nj"]
    ix = np.argsort(_ts)
    node = fout.create_dataset("node", (1,), dtype=ds_dt)
    node[0] = np.array(_ns, dtype=np.int)[ix]
    time = fout.create_dataset("time", (1,), dtype=ts_dt)
    time[0] = np.array(_ts, dtype=np.float)[ix]
    # fout.create_dataset('population', data=pop.set_index('county').loc[knames].values.squeeze())
