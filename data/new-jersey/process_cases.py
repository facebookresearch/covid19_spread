#!/usr/bin/env python3

import h5py
import numpy as np
import os
import pandas as pd
import sys
from scraper import get_latest
from collections import defaultdict as ddict
from itertools import count


def main(infile=None, counts_only=False):
    usecols = [
        "Date",
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
    ]
    if infile is not None:
        df = pd.read_csv(infile, header=0, usecols=usecols)
    else:
        df = get_latest()[usecols]

    window = 7
    smooth = "SMOOTH" in os.environ and os.environ["SMOOTH"] == "1"
    print("Smoothing data =", smooth)

    if smooth:
        fout = "timeseries_smooth.h5"
    else:
        fout = "timeseries.h5"

    last_date = str(pd.to_datetime(df["Date"]).max().date())
    del df["Date"]

    df = df.sort_values(by="Start day")
    # df = df.dropna()

    # Dump the CSV for AR models, etc.
    clone = df[[c for c in df.columns if c not in {"Start day", "Unknown"}]].copy()
    clone.index = pd.date_range(end=last_date, periods=len(clone))
    clone.cummax().transpose().to_csv("data_cases.csv", index_label="region")

    if counts_only:
        return

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

    for county in counties:
        if smooth:
            _ws = df[county]
            ws = _ws.rolling(window=window).mean().to_numpy()
            ws[ws != ws] = 0
            ws[0] = _ws.iloc[0]
        else:
            ws = df[county].to_numpy()
        # ws = df[county].to_numpy()
        ix = np.where(ws > 0)[0]
        print(county, ws[-1])
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
                # print(es[-1], _es[0], _es[-1])
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

    script_dir = os.path.dirname(os.path.realpath(__file__))
    pop = pd.read_csv(f"{script_dir}/new_jersey-population.csv", header=None)
    pop.columns = ["county", "population"]

    str_dt = h5py.special_dtype(vlen=str)
    ds_dt = h5py.special_dtype(vlen=np.dtype("int"))
    ts_dt = h5py.special_dtype(vlen=np.dtype("float32"))
    with h5py.File(fout, "w") as hf:
        hf.attrs["basedate"] = last_date
        _dnames = hf.create_dataset("nodes", (len(knames),), dtype=str_dt)
        _dnames[:] = knames
        _cnames = hf.create_dataset("cascades", (1,), dtype=str_dt)
        _cnames[:] = ["covid19_nj"]
        ix = np.argsort(_ts)
        node = hf.create_dataset("node", (1,), dtype=ds_dt)
        node[0] = np.array(_ns, dtype=np.int)[ix]
        time = hf.create_dataset("time", (1,), dtype=ts_dt)
        time[0] = np.array(_ts, dtype=np.float)[ix]
        # hf.create_dataset('population', data=pop.set_index('county').loc[knames].values.squeeze())
        gt = df.set_index("Start day").transpose()
        gt = gt[sorted(gt.columns)]
        hf["ground_truth"] = gt.reindex(knames).values


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else None)
