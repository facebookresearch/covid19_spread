#!/usr/bin/env python3

import numpy as np
import os
import pandas as pd
import sys

from datetime import timedelta
from common import load_data


def load_ground_truth(path):
    nodes, ns, ts, _ = load_data(path)
    nodes = [n for n in nodes if n != "Unknown"]
    gt = {n: len(np.where(ns == i)[0]) - 1 for i, n in enumerate(nodes)}
    tmax = int(np.ceil(ts.max()))
    counts = {n: np.zeros(tmax) for n in nodes}
    for t in range(1, tmax + 1):
        ix2 = np.where(ts < t)[0]
        for i, n in enumerate(nodes):
            ix1 = np.where(ns == i)[0]
            counts[n][t - 1] = len(np.intersect1d(ix1, ix2))
    return gt, nodes, counts


gt, cols, counts = load_ground_truth(sys.argv[1])
prefix = sys.argv[2]
basedate = pd.to_datetime(sys.argv[3])
suffix = sys.argv[4]

maes = {"Measure": ["MAE", "MAPE", "MASE"]}
ix = basedate.strftime("%m/%d")
for d in range(1, 8):
    pdate = basedate - timedelta(d)
    fname = f"{prefix}-{pdate.strftime('%Y%m%d')}{suffix}.csv"
    if not os.path.exists(fname):
        print(f"Skipping {fname}")
        continue
    df_pred = pd.read_csv(fname, usecols=cols + ["date"])
    df_pred = df_pred.set_index("date")
    vals = df_pred.loc[ix]

    pred_triv = {n: np.abs(c[-(d + 1)] - c[-d]) * d for n, c in counts.items()}

    print()
    print(pdate)
    errs = np.zeros(len(cols))
    gts = np.ones(len(cols))
    trivs = np.zeros(len(cols))
    for i, c in enumerate(cols):
        err = abs(gt[c] - vals[i])
        trivs[i] = abs(gt[c] - pred_triv[c])
        print(
            d,
            c,
            round(err, 2),
            gt[c],
            vals[i],
            round(err / gt[c], 3),
            trivs[i],
            pred_triv[c],
        )
        errs[i] = err
        # make sure errors and gts are aligned
        gts[i] = max(1, gt[c])
    maes[pdate.strftime("%m/%d")] = [
        np.mean(errs),
        np.mean(errs / gts),
        np.mean(errs / trivs),
    ]

df = pd.DataFrame(maes).round(2)

print()
print(df)
# print("MASE", mae / mae_triv)
