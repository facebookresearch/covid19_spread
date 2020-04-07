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
    return gt, nodes


gt, cols = load_ground_truth(sys.argv[1])
prefix = sys.argv[2]
basedate = pd.to_datetime(sys.argv[3])
suffix = sys.argv[4]
# print(df_pred)

maes = {"Measure": ["MAE", "MAPE"]}
ix = basedate.strftime("%m/%d")
for d in range(1, 8):
    pdate = basedate - timedelta(d)
    fname = f"{prefix}-{pdate.strftime('%Y%m%d')}{suffix}.csv"
    if not os.path.exists(fname):
        print(f"Skipping {fname}")
        continue
    df_pred = pd.read_csv(fname, usecols=cols + ["date"])
    df_pred = df_pred.set_index("date")
    # print(d, pdate, df_pred.loc[ix])

    # dates = df_pred.pop("date")
    # vals = df_pred[cols].iloc[offset].to_numpy()
    vals = df_pred.loc[ix]

    # triv = np.mean(np.abs(np.diff(df_gt.counts)))

    print()
    print(pdate)
    errs = np.zeros(len(cols))
    for i, c in enumerate(cols):
        err = abs(gt[c] - vals[i])
        print(d, c, err, gt[c], vals[i])
        errs[i] = err
        vals[i] = gt[c]
    maes[pdate.strftime("%m/%d")] = [np.mean(errs), np.mean(errs / vals)]

df = pd.DataFrame(maes).round(2)

print()
print(df)
# print("MASE", mae / mae_triv)
