#!/usr/bin/env python3

import numpy as np
import os
import pandas as pd
import sys

from datetime import timedelta
from common import load_data


verbose = True


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

maes = {"Measure": ["MAE", "MAPE", "NAIV", "MASE"]}
ix = basedate.strftime("%m/%d")
for d in range(1, 8):
    pdate = basedate - timedelta(d)
    fname = f"{prefix}-{pdate.strftime('%Y%m%d')}{suffix}.csv"
    if not os.path.exists(fname):
        print(f"Skipping {fname}")
        vals = [np.nan for _ in cols]
    else:
        df_pred = pd.read_csv(fname, usecols=cols + ["date"])
        df_pred = df_pred.set_index("date")
        df_pred = df_pred[cols]
        vals = df_pred.loc[ix]

    pred_triv = {
        n: c[-(d + 1)] + d * np.abs(c[-(d + 1)] - c[-(d + 2)])
        for n, c in counts.items()
    }

    print()
    print(pdate)
    log = []
    gts = np.ones(len(cols))
    errs = np.zeros(len(cols))
    errs_triv = np.zeros(len(cols))
    for i, c in enumerate(cols):
        if gt[c] < 10:
            continue
        err = gt[c] - vals[i]
        errs_triv[i] = abs(gt[c] - pred_triv[c])
        log.append(
            [
                d,
                c,
                counts[c][-(d + 2)],
                counts[c][-(d + 1)],
                gt[c],
                vals[i],
                pred_triv[c],
                err,
                errs_triv[i],
                err / errs_triv[i],
            ]
        )
        errs[i] = abs(err)
        # make sure errors and gts are aligned
        gts[i] = max(1, gt[c])
    maes[pdate.strftime("%m/%d")] = [
        np.mean(errs),
        np.mean(errs / gts),
        np.mean(errs_triv),
        np.mean(errs) / np.mean(errs_triv),
    ]

    if verbose:
        print(
            pd.DataFrame(
                log,
                columns=[
                    "Day",
                    "County",
                    "D-2",
                    "D-1",
                    "GT",
                    "Pred",
                    "Naiv",
                    "MAE",
                    "MAE Naiv",
                    "MAE Ratio",
                ],
            ).round(2)
        )

df = pd.DataFrame(maes).round(2)

print()
print(df)

print(" | ".join(f"{d+1}d = {m}" for d, m in enumerate(df.iloc[0][1:])))
# print("MASE", mae / mae_triv)
