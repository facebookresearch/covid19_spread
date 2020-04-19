#!/usr/bin/env python3
import numpy as np
import pandas as pd

from datetime import timedelta
from common import load_data


def load_ground_truth(path):
    nodes, ns, ts, basedate = load_data(path)
    nodes = [n for n in nodes if n != "Unknown"]
    gt = {n: len(np.where(ns == i)[0]) - 1 for i, n in enumerate(nodes)}
    tmax = int(np.ceil(ts.max()))
    counts = {n: np.zeros(tmax) for n in nodes}
    for t in range(1, tmax + 1):
        ix2 = np.where(ts < t)[0]
        for i, n in enumerate(nodes):
            ix1 = np.where(ns == i)[0]
            counts[n][t - 1] = len(np.intersect1d(ix1, ix2))
    return gt, nodes, counts, pd.to_datetime(basedate)


def compute_metrics(f_ground_truth, f_predictions, mincount=10):
    gt, cols, counts, basedate = load_ground_truth(f_ground_truth)
    df_pred = pd.read_csv(f_predictions, usecols=cols + ["date"], parse_dates=["date"])
    df_pred = df_pred.set_index("date")
    df_pred = df_pred[cols]
    print(df_pred)
    maes = {"Measure": ["MAE", "MAPE", "NAIV", "MASE"]}
    for d in range(len(df_pred)):
        pdate = basedate - timedelta(d)
        preds = df_pred.loc[pdate]

        pred_triv = {
            n: c[-(d + 2)] + (d + 1) * np.abs(c[-(d + 2)] - c[-(d + 3)])
            for n, c in counts.items()
        }
        print(preds)
        print(pred_triv)

        gts = np.ones(len(cols))
        errs = np.zeros(len(cols))
        errs_triv = np.zeros(len(cols))
        for i, n in enumerate(cols):
            _gt = counts[n][-(d + 1)]
            if _gt < mincount:
                continue
            err = _gt - preds[n]
            errs_triv[i] = abs(_gt - pred_triv[n])
            errs[i] = abs(err)
            # make sure errors and gts are aligned
            gts[i] = max(1, _gt)
            maes[pdate.strftime("%Y-%m-%d")] = [
                np.mean(errs),
                np.mean(errs / gts),
                np.mean(errs_triv),
                np.mean(errs) / np.mean(errs_triv),
            ]
    df = pd.DataFrame(maes)
    return df
