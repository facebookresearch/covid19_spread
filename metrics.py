#!/usr/bin/env python3
import numpy as np
import pandas as pd

from datetime import timedelta
from load import load_data


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
    df = pd.DataFrame(counts)
    df.index = pd.date_range(end=basedate, periods=len(df))
    return df


def load_ground_truth_csv(path):
    df = pd.read_csv(path)
    df = df.rename(columns={"region": "date"})
    df.set_index("date", inplace=True)
    df = df.transpose()
    df.index = pd.to_datetime(df.index)
    return df


def rmse(pred, gt):
    return (pred - gt).pow(2).mean(axis=1).pow(1. / 2)


def mae(pred, gt):
    return (pred - gt).abs().mean(axis=1)


def sae(pred, gt):
    return (pred - gt).abs().std(axis=1)


def mape(pred, gt):
    return ((pred - gt).abs() / gt.clip(1)).mean(axis=1)


def compute_metrics(f_ground_truth, f_predictions, mincount=10):
    if f_ground_truth.endswith(".h5"):
        df_true = load_ground_truth(f_ground_truth)
    elif f_ground_truth.endswith(".csv"):
        df_true = load_ground_truth_csv(f_ground_truth)
    else:
        raise RuntimeError(f"Unrecognized extension: {f_ground_truth}")
    cols = df_true.columns.to_numpy().tolist()
    df_pred = pd.read_csv(f_predictions, usecols=cols + ["date"], parse_dates=["date"])
    df_pred = df_pred.set_index("date")
    df_pred = df_pred[cols]
    z = len(df_pred)
    print(df_pred.round(2))

    basedate = df_pred.index[0]
    pdate = basedate - timedelta(1)
    # print(basedate, pdate)
    diff = df_true.loc[pdate] - df_true.loc[basedate - timedelta(2)]
    naive = [df_true.loc[pdate] + d * diff for d in range(1, z + 1)]
    naive = pd.DataFrame(naive)
    naive.index = df_pred.index
    # print(naive)

    gt = df_true.loc[df_pred.index]
    # print(df_true.loc[pdate], gt, df_pred)
    # err = df_pred - gt
    # rmse = (err ** 2).mean(axis=1).pow(1. / 2)
    # mae = err.abs().mean(axis=1)
    # mae_naive = (naive - gt).abs().mean(axis=1)
    # rmse_naive = (naive - gt).pow(2).mean(axis=1).pow(1. / 2)
    # mape = (err.abs() / gt.clip(1)).mean(axis=1)
    # mae_mase = mae / mae_naive
    # rmse_mase = rmse / rmse_naive
    metrics = pd.DataFrame(
        [
            rmse(df_pred, gt),
            mae(df_pred, gt),
            mape(df_pred, gt),
            rmse(naive, gt),
            mae(naive, gt),
        ],
        columns=df_pred.index.to_numpy(),
    )
    metrics["Measure"] = ["RMSE", "MAE", "MAPE", "RMSE_NAIVE", "MAE_NAIVE"]
    metrics.set_index("Measure", inplace=True)
    metrics.loc["MAE_MASE"] = metrics.loc["MAE"] / metrics.loc["MAE_NAIVE"]
    metrics.loc["RMSE_MASE"] = metrics.loc["RMSE"] / metrics.loc["RMSE_NAIVE"]
    return metrics


def _compute_metrics(f_ground_truth, f_predictions, mincount=0):
    gt, cols, counts, basedate = load_ground_truth(f_ground_truth)
    df_pred = pd.read_csv(f_predictions, usecols=cols + ["date"], parse_dates=["date"])
    df_pred = df_pred.set_index("date")
    df_pred = df_pred[cols]
    print(df_pred)
    z = len(df_pred)

    maes = {"Measure": ["MAE", "MAPE", "NAIV", "MASE"]}
    for d in range(len(df_pred)):
        pdate = basedate - timedelta(d)
        preds = df_pred.loc[pdate]

        pred_triv = {
            n: c[-(z + 1)] + (z - d) * np.abs(c[-(z + 1)] - c[-(z + 2)])
            for n, c in counts.items()
        }

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
