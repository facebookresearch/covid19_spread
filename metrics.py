#!/usr/bin/env python3
import numpy as np
import pandas as pd
import h5py
from datetime import timedelta
from load import load_data
import warnings
import sys


def load_ground_truth_h5(path):
    with h5py.File(path, "r") as hf:
        if "ground_truth" in hf.keys():
            assert "basedate" in hf.attrs, "`basedate` missing from HDF5 attrs!"
            basedate = pd.Timestamp(hf.attrs.get("gt_basedate", hf.attrs["basedate"]))
            ground_truth = pd.DataFrame(hf["ground_truth"][:])
            ground_truth.columns = pd.date_range(
                end=basedate, periods=ground_truth.shape[1]
            )
            ground_truth["county"] = hf["nodes"][:]
            # Ignore any Unknown counts
            ground_truth = ground_truth[~ground_truth["county"].str.contains("Unknown")]
            return ground_truth.set_index("county").transpose().sort_index()

    warnings.warn(
        (
            "Using raw HDF5 data as ground truth is deprecated.  You should"
            " instead create a `ground_truth` field in the HDF5 dataset.  This can give "
            "misleading results if you are smoothing the data."
        ),
        DeprecationWarning,
    )

    nodes, ns, ts, basedate = load_data(path)
    nodes = [n for n in nodes if n != "Unknown"]
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


def load_ground_truth(f_ground_truth):
    if f_ground_truth.endswith(".h5"):
        return load_ground_truth_h5(f_ground_truth)
    elif f_ground_truth.endswith(".csv"):
        return load_ground_truth_csv(f_ground_truth)
    else:
        raise RuntimeError(f"Unrecognized extension: {f_ground_truth}")


def rmse(pred, gt):
    return (pred - gt).pow(2).mean(axis=1).pow(1.0 / 2)


def mae(pred, gt):
    return (pred - gt).abs().mean(axis=1)


def mape(pred, gt):
    return ((pred - gt).abs() / gt.clip(1)).mean(axis=1)


def max_mae(pred, gt):
    return (pred - gt).abs().max(axis=1)


def compute_metrics(f_ground_truth, f_predictions, mincount=0, nanfill=False):
    df_true = load_ground_truth(f_ground_truth)
    df_pred = pd.read_csv(f_predictions, parse_dates=["date"], index_col="date")
    return _compute_metrics(df_true, df_pred, mincount, nanfill=nanfill)


def _compute_metrics(df_true, df_pred, mincount=0, nanfill=False):
    if nanfill:
        cols = sorted(set(df_true.columns).difference(set(df_pred.columns)))
        zeros = pd.DataFrame(np.zeros((len(df_pred), len(cols))), columns=cols)
        zeros.index = df_pred.index
        df_pred = pd.concat([df_pred, zeros], axis=1)

    common_cols = list(set(df_true.columns).intersection(set(df_pred.columns)))
    df_pred = df_pred[common_cols]
    df_true = df_true[common_cols]
    z = len(df_pred)
    # print(df_pred.round(2))

    basedate = df_pred.index.min()
    pdate = basedate - timedelta(1)

    diff = df_true.loc[pdate] - df_true.loc[basedate - timedelta(2)]
    naive = [df_true.loc[pdate] + d * diff for d in range(1, z + 1)]
    naive = pd.DataFrame(naive)
    naive.index = df_pred.index

    ix = df_pred.index.intersection(df_true.index)

    df_pred = df_pred.loc[ix]
    naive = naive.loc[ix]
    gt = df_true.loc[ix]

    # compute state level MAE
    state_gt = gt.transpose().groupby(lambda x: x.split(", ")[-1]).sum()
    state_pred = df_pred.transpose().groupby(lambda x: x.split(", ")[-1]).sum()
    state_mae = (state_gt.sort_index() - state_pred.sort_index()).abs().mean(axis=0)

    metrics = pd.DataFrame(
        [
            rmse(df_pred, gt),
            mae(df_pred, gt),
            mape(df_pred, gt),
            rmse(naive, gt),
            mae(naive, gt),
            max_mae(df_pred, gt),
            max_mae(naive, gt),
            state_mae,
        ],
        columns=df_pred.index.to_numpy(),
    )
    metrics["Measure"] = [
        "RMSE",
        "MAE",
        "MAPE",
        "RMSE_NAIVE",
        "MAE_NAIVE",
        "STATE_MAE",
        "MAX_MAE",
        "MAX_NAIVE_MAE",
    ]
    metrics.set_index("Measure", inplace=True)
    if metrics.shape[1] > 0:
        metrics.loc["MAE_MASE"] = metrics.loc["MAE"] / metrics.loc["MAE_NAIVE"]
        metrics.loc["RMSE_MASE"] = metrics.loc["RMSE"] / metrics.loc["RMSE_NAIVE"]

        # Stack predictions onto last ground truth date.
        # We'll take the diff and compute MAE on the new daily counts
        stack = pd.concat(
            [df_true.loc[[df_pred.index.min() - timedelta(days=1)]], df_pred]
        )
        stack_diff = stack.diff().loc[ix]
        true_diff = df_true.diff().loc[ix]
        metrics.loc["MAE_DELTAS"] = mae(stack_diff, true_diff)
        metrics.loc["RMSE_DELTAS"] = rmse(stack_diff, true_diff)
    return metrics


if __name__ == "__main__":
    f_gt = sys.argv[1]
    f_pred = sys.argv[2]
    m = compute_metrics(f_gt, f_pred, 1)
    print(m)
