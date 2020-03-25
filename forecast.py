#!/usr/bin/env python3

# import warnings filter
from warnings import simplefilter

# ignore all future warnings
simplefilter(action="ignore", category=FutureWarning)

# HACK: set path to timelord submodule
import sys

sys.path.insert(0, "./timelord")
sys.path.insert(0, ".")

import argparse
import numpy as np
import pandas as pd
import torch as th
from datetime import timedelta
from scipy.stats import kstest
from common import load_data, load_model
from evaluation import simulate_mhp, goodness_of_fit
from tlc import Episode


def rmse(d, df_pred, target_date):
    df_gt = pd.read_csv(
        f"/checkpoint/maxn/data/covid19/nj-test-data-{target_date}.csv",
        usecols=["county", d],
    )
    df_nyu = pd.read_csv(
        f"/checkpoint/maxn/data/covid19/nj-pred-nyu.csv", usecols=["county", d]
    )
    df_nyu.columns = ["county", f"NYU {d}"]
    df_eval = pd.merge(df_gt, df_nyu, on="county")
    df_eval = pd.merge(df_eval, df_pred, on="county")
    rmse_nyu = np.sqrt(((df_eval[d] - df_eval[f"NYU {d}"]) ** 2).mean())
    rmse_mhp = np.sqrt(((df_eval[d] - df_eval[f"MHP {d}"]) ** 2).mean())
    # df_eval = df_eval[['county', 'confirmed', 'groundtruth', 'prediction', 'hawkes']]
    # df_eval.columns = ['County', f'Confirmed (d{int(t_obs)})', f'Confirmed (d{int(t_max)})', f'NYU (d{int(t_max)})', f'MHP (d{int(t_max)})']
    print(f"RMSE NYU ({d}) =", rmse_nyu)
    print(f"RMSE MHP ({d}) =", rmse_mhp)
    return df_eval


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Forecasting with Hawkes Embeddings")
    parser.add_argument(
        "-checkpoint",
        default="/tmp/timelord_model.bin",
        help="Path of checkpoint to load",
    )
    parser.add_argument("-dset", type=str, help="Forecasting dataset")
    parser.add_argument(
        "-basedate",
        type=str,
        help="Base date from which forecast dates are formated (%m%d format)",
    )
    parser.add_argument("-fout", type=str, help="Output file for forecasts")
    opt = parser.parse_args(sys.argv[1:])

    nodes, ns, ts, _ = load_data(opt.dset)
    M = len(nodes)
    mus, beta, S, U, V, A, scale, timescale = load_model(opt.checkpoint, M)
    base_date = pd.to_datetime(opt.basedate)

    # create episode
    nts = (ts - ts.min()) / timescale
    episode = Episode(th.from_numpy(nts).double(), th.from_numpy(ns).long(), False, M)
    t_obs = episode.timestamps[-1].item()

    # goodness of fit on observed data
    residuals = goodness_of_fit(episode, 0.01, mus, beta, A, nodes)
    ks, pval = zip(
        *[kstest(residuals[x], "expon") for x in range(M) if len(residuals[x]) > 2]
    )
    print("--- Goodness of fit ---")
    print(f"Avg. KS   = {np.mean(ks):.3f}")
    print(f"Avg. pval = {np.mean(pval):.3f}")
    print()

    # predictions
    sim_d = lambda d: simulate_mhp(t_obs, d, episode, mus, beta, A, timescale, nodes)
    d_eval = None
    for day in [1, 2, 3, 4, 5, 6, 7]:
        datestr = (base_date + timedelta(day)).strftime("%m/%d")
        df = sim_d(day)[["county", f"MHP d{day}"]]
        df.columns = ["county", datestr]
        if d_eval is None:
            d_eval = df
        else:
            d_eval = pd.merge(d_eval, df, on="county")
    print("--- Predictions ---")
    print(d_eval)

    if opt.fout is not None:
        d_eval.to_csv(opt.fout)

    """
    df_pred_d1 = sim_d(1)
    df_pred_d3 = sim_d(3)
    df_pred_d4 = sim_d(4)
    df_pred_d7 = sim_d(7)

    df_eval = rmse("d1", df_pred_d1, "0320")
    df_eval = pd.merge(
        df_eval, rmse("d3", df_pred_d3[["county", "MHP d3"]], "0322"), on="county"
    )
    df_eval = pd.merge(
        df_eval, rmse("d4", df_pred_d4[["county", "MHP d4"]], "0323"), on="county"
    )
    df_eval = pd.merge(df_eval, df_pred_d7[["county", "MHP d7"]], on="county")
    df_eval[
        [
            "county",
            "d1",
            "NYU d1",
            "MHP d1",
            "d3",
            "NYU d3",
            "MHP d3",
            "d4",
            "NYU d4",
            "MHP d4",
            "MHP d7",
        ]
    ].round(1)
    """
