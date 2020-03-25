#!/usr/bin/env python3

# HACK: set path to timelord submodule
import sys

sys.path.insert(0, "./timelord")
sys.path.insert(0, ".")


import numpy as np
import pandas as pd
import torch as th
from common import load_data, load_model
from evaluation import simulate_mhp
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
    nodes, ns, ts, _ = load_data("/checkpoint/maxn/data/covid19/nj.h5")
    M = len(nodes)
    mus, beta, S, U, V, A, scale, timescale = load_model("/tmp/timelord_model.bin", M)
    target_date = "0322"

    # create episode
    nts = (ts - ts.min()) / timescale
    episode = Episode(th.from_numpy(nts).double(), th.from_numpy(ns).long(), False, M)

    sim_d = lambda d: simulate_mhp(16, d, episode, mus, beta, A, timescale, nodes)
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