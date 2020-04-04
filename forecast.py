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
from common import load_data, load_model, print_model_stats
from evaluation import simulate_mhp, goodness_of_fit, ks_critical_value
from tlc import Episode


def main(args):
    parser = argparse.ArgumentParser(description="Forecasting with Hawkes Embeddings")
    parser.add_argument(
        "-checkpoint",
        default="/tmp/timelord_model.bin",
        help="Path of checkpoint to load",
    )
    parser.add_argument("-dset", type=str, help="Forecasting dataset")
    parser.add_argument(
        "-step-size", type=float, default=0.01, help="Step size for simulation"
    )
    parser.add_argument("-trials", type=int, default=50, help="Number of trials")
    parser.add_argument(
        "-basedate",
        type=str,
        help="Base date from which forecast dates are formated (%m%d format)",
    )
    parser.add_argument("-days", type=int, help="Number of days to forecast")
    parser.add_argument("-fout", type=str, help="Output file for forecasts")
    opt = parser.parse_args(args)

    nodes, ns, ts, _ = load_data(opt.dset)
    M = len(nodes)
    mus, beta, S, U, V, A, scale, timescale = load_model(opt.checkpoint, M)
    base_date = pd.to_datetime(opt.basedate)

    # A *= 0.99

    print_model_stats(mus, beta, S, U, V, A)

    # create episode
    nts = (ts - ts.min()) / timescale
    episode = Episode(th.from_numpy(nts).double(), th.from_numpy(ns).long(), False, M)
    t_obs = episode.timestamps[-1].item()
    print("max observation time: ", t_obs)

    # goodness of fit on observed data
    residuals = goodness_of_fit(episode, 0.001, mus, beta, A, nodes)
    ks, pval = zip(
        *[
            kstest(residuals[x], "expon") if len(residuals[x]) > 1 else (np.nan, np.nan)
            for x in range(M)
        ]
    )
    ks = np.array(ks)
    pval = np.array(pval)
    crit = [
        ks_critical_value(len(residuals[x]), 0.05) if len(residuals[x]) > 1 else np.nan
        for x in range(M)
    ]
    print("--- Goodness of fit ---")
    print(f"Avg. KS   = {np.mean(ks[ks == ks]):.3f}")
    print(f"Avg. pval = {np.mean(pval[pval == pval]):.3f}")
    print()

    assert len(ks) == len(nodes), (len(ks), len(nodes))
    for i, node in enumerate(nodes):
        print(
            f"{node:15s}: N = {len(residuals[i])}, KS = {ks[i]:.3f}, Crit = {crit[i]:.3f}, pval = {pval[i]:.3f}"
        )

    if opt.days is None or opt.days < 1:
        sys.exit(0)

    # predictions
    sim_d = lambda d: simulate_mhp(
        t_obs, d, episode, mus, beta, A, timescale, nodes, opt.step_size, opt.trials
    )
    d_eval = None
    # for day in range(1, opt.days + 1):
    datestrs = [
        (base_date + timedelta(d)).strftime("%m/%d") for d in range(opt.days + 1)
    ]
    # _day = int(day / timescale)
    d_eval = sim_d(opt.days)[["county"] + list(range(opt.days + 1))]
    d_eval.columns = ["county"] + datestrs
    # if d_eval is None:
    #    d_eval = df
    # else:
    #    d_eval = pd.merge(d_eval, df, on="county")
    # FIXME: this is wrong for data without an "Unknwon" column
    # compute sum without unknown
    # d_eval.set_index("county")
    d_eval = d_eval[d_eval["county"] != "Unknown"]
    # vals = d_eval.iloc[:-1, :][d_eval.columns[1:]].to_numpy()
    vals = d_eval[d_eval.columns[1:]].to_numpy()
    print("--- Predictions ---")
    d_eval = d_eval.append(
        pd.DataFrame(
            [["ALL REGIONS"] + vals.sum(axis=0).tolist()], columns=d_eval.columns
        ),
        ignore_index=True,
    )
    d_eval["KS"] = ks.tolist()[: len(d_eval)] + [np.mean(ks)]
    d_eval["pval"] = pval.tolist()[: len(d_eval)] + [np.mean(pval)]
    d_eval = d_eval.round(3)
    print(d_eval)

    # print("\n")
    # print(" --- Proportional Unknowns --- ")
    # unk = d_eval.loc[df["county"] == "Unknown"]
    # unk = unk[unk.columns[1:]].to_numpy()
    # # print(vals)
    # # print(unk)
    # # print(vals.sum(axis=0))
    # vals = vals + unk * vals / vals.sum(axis=0)
    # d_prop = pd.DataFrame(np.hstack([df["county"].to_numpy()[:-2, np.newaxis], vals]))
    # d_prop.columns = d_eval.columns
    # print(d_prop)

    if opt.fout is not None:
        d_eval.to_csv(opt.fout)


if __name__ == "__main__":
    main(sys.argv[1:])
