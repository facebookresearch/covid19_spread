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
from evaluation import simulate_mhp, goodness_of_fit
from tlc import Episode


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

    print_model_stats(mus, beta, S, U, V)

    # create episode
    nts = (ts - ts.min()) / timescale
    episode = Episode(th.from_numpy(nts).double(), th.from_numpy(ns).long(), False, M)
    t_obs = episode.timestamps[-1].item()
    print("max observation time: ", t_obs)

    # goodness of fit on observed data
    residuals = goodness_of_fit(episode, 0.001, mus, beta, A, nodes)
    ks, pval = zip(
        *[kstest(residuals[x], "expon") for x in range(M) if len(residuals[x]) > 2]
    )
    print("--- Goodness of fit ---")
    print(f"Avg. KS   = {np.mean(ks):.3f}")
    print(f"Avg. pval = {np.mean(pval):.3f}")
    print()

    # predictions
    sim_d = lambda d: simulate_mhp(
        t_obs, d, episode, mus, beta, A, timescale, nodes, 0.01, 10
    )
    d_eval = None
    for day in [1, 2, 3, 4, 5, 6, 7]:
        datestr = (base_date + timedelta(day)).strftime("%m/%d")
        _day = int(day / timescale)
        df = sim_d(_day)[["county", f"MHP d{_day}"]]
        df.columns = ["county", datestr]
        if d_eval is None:
            d_eval = df
        else:
            d_eval = pd.merge(d_eval, df, on="county")
    print("--- Predictions ---")
    print(d_eval)

    if opt.fout is not None:
        d_eval.to_csv(opt.fout)
