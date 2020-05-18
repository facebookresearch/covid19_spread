#!/usr/bin/env python3

# import warnings filter
from warnings import simplefilter

# ignore all future warnings
simplefilter(action="ignore", category=FutureWarning)

import sys
import argparse
import numpy as np
import pandas as pd
import torch as th
from datetime import timedelta
from scipy.stats import kstest
from common import print_model_stats
from load import load_data, load_model
from evaluation import simulate_mhp, goodness_of_fit, simulate_tl_mhp, ks_critical_value
from tlc import Episode
from timelord.ll import SparseEmbeddingSoftplus
import h5py


def evaluate_goodness_of_fit(episode, mus, beta, A, nodes):
    M = len(nodes)
    n_cases = np.array([len(episode.occurrences_of_dim(i)) - 1 for i in range(M)])

    # goodness of fit on observed data
    residuals = goodness_of_fit(episode, 0.001, mus, beta, A, nodes)
    ks, pval = zip(
        *[
            kstest(residuals[x], "expon")
            if len(residuals[x]) > 1 and nodes[x] != "Unknown"
            else (np.nan, np.nan)
            for x in range(M)
        ]
    )
    ks = np.array(ks)
    pval = np.array(pval)
    crit = [
        ks_critical_value(len(residuals[x]), 0.05)
        if len(residuals[x]) > 1 and nodes[x] != "Unknown"
        else np.nan
        for x in range(M)
    ]

    print("--- Goodness of fit ---")
    ix = ks == ks
    print(len(ix), ks[ix].shape, n_cases[ix].shape)
    print(f"Avg. KS   = {np.average(ks[ix], weights=np.log(1 + n_cases)[ix]):.3f}")
    ix = pval == pval
    print(f"Avg. pval = {np.average(pval[ix], weights=np.log(1 + n_cases)[ix]):.3f}")
    print()

    assert len(ks) == len(nodes), (len(ks), len(nodes))
    for i, node in enumerate(nodes):
        print(
            f"{node:15s}: N = {n_cases[i]}, "
            f"KS = {ks[i]:.3f}, Crit = {crit[i]:.3f}, pval = {pval[i]:.3f}"
        )
    return ks, pval, crit


def prepare_data(path, timescale=1):
    nodes, ns, ts, _basedate = load_data(path)
    M = len(nodes)
    nts = (ts - ts.min()) / timescale
    episode = Episode(th.from_numpy(nts).double(), th.from_numpy(ns).long(), False, M)
    assert episode.timestamps[0] == 0
    t_obs = episode.timestamps[-1].item()
    print("max observation time: ", t_obs)
    return nodes, episode, t_obs


def main(args):
    parser = argparse.ArgumentParser(description="Forecasting with Hawkes Embeddings")
    parser.add_argument(
        "-checkpoint",
        default="/tmp/timelord_model.bin",
        help="Path of checkpoint to load",
    )
    parser.add_argument("-dset", type=str, help="Forecasting dataset")
    parser.add_argument("-dset-true", type=str, help="Forecasting dataset")
    parser.add_argument(
        "-step-size", type=float, default=0.01, help="Step size for simulation"
    )
    parser.add_argument("-trials", type=int, default=50, help="Number of trials")
    parser.add_argument(
        "-basedate",
        type=str,
        help="Base date from which forecast dates are formated (%m%d format)",
    )
    parser.add_argument(
        "-tl-simulate",
        action="store_true",
        help="Use timelord's simulation implementation",
    )
    parser.add_argument(
        "-std-dev",
        default=-1,
        type=float,
        help="std deviation threshold for excluding simulations (-1 for none)",
    )
    parser.add_argument("-days", type=int, help="Number of days to forecast")
    parser.add_argument("-fout", type=str, help="Output file for forecasts")
    opt = parser.parse_args(args)

    if opt.basedate is None:
        with h5py.File(opt.dset, "r") as hf:
            assert "basedate" in hf.attrs, "Missing basedate!"
            opt.basedate = hf.attrs["basedate"]

    # -- data --
    nodes_true, episode_true, t_obs_true = prepare_data(opt.dset_true)
    nodes, episode, t_obs = prepare_data(opt.dset)
    M = len(nodes)

    # -- model --
    mus, beta, S, U, V, A, scale, timescale = load_model(opt.checkpoint, M)
    base_date = pd.to_datetime(opt.basedate)
    print_model_stats(mus, beta, S, U, V, A)
    assert timescale == 1

    # -- goodness of fit --
    ks, pval, crit = evaluate_goodness_of_fit(episode, mus, beta, A, nodes)

    if opt.days is None or opt.days < 1:
        sys.exit(0)

    # compute predictions
    sim_d = lambda d: simulate_mhp(
        t_obs,
        d,
        episode,
        mus,
        beta,
        A,
        timescale,
        nodes,
        opt.step_size,
        opt.trials,
        episode_true,
    )

    if opt.tl_simulate:
        model, model_opt = SparseEmbeddingSoftplus.from_checkpoint(opt.checkpoint)
        simulator = model.mk_simulator()
        timescale = model_opt.timescale
        sim_d = lambda d: simulate_tl_mhp(
            t_obs,
            d,
            episode,
            timescale,
            simulator,
            nodes,
            opt.trials,
            stddev=opt.std_dev,
        )

    # collect simulation data and prepare output
    d_eval = None
    datestrs = [
        (base_date + timedelta(d)).strftime("%m/%d") for d in range(opt.days + 1)
    ]
    d_eval = sim_d(opt.days)[["county"] + list(range(opt.days + 1))]
    d_eval.columns = ["county"] + datestrs
    # compute sum without unknown
    d_eval = d_eval[d_eval["county"] != "Unknown"]
    vals = d_eval[d_eval.columns[1:]].to_numpy()
    d_eval = d_eval.append(
        pd.DataFrame(
            [["ALL REGIONS"] + vals.sum(axis=0).tolist()], columns=d_eval.columns
        ),
        ignore_index=True,
    )
    # add KS stats to output
    d_eval["KS"] = ks.tolist()[: len(d_eval) - 1] + [np.mean(ks)]
    d_eval["pval"] = pval.tolist()[: len(d_eval) - 1] + [np.mean(pval)]
    d_eval = d_eval.round(3)

    print("--- Predictions ---")
    print(d_eval)

    if opt.fout is not None:
        # convert to format we send out
        d_eval = d_eval.set_index("county").transpose()
        d_eval.columns = d_eval.columns.rename("date")
        d_eval.to_csv(opt.fout, index_label="date")


if __name__ == "__main__":
    main(sys.argv[1:])
