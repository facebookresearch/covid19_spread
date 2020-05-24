#!/usr/bin/env python3

import train
import numpy as np
import torch as th
import random
from load import load_data
from tlc import Episode
from evaluation import simulate_tl_mhp, goodness_of_fit
import pandas
import copy
import h5py
from train import mk_parser
import cv
from metrics import load_ground_truth
from typing import List, Tuple, Dict, Any
from scipy.stats import kstest
from glob import glob
import os
import json


class MHPCV(cv.CV):
    def run_train(self, dset, model_params, model_out):
        model_params.dset = dset

        model_params = copy.deepcopy(model_params)
        # Fill out any default parameters...
        parser = mk_parser()
        defaults = parser.parse_args([])
        for k, v in vars(defaults).items():
            if not hasattr(model_params, k):
                setattr(model_params, k, v)

        seed = getattr(model_params, "seed", 0)
        np.random.seed(seed)
        th.manual_seed(seed)
        random.seed(seed)
        model_params.checkpoint = model_out
        trainer = train.CovidTrainer(model_params)
        trainer.train()
        return model_out

    def run_simulate(self, dset, model_params, checkpoint, sim_params):
        mdl, mdl_opt = train.CovidModel.from_checkpoint(checkpoint, map_location="cpu")
        nodes, ns, ts, basedate = load_data(dset)
        episode = Episode(
            th.from_numpy(ts).double(), th.from_numpy(ns).long(), True, mdl.nnodes
        )
        basedate = pandas.Timestamp(basedate)
        simulator = mdl.mk_simulator()
        t_obs = ts[-1]

        sim = simulate_tl_mhp(
            t_obs,
            sim_params["days"],
            episode,
            mdl_opt.timescale,
            simulator,
            nodes,
            sim_params["trials"],
            max_events=sim_params["max_events"],
        )

        sim = sim.set_index("county").transpose().sort_index()
        deltas = sim.diff(axis=0).fillna(0)
        deltas.index = pandas.date_range(start=basedate, periods=len(deltas))
        return deltas[deltas.index > basedate]

    def model_selection(self, basedir: str) -> List[cv.BestRun]:
        results = []
        for metrics_pth in glob(os.path.join(basedir, "*/metrics.csv")):
            metrics = pandas.read_csv(metrics_pth, index_col="Measure")
            metrics_json = json.load(open(metrics_pth.replace(".csv", ".json")))
            results.append(
                {
                    "job_pth": os.path.dirname(metrics_pth),
                    "mae": metrics.loc["MAE"].values[-1],
                    "avg_mae": metrics.loc["MAE"].mean(),
                    "ks": metrics_json["avg_ks"],
                    "pval": metrics_json["avg_pval"],
                }
            )
        df = pandas.DataFrame(results)
        return [
            cv.BestRun(df.sort_values(by="mae").iloc[0].job_pth, "best_mae"),
            cv.BestRun(df.sort_values(by="avg_mae").iloc[0].job_pth, "best_avg_mae"),
            cv.BestRun(df.sort_values(by="ks").iloc[0].job_pth, "best_ks"),
            cv.BestRun(df.sort_values(by="pval").iloc[-1].job_pth, "best_pval"),
        ]

    def compute_metrics(
        self, gt: str, forecast: str, model: str, metric_args: Dict[str, Any]
    ) -> Tuple[pandas.DataFrame, Dict[str, Any]]:
        df_val, json_val = super().compute_metrics(gt, forecast, model, metric_args)

        # Run KS Test
        mdl, mdl_opt = train.CovidModel.from_checkpoint(model, map_location="cpu")
        nodes, ns, ts, basedate = load_data(gt)
        M = len(nodes)
        nts = (ts - ts.min()) / mdl_opt.timescale
        episode = Episode(
            th.from_numpy(nts).double(), th.from_numpy(ns).long(), True, M
        )
        beta, A, mus = map(lambda x: x.numpy(), mdl.get_params())
        residuals = goodness_of_fit(episode, 0.001, mus, beta.item(), A, nodes)
        ks, pval = zip(
            *[
                kstest(residuals[x], "expon")
                if len(residuals[x]) > 1 and nodes[x] != "Unknown"
                else (np.nan, np.nan)
                for x in range(M)
            ]
        )
        ks = [
            {"loc": n, "ks": k, "pval": p}
            for n, k, p in zip(nodes, ks, pval)
            if "Unknown" not in n
        ]
        ks_df = pandas.DataFrame(ks)
        avg_ks = ks_df.mean().ks
        avg_pval = ks_df.mean().pval
        return df_val, {**json_val, "ks": ks, "avg_ks": avg_ks, "avg_pval": avg_pval}


CV_CLS = MHPCV
