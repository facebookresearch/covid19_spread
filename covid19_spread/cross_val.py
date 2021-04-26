#!/usr/bin/env python3
# Copyright (c) 2021-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
from typing import Dict, Any, List, Tuple
import pandas as pd
from datetime import timedelta
import torch as th
from tqdm import tqdm
import numpy as np
from .common import mk_absolute_paths
import yaml
from tensorboardX import SummaryWriter
from collections import namedtuple
from . import common, metrics
import os
from glob import glob
import shutil


BestRun = namedtuple("BestRun", ("pth", "name"))


def load_config(cfg_pth: str) -> Dict[str, Any]:
    return mk_absolute_paths(yaml.load(open(cfg_pth), Loader=yaml.FullLoader))


class CV:
    def run_simulate(
        self, dset: str, args: Dict[str, Any], model: Any, sim_params: Dict[str, Any]
    ) -> pd.DataFrame:
        """
        Run a simulation given a trained model.  This should return a pandas DataFrame with each
        column corresponding to a location and each row corresponding to a date.  The value
        of each cell is the forecasted cases per day (*not* cumulative cases)
        """
        args.fdat = dset
        if model is None:
            raise NotImplementedError

        cases, regions, basedate, device = self.initialize(args)
        tmax = cases.size(1)

        test_preds = model.simulate(tmax, cases, args.test_on, **sim_params)
        test_preds = test_preds.cpu().numpy()

        df = pd.DataFrame(test_preds.T, columns=regions)
        if basedate is not None:
            base = pd.to_datetime(basedate)
            ds = [base + timedelta(i) for i in range(1, args.test_on + 1)]
            df["date"] = ds

            df.set_index("date", inplace=True)

        return df

    def run_standard_deviation(
        self,
        dset,
        args,
        nsamples,
        intervals,
        orig_cases,
        model=None,
        batch_size=1,
        closed_form=False,
    ):
        with th.no_grad():
            args.fdat = dset
            if model is None:
                raise NotImplementedError

            cases, regions, basedate, device = self.initialize(args)

            tmax = cases.size(1)

            base = pd.to_datetime(basedate)

            def mk_df(arr):
                df = pd.DataFrame(arr, columns=regions)
                df.index = pd.date_range(base + timedelta(days=1), periods=args.test_on)
                return df

            if closed_form:
                preds, stds = model.simulate(
                    tmax, cases, args.test_on, deterministic=True, return_stds=True
                )
                stds = th.cat([x.narrow(-1, -1, 1) for x in stds], dim=-1)
                return mk_df(stds.cpu().numpy().T), mk_df(preds.cpu().numpy().T)

            samples = []

            if batch_size > 1:
                cases = cases.repeat(batch_size, 1, 1)
                nsamples = nsamples // batch_size

            for i in tqdm(range(nsamples)):
                test_preds = model.simulate(tmax, cases, args.test_on, False)
                test_preds = test_preds.cpu().numpy()
                samples.append(test_preds)
            samples = (
                np.stack(samples, axis=0)
                if batch_size <= 1
                else np.concatenate(samples, axis=0)
            )

            return mk_df(np.std(samples, axis=0).T), mk_df(np.mean(samples, axis=0).T)

    def run_train(self, dset, model_params, model_out):
        """
        Train a model
        """
        ...

    def preprocess(self, dset: str, preprocessed: str, preprocess_args: Dict[str, Any]):
        """
        Perform any kind of model specific pre-processing.
        """
        if "smooth" in preprocess_args:
            common.smooth(dset, preprocessed, preprocess_args["smooth"])
        else:
            shutil.copy(dset, preprocessed)

    def metric_df(self, basedir):
        runs = []
        for metrics_pth in glob(os.path.join(basedir, "*/metrics.csv")):
            metrics = pd.read_csv(metrics_pth, index_col="Measure")
            runs.append(
                {
                    "pth": os.path.dirname(metrics_pth),
                    "mae": metrics.loc["MAE"][-1],
                    "rmse": metrics.loc["RMSE"][-1],
                    "mae_deltas": metrics.loc["MAE_DELTAS"].mean(),
                    "rmse_deltas": metrics.loc["RMSE_DELTAS"].mean(),
                    "state_mae": metrics.loc["STATE_MAE"][-1],
                }
            )
        return pd.DataFrame(runs)

    def model_selection(self, basedir: str, config, module) -> List[BestRun]:
        """
        Evaluate a sweep returning a list of models to retrain on the full dataset.
        """
        df = self.metric_df(basedir)
        if "ablation" in config["train"]:
            ablations = []
            for _, row in df.iterrows():
                job_cfg = load_config(os.path.join(row.pth, f"{module}.yml"))
                if job_cfg["train"]["ablation"] is not None:
                    ablations.append(
                        ",".join(
                            os.path.basename(x) for x in job_cfg["train"]["ablation"]
                        )
                    )
                else:
                    ablations.append("null")
            df["ablation"] = ablations
            best_runs = []
            for key in ["mae", "rmse", "mae_deltas", "rmse_deltas"]:
                best = df.loc[df.groupby("ablation")[key].idxmin()]
                best_runs.extend(
                    [
                        BestRun(x.pth, f"best_{key}_{x.ablation}")
                        for _, x in best.iterrows()
                    ]
                )
            return best_runs

        return [
            BestRun(df.sort_values(by="mae").iloc[0].pth, "best_mae"),
            BestRun(df.sort_values(by="rmse").iloc[0].pth, "best_rmse"),
            BestRun(df.sort_values(by="mae_deltas").iloc[0].pth, "best_mae_deltas"),
            BestRun(df.sort_values(by="rmse_deltas").iloc[0].pth, "best_rmse_deltas"),
            BestRun(df.sort_values(by="state_mae").iloc[0].pth, "best_state_mae"),
        ]

    def compute_metrics(
        self, gt: str, forecast: str, model: Any, metric_args: Dict[str, Any]
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        return metrics.compute_metrics(gt, forecast).round(2), {}

    def setup_tensorboard(self, basedir):
        """
        Setup dir and writer for tensorboard logging
        """
        self.tb_writer = SummaryWriter(logdir=basedir)

    def run_prediction_interval(
        self, means_pth: str, stds_pth: str, intervals: List[float]
    ):
        ...
