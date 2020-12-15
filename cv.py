#!/usr/bin/env python3

import copy
import click
import importlib
import itertools
import json
import numpy as np
import pandas as pd
import os
import random
import shutil
import sys
import submitit
import tempfile
import torch as th
from tqdm import tqdm
import torch.nn.functional as F
import traceback
import re
import yaml
from argparse import Namespace
from collections import namedtuple
from datetime import datetime, timedelta
from functools import partial
from glob import glob, iglob
from typing import Dict, Any, List, Optional, Tuple
from tensorboardX import SummaryWriter
from contextlib import nullcontext, ExitStack
import common
import metrics
from lib import cluster
from lib.click_lib import DefaultGroup, OptionNArgs
from lib.slurm_pool_executor import SlurmPoolExecutor, JobStatus, get_db_client
from lib.mail import email_notebook
import sqlite3
from lib.context_managers import env_var
from lib.slack import get_client as get_slack_client


# FIXME: move snapshot to lib
from timelord import snapshot

BestRun = namedtuple("BestRun", ("pth", "name"))


def set_dict(d: Dict[str, Any], keys: List[str], v: Any):
    """
    update a dict using a nested list of keys.
    Ex:
        x = {'a': {'b': {'c': 2}}}
        set_dict(x, ['a', 'b'], 4) == {'a': {'b': 4}}
    """
    if len(keys) > 0:
        d[keys[0]] = set_dict(d[keys[0]], keys[1:], v)
        return d
    else:
        return v


def mk_executor(
    name: str, folder: str, extra_params: Dict[str, Any], ex=SlurmPoolExecutor, **kwargs
):
    executor = (ex or submitit.AutoExecutor)(folder=folder, **kwargs)
    executor.update_parameters(
        job_name=name,
        partition=cluster.PARTITION,
        gpus_per_node=extra_params.get("gpus", 0),
        cpus_per_task=extra_params.get("cpus", 3),
        mem=f'{cluster.MEM_GB(extra_params.get("memgb", 20))}GB',
        array_parallelism=extra_params.get("array_parallelism", 100),
        time=extra_params.get("timeout", 12 * 60),
    )
    return executor


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
        self, dset, args, nsamples, intervals, orig_cases, model=None, batch_size=1
    ):
        with th.no_grad():
            args.fdat = dset
            if model is None:
                raise NotImplementedError

            cases, regions, basedate, device = self.initialize(args)
            tmax = cases.size(1)
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

            base = pd.to_datetime(basedate)

            def mk_df(arr):
                df = pd.DataFrame(arr, columns=regions)
                df.index = pd.date_range(base + timedelta(days=1), periods=args.test_on)
                return df

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
                ablations.append(
                    ",".join(os.path.basename(x) for x in job_cfg["train"]["ablation"])
                )
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


def ensemble(basedirs, cfg, module, prefix, outdir):
    def _path(x):
        return os.path.join(basedir, prefix + x)

    means = []
    stds = []
    mean_deltas = []
    kwargs = {"index_col": "date", "parse_dates": ["date"]}
    for basedir in basedirs:
        if os.path.exists(_path(cfg["validation"]["output"])):
            means.append(pd.read_csv(_path(cfg["validation"]["output"]), **kwargs))
        if os.path.exists(_path("std.csv")):
            stds.append(pd.read_csv(_path("std.csv"), **kwargs))
            mean_deltas.append(pd.read_csv(_path("mean.csv"), **kwargs))
    if len(stds) > 0:
        # Average the variance, and take square root
        std = pd.concat(stds).pow(2).groupby(level=0).mean().pow(0.5)
        std.to_csv(os.path.join(outdir, prefix + "std.csv"))
        mean_deltas = pd.concat(mean_deltas).groupby(level=0).mean()
        mean_deltas.to_csv(os.path.join(outdir, prefix + "mean.csv"))

    assert len(means) > 0, "All ensemble jobs failed!!!!"
    mean = pd.concat(means).groupby(level=0).median()
    outfile = os.path.join(outdir, prefix + cfg["validation"]["output"])
    mean.to_csv(outfile, index_label="date")

    mod = importlib.import_module(module).CV_CLS()

    # -- metrics --
    metric_args = cfg[module].get("metrics", {})
    df_val, json_val = mod.compute_metrics(
        cfg[module]["data"], outfile, None, metric_args
    )
    df_val.to_csv(os.path.join(outdir, prefix + "metrics.csv"))
    with open(os.path.join(outdir, prefix + "metrics.json"), "w") as fout:
        json.dump(json_val, fout)
    print(df_val)

    if len(stds) > 0:
        pred_interval = cfg.get("prediction_interval", {})
        piv = mod.run_prediction_interval(
            os.path.join(outdir, prefix + "mean.csv"),
            os.path.join(outdir, prefix + "std.csv"),
            pred_interval.get("intervals", [0.99, 0.95, 0.8]),
        )
        piv.to_csv(os.path.join(outdir, prefix + "piv.csv"), index=False)


def run_cv(
    module: str,
    basedir: str,
    cfg: Dict[str, Any],
    prefix="",
    basedate=None,
    executor=None,
):
    """Runs cross validaiton for one set of hyperaparmeters"""
    try:
        basedir = basedir.replace("%j", submitit.JobEnvironment().job_id)
    except Exception:
        pass  # running locally, basedir is fine...
    os.makedirs(basedir, exist_ok=True)
    print(f"CWD = {os.getcwd()}")

    def _path(path):
        return os.path.join(basedir, path)

    log_configs(cfg, module, _path(prefix + f"{module}.yml"))

    n_models = cfg[module]["train"].get("n_models", 1)
    if n_models > 1:
        launcher = map if executor is None else executor.map_array
        fn = partial(
            run_cv, module, prefix=prefix, basedate=basedate, executor=executor
        )
        configs = [
            set_dict(copy.deepcopy(cfg), [module, "train", "n_models"], 1)
            for _ in range(n_models)
        ]
        basedirs = [os.path.join(basedir, f"job_{i}") for i in range(n_models)]
        with ExitStack() as stack:
            if executor is not None:
                stack.enter_context(executor.set_folder(os.path.join(basedir, "%j")))

            jobs = list(launcher(fn, basedirs, configs))
            launcher = (
                ensemble
                if executor is None
                else partial(executor.submit_dependent, jobs, ensemble)
            )
            ensemble_job = launcher(basedirs, cfg, module, prefix, basedir)
            if executor is not None:
                # Whatever jobs depend on "this" job, should be extended to the newly created jobs
                executor.extend_dependencies(jobs + [ensemble_job])
            return jobs + [ensemble_job]

    # setup input/output paths
    dset = cfg[module]["data"]
    val_in = _path(prefix + "filtered_" + os.path.basename(dset))
    val_out = _path(prefix + cfg["validation"]["output"])
    cfg[module]["train"]["fdat"] = val_in

    mod = importlib.import_module(module).CV_CLS()

    # -- store configs to reproduce results --
    log_configs(cfg, module, _path(prefix + f"{module}.yml"))

    ndays = cfg["validation"]["days"]
    if basedate is not None:
        # If we want to train from a particular basedate, then also subtract
        # out the different in days.  Ex: if ground truth contains data up to 5/20/2020
        # but the basedate is 5/10/2020, then drop an extra 10 days in addition to validation.days
        gt = metrics.load_ground_truth(dset)
        assert gt.index.max() >= basedate
        ndays += (gt.index.max() - basedate).days
    filter_validation_days(dset, val_in, ndays)
    # apply data pre-processing
    preprocessed = _path(prefix + "preprocessed_" + os.path.basename(dset))
    mod.preprocess(val_in, preprocessed, cfg[module].get("preprocess", {}))

    forecasts = []
    # weights = []
    mod.setup_tensorboard(basedir)
    # setup logging
    train_params = Namespace(**cfg[module]["train"])
    n_models = getattr(train_params, "n_models", 1)
    print(f"Training {n_models} models")
    # -- train --
    model = mod.run_train(
        preprocessed, train_params, _path(prefix + cfg[module]["output"])
    )

    # -- simulate --
    with th.no_grad():
        sim_params = cfg[module].get("simulate", {})
        # Returns the number of new cases for each day
        df_forecast_deltas = mod.run_simulate(
            preprocessed, train_params, model, sim_params=sim_params
        )
        df_forecast = rebase_forecast_deltas(val_in, df_forecast_deltas)

    mod.tb_writer.close()

    print(f"Storing validation in {val_out}")
    df_forecast.to_csv(val_out, index_label="date")

    # -- metrics --
    metric_args = cfg[module].get("metrics", {})
    df_val, json_val = mod.compute_metrics(
        cfg[module]["data"], val_out, model, metric_args
    )
    df_val.to_csv(_path(prefix + "metrics.csv"))
    with open(_path(prefix + "metrics.json"), "w") as fout:
        json.dump(json_val, fout)
    print(df_val)

    # -- prediction interval --
    if "prediction_interval" in cfg and prefix == "final_model_":
        try:
            with th.no_grad():
                # FIXME: refactor to use rebase_forecast_deltas
                gt = metrics.load_ground_truth(val_in)
                basedate = gt.index.max()
                prev_day = gt.loc[[basedate]]
                pred_interval = cfg.get("prediction_interval", {})
                df_std, df_mean = mod.run_standard_deviation(
                    preprocessed,
                    train_params,
                    pred_interval.get("nsamples", 100),
                    pred_interval.get("intervals", [0.99, 0.95, 0.8]),
                    prev_day.values.T,
                    model,
                    pred_interval.get("batch_size", 8),
                )
                df_std.to_csv(_path(f"{prefix}std.csv"), index_label="date")
                df_mean.to_csv(_path(f"{prefix}mean.csv"), index_label="date")
                piv = mod.run_prediction_interval(
                    _path(f"{prefix}mean.csv"),
                    _path(f"{prefix}std.csv"),
                    pred_interval.get("intervals", [0.99, 0.95, 0.8]),
                )
                piv.to_csv(_path(f"{prefix}piv.csv"), index=False)
        except NotImplementedError:
            pass  # naive...


def filter_validation_days(dset: str, val_in: str, validation_days: int):
    """Filters validation days and writes output to val_in path"""
    if dset.endswith(".csv"):
        common.drop_k_days_csv(dset, val_in, validation_days)
    elif dset.endswith(".h5"):
        common.drop_k_days(dset, val_in, validation_days)
    else:
        raise RuntimeError(f"Unrecognized dataset extension: {dset}")


def mk_absolute_paths(cfg):
    if isinstance(cfg, dict):
        return {k: mk_absolute_paths(v) for k, v in cfg.items()}
    elif isinstance(cfg, list):
        return list(map(mk_absolute_paths, cfg))
    else:
        return (
            os.path.realpath(cfg)
            if isinstance(cfg, str) and os.path.exists(cfg)
            else cfg
        )


def load_model(model_pth, cv, args):
    chkpnt = th.load(model_pth)
    cv.initialize(args)
    cv.func.load_state_dict(chkpnt)
    return cv.func


def copy_assets(cfg, dir):
    if isinstance(cfg, dict):
        return {k: copy_assets(v, dir) for k, v in cfg.items()}
    elif isinstance(cfg, list):
        return [copy_assets(x, dir) for x in cfg]
    elif isinstance(cfg, str) and os.path.exists(cfg):
        new_pth = os.path.join(dir, "assets", os.path.basename(cfg))
        shutil.copy(cfg, new_pth)
        return new_pth
    else:
        return cfg


def load_config(cfg_pth: str) -> Dict[str, Any]:
    return mk_absolute_paths(yaml.load(open(cfg_pth), Loader=yaml.FullLoader))


def log_configs(cfg: Dict[str, Any], module: str, path: str):
    """Logs configs for job for reproducibility"""
    with open(path, "w") as f:
        yaml.dump(cfg[module], f)


def attach_notebook(config, mode, module, basedir):
    """Run notebook for execution mode"""
    if "notebooks" in config and mode in config["notebooks"]:
        notebook_pth = config["notebooks"][mode]
        subject = f"[CV Notebook]: {config['region']}"
        with env_var({"CV_BASE_DIR": basedir, "CV_MODULE": module}):
            user = os.environ["USER"]
            email_notebook(notebook_pth, [f"{user}@fb.com"], subject)


def rebase_forecast_deltas(val_in, df_forecast_deltas):
    gt = metrics.load_ground_truth(val_in)
    # Ground truth for the day before our first forecast
    prev_day = gt.loc[[df_forecast_deltas.index.min() - timedelta(days=1)]]
    # Stack the first day ground truth on top of the forecasts
    common_cols = set(df_forecast_deltas.columns).intersection(set(gt.columns))
    stacked = pd.concat([prev_day[common_cols], df_forecast_deltas[common_cols]])
    # Cumulative sum to compute total cases for the forecasts
    df_forecast = stacked.sort_index().cumsum().iloc[1:]
    return df_forecast


def run_best(config, module, remote, basedir, basedate=None, executor=None):
    mod = importlib.import_module(module).CV_CLS()
    sweep_config = load_config(os.path.join(basedir, "cfg.yml"))
    best_runs = mod.model_selection(basedir, config=sweep_config[module], module=module)

    if remote and executor is None:
        executor = mk_executor(
            "model_selection", basedir, config[module].get("resources", {})
        )

    with open(os.path.join(basedir, "model_selection.json"), "w") as fout:
        json.dump([x._asdict() for x in best_runs], fout)

    cfg = copy.deepcopy(config)
    cfg["validation"]["days"] = 0
    best_runs_df = pd.DataFrame(best_runs)

    def run_cv_and_copy_results(tags, module, pth, cfg, prefix):
        try:
            jobs = run_cv(
                module, pth, cfg, prefix=prefix, basedate=basedate, executor=executor
            )

            def rest():
                for tag in tags:
                    shutil.copy(
                        os.path.join(pth, f'final_model_{cfg["validation"]["output"]}'),
                        os.path.join(
                            os.path.dirname(pth), f"forecasts/forecast_{tag}.csv"
                        ),
                    )

                    if "prediction_interval" in cfg:
                        piv_pth = os.path.join(
                            pth,
                            f'final_model_{cfg["prediction_interval"]["output_std"]}',
                        )
                        if os.path.exists(piv_pth):
                            shutil.copy(
                                piv_pth,
                                os.path.join(
                                    os.path.dirname(pth), f"forecasts/std_{tag}.csv"
                                ),
                            )

            if cfg[module]["train"].get("n_models", 1) > 1:
                executor.submit_dependent(jobs, rest)
            else:
                rest()
        except Exception as e:
            msg = f"*Final run failed for {tags}*\nbasedir = {basedir}\nException was: {e}"
            client = get_slack_client()
            client.chat_postMessage(channel="#cron_errors", text=msg)
            raise e

    for pth, tags in best_runs_df.groupby("pth")["name"].agg(list).items():
        os.makedirs(os.path.join(os.path.dirname(pth), f"forecasts"), exist_ok=True)
        name = ",".join(tags)
        print(f"Starting {name}: {pth}")
        job_config = load_config(os.path.join(pth, module + ".yml"))
        if "test" in cfg:
            job_config["train"]["test_on"] = cfg["test"]["days"]
        cfg[module] = job_config
        launcher = run_cv_and_copy_results
        if remote:
            launcher = partial(executor.submit, run_cv_and_copy_results)

        with executor.set_folder(pth) if remote else nullcontext():
            launcher(tags, module, pth, cfg, "final_model_")


@click.group(cls=DefaultGroup, default_command="cv")
def cli():
    pass


@cli.command()
@click.argument("chkpnts", nargs=-1)
@click.option("-remote", is_flag=True)
@click.option("-nsamples", type=click.INT)
@click.option("-batchsize", type=int)
def prediction_interval(chkpnts, remote, nsamples, batchsize):
    def f(chkpnt_pth):
        prefix = "final_model_" if "final_model_" in chkpnt_pth else ""
        chkpnt = th.load(chkpnt_pth)
        job_pth = os.path.dirname(chkpnt_pth)

        cfg_pth = os.path.join(job_pth, "../cfg.yml")
        if not os.path.exists(cfg_pth):
            cfg_pth = os.path.join(job_pth, "../../cfg.yml")
        cfg = load_config(cfg_pth)
        module = cfg["this_module"]
        job_config = load_config(os.path.join(job_pth, f"{prefix}{module}.yml"))
        opt = Namespace(**job_config["train"])
        mod = importlib.import_module(module).CV_CLS()
        new_cases, regions, basedate, device = mod.initialize(opt)
        model = mod.func
        model.load_state_dict(chkpnt)

        dset = os.path.join(
            job_pth, prefix + "preprocessed_" + os.path.basename(job_config["data"])
        )
        val_in = os.path.join(
            job_pth, prefix + "filtered_" + os.path.basename(job_config["data"])
        )

        gt = metrics.load_ground_truth(val_in)
        prev_day = gt.loc[[pd.to_datetime(basedate)]]
        pred_interval = cfg.get("prediction_interval", {})
        df_std, df_mean = mod.run_standard_deviation(
            dset,
            opt,
            nsamples or pred_interval.get("nsamples", 100),
            pred_interval.get("intervals", [0.99, 0.95, 0.8]),
            prev_day.values.T,
            model,
            batchsize or pred_interval.get("batch_size", 8),
        )
        df_std.to_csv(os.path.join(job_pth, f"{prefix}std.csv"), index_label="date")
        df_mean.to_csv(os.path.join(job_pth, f"{prefix}mean.csv"), index_label="date")
        pred_intervals = mod.run_prediction_interval(
            os.path.join(job_pth, f"{prefix}mean.csv"),
            os.path.join(job_pth, f"{prefix}std.csv"),
            pred_interval.get("intervals", [0.99, 0.95, 0.8]),
        )
        pred_intervals.to_csv(os.path.join(job_pth, f"{prefix}piv.csv"), index=False)

    if remote:
        now = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
        folder = os.path.expanduser(f"~/.covid19/logs/{now}")
        extra_params = {"gpus": 1, "cpus": 2, "memgb": 20, "timeout": 3600}
        ex = mk_executor(
            "prediction_interval", folder, extra_params, ex=submitit.AutoExecutor
        )
        ex.map_array(f, chkpnts)
        print(folder)
    else:
        list(map(f, chkpnts))


@cli.command()
@click.argument("config_pth")
@click.argument("sweep_dir")
@click.argument("module")
def notebook(config_pth, sweep_dir, module):
    config = load_config(config_pth)
    attach_notebook(config, "cv", module, sweep_dir)


@cli.command()
@click.argument("sweep_dirs", nargs=-1)
@click.argument("module")
@click.option("-remote", is_flag=True)
@click.option("-basedate", type=click.DateTime(), default=None)
def model_selection(sweep_dirs, module, remote, basedate):
    executor = None
    for sweep_dir in sweep_dirs:
        cfg = load_config(os.path.join(sweep_dir, "cfg.yml"))
        if executor is None:
            executor = mk_executor(
                "model_selection", sweep_dir, cfg[module].get("resources", {})
            )
        match = re.search(r"\d{4}-\d{2}-\d{2}", os.path.basename(sweep_dir))
        if basedate is None and match:
            basedate = pd.to_datetime(match.group(0))

        run_best(cfg, module, remote, sweep_dir, basedate, executor=executor)
    executor.launch(sweep_dir + "/workers", workers=4)


@cli.command()
@click.argument("config_pth")
@click.argument("module")
@click.option("-validate-only", type=click.BOOL, default=False)
@click.option("-remote", is_flag=True)
@click.option("-array-parallelism", type=click.INT, default=20)
@click.option("-max-jobs", type=click.INT, default=200)
@click.option("-basedir", default=None, help="Path to sweep base directory")
@click.option("-basedate", type=click.DateTime(), help="Date to treat as last date")
@click.option("-ablation", is_flag=True)
def cv(
    config_pth: str,
    module: str,
    validate_only: bool,
    remote: bool,
    array_parallelism: int,
    max_jobs: int,
    basedir: str,
    basedate: Optional[datetime] = None,
    executor=None,
    ablation=False,
):
    """
    Run cross validation pipeline for a given module.
    """
    # FIXME: This is a hack...
    in_backfill = executor is not None
    now = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    user = os.environ["USER"]

    cfg = load_config(config_pth)
    region = cfg["region"]
    cfg["this_module"] = module

    if basedir is None:
        if remote:
            basedir = f"{cluster.FS}/{user}/covid19/forecasts/{region}/{now}"
        else:
            basedir = f"/tmp/{user}/covid19/forecasts/{region}/{now}"

    os.makedirs(basedir, exist_ok=True)

    if not in_backfill:
        # Copy any asset files into `basedir/assets`
        os.makedirs(os.path.join(basedir, "assets"))
        cfg[module] = copy_assets(cfg[module], basedir)

    # Copy the dataset into the basedir
    shutil.copy(cfg[module]["data"], basedir)
    cfg[module]["data"] = os.path.join(basedir, os.path.basename(cfg[module]["data"]))

    with open(os.path.join(basedir, "cfg.yml"), "w") as fout:
        yaml.dump(cfg, fout)

    # if we are running an ablation, create new time features from ablation field
    # all list entries in are assumed to be a single ablation
    # all features in one list entry will be dropped from the full features to
    #   perform the ablation
    if ablation:
        feats = []
        all_feats = set(cfg[module]["train"]["time_features"][0])
        for x in cfg[module]["train"]["ablation"]:
            feats.append(list(all_feats - set(x)))
        cfg[module]["train"]["time_features"] = feats

    cfgs = []
    sweep_params = [
        ([module, "train", k], v)
        for k, v in cfg[module]["train"].items()
        if isinstance(v, list)
    ]
    sweep_params.extend(
        [
            ([module, "preprocess", k], v)
            for k, v in cfg[module].get("preprocess", {}).items()
            if isinstance(v, list)
        ]
    )
    if len(sweep_params) == 0:
        cfgs.append(cfg)
    else:
        random.seed(0)
        keys, values = zip(*sweep_params)
        for vals in itertools.product(*values):
            clone = copy.deepcopy(cfg)
            [set_dict(clone, ks, vs) for ks, vs in zip(keys, vals)]
            cfgs.append(clone)
        random.shuffle(cfgs)
        cfgs = cfgs[:max_jobs]

    print(f"Launching {len(cfgs)} jobs")
    if remote:
        extra = cfg[module].get("resources", {})
        if executor is None:
            executor = mk_executor(
                f"cv_{region}",
                basedir + "/%j",
                {**extra, "array_parallelism": array_parallelism},
            )
        launcher = executor.map_array
        basedirs = [f"{basedir}/%j" for _ in cfgs]
    else:
        launcher = map
        basedirs = [os.path.join(basedir, f"job_{i}") for i in range(len(cfgs))]

    with ExitStack() as stack:
        if not in_backfill:
            stack.enter_context(
                snapshot.SnapshotManager(
                    snapshot_dir=basedir + "/snapshot",
                    with_submodules=True,
                    exclude=["notebooks/*", "tests/*"],
                )
            )
        jobs = list(
            launcher(
                partial(run_cv, module, basedate=basedate, executor=executor),
                basedirs,
                cfgs,
            )
        )

        # Find the best model and retrain on the full dataset
        launcher = (
            partial(
                executor.submit_dependent,
                jobs,
                run_best,
                executor=copy.deepcopy(executor),
            )
            if remote
            else run_best
        )

        if not validate_only:
            job = launcher(cfg, module, remote, basedir, basedate=basedate)
            jobs.append(job)

        if remote:
            if not executor.nested:
                executor.submit_final_job(attach_notebook, cfg, "cv", module, basedir)
            executor.launch(basedir + "/workers", array_parallelism)

    print(basedir)
    return basedir, jobs


@cli.command()
@click.argument("config_pth")
@click.argument("module")
@click.option("-period", type=int, help="Number of days for sliding window")
@click.option(
    "-start-date", type=click.DateTime(), default="2020-04-01", help="Start date"
)
@click.option("-dates", default=None, multiple=True, type=click.DateTime())
@click.option("-validate-only", type=click.BOOL, default=False, is_flag=True)
@click.option("-remote", is_flag=True)
@click.option("-array-parallelism", type=click.INT, default=20)
@click.option("-max-jobs", type=click.INT, default=200)
@click.option("-ablation", is_flag=True)
@click.pass_context
def backfill(
    ctx: click.Context,
    config_pth: str,
    module: str,
    period: Optional[int] = None,
    start_date: Optional[datetime.date] = None,
    dates: Optional[List[datetime.date]] = None,
    validate_only: bool = False,
    remote: bool = False,
    array_parallelism: int = 20,
    max_jobs: int = 200,
    ablation: bool = False,
):
    """
    Run the cross validation pipeline over multiple time points.
    """
    config = mk_absolute_paths(load_config(config_pth))
    # allow to set backfill dates in config (function argument overrides)
    if not dates and "backfill" in config:
        dates = list(pd.to_datetime(config["backfill"]))
    assert (
        dates is not None or period is not None
    ), "Must specify either dates or period"
    gt = metrics.load_ground_truth(config[module]["data"])
    if not dates:
        assert period is not None
        dates = pd.date_range(
            start=start_date, end=gt.index.max(), freq=f"{period}D", closed="left"
        )
    print(
        "Running backfill for "
        + ", ".join(map(lambda x: x.strftime("%Y-%m-%d"), dates))
    )

    # setup experiment environment
    now = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    experiment_id = f'{config["region"]}/{now}'
    basedir = f'{cluster.FS}/{os.environ["USER"]}/covid19/forecasts/{experiment_id}'

    # setup executor
    extra_params = config[module].get("resources", {})
    executor = mk_executor(
        f'backfill_{config["region"]}',
        basedir,
        {**extra_params, "array_parallelism": array_parallelism},
    )
    print(f"Backfilling in {basedir}")
    # Copy any asset files into `basedir/assets`
    os.makedirs(os.path.join(basedir, "assets"))
    config[module] = copy_assets(config[module], basedir)
    with snapshot.SnapshotManager(
        snapshot_dir=basedir + "/snapshot",
        with_submodules=True,
        exclude=["notebooks/*", "tests/*"],
    ), tempfile.NamedTemporaryFile() as tfile:
        # Make sure that we use the CFG with absolute paths since we are now inside the snapshot directory
        with open(tfile.name, "w") as fout:
            yaml.dump(config, fout)
        for date in dates:
            print(f"Running CV for {date.date()}")
            cv_params = {
                k: v for k, v in ctx.params.items() if k in {p.name for p in cv.params}
            }
            cv_params["config_pth"] = tfile.name

            with executor.nest(), executor.set_folder(
                os.path.join(basedir, f"sweep_{date.date()}/%j")
            ):
                _, jobs = ctx.invoke(
                    cv,
                    basedir=os.path.join(basedir, f"sweep_{date.date()}"),
                    basedate=date,
                    executor=executor,
                    **cv_params,
                )

        if remote:
            executor.submit_final_job(
                attach_notebook, config, "backfill", module, basedir
            )
            executor.launch(basedir + "/workers", array_parallelism)


@cli.command()
@click.argument("sweep_dir")
def progress(sweep_dir):
    sweep_dir = os.path.realpath(sweep_dir)
    db_file = next(iglob(os.path.join(sweep_dir, "**/.job.db"), recursive=True))
    db_file = os.path.realpath(db_file)
    conn = get_db_client()
    with conn.cursor() as cur:
        df = pd.read_sql(
            f"SELECT status, worker_id FROM jobs WHERE id='{db_file}'", conn
        )
    msg = {
        "success": int((df["status"] == JobStatus.success.value).sum()),
        "failed": int((df["status"] == JobStatus.failure.value).sum()),
        "pending": int((df["status"] == JobStatus.pending.value).sum()),
        "running": int((df["status"] > len(JobStatus)).sum()),
    }
    print(json.dumps(msg, indent=4))


@cli.command()
@click.argument("sweep_dir")
@click.argument("workers", type=click.INT)
def add_workers(sweep_dir, workers):
    DB = os.path.abspath(glob(f"{sweep_dir}/**/.job.db", recursive=True)[0])
    cfg = load_config(glob(f"{sweep_dir}/**/cfg.yml", recursive=True)[0])
    extra_params = cfg[cfg["this_module"]].get("resources", {})
    executor = mk_executor(
        "add_workers", os.path.dirname(DB), extra_params, db_pth=os.path.realpath(DB)
    )
    executor.launch(f"{sweep_dir}/workers", workers)


@cli.command()
@click.argument("sweep_dir")
@click.option("-workers", type=click.INT)
def repair(sweep_dir, workers=None):
    db_file = next(iglob(os.path.join(sweep_dir, "**/.job.db"), recursive=True))
    conn = get_db_client()
    with conn.cursor() as cur:
        df = pd.read_sql(
            f"SELECT * FROM jobs WHERE id='{os.path.realpath(db_file)}'", conn
        )
        df = df.drop_duplicates(["pickle"]).copy()
        df.loc[
            (df["status"] == JobStatus.failure) | (df["status"] >= len(JobStatus)),
            "status",
        ] = JobStatus.pending
        cur.execute(f"DELETE FROM jobs WHERE id='{os.path.realpath(db_file)}'")
        cols = df.columns
        for _, row in df.iterrows():
            vals = tuple(row[c] for c in cols)
            cur.execute(
                f"INSERT INTO jobs ({', '.join(cols)}) VALUES({','.join(['%s' for _ in cols])})",
                vals,
            )
    conn.commit()
    cfg = load_config(next(iglob(f"{sweep_dir}/**/cfg.yml", recursive=True)))
    extra_params = cfg[cfg["this_module"]].get("resources", {})
    executor = mk_executor(
        "repair", sweep_dir, extra_params, db_pth=os.path.realpath(db_file)
    )
    executor.launch(os.path.join(sweep_dir, "workers"), workers or -1)


if __name__ == "__main__":
    cli()
