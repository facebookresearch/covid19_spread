#!/usr/bin/env python3

import argparse
import importlib
import pandas as pd
import os
import torch as th
import yaml
from argparse import Namespace
from datetime import datetime
from typing import Dict, Any, List, Optional
import common
import metrics
import itertools
import copy
import random
import submitit
from functools import partial
from glob import glob
from timelord import snapshot
from collections import namedtuple
import click
from lib.click_lib import DefaultGroup
import tempfile
import shutil


BestRun = namedtuple("BestRun", ("pth", "name"))


class CV:
    def run_simulate(self, dset, train_params, model, sim_params):
        ...

    def run_prediction_interval(self, args, nsamples, model=None):
        ...

    def run_train(self, dset, model_params, model_out):
        ...

    def model_selection(self, basedir: str) -> List[BestRun]:
        best_run, best_MAE = None, float("inf")
        for metrics_pth in glob(os.path.join(basedir, "*/metrics.csv")):
            metrics = pd.read_csv(metrics_pth, index_col="Measure")
            if metrics.loc["MAE"].values[-1] < best_MAE:
                best_MAE = metrics.loc["MAE"].values[-1]
                best_run = os.path.dirname(metrics_pth)
        return [BestRun(best_run, "best_mae")]


def run_cv(module: str, basedir: str, cfg: Dict[str, Any], prefix=""):
    """Runs cross validaiton for one set of hyperaparmeters"""
    try:
        basedir = basedir.replace("%j", submitit.JobEnvironment().job_id)
    except Exception:
        pass  # running locally, basedir is fine...

    os.makedirs(basedir, exist_ok=True)

    def _path(path):
        return os.path.join(basedir, path)

    # setup input/output paths
    dset = cfg[module]["data"]
    val_in = _path("filtered_" + os.path.basename(dset))
    val_out = _path(prefix + cfg["validation"]["output"])
    cfg[module]["train"]["fdat"] = val_in

    mod = importlib.import_module(module).CV_CLS()

    # -- store configs to reproduce results --
    log_configs(cfg, module, _path(f"{module}.yml"))

    filter_validation_days(dset, val_in, cfg["validation"]["days"])

    # -- train --
    train_params = Namespace(**cfg[module]["train"])
    model = mod.run_train(val_in, train_params, _path(prefix + cfg[module]["output"]))

    # -- simulate --
    with th.no_grad():
        sim_params = cfg[module].get("simulate", {})
        df_forecast = mod.run_simulate(
            val_in, train_params, model, sim_params=sim_params
        )
    print(f"Storing validation in {val_out}")
    df_forecast.to_csv(val_out)

    # -- metrics --
    if cfg["validation"]["days"] > 0:
        # Only compute metrics if this is the validation run.
        df_val = metrics.compute_metrics(cfg[module]["data"], val_out).round(2)
        df_val.to_csv(_path(prefix + "metrics.csv"))
        print(df_val)

    # -- prediction interval --
    if "prediction_interval" in cfg:
        with th.no_grad():
            df_mean, df_std = mod.run_prediction_interval(
                train_params, cfg["prediction_interval"]["nsamples"], model
            )
            df_mean.to_csv(_path(prefix + cfg["prediction_interval"]["output_mean"]))
            df_std.to_csv(_path(prefix + cfg["prediction_interval"]["output_std"]))


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


def load_config(cfg_pth: str) -> Dict[str, Any]:
    return mk_absolute_paths(yaml.load(open(cfg_pth), Loader=yaml.FullLoader))


def log_configs(cfg: Dict[str, Any], module: str, path: str):
    """Logs configs for job for reproducibility"""
    with open(path, "w") as f:
        yaml.dump(cfg[module], f)


def run_best(config, module, remote, basedir):
    mod = importlib.import_module(module).CV_CLS()
    best_runs = mod.model_selection(basedir)

    cfg = copy.deepcopy(config)
    cfg["validation"]["days"] = 0

    ngpus = cfg[module].get("resources", {}).get("gpus", 0)
    ncpus = cfg[module].get("resources", {}).get("cpus", 3)
    memgb = cfg[module].get("resources", {}).get("memgb", 20)
    timeout = cfg[module].get("resources", {}).get("timeout", 12 * 60)

    for run in best_runs:
        job_config = load_config(os.path.join(run.pth, module + ".yml"))
        cfg[module] = job_config
        cfg["validation"]["output"] = run.name + "_forecast.csv"
        launcher = run_cv
        if remote:
            executor = submitit.AutoExecutor(folder=run.pth)
            executor.update_parameters(
                name=run.name,
                gpus_per_node=ngpus,
                cpus_per_task=ncpus,
                mem_gb=memgb,
                timeout_min=timeout,
            )
            launcher = partial(executor.submit, run_cv)
        launcher(module, run.pth, cfg, prefix="final_model_")


@click.group(cls=DefaultGroup, default_command="cv")
def cli():
    pass


@cli.command()
@click.argument("config_pth")
@click.argument("module")
@click.option("-validate-only", type=click.BOOL, default=False)
@click.option("-remote", type=click.BOOL, default=False)
@click.option("-array-parallelism", type=click.INT, default=50)
@click.option("-max-jobs", type=click.INT, default=200)
@click.option("-basedir", default=None, help="Path to sweep base directory")
def cv(config_pth, module, validate_only, remote, array_parallelism, max_jobs, basedir):
    """
    Run cross validation pipeline for a given module.
    """
    now = datetime.now().strftime("%Y_%m_%d_%H_%M")
    user = os.environ["USER"]

    cfg = load_config(config_pth)
    region = cfg["region"]

    if basedir is None:
        if remote:
            basedir = f"/checkpoint/{user}/covid19/forecasts/{region}/{now}"
        else:
            basedir = f"/tmp/covid19/forecasts/{region}/{now}"

    os.makedirs(basedir, exist_ok=True)

    # Copy the dataset into the basedir
    shutil.copy(cfg[module]["data"], basedir)
    cfg[module]["data"] = os.path.join(basedir, os.path.basename(cfg[module]["data"]))

    with open(os.path.join(basedir, "cfg.yml"), "w") as fout:
        yaml.dump(cfg, fout)

    cfgs = []
    sweep_params = [k for k, v in cfg[module]["train"].items() if isinstance(v, list)]
    if len(sweep_params) == 0:
        cfgs.append(cfg)
    else:
        random.seed(0)
        for vals in itertools.product(*[cfg[module]["train"][k] for k in sweep_params]):
            clone = copy.deepcopy(cfg)
            clone[module]["train"].update({k: v for k, v in zip(sweep_params, vals)})
            cfgs.append(clone)
        random.shuffle(cfgs)
        cfgs = cfgs[:max_jobs]

    if remote:
        ngpus = cfg[module].get("resources", {}).get("gpus", 0)
        ncpus = cfg[module].get("resources", {}).get("cpus", 3)
        memgb = cfg[module].get("resources", {}).get("memgb", 20)
        timeout = cfg[module].get("resources", {}).get("timeout", 12 * 60)
        executor = submitit.AutoExecutor(folder=basedir + "/%j")
        executor.update_parameters(
            name=f"cv_{region}",
            gpus_per_node=ngpus,
            cpus_per_task=ncpus,
            mem_gb=memgb,
            array_parallelism=array_parallelism,
            timeout_min=timeout,
        )
        launcher = executor.map_array
        basedirs = [f"{basedir}/%j" for _ in cfgs]
    else:
        basedirs = [os.path.join(basedir, f"job_{i}") for i in range(len(cfgs))]
        launcher = map

    with snapshot.SnapshotManager(
        snapshot_dir=basedir + "/snapshot", with_submodules=True, exclude=["data/*"]
    ):
        jobs = list(launcher(partial(run_cv, module), basedirs, cfgs))

        # Find the best model and retrain on the full dataset
        launcher = run_best
        if remote:
            executor = submitit.AutoExecutor(folder=basedir)
            executor.update_parameters(
                name="model_selection", cpus_per_task=1, mem_gb=2, timeout_min=20
            )
            # Launch the model selection job *after* the sweep finishs
            sweep_job = jobs[0].job_id.split("_")[0]
            executor.update_parameters(
                additional_parameters={"dependency": f"afterany:{sweep_job}"}
            )
            launcher = partial(executor.submit, run_best) if remote else run_best
        if not validate_only:
            launcher(cfg, module, remote, basedir)

    print(basedir)


@cli.command()
@click.argument("config_pth")
@click.argument("module")
@click.option(
    "-period", type=int, help="Number of days for sliding window", required=True
)
@click.option(
    "-start-date", type=click.DateTime(), default="2020-04-01", help="Start date"
)
@click.option("-validate-only", type=click.BOOL, default=False)
@click.option("-remote", type=click.BOOL, default=False)
@click.option("-array-parallelism", type=click.INT, default=50)
@click.option("-max-jobs", type=click.INT, default=200)
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
    array_parallelism: int = 50,
    max_jobs: int = 200,
):
    """
    Run the cross validation pipeline over multiple time points.
    """
    config = load_config(config_pth)
    assert (
        dates is not None or period is not None
    ), "Must specify either dates or period"
    gt = metrics.load_ground_truth(config[module]["data"])
    if dates is None:
        dates = pd.date_range(
            start=start_date, end=gt.index.max(), freq=f"{period}D", closed="left"
        )

    now = datetime.now().strftime("%Y_%m_%d_%H_%M")
    basedir = (
        f'/checkpoint/{os.environ["USER"]}/covid19/forecasts/{config["region"]}/{now}'
    )
    print(f"Backfilling in {basedir}")
    ext = os.path.splitext(config[module]["data"])[-1]
    for date in dates:
        print(f"Running CV for {date.date()}")
        with tempfile.TemporaryDirectory() as tdir:
            tfile = os.path.join(tdir, os.path.basename(config[module]["data"]))
            tconfig = os.path.join(tdir, "config.yml")
            days = (gt.index.max() - date).days
            filter_validation_days(config[module]["data"], tfile, days)
            current_config = copy.deepcopy(config)
            current_config[module]["data"] = tfile
            with open(tconfig, "w") as fout:
                yaml.dump(current_config, fout)
            cv_params = {
                k: v for k, v in ctx.params.items() if k in {p.name for p in cv.params}
            }
            cv_params["config_pth"] = tconfig
            ctx.invoke(
                cv, basedir=os.path.join(basedir, f"sweep_{date.date()}"), **cv_params
            )


if __name__ == "__main__":
    cli()
