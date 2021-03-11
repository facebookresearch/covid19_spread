#!/usr/bin/env python3
# Copyright (c) 2021-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import copy
from importlib.machinery import SourceFileLoader
import click
import importlib
import itertools
import json
import numpy as np
import pandas as pd
import os
import random
import shutil
import submitit
import tempfile
import torch as th
from tqdm import tqdm
import re
import yaml
from argparse import Namespace
from collections import namedtuple
from datetime import datetime, timedelta
from functools import partial
from glob import glob, iglob
from typing import Dict, Any, List, Optional, Tuple
from contextlib import nullcontext, ExitStack
from covid19_spread import common
from covid19_spread import metrics
from covid19_spread.lib import cluster
from covid19_spread.lib.click_lib import DefaultGroup
from covid19_spread.lib.slurm_pool_executor import (
    SlurmPoolExecutor,
    JobStatus,
    get_db_client,
    TransactionManager,
)
from covid19_spread.lib.slack import get_client as get_slack_client
from submitit.helpers import RsyncSnapshot
from covid19_spread.cross_val import CV, load_config, BestRun


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


def ensemble(basedirs, cfg, module, prefix, outdir):
    def _path(x):
        return os.path.join(basedir, prefix + x)

    means = []
    stds = []
    mean_deltas = []
    kwargs = {"index_col": "date", "parse_dates": ["date"]}
    stdfile = "std_closed_form.csv"
    meanfile = "mean_closed_form.csv"
    for basedir in basedirs:
        if os.path.exists(_path(cfg["validation"]["output"])):
            means.append(pd.read_csv(_path(cfg["validation"]["output"]), **kwargs))
        if os.path.exists(_path(stdfile)):
            stds.append(pd.read_csv(_path(stdfile), **kwargs))
            mean_deltas.append(pd.read_csv(_path(meanfile), **kwargs))
    if len(stds) > 0:
        # Average the variance, and take square root
        std = pd.concat(stds).pow(2).groupby(level=0).mean().pow(0.5)
        std.to_csv(os.path.join(outdir, prefix + stdfile))
        mean_deltas = pd.concat(mean_deltas).groupby(level=0).mean()
        mean_deltas.to_csv(os.path.join(outdir, prefix + meanfile))

    assert len(means) > 0, "All ensemble jobs failed!!!!"

    mod = importlib.import_module(module, package="covid19_spread").CV_CLS()

    if len(stds) > 0:
        pred_interval = cfg.get("prediction_interval", {})
        piv = mod.run_prediction_interval(
            os.path.join(outdir, prefix + meanfile),
            os.path.join(outdir, prefix + stdfile),
            pred_interval.get("intervals", [0.99, 0.95, 0.8]),
            gaussian=True,
        )
        piv.to_csv(os.path.join(outdir, prefix + "piv.csv"), index=False)

    mean = pd.concat(means).groupby(level=0).median()
    outfile = os.path.join(outdir, prefix + cfg["validation"]["output"])
    mean.to_csv(outfile, index_label="date")

    # -- metrics --
    metric_args = cfg[module].get("metrics", {})
    df_val, json_val = mod.compute_metrics(
        cfg[module]["data"], outfile, None, metric_args
    )
    df_val.to_csv(os.path.join(outdir, prefix + "metrics.csv"))
    with open(os.path.join(outdir, prefix + "metrics.json"), "w") as fout:
        json.dump(json_val, fout)
    print(df_val)


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

    mod = importlib.import_module("covid19_spread." + module).CV_CLS()

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
        df_forecast = common.rebase_forecast_deltas(val_in, df_forecast_deltas)

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
                    closed_form=True,
                )
                df_std.to_csv(_path(f"{prefix}std_closed_form.csv"), index_label="date")
                df_mean.to_csv(
                    _path(f"{prefix}mean_closed_form.csv"), index_label="date"
                )
                piv = mod.run_prediction_interval(
                    _path(f"{prefix}mean_closed_form.csv"),
                    _path(f"{prefix}std_closed_form.csv"),
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


def log_configs(cfg: Dict[str, Any], module: str, path: str):
    """Logs configs for job for reproducibility"""
    with open(path, "w") as f:
        yaml.dump(cfg[module], f)


def run_best(config, module, remote, basedir, basedate=None, executor=None):
    mod = importlib.import_module("covid19_spread." + module).CV_CLS()
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
        os.makedirs(os.path.join(os.path.dirname(pth), "forecasts"), exist_ok=True)
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
@click.option("-closed-form", is_flag=True)
def prediction_interval(chkpnts, remote, nsamples, batchsize, closed_form):
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
        mod = importlib.import_module("covid19_spread." + module).CV_CLS()
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
            closed_form=closed_form,
        )
        suffix = "_closed_form" if closed_form else ""
        df_std.to_csv(
            os.path.join(job_pth, f"{prefix}std{suffix}.csv"), index_label="date"
        )
        df_mean.to_csv(
            os.path.join(job_pth, f"{prefix}mean{suffix}.csv"), index_label="date"
        )
        pred_intervals = mod.run_prediction_interval(
            os.path.join(job_pth, f"{prefix}mean{suffix}.csv"),
            os.path.join(job_pth, f"{prefix}std{suffix}.csv"),
            pred_interval.get("intervals", [0.99, 0.95, 0.8]),
            gaussian=True,
        )
        pred_intervals.to_csv(
            os.path.join(job_pth, f"{prefix}piv{suffix}.csv"), index=False
        )

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
    user = cluster.USER

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
                RsyncSnapshot(
                    snapshot_dir=basedir + "/snapshot",
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
    basedir = f"{cluster.FS}/{cluster.USER}/covid19/forecasts/{experiment_id}"

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
    with RsyncSnapshot(
        snapshot_dir=basedir + "/snapshot", exclude=["notebooks/*", "tests/*"],
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
            executor.launch(basedir + "/workers", array_parallelism)


@cli.command()
@click.argument("paths", nargs=-1)
def ensemble_jobs(paths):
    for path in paths:
        ms = json.load(open(os.path.join(path, "model_selection.json")))
        ms = {x["name"]: x["pth"] for x in ms}
        jobs = [
            x for x in glob(os.path.join(ms["best_mae"], "job_*")) if os.path.isdir(x)
        ]
        cfg = load_config(os.path.join(path, "cfg.yml"))
        cfg["prediction_interval"]["intervals"] = [0.95, 0.8, 0.5]
        ensemble(jobs, cfg, cfg["this_module"], "final_model_", ms["best_mae"])


@cli.command()
@click.argument("sweep_dirs", nargs=-1)
def progress(sweep_dirs):
    for sweep_dir in sweep_dirs:
        sweep_dir = os.path.realpath(sweep_dir)
        db_file = next(iglob(os.path.join(sweep_dir, "**/.job.db"), recursive=True))
        db_file = os.path.realpath(db_file)
        conn = get_db_client()
        df = pd.read_sql(
            f"SELECT status, worker_id FROM jobs WHERE id='{db_file}'", conn
        )
        msg = {
            "sweep_dir": sweep_dir,
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
@click.option("-reset-running", is_flag=True, default=False)
def repair(sweep_dir, workers=None, reset_running=False):
    db_file = next(iglob(os.path.join(sweep_dir, "**/.job.db"), recursive=True))
    txn_manager = TransactionManager(os.path.realpath(db_file))
    cond = ""
    if reset_running:
        cond = f" OR status >= {len(JobStatus)}"
    txn_manager.run(
        lambda conn: conn.execute(
            f"""
    UPDATE jobs SET status={JobStatus.pending}
    WHERE id='{os.path.realpath(db_file)}' AND (status={JobStatus.failure} {cond})
    """
        )
    )
    if workers is not None:
        cfg = load_config(next(iglob(f"{sweep_dir}/**/cfg.yml", recursive=True)))
        extra_params = cfg[cfg["this_module"]].get("resources", {})
        executor = mk_executor(
            "repair", sweep_dir, extra_params, db_pth=os.path.realpath(db_file)
        )
        executor.launch(os.path.join(sweep_dir, "workers"), workers or -1)


@cli.command()
@click.argument("sweep_dir")
@click.option(
    "--type",
    "-t",
    type=click.Choice(["failure", "running", "pending", "success"]),
    required=True,
)
def list_jobs(sweep_dir, type):
    db_file = next(iglob(os.path.join(sweep_dir, "**/.job.db"), recursive=True))
    db_file = os.path.realpath(db_file)
    conn = get_db_client()
    if type == "running":
        cond = f"status >= {len(JobStatus)}"
    else:
        cond = f"status = {getattr(JobStatus, type)}"
    with conn.cursor() as cur:
        cur.execute(
            f"""
        SELECT pickle, worker_id FROM jobs WHERE id='{db_file}' AND {cond}
        """
        )
        for row in cur:
            print(row)


if __name__ == "__main__":
    cli()
