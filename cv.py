#!/usr/bin/env python3

import argparse
import importlib
import pandas as pd
import os
import torch as th
import yaml
from argparse import Namespace
from datetime import datetime
from typing import Dict, Any, List
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


BestRun = namedtuple('BestRun', ('pth', 'name'))

class CV:
    def run_simulate(self, dset, train_params, model, sim_params):
        ...
    
    def run_prediction_interval(self, args, nsamples, model=None):
        ...

    def run_train(self, dset, model_params, model_out):
        ...
    
    def model_selection(self, basedir: str) -> List[BestRun]:
        best_run, best_MAE = None, float('inf')
        for metrics_pth in glob(os.path.join(basedir, '*/metrics.csv')):
            metrics = pd.read_csv(metrics_pth, index_col='Measure')
            if metrics.loc['MAE'].values[-1] < best_MAE:
                best_MAE = metrics.loc['MAE'].values[-1]
                best_run = os.path.dirname(metrics_pth)
        return [BestRun(best_run, 'best_mae')]


def cv(opt: argparse.Namespace, basedir: str, cfg: Dict[str, Any], prefix=''):
    try:
        basedir = basedir.replace("%j", submitit.JobEnvironment().job_id)
    except Exception:
        pass  # running locally, basedir is fine...

    os.makedirs(basedir, exist_ok=True)

    def _path(path):
        return os.path.join(basedir, path)

    # setup input/output paths
    dset = cfg[opt.module]["data"]
    val_in = _path("filtered_" + os.path.basename(dset))
    val_out = _path(prefix + cfg["validation"]["output"])
    cfg[opt.module]["train"]["fdat"] = val_in

    mod = importlib.import_module(opt.module).CV_CLS()

    filter_validation_days(dset, val_in, cfg["validation"]["days"])

    # -- train --
    train_params = Namespace(**cfg[opt.module]["train"])
    model = mod.run_train(val_in, train_params, _path(prefix + cfg[opt.module]["output"]))

    # -- simulate --
    with th.no_grad():
        sim_params = cfg[opt.module].get('simulate', {})
        df_forecast = mod.run_simulate(val_in, train_params, model, sim_params=sim_params)
    print(f"Storing validation in {val_out}")
    df_forecast.to_csv(val_out)

    # -- metrics --
    if cfg["validation"]["days"] > 0:
        # Only compute metrics if this is the validation run.
        df_val = metrics.compute_metrics(cfg[opt.module]["data"], val_out).round(2)
        df_val.to_csv(_path(prefix + "metrics.csv"))
        print(df_val)

    # -- store configs to reproduce results --
    log_configs(cfg, opt.module, _path(f"{opt.module}.yml"))

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
        return os.path.realpath(cfg) if isinstance(cfg, str) and os.path.exists(cfg) else cfg


def load_config(cfg_pth: str) -> Dict[str, Any]:
    return mk_absolute_paths(yaml.load(open(cfg_pth), Loader=yaml.FullLoader))


def log_configs(cfg: Dict[str, Any], module: str, path: str):
    """Logs configs for job for reproducibility"""
    with open(path, "w") as f:
        yaml.dump(cfg[module], f)


def run_best(basedir, cfg, opt):
    mod = importlib.import_module(opt.module).CV_CLS()
    best_runs = mod.model_selection(basedir)

    cfg = copy.deepcopy(cfg)
    cfg['validation']['days'] = 0

    ngpus = cfg[opt.module].get("resources", {}).get("gpus", 0)
    ncpus = cfg[opt.module].get("resources", {}).get("cpus", 3)
    memgb = cfg[opt.module].get("resources", {}).get("memgb", 20)
    timeout = cfg[opt.module].get("resources", {}).get("timeout", 12 * 60)

    for run in best_runs:
        job_config = load_config(os.path.join(run.pth, opt.module + '.yml'))
        cfg[opt.module] = job_config
        cfg["validation"]["output"] = run.name + '_forecast.csv'
        launcher = cv
        if opt.remote:
            executor = submitit.AutoExecutor(folder=run.pth)
            executor.update_parameters(
                name=run.name,
                gpus_per_node=ngpus,
                cpus_per_task=ncpus,
                mem_gb=memgb,
                timeout_min=timeout,
            )
            launcher = partial(executor.submit, cv)
        launcher(opt, run.pth, cfg, prefix='final_model_')



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config", help="config file (yaml)")
    parser.add_argument("module", help="config file (yaml)")
    parser.add_argument(
        '-validate-only', 
        action='store_true',
        help='Only run validation jobs (skip model selection and retraining)'
    )
    parser.add_argument(
        "-remote", action="store_true", help="Run jobs remotely on SLURM"
    )
    parser.add_argument("-array-parallelism", type=int, default=50)
    parser.add_argument("-max-jobs", type=int, default=200)
    opt = parser.parse_args()

    now = datetime.now().strftime("%Y_%m_%d_%H_%M")
    user = os.environ["USER"]

    cfg = load_config(opt.config)
    region = cfg["region"]

    if opt.remote:
        basedir = f"/checkpoint/{user}/covid19/forecasts/{region}/{now}"
    else:
        basedir = f"/tmp/covid19/forecasts/{region}/{now}"

    os.makedirs(basedir, exist_ok=True)
    with open(os.path.join(basedir, 'cfg.yml'), 'w') as fout:
        yaml.dump(cfg, fout)

    cfgs = []
    sweep_params = [
        k for k, v in cfg[opt.module]["train"].items() if isinstance(v, list)
    ]
    if len(sweep_params) == 0:
        cfgs.append(cfg)
    else:
        random.seed(0)
        for vals in itertools.product(
            *[cfg[opt.module]["train"][k] for k in sweep_params]
        ):
            clone = copy.deepcopy(cfg)
            clone[opt.module]["train"].update(
                {k: v for k, v in zip(sweep_params, vals)}
            )
            cfgs.append(clone)
        random.shuffle(cfgs)
        cfgs = cfgs[: opt.max_jobs]

    if opt.remote:
        ngpus = cfg[opt.module].get("resources", {}).get("gpus", 0)
        ncpus = cfg[opt.module].get("resources", {}).get("cpus", 3)
        memgb = cfg[opt.module].get("resources", {}).get("memgb", 20)
        timeout = cfg[opt.module].get("resources", {}).get("timeout", 12 * 60)
        executor = submitit.AutoExecutor(folder=basedir + "/%j")
        executor.update_parameters(
            name=f"cv_{region}",
            gpus_per_node=ngpus,
            cpus_per_task=ncpus,
            mem_gb=memgb,
            slurm_array_parallelism=opt.array_parallelism,
            timeout_min=timeout,
        )
        launcher = executor.map_array
        basedirs = [f"{basedir}/%j" for _ in cfgs]
    else:
        basedirs = [os.path.join(basedir, f"job_{i}") for i in range(len(cfgs))]
        launcher = map

    with snapshot.SnapshotManager(snapshot_dir=basedir + "/snapshot", with_submodules=True):
        jobs = list(launcher(partial(cv, opt), basedirs, cfgs))

        # Find the best model and retrain on the full dataset
        launcher = run_best
        if opt.remote:
            executor = submitit.AutoExecutor(folder=basedir)
            executor.update_parameters(name="model_selection", cpus_per_task=1, mem_gb=2, timeout_min=20)
            # Launch the model selection job *after* the sweep finishs
            sweep_job = jobs[0].job_id.split('_')[0]
            executor.update_parameters(slurm_additional_parameters={'dependency': f'afterany:{sweep_job}'})
            launcher = partial(executor.submit, run_best) if opt.remote else run_best
        if not opt.validate_only:
            launcher(basedir, cfg, opt)

    print(basedir)
    
