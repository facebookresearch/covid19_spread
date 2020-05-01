#!/usr/bin/env python3

import argparse
import importlib
import pandas as pd
import os
import torch as th
import yaml
from argparse import Namespace
from datetime import datetime
from typing import Dict, Any
import common
import metrics
import itertools
import copy
import random
import submitit
from functools import partial


def cv(opt: argparse.Namespace, basedir: str, cfg: Dict[str, Any]):
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
    val_out = _path("validation.csv")
    cfg[opt.module]["train"]["fdat"] = val_in

    filter_validation_days(dset, val_in, cfg["validation_days"])

    # -- train --
    train_params = Namespace(**cfg[opt.module]["train"])
    mod = importlib.import_module(opt.module)
    model = mod.run_train(val_in, train_params, _path(cfg[opt.module]["output"]))

    # -- simulate --
    with th.no_grad():
        df_forecast = mod.run_simulate(val_in, train_params, model)
    print(f"Storing forecast in {val_out}")
    df_forecast.to_csv(val_out)

    # -- metrics --
    df_val = metrics.compute_metrics(cfg[opt.module]["data"], val_out).round(2)
    df_val.to_csv(_path("metrics.csv"))
    print(df_val)

    # -- store configs to reproduce results -- 
    log_configs(cfg, opt.module, _path(f"{opt.module}.yml"))

    # -- prediction interval --
    if "prediction_interval" in cfg:
        with th.no_grad():
            df_mean, df_std = mod.run_prediction_interval(
                train_params, cfg["prediction_interval"]["nsamples"], model
            )
            df_mean.to_csv(_path(cfg["prediction_interval"]["output_mean"]))
            df_std.to_csv(_path(cfg["prediction_interval"]["output_std"]))


def filter_validation_days(dset: str, val_in: str, validation_days: int):
    """Filters validation days and writes output to val_in path"""
    if dset.endswith(".csv"):
        common.drop_k_days_csv(dset, val_in, cfg["validation_days"])
    elif dset.endswith(".h5"):
        common.drop_k_days(dset, val_in, cfg["validation_days"])
    else:
        raise RuntimeError(f"Unrecognized dataset extension: {dset}")


def log_configs(cfg: Dict[str, Any], module: str, path: str):
    """Logs configs for job for reproducibility"""
    with open(path, "w") as f:
        yaml.dump(cfg[module], f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config", help="config file (yaml)")
    parser.add_argument("module", help="config file (yaml)")
    parser.add_argument("-basedate", help="Base date for forecasting")
    parser.add_argument(
        "-remote", action="store_true", help="Run jobs remotely on SLURM"
    )
    parser.add_argument("-timeout-min", type=int, default=12 * 60)
    parser.add_argument("-array-parallelism", type=int, default=50)
    parser.add_argument("-mem-gb", type=int, default=20)
    parser.add_argument("-ncpus", type=int, default=20)
    parser.add_argument("-max-jobs", type=int, default=200)
    opt = parser.parse_args()

    now = datetime.now().strftime("%Y_%m_%d_%H_%M")
    user = os.environ["USER"]

    cfg = yaml.load(open(opt.config), Loader=yaml.FullLoader)
    region = cfg["region"]

    if opt.remote:
        basedir = f"/checkpoint/{user}/covid19/forecasts/{region}/{now}"
    else:
        basedir = f"/tmp/covid19/forecasts/{region}/{now}"

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
        executor = submitit.AutoExecutor(folder=basedir + "/%j")
        executor.update_parameters(
            name=f"cv_{region}",
            gpus_per_node=ngpus,
            cpus_per_task=ncpus,
            mem_gb=memgb,
            slurm_array_parallelism=opt.array_parallelism,
            timeout_min=12 * 60,
        )
        launcher = executor.map_array
        basedirs = [f"{basedir}/%j" for _ in cfgs]
    else:
        basedirs = [os.path.join(basedir, f"job_{i}") for i in range(len(cfgs))]
        launcher = map

    list(launcher(partial(cv, opt), basedirs, cfgs))
    print(basedir)