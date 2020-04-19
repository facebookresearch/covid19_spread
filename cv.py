#!/usr/bin/env python3

import argparse
import importlib
import pandas as pd
import os
import torch as th
import yaml
from argparse import Namespace
from datetime import datetime

import common
import metrics


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
    opt = parser.parse_args()

    now = datetime.now().strftime("%Y_%m_%d_%H_%M")
    user = os.environ["USER"]

    cfg = yaml.load(open(opt.config), Loader=yaml.FullLoader)

    if opt.remote:
        basedir = f"/checkpoint/{user}/covid19/forecasts/{region}/{now}"
    else:
        basedir = "/tmp"

    # setup input/output paths
    val_in = os.path.join(basedir, cfg["validation"]["input"])
    val_out = os.path.join(basedir, cfg["validation"]["output"])
    metrics_out = os.path.join(basedir, cfg["metrics"]["output"])
    model_out = os.path.join(basedir, cfg[opt.module]["output"])

    # -- filter --
    common.drop_k_days(
        cfg["data"], cfg["validation"]["input"], cfg["validation"]["days"]
    )

    # -- train --
    train_params = Namespace(**cfg[opt.module]["train"])
    mod = importlib.import_module(cfg[opt.module]["module"])
    model = mod.run_train(train_params, model_out)

    # -- simulate --
    with th.no_grad():
        df_forecast = mod.run_simulate(train_params, model)
    df_forecast.to_csv(val_out)

    # -- metrics --
    df_val = metrics.compute_metrics(cfg["data"], val_out)
    df_val.to_csv(metrics_out)
    print(df_val)
