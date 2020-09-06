#!/usr/bin/env python3

import sys
import yaml
import json
import argparse
from datetime import date, datetime
from sir import main as sir
from rmse import main as rmse
from train import main as train
from forecast import main as forecast
import os
import contextlib
import itertools
import submitit
from timelord import snapshot


DFLT_PARAMS = [
    "-max-events",
    1000000,
    "-sparse",
    "-scale",
    "1",
    "-optim",
    "lbfgs",
    "-quiet",
    "-fresh",
    "-epochs",
    200,
]


def forecast_train(
    train_params, cwd, dataset_true, basedate, job_dir, days=7, trials=100, log=None
):
    with contextlib.ExitStack() as stack:
        if log is not None:
            stderr = stack.enter_context(open(log + ".stderr", "w"))
            stack.enter_context(contextlib.redirect_stderr(stderr))
            stdout = stack.enter_context(open(log + ".stdout", "w"))
            stack.enter_context(contextlib.redirect_stdout(stdout))

        print(f"train_params: {json.dumps(train_params)}")
        checkpoint = os.path.join(job_dir, "model.bin")
        with_intensity = (
            [] if train_params.get("base_intensity", True) else ["-no-baseint"]
        )

        dataset = os.path.join(cwd, train_params.get("data"))
        NON_DFLT = [
            "-checkpoint",
            checkpoint,
            "-dset",
            dataset,
            "-dim",
            train_params.get("dim", 10),
            "-const-beta",
            train_params.get("const_beta", -1),
            "-max-events",
            train_params.get("max_events", 5000),
            "-alpha-scale",
            train_params.get("alpha_scale", -10),
            "-weight-decay",
            train_params.get("weight_decay", 0),
            "-maxcor",
            train_params.get("maxcor", 50),
        ] + with_intensity
        train_params = list(map(str, DFLT_PARAMS + NON_DFLT))

        rmse(train_params)
        train(train_params)

        forecast_params = [
            # "-tl-simulate",
            "-dset",
            dataset,
            "-dset-true",
            dataset_true,
            "-checkpoint",
            checkpoint,
            "-basedate",
            basedate,
            "-days",
            days,
            "-trials",
            trials,
            "-fout",
            os.path.join(job_dir, "forecasts.csv"),
        ]
        forecast(list(map(str, forecast_params)))


def load_config(pth):
    if pth.endswith(".yml") or pth.endswith("yaml"):
        return yaml.load(open(pth), Loader=yaml.FullLoader)
    elif pth.endswiith(".json"):
        return json.load(open(pth))
    else:
        raise ValueError(f"Unrecognized grid file extension: {pth}")


def run_sir(data, population, region, base, **kwargs):
    doubling_times = kwargs.get("doubling_times", [2, 3, 4, 10])
    os.makedirs(f"{base}/sir", exist_ok=True)
    args = (
        [
            "-fdat",
            data,
            "-fpop",
            population,
            "-dout",
            f"{base}/sir",
            "-days",
            kwargs.get("days", 60),
            "-keep",
            kwargs.get("keep", 7),
            "-window",
            kwargs.get("window", 5),
            "-fsuffix",
            region,
        ]
        + ["-doubling-times"]
        + doubling_times
    )
    sir(list(map(str, args)))


def main(args):
    parser = argparse.ArgumentParser()
    parser.add_argument("config", help="config file (json or yml)")
    parser.add_argument("-basedate", help="Base date for forecasting")
    parser.add_argument(
        "-remote", action="store_true", help="Run jobs remotely on SLURM"
    )
    parser.add_argument("-timeout-min", type=int, default=12 * 60)
    parser.add_argument("-array-parallelism", type=int, default=50)
    parser.add_argument("-mem-gb", type=int, default=20)
    parser.add_argument("-ncpus", type=int, default=20)
    parser.add_argument("-partition", type=str, default="learnfair,scavenge")
    parser.add_argument("-comment", type=str, default=None)
    parser.add_argument("-mail-to", type=str, default=None)
    opt = parser.parse_args(args)

    config = load_config(opt.config)
    now = datetime.now().strftime("%Y_%m_%d_%H_%M")
    user = os.environ["USER"]

    region = os.path.splitext(os.path.basename(opt.config))[0]

    base = f"/checkpoint/{user}/covid19/forecasts/{region}/{now}"

    dataset_true = os.path.realpath(config["data_true"])
    cwd = os.path.realpath(".")

    print("Running SIR model...")
    run_sir(data=dataset_true, base=base, region=region, **config["sir"])

    keys = config["grid"].keys()
    values = list(itertools.product(*[config["grid"][k] for k in keys]))

    grid = [{k: v for k, v in zip(keys, vs)} for vs in values]

    basedate = opt.basedate or str(date.today())

    def run_job(d):
        kwargs = {**config["forecast"], "basedate": basedate}
        try:
            job_env = submitit.JobEnvironment()
            job_dir = os.path.join(base, str(submitit.JobEnvironment().job_id))
        except Exception:
            job_dir = os.path.join(base, "_".join([f"{k}_{v}" for k, v in d.items()]))
            kwargs["log"] = f"{job_dir}/log"
        os.makedirs(job_dir, exist_ok=True)
        forecast_train(d, cwd=cwd, dataset_true=dataset_true, job_dir=job_dir, **kwargs)

    if opt.remote:
        mail_to = {}
        if opt.mail_to is not None:
            mail_to["mail_type"] = "END"
            mail_to["mail_user"] = opt.mail_to

        executor = submitit.AutoExecutor(folder=base + "/%j")
        executor.update_parameters(
            name=f"{region}-sweep-mhp",
            gpus_per_node=1,
            cpus_per_task=opt.ncpus,
            mem_gb=opt.mem_gb,
            array_parallelism=opt.array_parallelism,
            timeout_min=opt.timeout_min,
            partition=opt.partition,
            comment=opt.comment,
            additional_parameters=mail_to,
        )
        with snapshot.SnapshotManager(
            snapshot_dir=base + "/snapshot", with_submodules=True
        ):
            jobs = executor.map_array(run_job, grid)
        print(base)
        return base, jobs
    else:
        for d in grid:
            run_job(d)


if __name__ == "__main__":
    main(sys.argv[1:])
