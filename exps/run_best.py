#!/usr/bin/env python3

import argparse
from glob import glob
import os
import re
import json
import pandas
from exps.run_experiment import run_experiment
import submitit
import sys
import shutil
from subprocess import check_output


def train_and_forecast(name, **kwargs):
    run_experiment(**kwargs)
    user = os.environ["USER"]
    forecast_dir = f"/checkpoint/{user}/covid19/forecasts/usa"
    os.makedirs(forecast_dir, exist_ok=True)
    df = pandas.read_csv(
        os.path.join(kwargs["folder"], "full_data_forecasts.csv"), index_col="location"
    )
    basedate = pandas.to_datetime(df.columns).min().date()
    df.to_csv(forecast_dir + f"/{name}_forecast_{basedate}.csv")


def launch_job(job_dir, name, launched):
    if job_dir in launched:
        return launched

    executor = submitit.AutoExecutor(folder=job_dir)
    executor.update_parameters(
        name=name, gpus_per_node=1, cpus_per_task=3, mem_gb=20, timeout_min=12 * 60,
    )
    params = json.load(open(os.path.join(job_dir, "params.json")))
    job = executor.submit(
        train_and_forecast,
        name=name,
        grid={},
        days=7,
        crossval=False,
        folder=job_dir,
        pdict=params["params"],
        chkpnt_name="model_full_data.bin",
    )
    launched.add(job_dir)
    return launched


def launch_best(sweep_dir):
    results = []
    for d in os.listdir(sweep_dir):
        if re.match("\d+_\d+", d):
            res_pth = os.path.join(sweep_dir, d, "eval.json")
            if not os.path.exists(res_pth):
                continue
            results.append(json.load(open(res_pth)))
            log = check_output(
                f'cat {os.path.join(sweep_dir,d)}/{d}*.out | grep json_stats | grep -o "\\{{.*\\}}"',
                shell=True,
            )
            log = pandas.DataFrame(
                [json.loads(l) for l in log.decode("utf-8").strip().split("\n")]
            )
            results[-1]["best_ll"] = log.non_global_ll.max()
    df = pandas.DataFrame(results)
    print(f"Evaluated {len(df)} jobs")

    # Launch job by best MAE
    mae_cols = [c for c in df.columns if re.match("day_\d+_mean", c)]
    mae_col = sorted(mae_cols, key=lambda x: int(re.search("\d+", x).group(0)))[-1]
    launched = launch_job(
        os.path.dirname(df.sort_values(by=mae_col).iloc[0].pth),
        f"best_{mae_col}",
        set(),
    )

    # Launch job by best KS
    launched = launch_job(
        os.path.dirname(df.sort_values(by="ks").iloc[0].pth), "best_ks", launched
    )

    # Launch job by best pval
    launched = launch_job(
        os.path.dirname(df.sort_values(by="pval").iloc[-1].pth), "best_pval", launched
    )

    # Launch job by best LL
    launched = launch_job(
        os.path.dirname(df.sort_values(by="best_ll").iloc[-1].pth), "best_ll", launched
    )

    print(json.dumps(list(launched), indent=2))


def main(args):
    parser = argparse.ArgumentParser()
    parser.add_argument("sweep_dir")
    opt = parser.parse_args(args)

    dependent_jobs = []
    for d in os.listdir(opt.sweep_dir):
        if re.match("\d+_\d+", d):
            dependent_jobs.append(d)

    dep = f'afterany:{dependent_jobs[0].strip().split("_")[0]}'

    # dep = ','.join([f'afterany:{d}' for d in dependent_jobs])

    executor = submitit.AutoExecutor(folder=opt.sweep_dir)
    executor.update_parameters(
        name="launch_best_runs",
        cpus_per_task=1,
        mem_gb=2,
        timeout_min=20,
        slurm_additional_parameters={"dependency": dep},
    )
    executor.submit(launch_best, opt.sweep_dir)


if __name__ == "__main__":
    main(sys.argv[1:])
