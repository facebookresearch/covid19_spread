#!/usr/bin/env python3


import sys
import os

script_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, os.path.join(script_dir, "../.."))
sys.path.insert(0, os.path.join(script_dir, ".."))
import recurring
import cv
import argparse
from scraper import get_latest
import process_cases
import tempfile
import numpy as np
from subprocess import check_call, check_output
from glob import glob
import sqlite3
import pandas
import sweep

MAIL_TO = ["mattle@fb.com", "lematt1991@gmail.com"]
if os.environ.get("__PROD__") == "1":
    MAIL_TO.append("maxn@fb.com")

print(f"MAIL_TO == {MAIL_TO}")


class NJRecurring(recurring.Recurring):
    script_dir = script_dir

    def get_id(self):
        return "new-jersey"

    def command(self):
        return f"python {os.path.realpath(__file__)}"

    # Create the new timeseries.h5 dataset
    def update_data(self, env_vars={"SMOOTH": "1"}, counts_only=False):
        repo = "git@github.com:fairinternal/covid19_spread.git"
        user = os.environ["USER"]
        data_dir = f"/checkpoint/{user}/covid19/data/nj_data"
        if not os.path.exists(data_dir):
            check_call(["git", "clone", repo, data_dir])
        check_call(["git", "pull"], cwd=data_dir)
        latest_pth = sorted(glob(f"{data_dir}/data/new-jersey/data-202*.csv"))[-1]
        with recurring.env_var(env_vars):
            process_cases.main(latest_pth, counts_only=counts_only)

    def latest_date(self):
        df = pandas.read_csv("data_cases.csv", index_col="region")
        return pandas.to_datetime(df.columns).max().date()

    def launch_job(self, cv_config="nj_prod", module="mhp", **kwargs):
        return super().launch_job(cv_config=cv_config, module=module, **kwargs)


class NJARRecurring(NJRecurring):
    def get_id(self):
        return "new-jersey-ar"

    def command(self):
        return super().command() + f" --kind ar"

    def launch_job(self, **kwargs):
        return super().launch_job(cv_config="nj_prod", module="ar", **kwargs)

    def module(self):
        return "ar"

    # Create the new timeseries.h5 dataset
    def update_data(self):
        super().update_data(env_vars={}, counts_only=True)


class NJSweepRecurring(NJRecurring):
    def get_id(self):
        return "new-jersey-sweep.py"

    def launch_job(self):
        # Launch the sweep
        with recurring.chdir(os.path.join(script_dir, "../../")):
            base_dir, jobs = sweep.main(
                [
                    os.path.join("grids/new-jersey.yml"),
                    "-remote",
                    "-ncpus",
                    "40",
                    "-timeout-min",
                    "60",
                    "-partition",
                    "learnfair,scavenge",
                    "-comment",
                    "COVID-19 NJ Forecast",
                    "-mail-to",
                    ",".join(MAIL_TO),
                ]
            )

        with tempfile.NamedTemporaryFile() as tfile:
            with open(tfile.name, "w") as fout:
                print(f"Started NJ MHP sweep for {self.latest_date()}", file=fout)
                print(f"Sweep directory: {base_dir}", file=fout)
                print(f"SLURM job ID: {jobs[0].job_id.split('_')[0]}", file=fout)
            check_call(
                f'mail -s "Started NJ MHP sweep!" {" ".join(MAIL_TO)} < {tfile.name}',
                shell=True,
            )
        return base_dir

    def update_data(self):
        super().update_data(env_vars={"SMOOTH": "1"})
        super().update_data(env_vars={})

    def schedule(self) -> str:
        """Cron schedule"""
        # run every 5 minutes, offset by 2 minutes.  This avoids conflicts
        # with the AR data prep pipeline
        return "2-59/5 * * * *"

    def command(self):
        return super().command() + f" --kind sweep"


def main(args):
    kinds = {
        "cv": NJRecurring,
        "sweep": NJSweepRecurring,
        "ar": NJARRecurring,
    }

    parser = argparse.ArgumentParser()
    parser.add_argument("--install", action="store_true")
    parser.add_argument("--kind", choices=list(kinds.keys()), default="cv")
    opt = parser.parse_args()

    job = kinds[opt.kind]()

    if opt.install:
        job.install()  # install cron job
    else:
        job.refresh()  # run sweep if new data is available


if __name__ == "__main__":
    main(sys.argv[1:])
