#!/usr/bin/env python3


import sys
import os

script_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, os.path.join(script_dir, "../.."))
sys.path.insert(0, os.path.join(script_dir, ".."))
import recurring
import cv
import sweep
import argparse
import tempfile
import numpy as np
from subprocess import check_call, check_output
import sqlite3
import process_cases
import pandas
from lib.slack import get_client as get_slack_client


MAIL_TO = ["mattle@fb.com"]
if os.environ.get("__PROD__") == "1":
    MAIL_TO.append("maxn@fb.com")

print(f"MAIL_TO == {MAIL_TO}")


class NYARRecurring(recurring.Recurring):
    script_dir = script_dir

    def get_id(self):
        return "nystate-ar"

    def command(self):
        return f"python {os.path.realpath(__file__)} --kind ar"

    def module(self):
        return "ar"

    def schedule(self):
        return "2-59/5 * * * *"

    # Create the new timeseries.h5 dataset
    def update_data(self):
        URL = "https://health.data.ny.gov/api/views/xdss-u53e/rows.csv?accessType=DOWNLOAD"
        with recurring.env_var({"SMOOTH": "1"}):
            process_cases.main(fout="timeseries", fin=URL)

    def latest_date(self):
        df = pandas.read_csv("data_cases.csv", index_col="region")
        return pandas.to_datetime(df.columns).max().date()

    def launch_job(self, **kwargs):
        client = get_slack_client()
        msg = f"*New Data Available for New York: {self.latest_date()}*"
        client.chat_postMessage(channel="#new-data", text=msg)
        return super().launch_job(module="ar", cv_config="ny_prod", **kwargs)


class NYSweepRecurring(NYARRecurring):
    def get_id(self):
        return "nystate_mhp_sweep"

    def module(self):
        return "mhp"

    def command(self):
        return f"python {os.path.realpath(__file__)} --kind sweep"

    def update_data(self):
        with recurring.chdir(os.path.join(script_dir, "../../")):
            check_call(["make", "data-ny"])

    def latest_date(self):
        df = pandas.read_csv(
            os.path.join(script_dir, "data-new.csv"), index_col="region"
        )
        return pandas.to_datetime(df.columns).max().date()

    def launch_job(self, **kwargs):
        with recurring.chdir(os.path.join(script_dir, "../../")):
            base_dir_nyc, jobs_nyc = sweep.main(
                [
                    os.path.join("grids/nyc.yml"),
                    "-remote",
                    "-ncpus",
                    "40",
                    "-timeout-min",
                    "60",
                    "-partition",
                    "learnfair,scavenge",
                    "-comment",
                    "COVID-19 NYC Forecast",
                    "-mail-to",
                    ",".join(MAIL_TO),
                ]
            )

            base_dir_nys, jobs_nys = sweep.main(
                [
                    os.path.join("grids/nys.yml"),
                    "-remote",
                    "-ncpus",
                    "40",
                    "-timeout-min",
                    "60",
                    "-partition",
                    "learnfair,scavenge",
                    "-comment",
                    "COVID-19 NY state Forecast",
                    "-mail-to",
                    ",".join(MAIL_TO),
                ]
            )
        with tempfile.NamedTemporaryFile() as tfile:
            with open(tfile.name, "w") as fout:
                print(f"Started NYS MHP sweep for {self.latest_date()}", file=fout)
                print(f"NYS Sweep directory: {base_dir_nys}", file=fout)
                print(
                    f"NYS SLURM job ID: {jobs_nys[0].job_id.split('_')[0]}", file=fout
                )
                print(f"NYC Sweep directory: {base_dir_nyc}", file=fout)
                print(
                    f"NYC SLURM job ID: {jobs_nyc[0].job_id.split('_')[0]}", file=fout
                )
            check_call(
                f'mail -s "Started NYS MHP sweep!" {" ".join(MAIL_TO)} < {tfile.name}',
                shell=True,
            )
        return base_dir_nys


def main(args):
    kinds = {
        "ar": NYARRecurring,
        "sweep": NYSweepRecurring,
    }

    parser = argparse.ArgumentParser()
    parser.add_argument("--install", action="store_true")
    parser.add_argument("--kind", choices=list(kinds.keys()), default="ar")
    opt = parser.parse_args()

    job = kinds[opt.kind]()

    if opt.install:
        job.install()  # install cron job
    else:
        job.refresh()  # run sweep if new data is available


if __name__ == "__main__":
    main(sys.argv[1:])
