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
import sqlite3
import pandas


class NJRecurring(recurring.Recurring):
    script_dir = script_dir

    def get_id(self):
        return "new-jersey"

    def command(self):
        return f"python {os.path.realpath(__file__)}"

    # Create the new timeseries.h5 dataset
    def update_data(self, env_vars={"SMOOTH": "1"}):
        df = get_latest()
        date_fmt = df["Date"].max().date().strftime("%Y%m%d")
        csv_file = f"{script_dir}/data-{date_fmt}.csv"
        df.to_csv(csv_file)
        process_cases.main(csv_file)

    def latest_date(self):
        df = pandas.read_csv("data_cases.csv", index_col="region")
        return pandas.to_datetime(df.columns).max().date()

    def launch_job(self, cv_config="nj", module="mhp", **kwargs):
        return super().launch_job(cv_config="nj", module="mhp", **kwargs)


class NJARRecurring(NJRecurring):
    def get_id(self):
        return "new-jersey-ar"

    def command(self):
        return super().command() + f" --kind ar"

    def launch_job(self, **kwargs):
        return super().launch_job(cv_config="nj", module="ar", **kwargs)

    def module(self):
        return "ar"

    # Create the new timeseries.h5 dataset
    def update_data(self):
        super().update_data(env_vars={})


class NJSweepRecurring(NJRecurring):
    def get_id(self):
        return "new-jersey-sweep.py"

    def launch_job(self):
        # Launch the sweep
        date = self.latest_date().strftime("%Y%m%d")
        output = check_output(
            ["make", "grid-nj", f"DATE={date}"], cwd=os.path.join(script_dir, "../../")
        )
        return output.decode("utf-8").strip().split("\n")[-1]

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
