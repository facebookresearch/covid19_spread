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


class NJRecurring(recurring.Recurring):
    script_dir = script_dir

    def get_id(self):
        return "new-jersey"

    def marker(self):
        return "___NJ_SWEEP_JOB___"

    def command(self):
        return f"python {os.path.realpath(__file__)}"

    # Create the new timeseries.h5 dataset
    def update_data(self):
        df = get_latest()
        date_fmt = df["Date"].max().date().strftime("%Y%m%d")
        df.to_csv(f"{script_dir}/data-{date_fmt}.csv")
        with tempfile.NamedTemporaryFile(suffix=".csv") as tfile, recurring.env_var(
            "SMOOTH", "1"
        ):
            df = df.reset_index().rename(columns={"index": "Date"})
            df["Start day"] = np.arange(1, len(df) + 1)
            df.to_csv(tfile.name)
            process_cases.main(tfile.name)

    def latest_date(self):
        return get_latest()["Date"].max()


class NJSweepRecurring(NJRecurring):
    def get_id(self):
        return "new-jersey-sweep.py"

    def marker(self):
        return "___NJ_SWEEP_PY_JOB___"

    def launch_job(self, latest_date):
        # Launch the sweep
        date = latest_date.strftime("%Y%m%d")
        output = check_output(
            ["make", "grid-nj", f"DATE={date}"], cwd=os.path.join(script_dir, "../../")
        )
        return output.decode("utf-8").strip().split("\n")[-1]

    def command(self):
        return super().command() + f" --kind sweep"


def main(args):
    parser = argparse.ArgumentParser()
    parser.add_argument("--install", action="store_true")
    parser.add_argument("--kind", choices=["cv", "sweep"], default="cv")
    opt = parser.parse_args()
    job = NJRecurring() if opt.kind == "cv" else NJSweepRecurring()

    if opt.install:
        job.install()  # install cron job
    else:
        job.refresh()  # run sweep if new data is available


if __name__ == "__main__":
    main(sys.argv[1:])