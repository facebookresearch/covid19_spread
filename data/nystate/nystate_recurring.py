#!/usr/bin/env python3


import sys
import os

script_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, os.path.join(script_dir, "../.."))
sys.path.insert(0, os.path.join(script_dir, ".."))
import recurring
import cv
import argparse
import tempfile
import numpy as np
from subprocess import check_call, check_output
import sqlite3
import process_cases
import pandas


class NYARRecurring(recurring.Recurring):
    script_dir = script_dir

    def get_id(self):
        return "nystate-ar"

    def command(self):
        return f"python {os.path.realpath(__file__)} --kind ar"

    def module(self):
        return "ar"

    # Create the new timeseries.h5 dataset
    def update_data(self):
        URL = "https://health.data.ny.gov/api/views/xdss-u53e/rows.csv?accessType=DOWNLOAD"
        process_cases.main(fout="timeseries", fin=URL)

    def latest_date(self):
        df = pandas.read_csv("data_cases.csv", index_col="region")
        return pandas.to_datetime(df.columns).max().date()

    def launch_job(self, **kwargs):
        return super().launch_job(module="ar", cv_config="ny", **kwargs)


def main(args):
    kinds = {
        "ar": NYARRecurring,
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
