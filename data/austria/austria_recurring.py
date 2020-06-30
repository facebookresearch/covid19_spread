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
import pandas


MAIL_TO = ["mattle@fb.com", "lematt1991@gmail.com"]
if os.environ.get("__PROD__") == "1":
    MAIL_TO.append("maxn@fb.com")

print(f"MAIL_TO == {MAIL_TO}")


class AustriaRecurring(recurring.Recurring):
    script_dir = script_dir

    def get_id(self):
        return "austria-ar"

    def schedule(self) -> str:
        """Cron schedule"""
        return "0 9 * * Thu,Sun"  # run at Noon (EST) every thursday and sunday

    def command(self):
        return f"python {os.path.realpath(__file__)} --kind ar"

    def module(self):
        return "ar"

    def update_data(self):
        # Avoid `git pull` since it could fail if the repo isn't in a clean state
        df = pandas.read_csv(
            "https://raw.githubusercontent.com/fairinternal/covid19_spread/master/data/austria/data.csv?token=ADEIXC5LV6KSB7ZAIR2Z5RS67ISSU",
            index_col="region",
        )
        df = df.cummax(axis=1)
        df.to_csv("data.csv", index_label="region")

    def latest_date(self):
        df = pandas.read_csv("data.csv", index_col="region")
        return pandas.to_datetime(df.columns).max().date()

    def launch_job(self, **kwargs):
        return super().launch_job(module="ar", cv_config="at", **kwargs)


class AustriaDaily(AustriaRecurring):
    def schedule(self) -> str:
        """Cron schedule"""
        return "0 9 * * Mon,Tue,Wed,Fri,Sat"  # run on alternate days

    def get_id(self):
        return "austria-ar-daily"

    def module(self):
        return "ar_daily"

    def command(self):
        return f"python {os.path.realpath(__file__)} --kind ar-daily"


def main(args):
    kinds = {"ar": AustriaRecurring, "ar-daily": AustriaDaily}

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
