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
from subprocess import check_call, check_output
import pandas
from lib.slack import get_client as get_slack_client
from datetime import date, datetime, timedelta


MAIL_TO = ["mattle@fb.com"]
if os.environ.get("__PROD__") == "1":
    MAIL_TO.append("maxn@fb.com")

print(f"MAIL_TO == {MAIL_TO}")


class USARRecurring(recurring.Recurring):
    script_dir = script_dir

    def get_id(self):
        return "us-bar"

    def command(self):
        return f"python {os.path.realpath(__file__)} --kind ar"

    def module(self):
        return "bar_time_features"

    def schedule(self):
        return "*/5 * * * *"

    def update_data(self):
        check_call(["python", "convert.py", "-source", "nyt", "-metric", "cases"])

    def latest_date(self):
        df = pandas.read_csv("data_cases.csv", index_col="region")
        max_date = pandas.to_datetime(df.columns).max().date()
        if max_date < (date.today() - timedelta(days=1)) and datetime.now().hour > 17:
            expected_date = date.today() - timedelta(days=1)
            client = get_slack_client()
            msg = f"*WARNING: new data for {expected_date} is still not available!*"
            client.chat_postMessage(channel="#cron_errors", text=msg)
        return pandas.to_datetime(df.columns).max().date()

    def launch_job(self, **kwargs):
        # Make clean with features
        check_call(["make", "clean"], cwd=script_dir)
        check_call(["python", "convert.py", "-source", "nyt"], cwd=script_dir)
        check_call(["make", "data_cases.csv", "-j", "5"], cwd=script_dir)
        client = get_slack_client()
        msg = f"*New Data Available for US: {self.latest_date()}*"
        client.chat_postMessage(channel="#new-data", text=msg)
        return super().launch_job(
            module="bar", cv_config="us_prod", array_parallelism=90, **kwargs
        )


def main(args):
    kinds = {
        "ar": USARRecurring,
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
