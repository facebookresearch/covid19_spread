#!/usr/bin/env python3


import sys
import os
from .. import recurring
import argparse
from subprocess import check_call, check_output
import pandas
from ...lib.slack import get_client as get_slack_client
from datetime import date, datetime, timedelta

SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))


class USARRecurring(recurring.Recurring):
    script_dir = SCRIPT_DIR

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
        check_call(["make", "clean"], cwd=SCRIPT_DIR)
        check_call(["python", "convert.py", "-source", "nyt"], cwd=SCRIPT_DIR)
        check_call(["make", "data_cases.csv", "-j", "5"], cwd=SCRIPT_DIR)
        client = get_slack_client()
        msg = f"*New Data Available for US: {self.latest_date()}*"
        client.chat_postMessage(channel="#new-data", text=msg)
        return super().launch_job(
            module="bar", cv_config="us_prod", array_parallelism=90, **kwargs
        )
