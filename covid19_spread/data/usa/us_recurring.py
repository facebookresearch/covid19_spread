#!/usr/bin/env python3
# Copyright (c) 2021-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
from .. import recurring
import pandas
from ...lib.slack import get_client as get_slack_client
from datetime import date, datetime, timedelta
from .convert import main as convert

SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))


class USARRecurring(recurring.Recurring):
    script_dir = SCRIPT_DIR

    def get_id(self):
        return "us-bar"

    def command(self):
        return f"recurring run us"

    def module(self):
        return "bar_time_features"

    def schedule(self):
        return "*/5 * * * *"

    def update_data(self):
        convert("cases", with_features=False, source="nyt", resolution="county")

    def latest_date(self):
        df = pandas.read_csv(f"{SCRIPT_DIR}/data_cases.csv", index_col="region")
        max_date = pandas.to_datetime(df.columns).max().date()
        if max_date < (date.today() - timedelta(days=1)) and datetime.now().hour > 17:
            expected_date = date.today() - timedelta(days=1)
            client = get_slack_client()
            msg = f"*WARNING: new data for {expected_date} is still not available!*"
            client.chat_postMessage(channel="#cron_errors", text=msg)
        return pandas.to_datetime(df.columns).max().date()

    def launch_job(self, **kwargs):
        # Make clean with features
        convert("cases", with_features=True, source="nyt", resolution="county")
        client = get_slack_client()
        msg = f"*New Data Available for US: {self.latest_date()}*"
        client.chat_postMessage(channel="#new-data", text=msg)
        return super().launch_job(
            module="bar", cv_config="us_prod", array_parallelism=90, **kwargs
        )
