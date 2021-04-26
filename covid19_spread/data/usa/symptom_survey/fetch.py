#!/usr/bin/env python3
# Copyright (c) 2021-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import pandas as pd
import sys
from datetime import timedelta
from delphi_epidata import Epidata
import covidcast

# Fetch data

SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))


def main(geo_value, source, signal):
    # grab start and end date from metadata
    df = covidcast.metadata().drop(columns=["time_type", "min_lag", "max_lag"])
    df.min_time = pd.to_datetime(df.min_time)
    df.max_time = pd.to_datetime(df.max_time)
    df = df.query(
        f"data_source == '{source}' and signal == '{signal}' and geo_type == '{geo_value}'"
    )
    assert len(df) == 1
    base_date = df.iloc[0].min_time - timedelta(1)
    end_date = df.iloc[0].max_time
    dfs = []
    current_date = base_date
    while current_date < end_date:
        current_date = current_date + timedelta(1)
        date_str = current_date.strftime("%Y%m%d")
        os.makedirs(os.path.join(SCRIPT_DIR, geo_value, source), exist_ok=True)
        fout = f"{SCRIPT_DIR}/{geo_value}/{source}/{signal}-{date_str}.csv"

        # d/l only if we don't have the file already
        if os.path.exists(fout):
            dfs.append(pd.read_csv(fout))
            continue

        for _ in range(3):
            res = Epidata.covidcast(source, signal, "day", geo_value, [date_str], "*")
            print(date_str, res["result"], res["message"])
            if res["result"] == 1:
                break

        assert res["result"] == 1, ("CLI", res["message"], res)

        df = pd.DataFrame(res["epidata"])
        df.rename(
            columns={
                "geo_value": geo_value,
                "time_value": "date",
                "value": signal,
                "direction": f"{signal}_direction",
                "stderr": f"{signal}_stderr",
                "sample_size": f"{signal}_sample_size",
            },
            inplace=True,
        )
        df.to_csv(fout, index=False)
        dfs.append(df)
    pd.concat(dfs).to_csv(f"{SCRIPT_DIR}/{geo_value}/{source}/{signal}.csv")


SIGNALS = [
    ("fb-survey", "smoothed_hh_cmnty_cli"),
    ("fb-survey", "smoothed_wcli"),
    ("doctor-visits", "smoothed_adj_cli"),
    ("fb-survey", "smoothed_wcovid_vaccinated_or_accept"),
    ("fb-survey", "smoothed_wearing_mask"),
    ("fb-survey", "smoothed_wearing_mask_7d"),
    ("fb-survey", "smoothed_wothers_masked"),
    ("fb-survey", "smoothed_wcovid_vaccinated_or_accept"),
]

if __name__ == "__main__":
    main(sys.argv[1], *sys.argv[2].split("/"))
