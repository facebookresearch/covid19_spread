#!/usr/bin/env python3
# Copyright (c) 2021-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import requests
import json
import pandas
from subprocess import check_call
import os
from glob import glob
import zipfile
import re
from fuzzywuzzy import process
import sys
import argparse


LOC_MAP = {
    "Stadt Steyr": "Steyr(Stadt)",
    "Stadt Wels": "Wels(Stadt)",
    "Ried": "Ried im Innkreis",
    "Braunau": "Braunau am Inn",
    "Stadt Linz": "Linz(Stadt)",
    "Kirchdorf": "Kirchdorf an der Krems",
}


def merge(base, full_history, newest):
    date = str(newest["Timestamp"][0].date())
    print(f"Merging {date}")
    df = newest.rename(columns={"Bezirk": "loc", "Anzahl": date}).set_index("loc")[
        [date]
    ]

    timestamp = str(newest["Timestamp"][0])
    new_history = df.rename(columns={date: timestamp})

    if full_history is None:
        full_history = new_history

    if timestamp not in full_history.columns:
        full_history = full_history.merge(
            new_history, left_index=True, right_index=True, how="outer"
        )

    df.index = [LOC_MAP.get(c, c) for c in df.index]

    merged = base.merge(df, left_index=True, right_index=True)

    if f"{date}_x" in merged.columns:
        # Resolve merged columns
        merged[date] = merged[f"{date}_y"].combine_first(merged[f"{date}_x"])
        del merged[f"{date}_y"]
        del merged[f"{date}_x"]
        merged = merged[sorted(merged.columns)]

    if len(merged[merged[date].isnull()]) > 0:
        user = os.environ["USER"]
        nulls = list(merged[merged[date].isnull()].index)
        subject = f"WARNING: Austria dataset has null values for {date}"
        msg = f"NULL values include: {nulls}"
        check_call(f'echo "{msg}" | mail -s "{subject}" {user}@fb.com', shell=True)

    return merged, full_history


def backfill(base, full_history):
    script_dir = os.path.dirname(os.path.realpath(__file__))
    if not os.path.exists(f"{script_dir}/coronaDAT"):
        repo = "https://github.com/statistikat/coronaDAT.git"
        check_call(["git", "clone", repo, f"{script_dir}/coronaDAT"])
    check_call(["git", "pull"], cwd=f"{script_dir}/coronaDAT")
    dfs = []
    for csv in glob("coronaDAT/**/*csv.zip", recursive=True):
        if not re.match("\d+_\d+.*csv.zip", os.path.basename(csv)):
            continue
        with zipfile.ZipFile(csv) as zfile:
            with zfile.open("Bezirke.csv") as f:
                date = re.search("^(\d+)_.*", os.path.basename(csv)).group(1)
                time = re.search("^\d+_(\d+)_.*", os.path.basename(csv)).group(1)
                time = ":".join([time[i : i + 2] for i in range(0, len(time), 2)])
                if (
                    pandas.to_datetime(date)
                    <= pandas.to_datetime(full_history.columns).max()
                ):
                    continue
                df = pandas.read_csv(f, sep=";", usecols=["Bezirk", "Anzahl"])
                df["Timestamp"] = pandas.to_datetime(f"{date} {time}")
                base, full_history = merge(base, full_history, df)
    return base, full_history


def main(args):
    parser = argparse.ArgumentParser()
    parser.add_argument("--backfill", action="store_true")
    opt = parser.parse_args()

    check_call(["git", "pull"])

    df = pandas.read_csv("data.csv", index_col="region")
    if os.path.exists("full_history.csv"):
        full_history = pandas.read_csv("full_history.csv", index_col="region")
    else:
        full_history = None

    last_date = pandas.to_datetime(df.columns).max()

    if opt.backfill:
        df, full_history = backfill(df, full_history)
    else:
        newdata = pandas.read_csv(
            "https://info.gesundheitsministerium.at/data/Bezirke.csv",
            sep=";",
            parse_dates=["Timestamp"],
        )
        df, full_history = merge(df, full_history, newdata)

    full_history.to_csv("full_history.csv", index_label="region")

    df = df.cummax(axis=1)
    df.to_csv("data.csv", index_label="region")
    newest = str(pandas.to_datetime(df.columns).max().date())
    if pandas.to_datetime(df.columns).max().date() > last_date.date():
        check_call(["git", "add", "data.csv", "full_history.csv"])
        check_call(["git", "commit", "-m", f"Updating Austria data: {newest}"])
        check_call(["git", "push"])
    else:
        print(f"Already updated latest data: {newest}")


if __name__ == "__main__":
    main(sys.argv[1:])
