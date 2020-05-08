#!/usr/bin/env python3

import click
from glob import glob
import numpy as np
import pandas
import os
import re
import json
import sqlite3
from subprocess import check_call
import datetime
import itertools
import tempfile
import shutil
from data.usa.process_cases import get_nyt
import requests
from xml.etree import ElementTree


DB = "/checkpoint/mattle/covid19/forecast.db"


class MaxBy:
    def __init__(self):
        self.max_key = None
        self.max_val = None

    def step(self, key, value):
        if self.max_val is None or value > self.max_val:
            self.max_val = value
            self.max_key = key

    def finalize(self):
        return self.max_key


def adapt_date(x):
    if isinstance(x, datetime.datetime):
        x = x.date()
    return str(x)


def convert_date(s):
    return datetime.datetime.strptime(s.decode("utf-8"), "%Y-%m-%d").date()


sqlite3.register_adapter(datetime.date, adapt_date)
sqlite3.register_adapter(datetime.datetime, adapt_date)
sqlite3.register_converter("date", convert_date)


def regexp(expr, item):
    reg = re.compile(expr)
    return reg.search(item) is not None


def mk_db():
    conn = sqlite3.connect(DB)
    res = conn.execute(
        """
    CREATE TABLE IF NOT EXISTS infections(
        date date,
        loc1 text,
        loc2 text,
        loc3 text,
        counts real,
        id text,
        forecast_date date,
        UNIQUE(id, forecast_date, date, loc1, loc2, loc3) ON CONFLICT REPLACE
    );
    """
    )
    conn.execute("CREATE INDEX date_idx ON infections(date);")
    conn.execute("CREATE INDEX loc_1_idx ON infections(loc1);")
    conn.execute("CREATE INDEX loc_2_idx ON infections(loc2);")
    conn.execute("CREATE INDEX loc_3_idx ON infections(loc3);")
    conn.execute("CREATE INDEX id_idx ON infections(id);")
    conn.execute("CREATE INDEX forecast_date_idx ON infections(forecast_date);")
    res = conn.execute(
        """
    CREATE TABLE IF NOT EXISTS deaths(
        date date,
        loc1 text,
        loc2 text,
        loc3 text,
        counts real,
        id text,
        forecast_date date,
        UNIQUE(id, forecast_date, date, loc1, loc2, loc3) ON CONFLICT REPLACE
    );
    """
    )
    conn.execute("CREATE INDEX date_deaths_idx ON deaths(date);")
    conn.execute("CREATE INDEX loc_1_deaths_idx ON deaths(loc1);")
    conn.execute("CREATE INDEX loc_2_deaths_idx ON deaths(loc2);")
    conn.execute("CREATE INDEX loc_3_deaths_idx ON deaths(loc3);")
    conn.execute("CREATE INDEX id_deaths_idx ON deaths(id);")
    conn.execute("CREATE INDEX forecast_date_deaths_idx ON deaths(forecast_date);")


@click.group()
def cli():
    pass


LOC_MAP = {"new-jersey": "New Jersey", "nys": "New York"}


def sync_max_forecasts(conn, remote_dir, local_dir):
    check_call(
        ["scp", f"{remote_dir}/new-jersey/forecast-[0-9]*_fast.csv", "."],
        cwd=f"{local_dir}/new-jersey",
    )
    check_call(
        ["scp", f"{remote_dir}/new-jersey/forecast-[0-9]*_slow.csv", "."],
        cwd=f"{local_dir}/new-jersey",
    )
    check_call(
        ["scp", f"{remote_dir}/nys/forecast-[0-9]*_fast.csv", "."],
        cwd=f"{local_dir}/nys",
    )
    check_call(
        ["scp", f"{remote_dir}/nys/forecast-[0-9]*_slow.csv", "."],
        cwd=f"{local_dir}/nys",
    )
    files = glob(f"local_dir/new-jersey/forecast-*_(fast|slow).csv")
    for state, ty in itertools.product(["new-jersey", "nys"], ["slow", "fast"]):
        files = glob(f"{local_dir}/{state}/forecast-*_{ty}.csv")
        for f in files:
            forecast_date = re.search("forecast-(\d+)_", f).group(1)
            forecast_date = datetime.datetime.strptime(forecast_date, "%Y%m%d").date()
            res = conn.execute(
                f"SELECT COUNT(1) FROM infections WHERE forecast_date=? AND id=?",
                (forecast_date, f"{state}_{ty}"),
            )
            if res.fetchone()[0] == 0:
                df = pandas.read_csv(f)
                df = df[df["date"].str.match("\d{2}/\d{2}")]
                df["date"] = pandas.to_datetime(df["date"] + "/2020")
                df = df.melt(id_vars=["date"], value_name="counts", var_name="location")
                df = df.rename(columns={"location": "loc3"})
                df["forecast_date"] = forecast_date
                df["id"] = f"{state}_{ty}"
                df["loc2"] = LOC_MAP[state]
                df["loc1"] = "United States"
                state_agg = (
                    df.groupby(["loc2", "date", "loc1", "id", "forecast_date"])
                    .counts.sum()
                    .reset_index()
                )
                df = pandas.concat([df, state_agg])
                df.to_sql(name="infections", index=False, con=conn, if_exists="append")


def sync_nyt(conn):
    # Sync the NYTimes ground truth data
    conn.execute("DELETE FROM infections WHERE id='nyt_ground_truth'")
    conn.execute("DELETE FROM deaths WHERE id='nyt_ground_truth'")

    def dump(df, metric):
        df = df.reset_index().melt(
            id_vars=["date"], value_name="counts", var_name="loc2"
        )
        df["loc3"] = df["loc2"].apply(lambda x: x.split("_")[1])
        df["loc2"] = df["loc2"].apply(lambda x: x.split("_")[0])
        df["loc1"] = "United States"
        df["date"] = pandas.to_datetime(df["date"])
        df["id"] = "nyt_ground_truth"
        # Aggregate to state level
        state = df.groupby(["loc1", "loc2", "date", "id"]).counts.sum().reset_index()
        # Aggregate to country level
        country = df.groupby(["loc1", "date", "id"]).counts.sum().reset_index()
        df = pandas.concat([df, state, country])
        df.to_sql(name=metric, index=False, con=conn, if_exists="append")

    dump(get_nyt(metric="cases"), "infections")
    dump(get_nyt(metric="deaths"), "deaths")


def get_ihme_file(dir):
    """
    There's some inconsistent naming conventions for the CSV files containing the forecasts.
    This function tries to resolve these inconsistencies.
    """
    csvs = glob(os.path.join(dir, "*/*.csv"))
    if len(csvs) > 1:
        csvs = [f for f in csvs if "hospitalization" in os.path.basename(f).lower()]
    if len(csvs) == 1:
        return csvs[0]
    import pdb

    pdb.set_trace()
    if len(csvs) == 0:
        raise ValueError("No CSVs found in IHME zip!")
    else:
        raise ValueError(f"Ambiguous CSVs in IHME zip!  Found {len(csvs)} CSV files")


def sync_ihme(conn):
    marker = None
    while True:
        url = "https://ihmecovid19storage.blob.core.windows.net/archive?comp=list"
        if marker:
            url += f"&marker={marker}"
        req = requests.get(url)
        req.raise_for_status()
        tree = ElementTree.fromstring(req.content)

        basedir = f'/checkpoint/{os.environ["USER"]}/covid19/data/ihme'
        os.makedirs(basedir, exist_ok=True)
        states = pandas.read_csv(
            "https://raw.githubusercontent.com/jasonong/List-of-US-States/master/states.csv"
        )
        states = states[["State"]].rename(columns={"State": "loc2"})
        for elem in tree.findall(".//Blob"):
            if elem.find("Url").text.endswith("ihme-covid19.zip"):
                forecast_date = datetime.datetime.strptime(
                    elem.find("Url").text.split("/")[-2], "%Y-%m-%d"
                ).date()

                os.makedirs(os.path.join(basedir, str(forecast_date)), exist_ok=True)
                zip_file = os.path.join(basedir, str(forecast_date), "ihme-covid19.zip")
                if not os.path.exists(zip_file):
                    check_call(["wget", "-O", zip_file, elem.find("Url").text])
                    shutil.unpack_archive(
                        zip_file, extract_dir=os.path.join(basedir, str(forecast_date))
                    )
                stats_file = get_ihme_file(os.path.join(basedir, str(forecast_date)))
                stats = pandas.read_csv(stats_file).rename(
                    columns={"date_reported": "date"}
                )
                df = states.merge(stats, left_on="loc2", right_on="location_name")[
                    ["loc2", "date", "deaths_mean"]
                ]
                df = df.dropna().rename(columns={"deaths_mean": "counts"})

                # Unfortunately, they don't explictly say what the forecast date is.  Here we try to infer it.
                if "confirmed_infections" in stats.columns:
                    # If we have a confirmed_infectiosn column.  Take the last date this is non-null for
                    forecast_date = stats[~stats["confirmed_infections"].isnull()][
                        "date"
                    ].max()
                else:
                    continue  # not sure this is sufficient for determining forecast_date
                    # This is a pretty hacky way of determining what the actual forecast date is
                    # Find the latest date that has all whole number `deaths_mean` and at least
                    # one non-zero deaths_mean
                    stats["deaths_even"] = stats["deaths_mean"] % 1 == 0
                    stats["deaths_gt_0"] = stats["deaths_mean"] > 0
                    grouped = stats.groupby("date")["deaths_even"].all().reset_index()
                    grouped = grouped.merge(
                        stats.groupby("date")["deaths_gt_0"].any().reset_index()
                    )
                    forecast_date = grouped[
                        grouped["deaths_even"] & grouped["deaths_gt_0"]
                    ]["date"].max()
                print(forecast_date)

                df["loc1"] = "United States"
                df["forecast_date"] = forecast_date
                df["id"] = "IHME"
                df.to_sql(name="deaths", index=False, con=conn, if_exists="append")
        marker = tree.find("NextMarker").text
        if marker is None:
            break


@click.command()
def sync_forecasts():
    if not os.path.exists(DB):
        mk_db()
    conn = sqlite3.connect(DB)
    conn.create_function("REGEXP", 2, regexp)
    remote_dir = "devfairh1:/private/home/maxn/covid19_spread/forecasts"
    local_dir = f'/checkpoint/{os.environ["USER"]}/covid19/forecasts'
    sync_max_forecasts(conn, remote_dir, local_dir)
    sync_nyt(conn)
    sync_ihme(conn)

    check_call(
        [
            "scp",
            f'/checkpoint/{os.environ["USER"]}/covid19/forecast.db',
            f'devfairh1:/checkpoint/{os.environ["USER"]}/covid19/forecast.db',
        ]
    )


if __name__ == "__main__":
    cli.add_command(sync_forecasts)
    cli()
