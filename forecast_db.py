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


DB = f'/private/home/{os.environ["USER"]}/covid19_spread/forecasts/forecast.db'


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


def mk_db():
    conn = sqlite3.connect(DB)
    # forecast_date is the last date that we have ground truth data for.
    # i.e. the last date the model sees during training
    res = conn.execute(
        """
    CREATE TABLE IF NOT EXISTS infections(
        date date NOT NULL,
        loc1 text,
        loc2 text,
        loc3 text,
        counts real NOT NULL,
        id text NOT NULL,
        forecast_date date
    );
    """
    )
    conn.execute(
        "CREATE UNIQUE INDEX unique_infections ON infections(id, forecast_date, date, ifnull(loc1, 0), ifnull(loc2, 0), ifnull(loc3, 0))"
    )
    conn.execute("CREATE INDEX date_idx ON infections(date);")
    conn.execute("CREATE INDEX loc_idx ON infections(loc1, loc2, loc3);")
    conn.execute("CREATE INDEX id_idx ON infections(id);")
    conn.execute("CREATE INDEX forecast_date_idx ON infections(forecast_date);")
    res = conn.execute(
        """
    CREATE TABLE IF NOT EXISTS deaths(
        date date NOT NULL,
        loc1 text,
        loc2 text,
        loc3 text,
        counts real NOT NULL,
        id text NOT NULL,
        forecast_date date
    );
    """
    )
    conn.execute(
        "CREATE UNIQUE INDEX unique_deaths ON deaths(id, forecast_date, date, ifnull(loc1, 0), ifnull(loc2, 0), ifnull(loc3, 0))"
    )
    conn.execute("CREATE INDEX date_deaths_idx ON deaths(date);")
    conn.execute("CREATE INDEX loc_deaths_idx ON deaths(loc1, loc2, loc3);")
    conn.execute("CREATE INDEX id_deaths_idx ON deaths(id);")
    conn.execute("CREATE INDEX forecast_date_deaths_idx ON deaths(forecast_date);")


@click.group()
def cli():
    pass


LOC_MAP = {"new-jersey": "New Jersey", "nystate": "New York"}


def to_sql(conn, df, table):
    df.to_sql("temp____", conn, if_exists="replace", index=False)
    cols = ", ".join(df.columns)
    conn.execute(f"INSERT OR REPLACE INTO {table}({cols}) SELECT {cols} FROM temp____")


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
        ["scp", f"{remote_dir}/nystate/forecast-[0-9]*_fast.csv", "."],
        cwd=f"{local_dir}/nystate",
    )
    check_call(
        ["scp", f"{remote_dir}/nystate/forecast-[0-9]*_slow.csv", "."],
        cwd=f"{local_dir}/nystate",
    )
    files = glob(f"local_dir/new-jersey/forecast-*_(fast|slow).csv")
    for state, ty in itertools.product(["new-jersey", "nystate"], ["slow", "fast"]):
        files = glob(f"{local_dir}/{state}/forecast-*_{ty}.csv")
        for f in files:
            forecast_date = re.search("forecast-(\d+)_", f).group(1)
            forecast_date = datetime.datetime.strptime(forecast_date, "%Y%m%d").date()
            forecast_date -= datetime.timedelta(days=1)
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
                df = df[df["loc3"] != "ALL REGIONS"]
                to_sql(conn, df, "infections")


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
        to_sql(conn, df, metric)

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
                # Filter out only the US states
                df = states.merge(stats, left_on="loc2", right_on="location_name")[
                    ["loc2", "date", "totdea_mean"]
                ]
                df = df[~df["totdea_mean"].isnull()].rename(
                    columns={"totdea_mean": "counts"}
                )

                # Unfortunately, they don't explictly say what the forecast date is.  Here we try to infer it.
                if "confirmed_infections" in stats.columns:
                    # If we have a confirmed_infectiosn column.  Take the last date this is non-null for
                    forecast_date = stats[~stats["confirmed_infections"].isnull()][
                        "date"
                    ].max()
                else:
                    # continue  # not sure this is sufficient for determining forecast_date
                    # This is a pretty hacky way of determining what the actual forecast date is
                    # Find the latest date that has all whole number `totdea_mean` and at least
                    # one non-zero totdea_mean
                    temp = df.copy()
                    temp["nonzero"] = temp["counts"] > 0
                    temp["round"] = temp["counts"] % 1 == 0

                    grouped = temp.groupby("date")["round"].all().reset_index()
                    grouped = grouped.merge(
                        temp.groupby("date")["nonzero"].any().reset_index()
                    )
                    forecast_date = grouped[grouped["round"] & grouped["nonzero"]][
                        "date"
                    ].max()
                print(forecast_date)
                df["loc1"] = "United States"
                df["forecast_date"] = forecast_date
                df["id"] = "IHME"
                to_sql(conn, df, "deaths")
        marker = tree.find("NextMarker").text
        if marker is None:
            break


def sync_los_alamos(conn):
    url = "https://covid-19.bsvgateway.org"
    req = requests.get(f"{url}/forecast/forecast_metadata.json").json()

    def fmt(df_):

        df = df_[["dates", "q.50", "state"]].rename(
            columns={"dates": "date", "q.50": "counts", "state": "loc2"}
        )
        df["loc1"] = "United States"
        df["forecast_date"] = df_["fcst_date"].unique().item()
        df["id"] = "los_alamos"
        return df

    for date in req["us"]["files"].keys():
        cases = fmt(
            pandas.read_csv(
                os.path.join(
                    url, req["us"]["files"][date]["quantiles_confirmed"].lstrip("./")
                )
            )
        )
        deaths = fmt(
            pandas.read_csv(
                os.path.join(
                    url, req["us"]["files"][date]["quantiles_deaths"].lstrip("./")
                )
            )
        )
        to_sql(conn, deaths, "deaths")
        to_sql(conn, cases, "infections")


def dump_to_csv(conn, distribute):
    basedir = f'/checkpoint/{os.environ["USER"]}/covid19/csvs'

    def f(metric):
        q = f"SELECT counts, loc2, loc3, date, forecast_date, id FROM {metric}"
        df = pandas.read_sql(q, conn)
        for (model, forecast_date), group in df.fillna("").groupby(
            ["id", "forecast_date"]
        ):
            if forecast_date != "":
                forecast_date = "_" + forecast_date

            group = group[group["date"] >= group["forecast_date"]].copy()

            group["location"] = group.apply(
                lambda x: x.loc2 + (", " + x.loc3 if x.loc3 else ""), axis=1
            )
            group = group.pivot_table(
                columns=["location"], values=["counts"], index="date"
            )
            group.columns = group.columns.get_level_values(-1)
            outfile = os.path.join(basedir, metric, model, f"counts{forecast_date}.csv")
            os.makedirs(os.path.dirname(outfile), exist_ok=True)
            group.to_csv(outfile)

    f("deaths")
    f("infections")
    check_call(
        ["rsync", "--delete", "-av", basedir, f"devfairh1:{os.path.dirname(basedir)}"]
    )


@click.command()
@click.option(
    "--distribute",
    type=click.BOOL,
    default=False,
    help="Distribute across clusters (H1/H2)",
)
def sync_forecasts(distribute=False):
    if not os.path.exists(DB):
        mk_db()
    conn = sqlite3.connect(DB)
    remote_dir = "devfairh1:/private/home/maxn/covid19_spread/forecasts"
    local_dir = f'/checkpoint/{os.environ["USER"]}/covid19/forecasts'
    # sync_max_forecasts(conn, remote_dir, local_dir)
    sync_nyt(conn)
    sync_ihme(conn)
    sync_los_alamos(conn)
    conn.execute("REINDEX;")
    if distribute:
        DEST_DB = f"devfairh1:/private/home/{os.environ['USER']}/covid19_spread/forecasts/forecast.db"
        check_call(["scp", DB, DEST_DB])
    dump_to_csv(conn, distribute)


if __name__ == "__main__":
    cli.add_command(sync_forecasts)
    cli()
