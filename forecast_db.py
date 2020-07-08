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
from bs4 import BeautifulSoup
import yaml
from lib import cluster
from data.recurring import DB as RECURRING_DB


DB = os.path.join(os.path.dirname(os.path.realpath(__file__)), "forecasts/forecast.db")


CLUSTERS = {
    "H1": "devfairh2",
    "H2": "devfairh1",
}


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
    os.makedirs(os.path.dirname(DB), exist_ok=True)
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
        "CREATE UNIQUE INDEX unique_infections ON infections(id, ifnull(forecast_date, 0), date, ifnull(loc1, 0), ifnull(loc2, 0), ifnull(loc3, 0))"
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
        "CREATE UNIQUE INDEX unique_deaths ON deaths(id, ifnull(forecast_date, 0), date, ifnull(loc1, 0), ifnull(loc2, 0), ifnull(loc3, 0))"
    )
    conn.execute("CREATE INDEX date_deaths_idx ON deaths(date);")
    conn.execute("CREATE INDEX loc_deaths_idx ON deaths(loc1, loc2, loc3);")
    conn.execute("CREATE INDEX id_deaths_idx ON deaths(id);")
    conn.execute("CREATE INDEX forecast_date_deaths_idx ON deaths(forecast_date);")
    conn.execute("CREATE TABLE gt_mapping (id text, gt text);")


@click.group()
def cli():
    pass


LOC_MAP = {"new-jersey": "New Jersey", "nystate": "New York"}


def to_sql(conn, df, table):
    cols = ["date", "loc1", "loc2", "loc3", "forecast_date", "counts", "id"]
    df = df[[c for c in cols if c in df.columns]]
    df.to_sql("temp____", conn, if_exists="replace", index=False)
    cols = ", ".join(df.columns)
    conn.execute(f"INSERT OR REPLACE INTO {table}({cols}) SELECT {cols} FROM temp____")
    conn.commit()


def create_gt_mapping(conn):
    df = pandas.DataFrame(
        [
            {"id": "yyg", "gt": "jhu_ground_truth"},
            {"id": "mit-delphi", "gt": "jhu_ground_truth"},
            {"id": "cv_ar", "gt": "nyt_ground_truth"},
            {"id": "cv_ar_daily", "gt": "nyt_ground_truth"},
            {"id": "new-jersey_fast", "gt": "nyt_ground_truth"},
            {"id": "new-jersey_slow", "gt": "nyt_ground_truth"},
            {"id": "nystate_fast", "gt": "nyt_ground_truth"},
            {"id": "nystate_slow", "gt": "nyt_ground_truth"},
            {"id": "los_alamos", "gt": "jhu_ground_truth"},
        ]
    )
    df.to_sql("temp___", conn, if_exists="replace", index=False)
    cols = "id, gt"
    conn.execute(
        f"INSERT OR REPLACE INTO gt_mapping({cols}) SELECT {cols} FROM temp___"
    )
    conn.commit()


def sync_max_forecasts(conn):
    base = "/checkpoint/mattle/covid19/final_forecasts"
    for state, ty in itertools.product(["new-jersey", "nystate"], ["slow", "fast"]):
        files = glob(f"{base}/{state}/maxn/forecast-*_{ty}.csv")
        for f in files:
            df = pandas.read_csv(f)
            df = df[df["date"].str.match("\d{2}/\d{2}")]
            df["date"] = pandas.to_datetime(df["date"] + "/2020")
            forecast_date = df["date"].min().date()
            res = conn.execute(
                f"SELECT COUNT(1) FROM infections WHERE forecast_date=? AND id=?",
                (forecast_date, f"{state}_{ty}"),
            )
            if res.fetchone()[0] == 0:
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
    if any(["Best_mask_hospitalization_all_locs.csv" in f for f in csvs]):
        csvs = [f for f in csvs if "Best_mask_hospitalization_all_locs.csv" in f]
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


def sync_reich_forecast(conn, name, mdl_id):
    user = os.environ["USER"]
    data_dir = f"{cluster.FS}/{user}/covid19/data/covid19-forecast-hub"
    os.makedirs(os.path.dirname(data_dir), exist_ok=True)
    if not os.path.exists(data_dir):
        check_call(
            [
                "git",
                "clone",
                "https://github.com/reichlab/covid19-forecast-hub.git",
                data_dir,
            ]
        )
    check_call(["git", "pull"], cwd=data_dir)
    loc_codes = pandas.read_csv(f"{data_dir}/data-locations/locations.csv")

    for pth in glob(f"{data_dir}/data-processed/{name}/*.csv"):
        value = pandas.read_csv(pth, dtype={"location": str})
        value = value[
            (value["type"] == "point")
            & (value["target"].str.match("\d wk ahead cum death"))
            & (value["location"].str.match("\d\d"))
        ].copy()
        value = value.merge(loc_codes, on="location")
        value["days"] = value["target"].str.extract("(\d+) wk")
        value = value[value["days"] == value["days"].max()]

        value = value.rename(
            columns={
                "target_end_date": "date",
                "location_name": "loc2",
                "value": "counts",
            }
        )
        value["loc1"] = "United States"
        value["id"] = mdl_id
        value = value.drop(columns=["target", "location", "type", "quantile"])
        value = value[["date", "loc1", "loc2", "counts", "id", "forecast_date"]]
        to_sql(conn, value, "deaths")


def sync_mit(conn):
    sync_reich_forecast(conn, "MIT_CovidAnalytics-DELPHI", "mit-delphi")


def sync_yyg(conn):
    sync_reich_forecast(conn, "YYG-ParamSearch", "yyg")


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

    def f(metric, deltas=False):
        q = f"SELECT counts, loc2, loc3, date, forecast_date, id FROM {metric}"
        if deltas:
            metric = f"{metric}_deltas"
        df = pandas.read_sql(q, conn, parse_dates=["date", "forecast_date"])
        for (model, forecast_date), group in df.fillna("").groupby(
            ["id", "forecast_date"]
        ):
            dt = pandas.to_datetime(forecast_date if forecast_date else 0)
            if forecast_date != "":
                forecast_date = "_" + str(forecast_date.date())

            group["location"] = group.apply(
                lambda x: x.loc2 + (", " + x.loc3 if x.loc3 else ""), axis=1
            )
            group = group.pivot_table(
                columns=["location"], values=["counts"], index="date"
            )
            group.columns = group.columns.get_level_values(-1)

            if deltas:
                group = group.diff()
                if dt not in group.index:
                    print(
                        f"Warning: forecast_date not in forecast for {model}, {forecast_date}"
                    )
            group = group[group.index > dt]

            outfile = os.path.join(basedir, metric, model, f"counts{forecast_date}.csv")
            os.makedirs(os.path.dirname(outfile), exist_ok=True)
            group.to_csv(outfile)

    f("deaths", True)
    f("infections", True)
    f("deaths", False)
    f("infections", False)

    if distribute:
        ssh_alias = CLUSTERS[cluster.FAIR_CLUSTER]
        check_call(
            [
                "rsync",
                "--delete",
                "-av",
                basedir,
                f"{ssh_alias}:{os.path.dirname(basedir)}",
            ]
        )


def sync_matts_forecasts(conn):
    if not os.path.exists(RECURRING_DB):
        return
    loc_map = {
        "new-jersey": {"loc1": "United States", "loc2": "New Jersey"},
        "nystate": {"loc1": "United States", "loc2": "New York"},
        "at": {"loc1": "Austria"},
    }
    module_map = {"ar": "ar", "ar_daily": "ar"}
    _conn = sqlite3.connect(RECURRING_DB)
    q = """
    SELECT path, basedate, module
    FROM sweeps
    WHERE module IN ('ar', 'ar_daily')
    """
    for sweep_pth, basedate, module in _conn.execute(q):
        if not os.path.exists(
            os.path.join(sweep_pth, "forecasts/forecast_best_mae.csv")
        ):
            continue
        cfg = yaml.safe_load(open(os.path.join(sweep_pth, "cfg.yml")))
        print(f"{basedate}, {module}")
        df = pandas.read_csv(
            os.path.join(sweep_pth, "forecasts/forecast_best_mae.csv"),
            parse_dates=["date"],
        )
        df = df.melt(id_vars=["date"], value_name="counts", var_name="location")
        loc = loc_map[cfg["region"]]
        for k, v in loc.items():
            df[k] = v
        df = df.rename(columns={"location": f"loc{len(loc) + 1}"})
        df["id"] = f"cv_{module_map[module]}"
        df["forecast_date"] = basedate
        to_sql(conn, df, "infections")


def sync_austria_gt(conn):
    user = os.environ["USER"]
    data_dir = f"/checkpoint/{user}/covid19/data/covid19_spread"
    if not os.path.exists(data_dir):
        check_call(
            ["git", "clone", "git@github.com:fairinternal/covid19_spread.git", data_dir]
        )
    check_call(["git", "pull"], cwd=data_dir)
    df = pandas.read_csv(
        f"{data_dir}/data/austria/data.csv", index_col="region"
    ).transpose()
    df.index = pandas.to_datetime(df.index)
    df.index.name = "date"
    df = df.reset_index()
    df = df.melt(id_vars=["date"], value_name="counts", var_name="loc2")
    df["loc1"] = "Austria"
    df["id"] = "austria_ground_truth"
    to_sql(conn, df, "infections")


def sync_jhu(conn):
    user = os.environ["USER"]
    data_pth = f"{cluster.FS}/{user}/covid19/data/jhu_covid_data"
    if not os.path.exists(data_pth):
        check_call(
            ["git", "clone", "https://github.com/CSSEGISandData/COVID-19.git", data_pth]
        )
    check_call(["git", "pull"], cwd=data_pth)
    col_map = {
        "Country/Region": "loc1",
        "Province/State": "loc2",
        "Last Update": "date",
        "Last_Update": "date",
        "Admin2": "loc3",
        "Province_State": "loc2",
        "Country_Region": "loc1",
    }
    for file in glob(
        f"{data_pth}/csse_covid_19_data/csse_covid_19_daily_reports/*.csv"
    ):
        print(file)
        try:
            df = pandas.read_csv(file)
            df = df.rename(columns=col_map)
            df["date"] = pandas.to_datetime(df["date"])
            df["id"] = "jhu_ground_truth"
            df["date"] = df["date"].dt.date
            to_sql(
                conn,
                df[~df["Confirmed"].isnull()].rename(columns={"Confirmed": "counts"}),
                "infections",
            )
            to_sql(
                conn,
                df[~df["Deaths"].isnull()].rename(columns={"Deaths": "counts"}),
                "deaths",
            )
        except Exception as e:
            import pdb

            pdb.set_trace()


@click.command()
@click.option(
    "--distribute", is_flag=True, help="Distribute across clusters (H1/H2)",
)
def sync_forecasts(distribute=False):
    if not os.path.exists(DB):
        mk_db()
    conn = sqlite3.connect(DB)
    sync_jhu(conn)
    sync_austria_gt(conn)
    sync_matts_forecasts(conn)
    sync_max_forecasts(conn)
    sync_nyt(conn)
    sync_ihme(conn)
    sync_los_alamos(conn)
    sync_mit(conn)
    sync_yyg(conn)
    conn.execute("REINDEX;")
    if distribute:
        ssh_alias = CLUSTERS[cluster.FAIR_CLUSTER]
        DEST_DB = f"{ssh_alias}:/private/home/{os.environ['USER']}/covid19_spread/forecasts/forecast.db"
        check_call(["scp", DB, DEST_DB])
    dump_to_csv(conn, distribute)


if __name__ == "__main__":
    cli.add_command(sync_forecasts)
    cli()
