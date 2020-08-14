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
from data.usa.process_cases import get_nyt, get_google
import requests
from xml.etree import ElementTree
import yaml
from lib import cluster
from data.recurring import DB as RECURRING_DB
from sqlalchemy import create_engine
import psycopg2
from datetime import timedelta


DB = os.path.join(os.path.dirname(os.path.realpath(__file__)), "forecasts/forecast.db")


CLUSTERS = {
    "H1": "devfairh2",
    "H2": "devfairh1",
}


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
        "CREATE UNIQUE INDEX unique_infections ON infections(id, date, COALESCE(forecast_date, '1970-01-01'), COALESCE(loc1, ''), COALESCE(loc2, ''), COALESCE(loc3, ''))"
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
        "CREATE UNIQUE INDEX unique_deaths ON deaths(id, date, COALESCE(forecast_date, '1970-01-01'), COALESCE(loc1, ''), COALESCE(loc2, ''), COALESCE(loc3, ''))"
    )
    conn.execute("CREATE INDEX date_deaths_idx ON deaths(date);")
    conn.execute("CREATE INDEX loc_deaths_idx ON deaths(loc1, loc2, loc3);")
    conn.execute("CREATE INDEX id_deaths_idx ON deaths(id);")
    conn.execute("CREATE INDEX forecast_date_deaths_idx ON deaths(forecast_date);")
    conn.execute("CREATE TABLE gt_mapping (id text, gt text, UNIQUE(id, gt));")
    # conn_.commit()


def update_repo(repo):
    user = os.environ["USER"]
    match = re.search(r"([^(\/|:)]+)/([^(\/|:)]+)\.git", repo)
    name = f"{match.group(1)}_{match.group(2)}"
    data_pth = f"{cluster.FS}/{user}/covid19/data/{name}"
    if not os.path.exists(data_pth):
        check_call(["git", "clone", repo, data_pth])
    check_call(["git", "pull"], cwd=data_pth)
    return data_pth


@click.group()
def cli():
    pass


LOC_MAP = {"new-jersey": "New Jersey", "nystate": "New York"}


def to_sql(conn, df, table):
    cols = ["date", "loc1", "loc2", "loc3", "forecast_date", "counts", "id"]
    df = df[[c for c in cols if c in df.columns]]
    if hasattr(df["date"], "dt"):
        df["date"] = df["date"].dt.date
    if "forecast_date" in df.columns and hasattr(df["forecast_date"], "dt"):
        df["forecast_date"] = df["forecast_date"].dt.date
    df.to_sql("temp____", conn, if_exists="replace", index=False)
    cols = ", ".join(df.columns)
    conn.execute(f"INSERT OR REPLACE INTO {table}({cols}) SELECT {cols} FROM temp____")


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
            df = df[df["date"].str.match(r"\d{2}/\d{2}")]
            df["date"] = pandas.to_datetime(df["date"] + "/2020")
            forecast_date = df["date"].min().date()
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
                df["date"] = pandas.to_datetime(df["date"])
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
                df["forecast_date"] = pandas.to_datetime(forecast_date)
                df["id"] = "IHME"
                to_sql(conn, df, "deaths")
        marker = tree.find("NextMarker").text
        if marker is None:
            break


def sync_reich_forecast(conn, name, mdl_id):
    data_dir = update_repo("https://github.com/reichlab/covid19-forecast-hub.git")
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
        value["forecast_date"] = pandas.to_datetime(value["forecast_date"])
        value["date"] = pandas.to_datetime(value["date"])
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
        df["forecast_date"] = pandas.to_datetime(df_["fcst_date"].unique().item())
        df["date"] = pandas.to_datetime(df["date"])
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


def to_csv(df, metric, model, forecast_date, deltas=False):
    dt = pandas.to_datetime(forecast_date if forecast_date else 0)
    basedir = f'/checkpoint/{os.environ["USER"]}/covid19/csvs'
    if forecast_date != "":
        forecast_date = "_" + str(forecast_date.date())

    df["location"] = df.apply(
        lambda x: ((x.loc3 + ", ") if x.loc3 else "") + x.loc2, axis=1
    )
    df = df.pivot_table(columns=["location"], values=["counts"], index="date")
    df.columns = df.columns.get_level_values(-1)

    if deltas:
        df = df.diff()
        if dt not in df.index:
            print(
                f"Warning: forecast_date not in forecast for {model}, {forecast_date}"
            )
    df = df[df.index > dt]

    outfile = os.path.join(basedir, metric, model, f"counts{forecast_date}.csv")
    os.makedirs(os.path.dirname(outfile), exist_ok=True)
    df.to_csv(outfile)


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
            to_csv(group, metric, model, forecast_date, deltas)

    # f("deaths", True)
    # f("infections", True)
    f("infections", False)
    f("deaths", False)

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
        df["forecast_date"] = pandas.to_datetime(basedate)
        to_sql(conn, df, "infections")


def sync_austria_gt(conn):
    data_dir = update_repo("git@github.com:fairinternal/covid19_spread.git")
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
    data_pth = update_repo("https://github.com/CSSEGISandData/COVID-19.git")
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


def sync_columbia(conn):
    data_dir = update_repo("git@github.com:shaman-lab/COVID-19Projection.git")
    fips_dir = update_repo("git@github.com:kjhealy/fips-codes.git")
    fips = pandas.read_csv(
        f"{fips_dir}/county_fips_master.csv", encoding="latin1", dtype={"fips": str}
    )
    fips["fips"] = fips["fips"].apply(lambda x: x.zfill(5))
    fips = fips.drop_duplicates(["fips"])
    usa_facts = pandas.read_sql(
        "SELECT * FROM infections WHERE id='usafacts_ground_truth'",
        conn,
        parse_dates=["date"],
    )
    for file in glob(f"{data_dir}/Projection_*/Projection_*.csv"):
        print(file)
        df = pandas.read_csv(
            file, encoding="latin1", dtype={"fips": str}, parse_dates=["Date"]
        )
        df["fips"] = df["fips"].str.zfill(5)
        # Convert to cumulative counts
        df = df.pivot(index="Date", columns="fips", values="report_50").cumsum()
        df = df.reset_index().melt(id_vars=["Date"], value_name="report_50")

        merged = df.merge(fips[["fips", "county_name", "state_name"]], on="fips")
        merged = merged.rename(
            columns={
                "report_50": "counts",
                "state_name": "loc2",
                "county_name": "loc3",
                "Date": "date",
            }
        )
        merged = merged[["counts", "loc2", "loc3", "date"]]
        merged["loc1"] = "United States"
        merged["loc3"] = merged["loc3"].str.replace(" (County|Municipality|Parish)", "")
        merged["forecast_date"] = merged["date"].min()

        first_day = merged["date"].min()
        prev_day = usa_facts[usa_facts["date"] == first_day - timedelta(days=1)]

        with_base = merged.merge(prev_day, on=["loc2", "loc3"], suffixes=("", "_y"))
        with_base["counts"] += with_base["counts_y"]
        with_base = with_base[
            ["loc1", "loc2", "loc3", "date", "forecast_date", "id", "counts"]
        ]
        with_base = pandas.concat([with_base, prev_day])
        with_base["forecast_date"] = first_day
        name = re.search("Projection_(.*).csv", os.path.basename(file)).group(1)
        with_base["id"] = f"columbia_{name}"
        # to_csv(with_base, "infections", f"columbia_{name}", first_day)
        to_sql(conn, with_base, "infections")
        conn.execute(
            f"INSERT INTO gt_mapping(id, gt) VALUES ('columbia_{name}','infections') ON CONFLICT DO NOTHING"
        )


def sync_usa_facts(conn):
    fips_dir = update_repo("git@github.com:kjhealy/fips-codes.git")
    fips = pandas.read_csv(
        f"{fips_dir}/county_fips_master.csv", encoding="latin1", dtype={"fips": str}
    )
    fips["fips"] = fips["fips"].str.zfill(5)
    for table, url in [
        (
            "infections",
            "https://usafactsstatic.blob.core.windows.net/public/data/covid-19/covid_confirmed_usafacts.csv",
        ),
        (
            "deaths",
            "https://usafactsstatic.blob.core.windows.net/public/data/covid-19/covid_deaths_usafacts.csv",
        ),
    ]:
        df = pandas.read_csv(url, dtype={"countyFIPS": str})
        df["countyFIPS"] = df["countyFIPS"].str.zfill(5)
        df = df.set_index("countyFIPS")
        df = df[[c for c in df.columns if re.match(r"\d+\/\d+\/\d+", c)]].transpose()
        df = (
            df.reset_index()
            .melt(id_vars=["index"])
            .rename(columns={"index": "date", "countyFIPS": "fips", "value": "counts"})
        )
        df["date"] = pandas.to_datetime(df["date"])
        df = df.merge(fips, on=["fips"])[
            ["date", "counts", "state_name", "county_name"]
        ]
        df = df.rename(columns={"state_name": "loc2", "county_name": "loc3"})
        df["loc1"] = "United States"
        df["loc3"] = df["loc3"].str.replace(" (County|Municipality|Parish)", "")
        df["id"] = "usafacts_ground_truth"
        to_sql(conn, df, table)


def sync_google(conn):
    for metric in ["cases", "deaths"]:
        df = get_google(metric=metric)
        df = df.reset_index().melt(
            id_vars=["date"], value_name="counts", var_name="loc"
        )
        df["loc1"] = "United States"
        df["loc2"] = df["loc"].apply(lambda x: x.split("_")[0])
        df["loc3"] = df["loc"].apply(lambda x: x.split("_")[1])
        df["id"] = "google_ground_truth"
        table = "infections" if metric == "cases" else metric
        to_sql(conn, df, table)
        to_csv(df, table, "google_ground_truth", "")


@click.command()
@click.option(
    "--distribute", is_flag=True, help="Distribute across clusters (H1/H2)",
)
def sync_forecasts(distribute=False):
    if not os.path.exists(DB):
        mk_db()
    conn = sqlite3.connect(DB)
    sync_google(conn)
    sync_usa_facts(conn)
    sync_columbia(conn)
    sync_jhu(conn)
    sync_austria_gt(conn)
    sync_matts_forecasts(conn)
    sync_max_forecasts(conn)
    sync_nyt(conn)
    sync_ihme(conn)
    sync_los_alamos(conn)
    sync_mit(conn)
    sync_yyg(conn)
    # print('Reindexing...')
    # conn.execute("REINDEX;")
    if distribute:
        ssh_alias = CLUSTERS[cluster.FAIR_CLUSTER]
        DEST_DB = f"{ssh_alias}:/private/home/{os.environ['USER']}/covid19_spread/forecasts/forecast.db"
        check_call(["scp", DB, DEST_DB])
    dump_to_csv(conn, distribute)


if __name__ == "__main__":
    cli.add_command(sync_forecasts)
    cli()
