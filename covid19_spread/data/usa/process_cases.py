#!/usr/bin/env python3

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(os.path.realpath(__file__)), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.realpath(__file__)), "../.."))
from common import update_repo
from episode import mk_episode, to_h5
from subprocess import check_call
import os
import csv
import pandas
import numpy as np
import h5py
import json
import re
from collections import defaultdict
import itertools
import shutil
import argparse
import datetime
from glob import glob


def get_index():
    index = pandas.read_csv(
        "https://storage.googleapis.com/covid19-open-data/v2/index.csv"
    )
    index = index[index["key"].str.match("^US_[A-Z]+_\d{5}$").fillna(False)]
    fips = pandas.read_csv(
        "https://raw.githubusercontent.com/kjhealy/fips-codes/master/state_and_county_fips_master.csv"
    )
    fips["fips"] = fips["fips"].astype(str).str.zfill(5)
    index = index.merge(fips, left_on="subregion2_code", right_on="fips")
    index["name"] = index["name"].str.replace(
        " (County|Municipality|Parish|Borough)", ""
    )
    return index


def get_nyt(metric="cases"):
    print("NYT")
    data_repo = update_repo("https://github.com/nytimes/covid-19-data.git")
    df = pandas.read_csv(
        os.path.join(data_repo, "us-counties.csv"), dtype={"fips": str}
    )
    index = get_index()
    df = df.merge(index[["fips", "subregion1_name", "name"]], on="fips")
    df["loc"] = df["subregion1_name"] + "_" + df["name"]
    pivot = df.pivot_table(values=metric, columns=["loc"], index="date")
    pivot = pivot.fillna(0)
    pivot.index = pandas.to_datetime(pivot.index)
    if metric == "deaths":
        return pivot

    # Swap out NYTimes NY state data with the NY DOH data.
    NYSTATE_URL = (
        "https://health.data.ny.gov/api/views/xdss-u53e/rows.csv?accessType=DOWNLOAD"
    )
    df = pandas.read_csv(NYSTATE_URL).rename(
        columns={"Test Date": "date", "Cumulative Number of Positives": "cases"}
    )
    df["loc"] = "New York_" + df["County"]
    df = df.pivot_table(values=metric, columns=["loc"], index="date")

    # The NYT labels each date as the date the report comes out, not the date the data corresponds to.
    # Add 1 day to the NYS DOH data to get it to align
    df.index = pandas.to_datetime(df.index) + datetime.timedelta(days=1)
    without_nystate = pivot[[c for c in pivot.columns if not c.startswith("New York")]]
    last_date = min(without_nystate.index.max(), df.index.max())
    df = df[df.index <= last_date]
    without_nystate = without_nystate[without_nystate.index <= last_date]
    assert (
        df.index.max() == without_nystate.index.max()
    ), "NYT and DOH data don't matchup yet!"
    # Only take NYT data up to the date for which we have nystate data
    without_nystate[without_nystate.index <= df.index.max()]
    return without_nystate.merge(
        df, left_index=True, right_index=True, how="outer"
    ).fillna(0)


def get_google(metric="cases"):
    index = pandas.read_csv(
        "https://storage.googleapis.com/covid19-open-data/v2/index.csv"
    )
    index = index[index["country_code"] == "US"]
    df = pandas.read_csv(
        "https://storage.googleapis.com/covid19-open-data/v2/epidemiology.csv",
        parse_dates=["date"],
    )
    fips = pandas.read_csv(
        "https://raw.githubusercontent.com/kjhealy/fips-codes/master/state_and_county_fips_master.csv"
    )
    fips["fips"] = fips["fips"].astype(str).str.zfill(5)
    index = index.merge(fips, left_on="subregion2_code", right_on="fips")
    merged = df.merge(index, on="key")
    merged = merged[~merged["subregion2_name"].isnull()]
    merged["loc"] = (
        merged["subregion1_name"]
        + "_"
        + merged["name"].str.replace(" (County|Municipality|Parish|Borough)", "")
    )
    value_col = "total_confirmed" if metric == "cases" else "total_deceased"
    pivot = merged.pivot(values=value_col, index="date", columns="loc")
    if pivot.iloc[-1].isnull().any():
        pivot = pivot.iloc[:-1]
    pivot.iloc[0] = pivot.iloc[0].fillna(0)
    pivot = pivot.fillna(method="ffill")
    return pivot


def get_jhu(metric="cases"):
    urls = {
        "cases": "https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_US.csv",
        "deaths": "https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_deaths_US.csv",
    }

    df = pandas.read_csv(urls[metric])
    date_cols = [c for c in df.columns if re.match("\d+/\d+/\d+", c)]
    df = df[~df["Admin2"].isnull()]
    US_TERRITORIES = {"AS", "GU", "MP", "PR", "VI", "UM"}
    # Strip out US territories
    df = df[~df["iso2"].isin(US_TERRITORIES)]
    df["loc"] = df["Province_State"] + "_" + df["Admin2"]
    df = df.set_index("loc")[date_cols].transpose()
    df.index = pandas.to_datetime(df.index)
    df.to_csv("jhu.csv")
    return df


SOURCES = {
    "nyt": get_nyt,
    "google": get_google,
    "jhu": get_jhu,
}


def main(args):
    parser = argparse.ArgumentParser()
    parser.add_argument("--smooth", type=int, default=1)
    parser.add_argument("--source", choices=list(SOURCES.keys()), default="nyt")
    parser.add_argument(
        "--mode",
        choices=["adjacent_states", "no_interaction", "deaths"],
        default="adjacent_states",
    )
    opt = parser.parse_args(args)

    if not os.path.exists("us-state-neighbors.json"):
        check_call(
            [
                "wget",
                "https://gist.githubusercontent.com/PrajitR/0afccfa4dc4febe59276/raw/7a73603f5346210ae34845c43094f0daabfd4d49/us-state-neighbors.json",
            ]
        )
    if not os.path.exists("states_hash.json"):
        check_call(
            [
                "wget",
                "https://gist.githubusercontent.com/mshafrir/2646763/raw/8b0dbb93521f5d6889502305335104218454c2bf/states_hash.json",
            ]
        )

    neighbors = json.load(open("us-state-neighbors.json", "r"))
    state_map = json.load(open("states_hash.json", "r"))

    # Convert abbreviated names to full state names
    neighbors = {
        state_map[k]: [state_map[v] for v in vs] for k, vs in neighbors.items()
    }

    df = SOURCES[opt.source](metric="deaths" if opt.mode == "deaths" else "cases")
    print(f"Latest date = {df.index.max()}")

    # Remove any unknowns
    df = df[[c for c in df.columns if "Unknown" not in c]]

    t = df.transpose()
    t.columns = [str(x.date()) for x in t.columns]
    t.reset_index().rename(columns={"loc": "county"}).to_csv(
        "ground_truth.csv", index=False
    )

    counter = itertools.count()
    county_ids = defaultdict(counter.__next__)

    outfile = f"timeseries_smooth_{opt.smooth}_days_mode_{opt.mode}.h5"

    episodes = []
    if opt.mode == "adjacent_states" or opt.mode == "deaths":
        for state, ns in neighbors.items():
            states = set([state] + ns)
            regex = "|".join(f"^{s}" for s in states)
            cols = [c for c in df.columns if re.match(regex, c)]
            episodes.append(mk_episode(df, cols, county_ids, opt.smooth))
    elif opt.mode == "no_interaction":
        for county in df.columns:
            ts, ns = episodes.append(mk_episode(df, [county], county_ids, opt.smooth))
            if ts is not None:
                episodes.append((ts, ns))

    to_h5(df, outfile, county_ids, episodes)


if __name__ == "__main__":
    main(sys.argv[1:])