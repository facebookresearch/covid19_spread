#!/usr/bin/env python3

import sys
import os

script_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, os.path.join(script_dir, "../.."))
import requests
import pandas
from datetime import date, timedelta, datetime
import numpy as np
from subprocess import check_call, check_output
from glob import glob
from lib.slack import get_client as get_slack_client


URL = "https://services7.arcgis.com/Z0rixLlManVefxqY/arcgis/rest/services/DailyCaseCounts/FeatureServer/0/query?f=json&where=1%3D1&returnGeometry=false&spatialRel=esriSpatialRelIntersects&outFields=*&orderByFields=TOTAL_CASES%20desc"
# UNK_URL = "https://services7.arcgis.com/Z0rixLlManVefxqY/arcgis/rest/services/survey123_cb9a6e9a53ae45f6b9509a23ecdf7bcf/FeatureServer/0/query?f=json&where=1=1&returnGeometry=false&outFields=*&resultOffset=0&resultRecordCount=1&resultType=standard&orderByFields=_date%20desc"
UNK_URL = "https://services7.arcgis.com/Z0rixLlManVefxqY/arcgis/rest/services/survey123_cb9a6e9a53ae45f6b9509a23ecdf7bcf/FeatureServer/0/query?f=json&where=unknown_positives%20IS%20NOT%20NULL&returnGeometry=false&spatialRel=esriSpatialRelIntersects&outFields=*&orderByFields=EditDate%20desc&r"


def get_latest_with_nyt(metric="cases"):
    """
    metric: str - 'cases' or 'deaths
    """
    # Fetch newest data from NJ DOH ESRI API
    unk = requests.get(UNK_URL).json()
    unk = unk["features"][0]["attributes"]
    data = requests.get(URL).json()
    df = pandas.DataFrame([x["attributes"] for x in data["features"]])
    # unk_time = datetime.fromtimestamp(unk["_date"] / 1000)
    unk_time = pandas.to_datetime(unk["CreationDate"], unit="ms")
    # Use NYT for historical data.  They lag behind by 1 day.
    nyt = pandas.read_csv(
        "https://raw.githubusercontent.com/nytimes/covid-19-data/master/us-counties.csv",
        parse_dates=["date"],
    )
    nyt = nyt[nyt["state"] == "New Jersey"]
    nyt = nyt.pivot_table(index="date", columns="county", values=metric).fillna(0)
    if (nyt.index.max() + timedelta(days=1)).date() == unk_time.date():
        # NJ has a newer date, append it on...
        df = df.rename(
            columns={
                "COUNTY_LAB": "county",
                "TOTAL_CASES": "cases",
                "TOTAL_DEATHS": "deaths",
            }
        )
        df["date"] = unk_time
        df = df.pivot_table(index="date", values=metric, columns="county")
        df.columns = [c.split(" County")[0] for c in df.columns]
        mapper = {"cases": "unknown_positives", "deaths": "unknown_deaths"}
        df["Unknown"] = unk[mapper[metric]]
        res = pandas.concat([nyt, df[nyt.columns]])
    else:
        res = nyt
    res.index.name = "date"
    res = res.reset_index().rename(columns={"date": "Date"})
    res["Start day"] = np.arange(1, len(res) + 1)
    return res


def get_latest():
    script_dir = os.path.dirname(os.path.realpath(__file__))
    latest_pth = sorted(glob(f"{script_dir}/data-202*.csv"))[-1]
    df = pandas.read_csv(latest_pth, parse_dates=["Date"], index_col=0).set_index(
        "Date"
    )
    # Fetch newest data from NJ DOH ESRI API
    unk = requests.get(UNK_URL).json()
    unk = unk["features"][0]["attributes"]
    data = requests.get(URL).json()
    new_df = pandas.DataFrame([x["attributes"] for x in data["features"]])
    unk_time = pandas.to_datetime(unk["CreationDate"], unit="ms")

    new_df = new_df.rename(
        columns={
            "COUNTY_LAB": "county",
            "TOTAL_CASES": "cases",
            "TOTAL_DEATHS": "deaths",
        }
    )
    new_df["date"] = unk_time
    new_df = new_df.pivot_table(index="date", values="cases", columns="county")
    new_df.columns = [c.split(" County")[0] for c in new_df.columns]
    new_df["Unknown"] = unk["unknown_positives"]
    new_df.index = new_df.index.floor("1d")

    diff_days = (new_df.index.item() - df.index.max()).components.days
    assert (
        diff_days <= 1
    ), f"Gap in data of {diff_days} days!!! prev day is {df.index.max().date()}, current date is {new_df.index.item().date()}"

    if new_df.index.item() > df.index.max():
        new_df["Start day"] = df["Start day"].max() + 1
        # Make sure we have all the same columns
        assert len(new_df.columns.intersection(df.columns)) == len(
            df.columns
        ), f"Inconsistent columns!!!!"
        df = pandas.concat([df, new_df])
    # Make sure there aren't any missing values
    assert not df.isnull().any().any(), "Found NaNs in data!!!!"
    # Check there are no gaps or redundant days in the dataset
    assert (pandas.Series(df.index).diff().dt.components.days[1:] == 1).all()
    counties = [c for c in df.columns if c not in {"Start day", "Unknown"}]
    assert not np.all(
        df[counties].diff().iloc[-1].values == 0
    ), "Today's values are the same as yesterday's!!!!"
    return df.rename_axis("Date").reset_index()


def main():
    print(f"Checking for new data at {datetime.now()}")
    script_dir = os.path.dirname(os.path.realpath(__file__))
    check_call(["git", "pull"], cwd=script_dir)
    df = get_latest()
    date_fmt = df["Date"].max().date().strftime("%Y%m%d")
    fout = os.path.join(script_dir, f"data-{date_fmt}.csv")
    update = not os.path.exists(fout)
    df.to_csv(fout)
    if update:
        client = get_slack_client()
        msg = f'*New Data Available for New Jersey: {df["Date"].max().date()}*'
        client.chat_postMessage(channel="#new-data", text=msg)
        check_call(["git", "add", fout], cwd=script_dir)
        check_call(
            ["git", "commit", "-m", f'Updating NJ data for {df["Date"].max().date()}'],
            cwd=script_dir,
        )
        check_call(["git", "push"], cwd=script_dir)
    else:
        print(f'Already have latest data for {df["Date"].max().date()}')


if __name__ == "__main__":
    main()
