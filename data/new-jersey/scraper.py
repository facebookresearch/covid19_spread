#!/usr/bin/env python3

import requests
import pandas
from datetime import date, timedelta, datetime
import os
import numpy as np


URL = "https://services7.arcgis.com/Z0rixLlManVefxqY/arcgis/rest/services/DailyCaseCounts/FeatureServer/0/query?f=json&where=1%3D1&returnGeometry=false&spatialRel=esriSpatialRelIntersects&outFields=*&orderByFields=TOTAL_CASES%20desc"
UNK_URL = "https://services7.arcgis.com/Z0rixLlManVefxqY/arcgis/rest/services/survey123_cb9a6e9a53ae45f6b9509a23ecdf7bcf/FeatureServer/0/query?f=json&where=1=1&returnGeometry=false&outFields=*&resultOffset=0&resultRecordCount=1&resultType=standard&orderByFields=_date%20desc"


def get_latest(metric="cases"):
    """
    metric: str - 'cases' or 'deaths
    """

    # Fetch newest data from NJ DOH ESRI API
    unk = requests.get(UNK_URL).json()
    unk = unk["features"][0]["attributes"]
    data = requests.get(URL).json()
    df = pandas.DataFrame([x["attributes"] for x in data["features"]])
    unk_time = datetime.fromtimestamp(unk["_date"] / 1000)

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
                "Number_COVID_Cases_Confirmed": "cases",
                "TOTAL_DEATHS": "deaths",
            }
        )
        df["date"] = unk_time
        df = df.pivot_table(index="date", values=metric, columns="county")
        df.columns = [c.split(" County")[0] for c in df.columns]
        mapper = {"cases": "unknown_positives", "deaths": "unknown_deaths"}
        df["Unknown"] = unk[mapper[metric]]
        res = pandas.concat([nyt, df[nyt.columns]])
        res.index = res.index.date
    else:
        res = nyt
    res = res.reset_index().rename(columns={"date": "Date"})
    res["Start day"] = np.arange(1, len(res) + 1)
    return res


def main():
    df = get_latest()
    date_fmt = df["Date"].max().date().strftime("%Y%m%d")
    df.to_csv(f"data-{date_fmt}.csv")


if __name__ == "__main__":
    main()
