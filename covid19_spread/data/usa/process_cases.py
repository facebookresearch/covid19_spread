import os
from covid19_spread.common import update_repo
import pandas
import re
import datetime


def get_index():
    index = pandas.read_csv(
        "https://storage.googleapis.com/covid19-open-data/v2/index.csv"
    )
    index = index[index["key"].str.match(r"^US_[A-Z]+_\d{5}$").fillna(False)]
    index["fips"] = index["subregion2_code"].astype(str).str.zfill(5)
    index["name"] = index["subregion2_name"]
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
    df.columns = [x + " County" for x in df.columns]
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
    index = get_index()
    df = pandas.read_csv(
        "https://storage.googleapis.com/covid19-open-data/v2/epidemiology.csv",
        parse_dates=["date"],
    )
    merged = df.merge(index, on="key")
    merged = merged[~merged["subregion2_name"].isnull()]
    merged["loc"] = merged["subregion1_name"] + "_" + merged["name"]
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
    df = df[~df["FIPS"].isnull()]
    df["FIPS"] = df["FIPS"].apply(lambda x: str(int(x)).zfill(5))
    index = get_index()
    index["loc"] = index["subregion1_name"] + "_" + index["name"]
    merged = df.merge(index[["fips", "loc"]], left_on="FIPS", right_on="fips")
    date_cols = [c for c in merged.columns if re.match("\d+/\d+/\d+", c)]
    transposed = merged[date_cols + ["loc"]].set_index("loc").transpose()
    transposed.index = pandas.to_datetime(transposed.index)
    return transposed.sort_index()


SOURCES = {
    "nyt": get_nyt,
    "google": get_google,
    "jhu": get_jhu,
}
