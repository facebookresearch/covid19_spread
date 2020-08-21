import pandas
import re
from datetime import datetime


index = pandas.read_csv("https://storage.googleapis.com/covid19-open-data/v2/index.csv")

index = index[index["country_code"] == "US"]
state_index = index[(index["key"].str.match("^US_[A-Z]+$")).fillna(False)]

fips = pandas.read_csv(
    "https://raw.githubusercontent.com/kjhealy/fips-codes/master/state_and_county_fips_master.csv"
)
fips["fips"] = fips["fips"].astype(str).str.zfill(5)
index = index.merge(fips, left_on="subregion2_code", right_on="fips")


def zscore(piv):
    # z-zcore
    piv = (piv - piv.mean(skipna=True)) / piv.std(skipna=True)
    piv = piv.fillna(method="ffill").fillna(method="bfill")
    # piv = piv.fillna(0)
    return piv


def zero_one(df):
    df = df.fillna(0)
    df = df.div(df.max(axis=1), axis=0)
    df = df.fillna(0)
    return df


def process_df(df, columns, resolution, func_normalize):
    idx = state_index if resolution == "state" else index
    merged = df.merge(idx, on="key")
    if resolution == "state":
        exclude = {"US_MP", "US_AS", "US_GU", "US_VI", "US_PR"}
        merged = merged[~merged["key"].isin(exclude)]
        merged["region"] = merged["subregion1_name"]
    else:
        merged["region"] = (
            merged["name"].str.replace(" (County|Municipality|Parish|Borough)", "")
            + ", "
            + merged["subregion1_name"]
        )
    piv = merged.pivot(index="date", columns="region", values=columns)
    piv = func_normalize(piv)

    dfs = []
    for k in piv.columns.get_level_values(0).unique():
        dfs.append(piv[k].transpose())
        dfs[-1]["type"] = k
    df = pandas.concat(dfs)
    df = df[["type"] + [c for c in df.columns if isinstance(c, datetime)]]
    df.columns = [str(c.date()) if isinstance(c, datetime) else c for c in df.columns]
    return df.fillna(0)  # in case all values are NaN


df = pandas.read_csv(
    "https://storage.googleapis.com/covid19-open-data/v2/weather.csv",
    parse_dates=["date"],
)
cols = ["average_temperature", "minimum_temperature", "maximum_temperature", "rainfall"]
weather = process_df(df, columns=cols, resolution="county", func_normalize=zscore)
weather.round(3).to_csv("weather_features_county.csv")

weather = process_df(df, columns=cols, resolution="state", func_normalize=zscore)
weather.round(3).to_csv("weather_features_state.csv")

df = pandas.read_csv(
    "https://storage.googleapis.com/covid19-open-data/v2/epidemiology.csv",
    parse_dates=["date"],
)
state_epi = process_df(
    df, columns=["new_confirmed"], resolution="state", func_normalize=zscore
)
state_epi.round(3).to_csv("epi_features_state.csv")

epi = process_df(
    df, columns=["new_confirmed"], resolution="county", func_normalize=zscore
)
epi.round(3).to_csv("epi_features_county.csv")

testing = process_df(
    df, columns=["new_tested"], resolution="state", func_normalize=zero_one
)
testing.round(3).to_csv("tested_total_state.csv")

df["ratio"] = df["new_confirmed"] / df["new_tested"]
testing = process_df(df, columns=["ratio"], resolution="state", func_normalize=zero_one)
testing.round(3).to_csv("tested_ratio_state.csv")

# gov response is not granular enough
# df = pandas.read_csv('https://storage.googleapis.com/covid19-open-data/v2/oxford-government-response.csv')
