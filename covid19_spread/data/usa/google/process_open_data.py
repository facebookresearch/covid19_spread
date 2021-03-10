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
    print(df)
    # df = df.div(df.max(axis=0), axis=1)
    df = df / df.max(axis=0)
    print(df)
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
    if func_normalize is not None:
        piv = func_normalize(piv)

    dfs = []
    for k in piv.columns.get_level_values(0).unique():
        dfs.append(piv[k].transpose())
        dfs[-1]["type"] = k
    df = pandas.concat(dfs)
    df = df[["type"] + [c for c in df.columns if isinstance(c, datetime)]]
    df.columns = [str(c.date()) if isinstance(c, datetime) else c for c in df.columns]
    return df.fillna(0)  # in case all values are NaN


# --- Hospitalizations ---
df = pandas.read_csv(
    "https://storage.googleapis.com/covid19-open-data/v2/hospitalizations.csv",
    parse_dates=["date"],
)
print(df)
state_hosp = process_df(
    df,
    columns=["current_hospitalized", "current_intensive_care", "current_ventilator"],
    resolution="state",
    # func_normalize=lambda x: zero_one(x.clip(0, None)).rolling(7, min_periods=1).mean(),
    func_normalize=lambda x: zero_one(x.clip(0, None)),
)
state_hosp.round(3).to_csv("hosp_features_state.csv")


# --- Weather features ---
df = pandas.read_csv(
    "https://storage.googleapis.com/covid19-open-data/v2/weather.csv",
    parse_dates=["date"],
)
cols = [
    "average_temperature",
    "minimum_temperature",
    "maximum_temperature",
    "rainfall",
    "relative_humidity",
    "dew_point",
]
weather = process_df(df, columns=cols, resolution="county", func_normalize=zscore)
weather.round(3).to_csv("weather_features_county.csv")

weather = process_df(df, columns=cols, resolution="state", func_normalize=zscore)
weather.round(3).to_csv("weather_features_state.csv")


# --- Epi features ---
df = pandas.read_csv(
    "https://storage.googleapis.com/covid19-open-data/v2/epidemiology.csv",
    # parse_dates=["date"],
)
# TODO: remove the following 2 lines.  For some reason, Google is listing some dates as "2564-M-d"
# This breaks pandas date parsing and causes downstream problems
df = df[~df["date"].str.startswith("2564")].copy()
df["date"] = pandas.to_datetime(df["date"])

state_epi = process_df(
    df,
    columns=["new_confirmed"],
    resolution="state",
    # func_normalize=lambda x: zero_one(x.clip(0, None)).rolling(7, min_periods=1).mean(),
    func_normalize=lambda x: zero_one(x.clip(0, None)),
)
state_epi.round(3).to_csv("epi_features_state.csv")
epi = process_df(
    df,
    columns=["new_confirmed"],
    resolution="county",
    # func_normalize=lambda x: zero_one(x.clip(0, None)).rolling(7, min_periods=1).mean(),
    func_normalize=lambda x: zero_one(x.clip(0, None)),
)
epi.round(3).to_csv("epi_features_county.csv")

testing = process_df(
    df,
    columns=["new_tested"],
    resolution="state",
    # func_normalize=lambda x: zero_one(x.clip(0, None)).rolling(7, min_periods=1).mean(),
    func_normalize=lambda x: zero_one(x.clip(0, None)),
)
testing.round(3).to_csv("tested_total_state.csv")

df["ratio"] = df["new_confirmed"] / df["new_tested"]
testing = process_df(df, columns=["ratio"], resolution="state", func_normalize=None,)
testing.round(3).to_csv("tested_ratio_state.csv")

# gov response is not granular enough
# df = pandas.read_csv('https://storage.googleapis.com/covid19-open-data/v2/oxford-government-response.csv')
