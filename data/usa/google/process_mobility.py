#!/usr/bin/env python3

import sys
import numpy as np
import pandas as pd

sys.path.append("../../../")
from common import standardize_county_name


cols = [
    "date",
    "region",
    "retail_and_recreation_percent_change_from_baseline",
    "grocery_and_pharmacy_percent_change_from_baseline",
    "parks_percent_change_from_baseline",
    "transit_stations_percent_change_from_baseline",
    "workplaces_percent_change_from_baseline",
    "residential_percent_change_from_baseline",
]


def get_county_mobility_google(fin=None):
    # Google LLC "Google COVID-19 Community Mobility Reports."
    # https://www.google.com/covid19/mobility/ Accessed: 2020-05-04.
    # unfortunately, this is only relative to mobility on a baseline date
    if fin is None:
        fin = "https://www.gstatic.com/covid19/mobility/Global_Mobility_Report.csv"
    df_Gmobility_global = pd.read_csv(fin, parse_dates=["date"])
    df_Gmobility_usa = df_Gmobility_global.query("country_region_code == 'US'")
    return df_Gmobility_usa


fin = sys.argv[1] if len(sys.argv) == 2 else None
df = get_county_mobility_google(fin)
df = df.dropna(subset=["sub_region_2"], axis=0)
df["sub_region_2"] = df["sub_region_2"].transform(standardize_county_name)
df["region"] = df[["sub_region_2", "sub_region_1"]].agg(", ".join, axis=1)
print(df["region"].head())
print(df["date"].min(), df["date"].max())

df = df[cols]
regions = []
for (name, _df) in df.groupby("region"):
    _df = _df.sort_values(by="date")
    _df = _df.drop_duplicates(subset="date")
    dates = _df["date"].to_list()
    assert len(dates) == len(np.unique(dates)), _df
    _df = _df.loc[:, ~_df.columns.duplicated()]
    _df = _df.drop(columns=["region", "date"]).transpose()
    _df = 1 + _df / 100
    # take 7 day average
    _df = _df.rolling(7, axis=1).mean()
    _df["region"] = [name] * len(_df)
    _df.columns = list(map(lambda x: x.strftime("%Y-%m-%d"), dates)) + ["region"]
    regions.append(_df.reset_index())

df = pd.concat(regions, axis=0, ignore_index=True)
cols = ["region"] + [x for x in df.columns if x != "region"]
df = df[cols]

# z-scores
# df.iloc[:, 2:] = (
#    df.iloc[:, 2:].values - df.iloc[:, 2:].mean(axis=1, skipna=True).values[:, None]
# ) / df.iloc[:, 2:].std(axis=1, skipna=True).values[:, None]

df = df.fillna(0)
df = df.rename(columns={"index": "type"})
print(df.head(), len(df))

df.round(3).to_csv("mobility_features.csv", index=False)
