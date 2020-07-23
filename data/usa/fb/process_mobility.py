#!/usr/bin/env python3

import sys
import numpy as np
import pandas as pd

sys.path.append("../../../")
from common import standardize_county_name

fips_map = pd.read_csv(
    "../county_fips_master.csv", index_col=["fips"], encoding="ISO-8859-1"
)

fips = {}
for f in np.unique(fips_map.index):
    try:
        cname = fips_map.loc[int(f)].county_name
        sname = fips_map.loc[int(f)].state_name
        fips[int(f)] = f"{cname}, {sname}"
    except:
        print(f"Skipping FIPS {f}")


cols = [
    "date",
    "region",
    "all_day_bing_tiles_visited_relative_change",
    "all_day_ratio_single_tile_users",
]


def rename_fips(f):
    f = int(f)
    if f in fips:
        return fips[f]
    else:
        return None


def get_county_mobility_fb(fin):
    df_mobility_global = pd.read_csv(fin, parse_dates=["ds"], header=0, delimiter="\t")
    df_mobility_usa = df_mobility_global.query("country == 'USA'")
    return df_mobility_usa


fin = sys.argv[1] if len(sys.argv) == 2 else None
df = get_county_mobility_fb(fin)
df = df.rename(columns={"ds": "date", "polygon_id": "region"})
print(df.columns)
print(df.iloc[0])
print(df.head())
df["region"] = df["region"].apply(rename_fips)
df = df.dropna(subset=["region"])
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
    _df.loc["all_day_ratio_single_tile_users"] = (
        _df.loc["all_day_ratio_single_tile_users"].diff().fillna(0)
    )
    _df["region"] = [name] * len(_df)
    _df.columns = list(map(lambda x: x.strftime("%Y-%m-%d"), dates)) + ["region"]
    regions.append(_df.reset_index())

df = pd.concat(regions, axis=0, ignore_index=True)
df = df.fillna(0)
cols = ["region"] + [x for x in df.columns if x != "region"]
df = df[cols]

df = df.rename(columns={"index": "type"})
print(df.head(), df.shape)

df.to_csv("mobility_features.csv", index=False)
