#!/usr/bin/env python3

import sys
import numpy as np
import pandas as pd
from hdx.hdx_configuration import Configuration
from hdx.data.dataset import Dataset

sys.path.append("../../../")
import shutil
from common import standardize_county_name
from glob import glob
import os


Configuration.create(hdx_site="prod", user_agent="A_Quick_Example", hdx_read_only=True)
dataset = Dataset.read_from_hdx("movement-range-maps")
resources = dataset.get_resources()
resource = [
    x for x in resources if os.path.basename(x["url"]).startswith("movement-range-data")
]
assert len(resource) == 1
resource = resource[0]
url, path = resource.download()
if os.path.exists("fb_mobility"):
    shutil.rmtree("fb_mobility")
shutil.unpack_archive(path, "fb_mobility", "zip")

fips_map = pd.read_csv("../county_fips_master.csv", encoding="ISO-8859-1")
# Washington DC is duplicated in this file for some reason...
fips_map = fips_map.drop_duplicates(["fips"]).set_index("fips")

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
    df_mobility_global = pd.read_csv(
        fin, parse_dates=["ds"], delimiter="\t", dtype={"polygon_id": str}
    )
    df_mobility_usa = df_mobility_global.query("country == 'USA'")
    return df_mobility_usa


# fin = sys.argv[1] if len(sys.argv) == 2 else None
txt_files = glob("fb_mobility/movement-range*.txt")
assert len(txt_files) == 1
fin = txt_files[0]
df = get_county_mobility_fb(fin)
df = df.rename(columns={"ds": "date", "polygon_id": "region"})
df["region"] = df["region"].apply(rename_fips)
df = df.dropna(subset=["region"])


def zscore(df):
    # z-scores
    df = (df.values - df.mean(skipna=True)) / df.std(skipna=True)
    return df


def process_df(df, cols):
    df = df[cols].copy()
    regions = []
    for (name, _df) in df.groupby("region"):
        _df = _df.sort_values(by="date")
        _df = _df.drop_duplicates(subset="date")
        dates = _df["date"].to_list()
        assert len(dates) == len(np.unique(dates)), _df
        _df = _df.loc[:, ~_df.columns.duplicated()]
        _df = _df.drop(columns=["region", "date"]).transpose()
        # take 7 day average
        _df = _df.rolling(7, min_periods=1, axis=1).mean()
        # convert relative change into absolute numbers
        _df.loc["all_day_bing_tiles_visited_relative_change"] += 1
        # standarize
        # _df.loc["all_day_ratio_single_tile_users"] = zscore(
        #    _df.loc["all_day_ratio_single_tile_users"]
        # )
        _df["region"] = [name] * len(_df)
        _df.columns = list(map(lambda x: x.strftime("%Y-%m-%d"), dates)) + ["region"]
        regions.append(_df.reset_index())

    df = pd.concat(regions, axis=0, ignore_index=True)
    cols = ["region"] + [x for x in df.columns if x != "region"]
    df = df[cols]

    df = df.rename(columns={"index": "type"})
    return df


county = process_df(df, cols)
# county["state"] = county["region"].apply(lambda x: x.split(", ")[-1])
state = df.copy()
state["region"] = state["region"].apply(lambda x: x.split(", ")[-1])
state = state.groupby(["region", "date"]).mean().reset_index()
state = process_df(state, cols)
# county2 = state.merge(
#     county[["state", "region"]].drop_duplicates(),
#     left_on="region",
#     right_on="state",
#     suffixes=("_x", None),
# )[county.columns]
# county2["type"] = county2["type"].apply(lambda x: x + "_state")
# county = pd.concat([county, county2]).drop(columns=["state"])


county = county.fillna(0)
state = state.fillna(0)
county.round(4).to_csv("mobility_features_county_fb.csv", index=False)
state.round(4).to_csv("mobility_features_state_fb.csv", index=False)
