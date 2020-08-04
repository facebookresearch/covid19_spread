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
resource = [x for x in resources if x.get_file_type() == "ZIP"]
assert len(resource) == 1
resource = resource[0]
url, path = resource.download()
if os.path.exists("fb_mobility"):
    shutil.rmtree("fb_mobility")
shutil.unpack_archive(path, "fb_mobility", "zip")

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


# fin = sys.argv[1] if len(sys.argv) == 2 else None
txt_files = glob("fb_mobility/movement-range*.txt")
assert len(txt_files) == 1
fin = txt_files[0]
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
    # convert relative change into absolute numbers
    _df.loc["all_day_bing_tiles_visited_relative_change"] += 1
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
print(df.head(), df.shape)

df.round(4).to_csv("mobility_features.csv", index=False)
