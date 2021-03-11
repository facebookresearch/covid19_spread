#!/usr/bin/env python3
# Copyright (c) 2021-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

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


# Configuration.create(hdx_site="prod", user_agent="A_Quick_Example", hdx_read_only=True)
# dataset = Dataset.read_from_hdx("movement-range-maps")
# resources = dataset.get_resources()
# resource = [x for x in resources if x.get_file_type() == "ZIP"]
# assert len(resource) == 1
# resource = resource[0]
# url, path = resource.download()
# if os.path.exists("fb_mobility"):
#    shutil.rmtree("fb_mobility")
# shutil.unpack_archive(path, "fb_mobility", "zip")


cols = [
    "date",
    "region",
    "all_day_bing_tiles_visited_relative_change",
    "all_day_ratio_single_tile_users",
]


def get_county_mobility_fb(fin):
    df_mobility_global = pd.read_csv(fin, parse_dates=["ds"], header=0, delimiter="\t")
    df_mobility = df_mobility_global.query("country == 'ESP'")
    print(df_mobility)
    return df_mobility


rename = {
    "Andaluc-a": "Andalucía",
    "Arag-n": "Aragón",
    "Castilla y Le-n": "Castilla y León",
    "Catalu-a": "Cataluña",
    "Pa-s Vasco": "País Vasco",
    "Ceuta y Melilla": "Ceuta",
    "Islas Canarias": "Canarias",
}


def standardize_name(x):
    if x in rename:
        x = rename[x]
    return x


# fin = sys.argv[1] if len(sys.argv) == 2 else None
txt_files = glob("fb_mobility/movement-range*.txt")
assert len(txt_files) == 1
fin = txt_files[0]
df = get_county_mobility_fb(fin)
df = df.rename(columns={"ds": "date", "polygon_name": "region"})
print(df.columns)
df["region"] = df["region"].str.replace(
    "(Comunidad Foral de|Comunidad de|Comunidad|Regi-n de|Principado de) ", ""
)

df["region"] = df["region"].apply(standardize_name)
print(df["region"].unique())
print(df.iloc[0])
print(df.head())
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
    _df = _df.rolling(7, min_periods=1, axis=1).mean()
    _df["region"] = [name] * len(_df)
    _df.columns = list(map(lambda x: x.strftime("%Y-%m-%d"), dates)) + ["region"]
    regions.append(_df.reset_index())

df = pd.concat(regions, axis=0, ignore_index=True)
cols = ["region"] + [x for x in df.columns if x != "region"]
df = df[cols]

# ceuta y melilla
melilla = df[df["region"] == "Ceuta"].copy()
melilla["region"] = "Melilla"
df = pd.concat([df, melilla], axis=0)

df = df.fillna(0)
df = df.rename(columns={"index": "type"})
print(df.head(), df.shape)

df.round(4).to_csv("mobility_features_fb.csv", index=False)
