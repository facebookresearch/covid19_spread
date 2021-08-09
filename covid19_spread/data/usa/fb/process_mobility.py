#!/usr/bin/env python3
# Copyright (c) 2021-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import pandas as pd
from hdx.hdx_configuration import Configuration
from hdx.data.dataset import Dataset
import shutil
from glob import glob
import os
from covid19_spread.data.usa.process_cases import get_index
import re


SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))


def main():
    Configuration.create(
        hdx_site="prod", user_agent="A_Quick_Example", hdx_read_only=True
    )
    dataset = Dataset.read_from_hdx("movement-range-maps")
    resources = dataset.get_resources()
    resource = [
        x
        for x in resources
        if re.match(".*/movement-range-data-\d{4}-\d{2}-\d{2}\.zip", x["url"])
    ]
    assert len(resource) == 1
    resource = resource[0]
    url, path = resource.download()
    if os.path.exists(f"{SCRIPT_DIR}/fb_mobility"):
        shutil.rmtree(f"{SCRIPT_DIR}/fb_mobility")
    shutil.unpack_archive(path, f"{SCRIPT_DIR}/fb_mobility", "zip")

    fips_map = get_index()
    fips_map["location"] = fips_map["name"] + ", " + fips_map["subregion1_name"]

    cols = [
        "date",
        "region",
        "all_day_bing_tiles_visited_relative_change",
        "all_day_ratio_single_tile_users",
    ]

    def get_county_mobility_fb(fin):
        df_mobility_global = pd.read_csv(
            fin, parse_dates=["ds"], delimiter="\t", dtype={"polygon_id": str}
        )
        df_mobility_usa = df_mobility_global.query("country == 'USA'")
        return df_mobility_usa

    # fin = sys.argv[1] if len(sys.argv) == 2 else None
    txt_files = glob(f"{SCRIPT_DIR}/fb_mobility/movement-range*.txt")
    assert len(txt_files) == 1
    fin = txt_files[0]
    df = get_county_mobility_fb(fin)
    df = df.rename(columns={"ds": "date", "polygon_id": "region"})

    df = df.merge(fips_map, left_on="region", right_on="fips")[
        list(df.columns) + ["location"]
    ]
    df = df.drop(columns="region").rename(columns={"location": "region"})

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
            _df["region"] = [name] * len(_df)
            _df.columns = list(map(lambda x: x.strftime("%Y-%m-%d"), dates)) + [
                "region"
            ]
            regions.append(_df.reset_index())

        df = pd.concat(regions, axis=0, ignore_index=True)
        cols = ["region"] + [x for x in df.columns if x != "region"]
        df = df[cols]

        df = df.rename(columns={"index": "type"})
        return df

    county = process_df(df, cols)
    state = df.copy()
    state["region"] = state["region"].apply(lambda x: x.split(", ")[-1])
    state = state.groupby(["region", "date"]).mean().reset_index()
    state = process_df(state, cols)
    county = county.fillna(0)
    state = state.fillna(0)
    county.round(4).to_csv(f"{SCRIPT_DIR}/mobility_features_county_fb.csv", index=False)
    state.round(4).to_csv(f"{SCRIPT_DIR}/mobility_features_state_fb.csv", index=False)


if __name__ == "__main__":
    main()
