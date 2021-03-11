#!/usr/bin/env python3
# Copyright (c) 2021-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import numpy as np
import pandas as pd
import sys
import torch as th
from datetime import timedelta
from os import listdir
from os.path import isfile, join


def process_time_features(df, pth, shift=0):
    mobility = pd.read_csv(pth)
    dates = pd.to_datetime(mobility.columns[2:])
    dates = dates[np.where(dates >= df.columns.min())[0]]
    mobility = mobility[
        mobility.columns[:2].to_list()
        + list(map(lambda d: d.strftime("%Y-%m-%d"), dates))
    ]

    print(np.unique(mobility["type"]))
    n_mobility_types = len(np.unique(mobility["type"]))
    mobility_types = {r: v for (r, v) in mobility.groupby("region")}
    print(f"Lengths df={len(df)}, mobility_types={len(mobility_types)}")
    print(mobility.head(), len(mobility), n_mobility_types)

    mob = {}
    skipped = 0
    print(dates.min(), dates.max(), df.columns)
    start_ix = np.where(dates.min() == df.columns)[0][0]
    print(start_ix)
    start_ix += shift
    print(start_ix)
    # FIXME: check via dates between google and fb are not aligned
    # end_ix = np.where(dates.max() == df.columns)[0][0]
    end_ix = start_ix + len(dates)
    end_ix = min(end_ix, df.shape[1])
    for region in df.index:
        query = region
        if query not in mobility_types:
            # print(region)
            skipped += 1
            continue
        _m = th.zeros(df.shape[1], n_mobility_types)
        _v = mobility_types[query].iloc[:, 2:].transpose()
        _v = _v[: end_ix - start_ix]
        try:
            _m[start_ix:end_ix] = th.from_numpy(_v.values)
            if end_ix < df.shape[1]:
                _m[end_ix:] = _m[end_ix - 1]
        except Exception as e:
            print(region, query, _m.size(), _v.shape)
            print(_v)
            raise e
        assert (_m == _m).all()
        mob[region] = _m
    th.save(mob, pth.replace(".csv", ".pt"))
    print(skipped, df.shape[0])


def process_county_features(df):
    df_feat = pd.read_csv("features.csv", index_col="region")
    df_feat = (df_feat - df_feat.min(axis=0)) / df_feat.max(axis=0)
    feat = {}
    for region in df.index:
        _v = df_feat.loc[region]
        # inc = _v["median_income"]
        # inc = inc - inc.min()
        # _v["median_income"] = inc / min(1, inc.max()) * 100
        # _v = (_v - _v.mean()) / _v.std()
        feat[region] = th.from_numpy(_v.values)
    th.save(feat, "county_features.pt")


if __name__ == "__main__":
    parser = argparse.ArgumentParser("AT data")
    parser.add_argument("-source", default="data.csv")
    opt = parser.parse_args()

    df = pd.read_csv(opt.source, index_col="region")
    dates = pd.to_datetime(df.columns)
    df.columns = dates
    # throw away all-zero columns, i.e., days with no cases
    counts = df.sum(axis=0)
    df = df.iloc[:, np.where(counts > 0)[0]]

    process_time_features(df, f"fb/mobility_features.csv", 5)
    process_time_features(df, f"google/mobility_features_google.csv", 5)
    process_time_features(df, f"symptom_survey/smoothed_cli.csv", 0)
    process_time_features(df, f"symptom_survey/smoothed_mc.csv", 5)
    process_time_features(df, f"symptom_survey/smoothed_dc.csv", 5)
    process_time_features(df, f"weather/weather_features.csv", 5)
