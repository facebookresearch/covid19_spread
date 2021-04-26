#!/usr/bin/env python3
# Copyright (c) 2021-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import numpy as np
import pandas as pd
import torch as th
from os import listdir
from os.path import isfile, join
from covid19_spread.data.usa.process_cases import SOURCES
import warnings
from covid19_spread.common import standardize_county_name
import os
import multiprocessing as mp

SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))

nyc_boroughs = [
    "Bronx, New York",
    "Kings, New York",
    "Queens, New York",
    "New York, New York",
    "Richmond, New York",
]


def county_id(county, state):
    return f"{county}, {state}"


def rename_nyc_boroughs(county_name):
    if county_name in nyc_boroughs:
        return "New York City, New York"
    else:
        return county_name


def merge_nyc_boroughs(df, ntypes):
    df["region"] = df["region"].transform(rename_nyc_boroughs)
    prev_len = len(df)
    df = df.groupby(["region", "type"]).mean()
    assert len(df) == prev_len - ntypes * 4, (prev_len, len(df))
    df = df.reset_index()
    print(df[df["region"] == "New York City, New York"])
    return df


def process_time_features(df, pth, shift=0, merge_nyc=False, input_resolution="county"):
    time_features = pd.read_csv(pth)
    if input_resolution == "county_state":
        # Expand state level time features to each county in `df`
        idx = df.rename_axis("county").reset_index()[["county"]]
        idx["region"] = idx["county"].apply(lambda x: x.split(", ")[-1])
        time_features = time_features.merge(idx, on="region").drop(columns="region")
        time_features = time_features.rename(columns={"county": "region"})
    time_feature_regions = time_features["region"].unique()
    ncommon = len(df.index.intersection(time_feature_regions))
    if ncommon != len(df):
        missing = set(df.index).difference(set(time_feature_regions))
        warnings.warn(
            f"{pth}: Missing time features for the following regions: {list(missing)}"
        )
    if ncommon != len(time_feature_regions):
        ignoring = set(time_feature_regions).difference(set(df.index))
        warnings.warn(
            f"{pth}: Ignoring time features for the following regions: {list(ignoring)}"
        )
        time_features = time_features[time_features["region"].isin(set(df.index))]
    if merge_nyc:
        time_features = merge_nyc_boroughs(
            time_features, len(time_features["type"].unique())
        )
    # Transpose to have two level columns (region, type) and dates as index
    time_features = time_features.set_index(["region", "type"]).transpose().sort_index()
    time_features.index = pd.to_datetime(time_features.index)
    # Trim prefix if it starts before the dates in `df`
    time_features = time_features.loc[time_features.index >= df.columns.min()]
    # Fill in dates that are missing in `time_features` that exist in `df`
    time_features = time_features.reindex(df.columns)
    # Shift time features UP by `shift` days
    time_features = time_features.shift(shift)
    # forward fill the missing values
    time_features = time_features.fillna(method="ffill")
    # Fill the beginning end with zeros if null
    time_features = time_features.fillna(0)
    time_features = time_features[time_features.columns.sort_values()]
    feature_tensors = {
        region: th.from_numpy(time_features[region].values)
        for region in time_features.columns.get_level_values(0).unique()
    }
    if input_resolution == "county_state":
        pth = pth.replace("state", "county_state")
    th.save(feature_tensors, pth.replace(".csv", ".pt"))


def run_par(fs, args, kwargs, max_par=None):
    if not isinstance(fs, list):
        fs = [fs] * len(args)
    if "MAX_PARALLELISM" in os.environ:
        max_par = int(os.environ["MAX_PARALLELISM"])
    print(f"Max parallelism = {max_par}")
    if max_par is not None and max_par <= 1:
        for _f, _args, _kwargs in zip(fs, args, kwargs):
            _f(*_args, **_kwargs)
        return
    with mp.Pool(max_par) as pool:
        results = [
            pool.apply_async(f, args=a, kwds=k) for f, a, k in zip(fs, args, kwargs)
        ]
        [r.get() for r in results]


def create_time_features():
    from .symptom_survey import prepare as ss_prepare
    from .fb import prepare as fb_prepare
    from .google import prepare as google_prepare
    from .testing import prepare as testing_prepare

    fs = [ss_prepare, fb_prepare, google_prepare, testing_prepare]
    run_par(fs, [()] * len(fs), [{}] * len(fs))


def main(metric, with_features, source, resolution):
    df = SOURCES[source](metric)
    df.index = pd.to_datetime(df.index)

    dates = df.index
    df.columns = [c.split("_")[1] + ", " + c.split("_")[0] for c in df.columns]

    # drop all zero columns
    df = df[df.columns[(df.sum(axis=0) != 0).values]]

    df = df.transpose()  # row for each county, columns correspond to dates...

    # make sure counts are strictly increasing
    df = df.cummax(axis=1)

    # throw away all-zero columns, i.e., days with no cases
    counts = df.sum(axis=0)
    df = df.iloc[:, np.where(counts > 0)[0]]

    if resolution == "state":
        df = df.groupby(lambda x: x.split(", ")[-1]).sum()
        df = df.drop(
            index=["Virgin Islands", "Northern Mariana Islands", "Puerto Rico", "Guam"],
            errors="ignore",
        )

    county_id = {c: i for i, c in enumerate(df.index)}

    df.to_csv(f"{SCRIPT_DIR}/data_{metric}.csv", index_label="region")
    df[df.index.str.endswith("New York")].to_csv(
        f"{SCRIPT_DIR}/data_{metric}_ny.csv", index_label="region"
    )
    df[df.index.str.endswith("Florida")].to_csv(
        f"{SCRIPT_DIR}/data_{metric}_fl.csv", index_label="region"
    )

    if resolution == "county":
        # Build state graph...
        adj = np.zeros((len(df), len(df)))
        for _, g in df.groupby(lambda x: x.split(", ")[-1]):
            idxs = np.array([county_id[c] for c in g.index])
            adj[np.ix_(idxs, idxs)] = 1

        print(adj)
        th.save(th.from_numpy(adj), f"{SCRIPT_DIR}/state_graph.pt")

    if with_features:
        create_time_features()
        res = resolution
        merge_nyc = metric == "deaths" and res == "county"

        features = [
            (f"{SCRIPT_DIR}/testing/ratio_features_{res}.csv", 0, res),
            (f"{SCRIPT_DIR}/testing/total_features_{res}.csv", 0, res),
            (f"{SCRIPT_DIR}/fb/mobility_features_{res}_fb.csv", 5, res),
            (f"{SCRIPT_DIR}/google/mobility_features_{res}_google.csv", 5, res),
            (f"{SCRIPT_DIR}/google/weather_features_{res}.csv", 5, res),
            (f"{SCRIPT_DIR}/google/epi_features_{res}.csv", 7, res),
            (f"{SCRIPT_DIR}/google/epi_features_{res}.csv", 7, res),
        ]
        if res == "state":
            features.append((f"{SCRIPT_DIR}/google/hosp_features_{res}.csv", 0, res))
            features.append((f"{SCRIPT_DIR}/shifted_features_{res}.csv", 0, res))
            features.append((f"{SCRIPT_DIR}/google/vaccination_state.csv", 0, "state"))
        else:
            features.append(
                (f"{SCRIPT_DIR}/google/vaccination_state.csv", 0, "county_state")
            )

        for signal, lag in [
            (f"{SCRIPT_DIR}/symptom_survey/doctor-visits_smoothed_adj_cli-{{}}.csv", 2),
            (f"{SCRIPT_DIR}/symptom_survey/fb-survey_smoothed_wcli-{{}}.csv", 0),
            (
                f"{SCRIPT_DIR}/symptom_survey/fb-survey_smoothed_hh_cmnty_cli-{{}}.csv",
                0,
            ),
            (
                f"{SCRIPT_DIR}/symptom_survey/fb-survey_smoothed_wearing_mask_all-{{}}.csv",
                5,
            ),
            (
                f"{SCRIPT_DIR}/symptom_survey/fb-survey_smoothed_wothers_masked-{{}}.csv",
                5,
            ),
            (
                f"{SCRIPT_DIR}/symptom_survey/fb-survey_smoothed_wcovid_vaccinated_or_accept-{{}}.csv",
                5,
            ),
            (f"{SCRIPT_DIR}/fb/mobility_features_{{}}_fb.csv", 5),
            (f"{SCRIPT_DIR}/google/mobility_features_{{}}_google.csv", 5),
        ]:
            if res == "county":
                features.append((signal.format("county"), lag, "county"))
                features.append((signal.format("state"), lag, "county_state"))
            else:
                features.append((signal.format("state"), lag, "state"))

        features = [(df, pth, lag, merge_nyc, r) for pth, lag, r in features]
        run_par([process_time_features] * len(features), features, [{}] * len(features))


if __name__ == "__main__":
    parser = argparse.ArgumentParser("US data")
    parser.add_argument("-metric", default="cases", choices=["cases", "deaths"])
    parser.add_argument("-with-features", default=False, action="store_true")
    parser.add_argument("-source", choices=SOURCES.keys(), default="nyt")
    parser.add_argument("-resolution", choices=["county", "state"], default="county")
    opt = parser.parse_args()
    main(opt.metric, opt.with_features, opt.source, opt.resolution)
