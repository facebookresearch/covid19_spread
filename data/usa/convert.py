#!/usr/bin/env python3

import argparse
import numpy as np
import pandas as pd
import sys
import torch as th
from datetime import timedelta
from os import listdir
from os.path import isfile, join
from process_cases import get_nyt

sys.path.append("../../")
from common import standardize_county_name

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


def read_population():
    poppath = "../population-data/US-states"
    fpops = [f for f in listdir(poppath) if isfile(join(poppath, f))]

    population = {}
    for fpop in fpops:
        state = fpop.split("-")[:-1]
        state = " ".join(map(lambda s: s.capitalize(), state))
        state = state.replace(" Of ", " of ")
        df = pd.read_csv(join(poppath, fpop), header=None)
        counties = df.iloc[:, 0].values
        counties = map(lambda c: county_id(standardize_county_name(c), state), counties)
        pop = df.iloc[:, 1].values
        population.update(zip(counties, pop))
    return population


def process_mobility(df, prefix, shift=0, merge_nyc=False):
    mobility = pd.read_csv(f"{prefix}/mobility_features.csv")
    dates = pd.to_datetime(mobility.columns[2:])
    dates = dates[np.where(dates >= df.columns.min())[0]]
    mobility = mobility[
        mobility.columns[:2].to_list()
        + list(map(lambda d: d.strftime("%Y-%m-%d"), dates))
    ]

    n_mobility_types = len(np.unique(mobility["type"]))
    mobility_types = {r: v for (r, v) in mobility.groupby("region")}
    if merge_nyc:
        mobility = merge_nyc_boroughs(mobility, n_mobility_types)
    print(mobility.head(), len(mobility))

    mob = {}
    skipped = 0
    start_ix = np.where(dates.min() == df.columns)[0][0] + shift
    # FIXME: check via dates between google and fb are not aligned
    # end_ix = np.where(dates.max() == df.columns)[0][0]
    end_ix = start_ix + len(dates)
    end_ix = min(end_ix, df.shape[1])
    for region in df.index:
        if region not in mobility_types:
            # print(region)
            skipped += 1
            continue
        _m = th.zeros(df.shape[1], n_mobility_types)
        _v = mobility_types[region].iloc[:, 2:].transpose()
        _v = _v[: end_ix - start_ix]
        _m[start_ix:end_ix] = th.from_numpy(_v.values)
        mob[region] = _m
    th.save(mob, f"{prefix}/mobility_features.pt")
    print(skipped, df.shape[0])


def process_symptom_survey(df, shift=1):
    symptoms = pd.read_csv(
        "symptom-survey/data-smoothed_cli-state.csv", index_col="region"
    )
    sym = {}
    skipped = 0
    dates = pd.to_datetime(symptoms.columns[1:])
    start_ix = np.where(dates.min() == df.columns)[0][0] + shift
    end_ix = start_ix + len(dates)
    end_ix = min(end_ix, df.shape[1])
    for region in df.index:
        _, state = region.split(", ")
        if state not in symptoms.index:
            print("Skipping {region}")
            skipped += 1
            continue
        _m = th.zeros(df.shape[1])
        _v = symptoms.loc[state]  # .rolling(7).mean()
        _v = _v[: end_ix - start_ix + 1]
        _m[start_ix:end_ix] = th.from_numpy(_v.values[1:])
        sym[region] = _m.unsqueeze(1)
    th.save(sym, "symptom-survey/features.pt")
    print(skipped, df.shape[0])


def process_county_features(df):
    df_feat = pd.read_csv("features.csv", index_col="region")
    feat = {}
    for region in df.index:
        _v = df_feat.loc[region]
        inc = _v["median_income"]
        inc = inc - inc.min()
        # _v["median_income"] = inc / min(1, inc.max()) * 100
        _v = (_v - _v.mean()) / _v.std()
        feat[region] = th.from_numpy(_v.values)
    th.save(feat, "county_features.pt")


def process_testing(df):
    tests = pd.read_csv("testing/testing_features.csv", index_col="region")
    ts = {}
    skipped = 0
    dates = pd.to_datetime(tests.columns[1:])
    start_ix = np.where(dates.min() == df.columns)[0][0]
    end_ix = start_ix + len(dates)
    for region in df.index:
        _, state = region.split(", ")
        if state not in tests.index:
            print("Skipping {region}")
            skipped += 1
            continue
        _m = th.zeros(df.shape[1])
        _v = tests.loc[state]  # .rolling(7).mean()
        _v = np.diff(np.log(_v.values[1:] + 1))
        _m[start_ix : end_ix - 1] = th.from_numpy(
            _v
        )  # make last zero (i.e., testing doesn't change)
        ts[region] = _m.unsqueeze(1)
    th.save(ts, "testing/features.pt")
    print(skipped, df.shape[0])


if __name__ == "__main__":
    parser = argparse.ArgumentParser("US data")
    parser.add_argument("-metric", default="cases", choices=["cases", "deaths"])
    parser.add_argument("-with-features", default=False, action="store_true")
    opt = parser.parse_args()

    population = read_population()
    df = get_nyt(opt.metric)
    df.index = pd.to_datetime(df.index)
    print(df.tail())

    # df_feat = pd.read_csv("features.csv", index_col="region")
    # state_policies = pd.read_csv("policy_features.csv")

    # HACK: for deaths we do not have borough-level information
    if opt.metric == "deaths":
        population["New York City, New York"] = sum(
            [population[b] for b in nyc_boroughs]
        )
        # nyc_boroughs[1] = "Kings, New York"
        # nyc_boroughs[3] = "New York, New York"
        # df_feat.loc["New York City, New York"] = np.mean([df_feat.loc[b] for b in boroughs])

    dates = df.index
    df.columns = [c.split("_")[1] + ", " + c.split("_")[0] for c in df.columns]
    print(df.columns)
    df = df[[c for c in df.columns if c in population]]

    # drop all zero columns
    df = df[df.columns[(df.sum(axis=0) != 0).values]]
    print(df.head())

    population_counties = list(population.keys())
    df_pop = pd.DataFrame.from_dict(
        {
            "county": population_counties,
            "population": [population[county] for county in population_counties],
        }
    )
    df_pop.to_csv("population.csv", index=False, header=False)
    df = df.transpose()  # row for each county, columns correspond to dates...
    county_id = {c: i for i, c in enumerate(df.index)}
    # make sure counts are strictly increasing
    df = df.cummax(axis=1)

    # throw away all-zero columns, i.e., days with no cases
    counts = df.sum(axis=0)
    df = df.iloc[:, np.where(counts > 0)[0]]

    df.to_csv(f"data_{opt.metric}.csv", index_label="region")

    df.groupby(lambda x: x.split(", ")[-1]).sum().to_csv(
        f"data_states_{opt.metric}.csv", index_label="region"
    )

    # Build state graph...
    adj = np.zeros((len(df), len(df)))
    for _, g in df.groupby(lambda x: x.split(", ")[-1]):
        idxs = np.array([county_id[c] for c in g.index])
        adj[np.ix_(idxs, idxs)] = 1

    print(adj)
    th.save(th.from_numpy(adj), "state_graph.pt")

    # process_county_features(df)
    if opt.with_features:
        merge_nyc = opt.metric == "deaths"
        process_testing(df)
        process_symptom_survey(df, 5)
        process_mobility(df, "google", 10, merge_nyc)
        process_mobility(df, "fb", 10, merge_nyc)


# n_policies = len(np.unique(state_policies["policy"]))
# state_policies = {s: v for (s, v) in state_policies.groupby("state")}
# pols = th.zeros(df.shape[0], df.shape[1], n_policies)
# for i, region in enumerate(df.index):
#     state = region.split(", ")[1]
#     _p = state_policies[state].iloc[:, 2:].transpose()
#     pols[i] = th.from_numpy(_p.values)
# th.save(pols, "policy_features.pt")
