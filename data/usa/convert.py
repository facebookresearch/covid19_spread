#!/usr/bin/env python3

import argparse
import numpy as np
import pandas as pd
import sys
import torch as th
from datetime import timedelta
from os import listdir
from os.path import isfile, join
from process_cases import SOURCES

# from process_cases import get_nyt

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


def process_time_features(df, pth, shift=0, merge_nyc=False, input_resolution="county"):
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
    if merge_nyc:
        mobility = merge_nyc_boroughs(mobility, n_mobility_types)
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
        if input_resolution == "county_state":
            _, query = region.split(", ")
        else:
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
    if input_resolution == "county_state":
        pth = pth.replace("state", "county_state")
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
    parser = argparse.ArgumentParser("US data")
    parser.add_argument("-metric", default="cases", choices=["cases", "deaths"])
    parser.add_argument("-with-features", default=False, action="store_true")
    parser.add_argument("-source", choices=SOURCES.keys(), default="nyt")
    parser.add_argument("-resolution", choices=["county", "state"], default="county")
    opt = parser.parse_args()
    population = read_population()
    df = SOURCES[opt.source](opt.metric)
    df.index = pd.to_datetime(df.index)

    # state_policies = pd.read_csv("policy_features.csv")

    # HACK: for deaths we do not have borough-level information
    if opt.metric == "deaths":
        population["New York City, New York"] = sum(
            [population[b] for b in nyc_boroughs]
        )

    dates = df.index
    df.columns = [c.split("_")[1] + ", " + c.split("_")[0] for c in df.columns]
    # df = df[[c for c in df.columns if c in population]]

    # drop all zero columns
    df = df[df.columns[(df.sum(axis=0) != 0).values]]

    population_counties = list(population.keys())
    df_pop = pd.DataFrame.from_dict(
        {
            "county": population_counties,
            "population": [population[county] for county in population_counties],
        }
    )

    df_pop = df_pop.set_index("county").rename(
        index={
            "Doña Ana, New Mexico": "Do̱a Ana, New Mexico",
            "LaSalle, Louisiana": "La Salle, Louisiana",
        }
    )

    df = df.transpose()  # row for each county, columns correspond to dates...
    county_id = {c: i for i, c in enumerate(df.index)}
    # make sure counts are strictly increasing
    df = df.cummax(axis=1)

    # throw away all-zero columns, i.e., days with no cases
    counts = df.sum(axis=0)
    df = df.iloc[:, np.where(counts > 0)[0]]

    if opt.resolution == "state":
        df = df.groupby(lambda x: x.split(", ")[-1]).sum()
        df = df.drop(
            index=["Virgin Islands", "Northern Mariana Islands", "Puerto Rico", "Guam"],
            errors="ignore",
        )
        df_pop = df_pop.groupby(lambda x: x.split(", ")[-1]).sum()

    assert df.shape[0] == len(df_pop.loc[df.index])
    df_pop = df_pop.loc[df.index]

    df_pop.to_csv(
        f"population_{opt.resolution}.csv", index_label="county", header=False
    )

    df.to_csv(f"data_{opt.metric}.csv", index_label="region")

    if opt.resolution == "county":
        # Build state graph...
        adj = np.zeros((len(df), len(df)))
        for _, g in df.groupby(lambda x: x.split(", ")[-1]):
            idxs = np.array([county_id[c] for c in g.index])
            adj[np.ix_(idxs, idxs)] = 1

        print(adj)
        th.save(th.from_numpy(adj), "state_graph.pt")

    # process_county_features(df)
    if opt.with_features:
        res = opt.resolution
        merge_nyc = opt.metric == "deaths" and res == "county"
        process_time_features(df, f"testing/ratio_features_{res}.csv", 0, merge_nyc)
        process_time_features(df, f"testing/total_features_{res}.csv", 0, merge_nyc)
        for signal, lag in [
            ("symptom-survey/doctor-visits_smoothed_adj_cli-{}.csv", 2),
            ("symptom-survey/fb-survey_smoothed_wcli-{}.csv", 0),
            ("symptom-survey/fb-survey_smoothed_hh_cmnty_cli-{}.csv", 0),
            # ("symptom-survey/fb-survey_smoothed_wearing_mask-{}.csv", 5),
            ("fb/mobility_features_{}_fb.csv", 5),
            ("google/mobility_features_{}_google.csv", 5),
        ]:
            if res == "county":
                process_time_features(
                    df, signal.format("county"), lag, merge_nyc, "county",
                )
                process_time_features(
                    df, signal.format("state"), lag, merge_nyc, "county_state",
                )
            else:
                process_time_features(
                    df, signal.format("state"), lag, merge_nyc, "state",
                )
        process_time_features(df, f"fb/mobility_features_{res}_fb.csv", 5, merge_nyc)
        process_time_features(
            df, f"google/mobility_features_{res}_google.csv", 5, merge_nyc
        )
        process_time_features(df, f"google/weather_features_{res}.csv", 5, merge_nyc)
        process_time_features(df, f"google/epi_features_{res}.csv", 7, merge_nyc)
        # process_time_features(df, f"shifted_features_{res}.csv", 0, merge_nyc)
        if res == "state":
            process_time_features(df, f"google/hosp_features_{res}.csv", 0, merge_nyc)
            process_time_features(df, f"shifted_features_{res}.csv", 0, merge_nyc)

# n_policies = len(np.unique(state_policies["policy"]))
# state_policies = {s: v for (s, v) in state_policies.groupby("state")}
# pols = th.zeros(df.shape[0], df.shape[1], n_policies)
# for i, region in enumerate(df.index):
#     state = region.split(", ")[1]
#     _p = state_policies[state].iloc[:, 2:].transpose()
#     pols[i] = th.from_numpy(_p.values)
# th.save(pols, "policy_features.pt")
