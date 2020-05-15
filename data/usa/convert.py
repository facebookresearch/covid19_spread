#!/usr/bin/env python3

import numpy as np
import pandas as pd
import sys
import torch as th
from datetime import timedelta
from os import listdir
from os.path import isfile, join
from process_cases import get_nyt


def county_id(county, state):
    return f"{county}, {state}"


def standardize_county(county):
    return (
        county.replace(" County", "")
        .replace(" Parish", "")
        .replace(" Municipality", "")
    )


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
        counties = map(lambda c: county_id(standardize_county(c), state), counties)
        pop = df.iloc[:, 1].values
        population.update(zip(counties, pop))
    return population


metric = sys.argv[1] if len(sys.argv) == 2 else "cases"
population = read_population()
df = get_nyt(metric)
df.index = pd.to_datetime(df.index)
print(df.head())

df_feat = pd.read_csv("features.csv", index_col="region")
state_policies = pd.read_csv("policy_features.csv")

# HACK: for deaths we do not have borough-level information
if metric == "deaths":
    boroughs = [
        "Bronx, New York",
        "Brooklyn, New York",
        "Queens, New York",
        "Manhattan, New York",
        "Richmond, New York",
    ]
    population["New York City, New York"] = sum([population[b] for b in boroughs])
    boroughs[1] = "Kings, New York"
    boroughs[3] = "New York, New York"
    df_feat.loc["New York City, New York"] = np.mean([df_feat.loc[b] for b in boroughs])

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
df = df.cummax(axis=1)
df.to_csv(f"data_{metric}.csv", index_label="region")

# Build state graph...
adj = np.zeros((len(df), len(df)))
for _, g in df.groupby(lambda x: x.split(", ")[-1]):
    idxs = np.array([county_id[c] for c in g.index])
    adj[np.ix_(idxs, idxs)] = 1

print(adj)
th.save(th.from_numpy(adj), "state_graph.pt")

for region in df.index:
    df_feat.loc[region]
df_feat = df_feat.loc[df.index]
inc = df_feat["median_income"]
inc = inc - inc.min()
df_feat["median_income"] = inc / inc.max() * 100
# df_feat = (df_feat - df_feat.mean(axis=0)) / df_feat.std(axis=0)
print(df_feat)
th.save(th.from_numpy(df_feat.values), "county_features.pt")

n_policies = len(np.unique(state_policies["policy"]))
state_policies = {s: v for (s, v) in state_policies.groupby("state")}
pols = th.zeros(df.shape[0], df.shape[1], n_policies)
for i, region in enumerate(df.index):
    state = region.split(", ")[1]
    _p = state_policies[state].iloc[:, 2:].transpose()
    pols[i] = th.from_numpy(_p.values)
th.save(pols, "policy_features.pt")
