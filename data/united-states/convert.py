#!/usr/bin/env python3

import numpy as np
import pandas as pd
import torch as th
from collections import defaultdict
from os import listdir
from os.path import isfile, join


def county_id(county, state):
    return f"{county}, {state}"


def standardize_county(county):
    return (
        county.replace(" County", "")
        .replace(" Parish", "")
        .replace(" Municipality", "")
    )


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

# print(population)

dparser = lambda x: pd.to_datetime(x, format="%Y-%m-%d")
df = pd.read_csv(
    "raw.csv",
    date_parser=dparser,
    parse_dates=["date"],
    usecols=["date", "state", "county", "fips", "cases"],
)
df = df.dropna()
df["fips"] = df["fips"].astype(int)

print(df.head())

dates = np.unique(df["date"])
counties = []
df_agg = df.groupby(["county", "fips", "state"])
for (name, _, state), group in df_agg:
    counties.append(county_id(name, state))


df_pop = pd.DataFrame(
    {"county": counties, "population": [population[c] for c in counties]}
)
df_pop.to_csv("population.csv", index=False, header=False)
index = []

cols = {
    pd.to_datetime(date).strftime("%Y-%m-%d"): np.zeros(len(counties), dtype=np.int)
    for date in dates
}
cols["region"] = counties

state_graph = defaultdict(list)
county_index = {c: i for i, c in enumerate(counties)}

for (name, fips, state), group in df_agg:
    group = group.sort_values(by="date")
    cid = county_id(name, state)
    cix = county_index[cid]
    print(cid, cix)
    _dates = group["date"].to_numpy()
    _cases = group["cases"].to_numpy()
    state_graph[state].append(cid)
    for i, _date in enumerate(_dates):
        dix = pd.to_datetime(_date).strftime("%Y-%m-%d")
        cols[dix][cix] = _cases[i]

df = pd.DataFrame(cols)
df.set_index("region", inplace=True)
df.to_csv("data.csv")

state_graph = dict(state_graph)
print(state_graph["New Jersey"])

adj = np.zeros((len(counties), len(counties)))
for (name, fips, state), group in df_agg:
    cid = county_id(name, state)
    cix = county_index[cid]
    ix = [county_index[c] for c in state_graph[state]]
    adj[cix, ix] = 1
print(adj)
th.save(th.from_numpy(adj), "state_graph.pt")
