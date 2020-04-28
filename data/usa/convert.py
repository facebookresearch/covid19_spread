#!/usr/bin/env python3

import numpy as np
import pandas as pd
import torch as th
from collections import defaultdict
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

df = get_nyt()

print(df.head())

dates = df.index
df.columns = [c.split("_")[1] + ", " + c.split("_")[0] for c in df.columns]
df = df[[c for c in df.columns if c in population]]

df_pop = pd.DataFrame.from_dict(
    {"county": df.columns, "population": [population[c] for c in df.columns]}
)
df_pop.to_csv("population.csv", index=False, header=False)
df = df.transpose()  # row for each county, columns correspond to dates...
county_id = {c: i for i, c in enumerate(df.index)}
df.to_csv("data.csv", index_label='region')

# Build state graph...
adj = np.zeros((len(df), len(df)))
for _, g in df.groupby(lambda x: x.split(', ')[-1]):
    idxs = np.array([county_id[c] for c in g.index])
    adj[np.ix_(idxs, idxs)] = 1

print(adj)
th.save(th.from_numpy(adj), "state_graph.pt")
