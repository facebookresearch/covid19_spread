#!/usr/bin/env python3


from subprocess import check_call
import os
import csv
import pandas
import numpy as np
import h5py
import json
import re
from collections import defaultdict
import itertools
import shutil


def get_nyt():
    CASES_URL = 'https://raw.githubusercontent.com/nytimes/covid-19-data/master/us-counties.csv'
    check_call(['wget', CASES_URL, '-O','us-counties.csv'])
    df = pandas.read_csv('us-counties.csv')
    df['loc'] = df['state'] + '_' + df['county']
    pivot = df.pivot_table(values='cases', columns=['loc'], index='date')
    pivot = pivot.fillna(0)
    pivot.index = pandas.to_datetime(pivot.index)
    return pivot

if not os.path.exists('us-state-neighbors.json'):
    check_call(['wget', 'https://gist.githubusercontent.com/PrajitR/0afccfa4dc4febe59276/raw/7a73603f5346210ae34845c43094f0daabfd4d49/us-state-neighbors.json'])
if not os.path.exists('states_hash.json'):
    check_call(['wget', 'https://gist.githubusercontent.com/mshafrir/2646763/raw/8b0dbb93521f5d6889502305335104218454c2bf/states_hash.json'])

neighbors = json.load(open('us-state-neighbors.json', 'r'))
state_map = json.load(open('states_hash.json','r'))

# Convert abbreviated names to full state names
neighbors = {state_map[k]: [state_map[v] for v in vs] for k, vs in neighbors.items()}

df = get_nyt()

counter = itertools.count()
county_ids = defaultdict(counter.__next__)

# If an h5 file already exists, use a consistent naming
if os.path.exists('timeseries.h5'):
    with h5py.File('timeseries.h5', 'r') as hf:
        shutil.copy('timeseries.h5', f'timeseries_{hf.attrs["basedate"]}.h5')
        nodes = hf['nodes'][:]
        for county in nodes:
            county_ids[county]

def mk_episode(counties):
    ats = np.arange(len(df)) + 1
    timestamps = []
    nodes = []
    county_id = 0

    for county in counties:
        ws = df[county].values
        ix = np.where(ws > 0)[0]
        if len(ix) < 1:
            continue
        ts = ats[ix]
        ws = np.diff([0] + ws[ix].tolist())
        es = []

        for i in range(len(ts)):
            w = int(ws[i])
            if w <= 0:
                continue
            tp = ts[i] - 1
            _es = sorted(np.random.uniform(tp, ts[i], w))
            if len(es) > 0:
                assert es[-1] < _es[0], (_es[0], es[-1])
            es += _es
        es = np.array(es)
        if len(es) > 0:
            timestamps.append(es)
            nodes.append(np.full(es.shape, county_ids[county]))
    times = np.concatenate(timestamps)
    idx = np.argsort(times)
    return times[idx], np.concatenate(nodes)[idx]


episodes = []
for state, ns in neighbors.items():
    states = set([state] + ns)
    regex = '|'.join(f'^{s}' for s in states)
    cols = [c for c in df.columns if re.match(regex, c)]
    episodes.append(mk_episode(cols))

str_dt = h5py.special_dtype(vlen=str)
ds_dt = h5py.special_dtype(vlen=np.dtype("int"))
ts_dt = h5py.special_dtype(vlen=np.dtype("float32"))

counties, cids = zip(*list(county_ids.items()))

times, entities = zip(*episodes)

with h5py.File('timeseries.h5', "w") as hf:
    hf.create_dataset('nodes', data=np.array(counties, dtype='O')[np.argsort(cids)], dtype=str_dt)
    hf.create_dataset('cascades', data=np.arange(len(episodes)))
    hf.create_dataset('node', data=entities, dtype=ds_dt)
    hf.create_dataset('time', data=times, dtype=ts_dt)
    hf.attrs['basedate'] = str(df.index.max().date())    

    # Group all events into a single episode
    processed = np.array([], dtype='int')
    all_times = []
    all_nodes = []
    for idx in range(len(times)):
        # Extract events corresponding to entities we haven't processed yet.
        mask = ~np.in1d(entities[idx], processed)
        all_nodes.append(entities[idx][mask])
        all_times.append(times[idx][mask])
        unique_nodes = np.unique(all_nodes[-1])
        assert np.intersect1d(processed, unique_nodes).size == 0
        processed = np.concatenate([processed, unique_nodes])

    all_times = np.concatenate(all_times)
    idx = all_times.argsort()
    all_times = all_times[idx]
    all_nodes = np.concatenate(all_nodes)[idx]
    hf.create_dataset('all_times', data=all_times, dtype='float64')
    hf.create_dataset('all_nodes', data=all_nodes)
    print(f'{len(counties)} counties')
