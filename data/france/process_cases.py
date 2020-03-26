import requests

# https://www.data.gouv.fr/fr/datasets/cas-confirmes-dinfection-au-covid-19-par-region/
# CANVEATS:
# - Only diagnosed cases - doesn't take into account suspicious but not tested cases
# - Numbers are reported for the day of diagnosis, not the day of contamination
# - Also depends on the number of tests that the french gouvernement can do/does
r = requests.get("https://www.data.gouv.fr/fr/datasets/r/fa9b8fc8-35d5-4e24-90eb-9abe586b0fa5", timeout=5)
r.encoding = 'utf-8'

import io
import csv
from datetime import datetime

f = io.StringIO(r.text)
reader = csv.reader(f)
header = next(reader)
assert header[0] == 'Date'
nodes = header[1:]
dataset = []
rows = [r for r in reader]
print('Processing from', rows[0][0], 'to', rows[-1][0])
# 8 march and 9 march are missing in the dataset for some reason, let's add those using
# Source: https://www.eficiens.com/nombre-de-cas-coronavirus-par-region/
rows.append([
    '2020-03-08',
    146,  # Auvergne-Rhône-Alpes
    112,  # Bourgogne-Franche-Comté
    59,  # Bretagne
    17,  # Centre-Val de Loire
    33,  # Corse ??
    262,  # Grand Est
    192,  # Hauts-de-France ??
    243,  # Ile-de-France
    22,  # Normandie
    29,  # Nouvelle-Aquitaine
    47,  # Occitanie
    25,  # Pays de la Loire
    37,  # Provence-Alpes-Côte d’Azur
    0,  # Guadeloupe ??
    0,  # Saint-Barthélémy ??
    0,  # Saint-Martin ??
    0,  # Guyane ??
    0,  # Martinique ??
    0,  # Mayotte ??
    0,  # La Réunion ??
])
rows.append([
    '2020-03-09',
    182,  # Auvergne-Rhône-Alpes
    119,  # Bourgogne-Franche-Comté
    71,  # Bretagne
    17,  # Centre-Val de Loire
    33,  # Corse
    310,  # Grand Est
    192,  # Hauts-de-France
    300,  # Ile-de-France
    25,  # Normandie
    39,  # Nouvelle-Aquitaine
    60,  # Occitanie
    26,  # Pays de la Loire
    52,  # Provence-Alpes-Côte d’Azur
    0,  # Guadeloupe ??
    0,  # Saint-Barthélémy ??
    0,  # Saint-Martin ??
    0,  # Guyane ??
    0,  # Martinique ??
    0,  # Mayotte ??
    0,  # La Réunion ??
])

first_outbreak_day = None
def date_to_timestamp(s):
    s = s.replace('/', '-')
    return datetime.timestamp(datetime.strptime(s, '%Y-%m-%d'))

for row in rows:
    date = row[0]
    if first_outbreak_day is None:
        first_outbreak_day = date
    assert len(row[1:]) == len(nodes)
    for node_idx, count in enumerate(row[1:]):
        dataset.append({
            'date': date,
            'days_since_outbreak': (date_to_timestamp(date) - date_to_timestamp(first_outbreak_day)) / (60 * 60 * 24),
            'node_idx': node_idx,
            'count': int(count) # Total count
        })

import random
_ts = []
_ns = []

# Split into individual cases instead of aggregates
for node_idx in range(len(nodes)):
    node_subset = [d for d in dataset if d['node_idx'] == node_idx]
    node_subset.sort(key=lambda d: d['days_since_outbreak'])
    node_cases = []
    # We skip initial number of cases on first day
    for prev, now in zip(node_subset, node_subset[1:]):
        new_cases = now['count'] - prev['count']
        ndays = now['days_since_outbreak']
        for _ in range(new_cases):
            _ts.append(random.uniform(ndays - 1, ndays))
            _ns.append(node_idx)

print('Dumping dataset with', len(_ts), 'cases')


# Now that we have all the data, convert it to the right format
import h5py
import numpy as np
import pandas as pd

from collections import defaultdict as ddict
from itertools import count


fout = "timeseries.h5"
str_dt = h5py.special_dtype(vlen=str)
ds_dt = h5py.special_dtype(vlen=np.dtype("int"))
ts_dt = h5py.special_dtype(vlen=np.dtype("float32"))
with h5py.File(fout, "w") as fout:
    _dnames = fout.create_dataset("nodes", (len(nodes),), dtype=str_dt)
    _dnames[:] = nodes
    _cnames = fout.create_dataset("cascades", (1,), dtype=str_dt)
    _cnames[:] = ["covid19_fr"]
    ix = np.argsort(_ts)
    node = fout.create_dataset("node", (1,), dtype=ds_dt)
    node[0] = np.array(_ns, dtype=np.int)[ix]
    time = fout.create_dataset("time", (1,), dtype=ts_dt)
    time[0] = np.array(_ts, dtype=np.float)[ix]
print('Done')