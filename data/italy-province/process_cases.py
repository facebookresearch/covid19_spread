import pandas as pd
from datetime import date
from datetime import datetime

link = 'https://raw.githubusercontent.com/pcm-dpc/COVID-19/master/dati-province/dpc-covid19-ita-province.csv' 

data = pd.read_csv(link)
print('run at: {}'.format(date.today()))
print('first case: 2020-01-29') # first case in Italy is January 29
# data.head()

_dates = (data.data).unique()
_nodes = data.codice_provincia.unique()
nodes = list(_nodes)
rows = []

for _date in _dates:
    row_tmp = [_date[:-9]] # remove hour info
    
    row = data.groupby('data').get_group(_date)
    # make sure region order is preserved
    assert (nodes == row['codice_provincia'].values).all()
    
    row_tmp += list(row['totale_casi'].values)
    rows.append(row_tmp)
    
# rows and nodes are ready to process
# below is copy of the code from France
dataset = []

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
    _cnames[:] = ["covid19_it"]
    ix = np.argsort(_ts)
    node = fout.create_dataset("node", (1,), dtype=ds_dt)
    node[0] = np.array(_ns, dtype=np.int)[ix]
    time = fout.create_dataset("time", (1,), dtype=ts_dt)
    time[0] = np.array(_ts, dtype=np.float)[ix]
print('Done')
