#!/usr/bin/env python3

import numpy as np
import pandas as pd
import sys
from datetime import timedelta


policy_temporal = [
    ("State of emergency", None),
    ("Date closed K-12 schools", None),
    ("Closed day cares", None),
    ("Date banned visitors to nursing homes", None),
    ("Stay at home/ shelter in place", "End stay at home/shelter in place"),
    ("Closed non-essential businesses", "Reopen businesses"),
    ("Closed restaurants except take out", "Reopen restaurants"),
    ("Closed gyms", "Repened gyms"),
    ("Closed movie theaters", "Reopened movie theaters"),
    (
        "Suspended elective medical/dental procedures",
        "Resumed elective medical procedures",
    ),
    ("Mandate face mask use by customers in essential businesses", None),
    ("Mandate face mask use by employees in essential businesses", None),
]


nstates = 51
df = pd.read_csv(sys.argv[1], index_col="State").iloc[:nstates, :]
global_start_date = pd.to_datetime(sys.argv[2])
global_end_date = pd.to_datetime(sys.argv[3])

days = (global_end_date - global_start_date).days + 1
ix = np.array([global_start_date + timedelta(d) for d in range(days)])
colnames = [i.strftime("%Y-%m-%d") for i in ix]

dfs = []
for (start_col, end_col) in policy_temporal:
    start = df[start_col]
    six = start != "0"
    start = start[six]
    if end_col is not None:
        end = df[end_col][six]
        end[end == "0"] = global_end_date
        end = pd.to_datetime(end)
    else:
        end = pd.Series([global_end_date] * len(start))
    start = pd.to_datetime(start)
    feature = {state: np.zeros(days) for state in df.index}
    for i in range(len(start)):
        _ix = (ix >= start.iloc[i]) & (ix <= end.iloc[i])
        feature[start.index[i]][_ix] = 1
    _df = pd.DataFrame(feature).transpose()
    _df["policy"] = [start_col] * nstates
    _df.columns = colnames + ["policy"]
    dfs.append(_df)
df = pd.concat(dfs, axis=0)
df = df[["policy"] + colnames]
df.index.set_names("state", inplace=True)
print(df)
df.to_csv("policy_features.csv")
