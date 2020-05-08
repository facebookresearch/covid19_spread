#!/usr/bin/env python3

import numpy as np
import pandas as pd
import sys

df = pd.read_csv(
    sys.argv[1], usecols=["dates", "state", "q.50", "fcst_date"], parse_dates=["dates"]
)

fcst_date = np.unique(df.fcst_date)
assert len(fcst_date) == 1
fcst_date = fcst_date[0]
print(fcst_date)

df = df[df.dates > fcst_date]

cols = {}
dates = np.unique(df["dates"])

for (state, _df) in df.groupby("state"):
    _df = _df.sort_values(by="dates")
    assert (dates == _df["dates"].values).all()
    cols[state] = _df["q.50"].values

df = pd.DataFrame(cols)
df["date"] = dates
df.set_index("date", inplace=True)
df.to_csv(f"predictions_{fcst_date}.csv")
