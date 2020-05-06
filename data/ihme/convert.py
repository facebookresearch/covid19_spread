#!/usr/bin/env python3

import numpy as np
import pandas as pd
import sys

basedate = sys.argv[1]
_df = pd.read_csv(f"raw_{basedate}.csv", parse_dates=["date"])

states = [
    "Alabama",
    "Alaska",
    "Arizona",
    "Arkansas",
    "California",
    "Colorado",
    "Connecticut",
    "Delaware",
    "Florida",
    "Georgia",
    "Hawaii",
    "Idaho",
    "Illinois",
    "Indiana",
    "Iowa",
    "Kansas",
    "Kentucky",
    "Louisiana",
    "Maine",
    "Maryland",
    "Massachusetts",
    "Michigan",
    "Minnesota",
    "Mississippi",
    "Missouri",
    "Montana",
    "Nebraska",
    "Nevada",
    "New Hampshire",
    "New Jersey",
    "New Mexico",
    "New York",
    "North Carolina",
    "North Dakota",
    "Ohio",
    "Oklahoma",
    "Oregon",
    "Pennsylvania",
    "Rhode Island",
    "South Carolina",
    "South Dakota",
    "Tennessee",
    "Texas",
    "Utah",
    "Vermont",
    "Virginia",
    "Washington",
    "West Virginia",
    "Wisconsin",
    "Wyoming",
]

cols = {}
dates = np.unique(_df["date"])

for state in states:
    df = _df[_df["location_name"] == state]
    df = df.sort_values(by="date")
    vals = np.zeros(len(dates))
    for i, d in enumerate(dates):
        ix = np.where(df["date"].values == d)[0]
        # print(i, d, ix)
        if len(ix) > 0:
            vals[i] = df["totdea_mean"].values[ix[0]]
    # print(state)
    # print(df["totdea_mean"].values)
    # print(vals)
    # print()
    cols[state] = vals

df = pd.DataFrame(cols).transpose()
df.columns = [pd.to_datetime(d).strftime("%Y-%m-%d") for d in dates]
df.index.set_names("state", inplace=True)
print(df.head())
df.to_csv(f"prediction-deaths-{basedate}.csv")
