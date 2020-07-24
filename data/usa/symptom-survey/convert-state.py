#!/usr/bin/env python3

import numpy as np
import pandas as pd
import sys

state_abbrvs = {
    "AL": "Alabama",
    "AK": "Alaska",
    "AZ": "Arizona",
    "AR": "Arkansas",
    "CA": "California",
    "CO": "Colorado",
    "CT": "Connecticut",
    "DE": "Delaware",
    "DC": "District of Columbia",
    "FL": "Florida",
    "GA": "Georgia",
    "HI": "Hawaii",
    "ID": "Idaho",
    "IL": "Illinois",
    "IN": "Indiana",
    "IA": "Iowa",
    "KS": "Kansas",
    "KY": "Kentucky",
    "LA": "Louisiana",
    "ME": "Maine",
    "MD": "Maryland",
    "MA": "Massachusetts",
    "MI": "Michigan",
    "MN": "Minnesota",
    "MS": "Mississippi",
    "MO": "Missouri",
    "MT": "Montana",
    "NE": "Nebraska",
    "NV": "Nevada",
    "NH": "New Hampshire",
    "NJ": "New Jersey",
    "NM": "New Mexico",
    "NY": "New York",
    "NC": "North Carolina",
    "ND": "North Dakota",
    "OH": "Ohio",
    "OK": "Oklahoma",
    "OR": "Oregon",
    "PA": "Pennsylvania",
    "PR": "Puerto Rico",
    "RI": "Rhode Island",
    "SC": "South Carolina",
    "SD": "South Dakota",
    "TN": "Tennessee",
    "TX": "Texas",
    "UT": "Utah",
    "VT": "Vermont",
    "VA": "Virginia",
    "WA": "Washington",
    "WV": "West Virginia",
    "WI": "Wisconsin",
    "WY": "Wyoming",
}


def expand_state(abbrv):
    return state_abbrvs[abbrv.upper()]


signal = sys.argv[1]

df = pd.read_csv(
    f"state/{signal}.csv",
    parse_dates=["date"],
    usecols=["state", "date", signal, f"{signal}_sample_size"],
)
df.dropna(axis=0, subset=["date"], inplace=True)

dates = np.unique(df["date"])
states = np.unique(df["state"])

cols = {
    pd.to_datetime(date).strftime("%Y-%m-%d"): np.zeros(len(states), dtype=np.float)
    for date in dates
}
cols["region"] = states


df_agg = df.groupby("state")
for _state, group in df_agg:
    six = np.where(states == _state)[0][0]
    group = group.sort_values(by="date")
    _dates = group["date"].to_numpy()
    _cases = group[signal].to_numpy()
    for i, _date in enumerate(_dates):
        dix = pd.to_datetime(_date).strftime("%Y-%m-%d")
        cols[dix][six] = _cases[i]

df = pd.DataFrame(cols)
df["region"] = df["region"].apply(expand_state)
df.set_index("region", inplace=True)
df.round(3).to_csv(f"data-{signal}-state.csv")
