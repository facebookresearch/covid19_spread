#!/usr/bin/env python3

import pandas as pd

state_abbrvs = {
    "AL": "Alabama",
    "AK": "Alaska",
    "AZ": "Arizona",
    "AS": "American Samoa",
    "AR": "Arkansas",
    "CA": "California",
    "CO": "Colorado",
    "CT": "Connecticut",
    "DE": "Delaware",
    "DC": "District of Columbia",
    "FL": "Florida",
    "GA": "Georgia",
    "GU": "Guam",
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
    "MP": "Northern Marianas",
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
    "VI": "Virgin Islands",
    "VT": "Vermont",
    "VA": "Virginia",
    "WA": "Washington",
    "WV": "West Virginia",
    "WI": "Wisconsin",
    "WY": "Wyoming",
}

df = pd.read_csv("testing.csv", parse_dates=["date"], index_col="date")
df["state"] = df["state"].apply(lambda x: state_abbrvs[x])
df["tests"] = df["positive"] + df["negative"]
df_aggr = df.groupby(by="state")
states = []
tests = []
for state, data in df_aggr:
    states.append(state)
    tests.append(data["tests"])
df = pd.concat(tests, axis=1)
df.columns = states
df = df.transpose().fillna(0)
df.index.set_names("region", inplace=True)
df.to_csv("testing_features.csv")
