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

df = pd.read_csv("testing_raw.csv", parse_dates=["date"], index_col="date")
df["state"] = df["state"].apply(lambda x: state_abbrvs[x])
df["total"] = df["positive"] + df["negative"]
df["ratio"] = df["positive"] / df["negative"]
df_aggr = df.groupby(by="state")


def zscore(df):
    df.iloc[:, 0:] = (
        df.iloc[:, 0:].values - df.iloc[:, 0:].mean(axis=1, skipna=True).values[:, None]
    ) / df.iloc[:, 0:].std(axis=1, skipna=True).values[:, None]
    df = df.fillna(0)
    return df


def zero_one(df):
    df = df.fillna(0)
    # df = df.sub(df.min(axis=1), axis=0)
    # df = df.div(df.max(axis=1), axis=0)
    df = df / df.values.max()
    df = df.fillna(0)
    return df


def write_features(key, func_smooth, func_normalize):
    states = []
    tests = []
    for state, data in df_aggr:
        states.append(state)
        # tests.append(data["tests"])
        tests.append(data[key])
    df = pd.concat(tests, axis=1)
    df.columns = states
    df = df.transpose()
    # df = df.diff(axis=1).clip(0, None).rolling(7, axis=1).mean()
    # df = df.rolling(7, axis=1).mean()
    df = func_smooth(df)
    if func_normalize is not None:
        df = func_normalize(df)

    df = df.fillna(0)
    print(df.head())
    df.index.set_names("region", inplace=True)
    df.round(3).to_csv(f"{key}_features.csv")


write_features("ratio", lambda _df: _df.rolling(7, axis=1).mean(), zscore)
write_features(
    "total", lambda _df: _df.diff(axis=1).rolling(7, axis=1).mean(), zero_one,
)
