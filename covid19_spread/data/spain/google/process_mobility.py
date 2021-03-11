#!/usr/bin/env python3
# Copyright (c) 2021-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import sys
import numpy as np
import pandas as pd
from datetime import datetime

sys.path.append("../../../")
from common import standardize_county_name


cols = [
    "date",
    "region",
    "retail_and_recreation_percent_change_from_baseline",
    "grocery_and_pharmacy_percent_change_from_baseline",
    "parks_percent_change_from_baseline",
    "transit_stations_percent_change_from_baseline",
    "workplaces_percent_change_from_baseline",
    "residential_percent_change_from_baseline",
]

rename = {
    "Canary Islands": "Canarias",
    "Balearic Islands": "Islas Baleares",
    "Valencian Community": "Valenciana",
    "Community of Madrid": "Madrid",
    "Region of Murcia": "Murcia",
    "Catalonia": "Cataluña",
    "Navarre": "Navarra",
    "Andalusia": "Andalucía",
    "Aragon": "Aragón",
    "Castile and León": "Castilla y León",
    "Castile-La Mancha": "Castilla-La Mancha",
    "Basque Country": "País Vasco",
}


def get_county_mobility_google(fin=None):
    # Google LLC "Google COVID-19 Community Mobility Reports."
    # https://www.google.com/covid19/mobility/ Accessed: 2020-05-04.
    # unfortunately, this is only relative to mobility on a baseline date
    if fin is None:
        fin = "https://www.gstatic.com/covid19/mobility/Global_Mobility_Report.csv"
    df_Gmobility_global = pd.read_csv(fin, parse_dates=["date"])
    df_Gmobility_usa = df_Gmobility_global.query("country_region_code == 'ES'")
    return df_Gmobility_usa


fin = sys.argv[1] if len(sys.argv) == 2 else None
df = get_county_mobility_google(fin)
df = df[df["sub_region_2"].isnull()]
df = df.dropna(subset=["sub_region_1"], axis=0)
print(df["sub_region_1"].unique())
df["region"] = df["sub_region_1"]
df["region"] = df["region"].apply(lambda x: rename.get(x, x))
df = df[cols]

val_cols = [c for c in df.columns if c not in {"region", "date"}]
ratio = (1 + df.set_index(["region", "date"]) / 100).reset_index()
piv = ratio.pivot(index="date", columns="region", values=val_cols)
piv = piv.rolling(7, min_periods=1).mean().transpose()
piv.iloc[0] = piv.iloc[0].fillna(0)
# piv = piv.fillna(method="ffill")
piv = piv.fillna(0)

dfs = []
for k in piv.index.get_level_values(0).unique():
    df = piv.loc[k].copy()
    df["type"] = k
    dfs.append(df)
df = pd.concat(dfs)
df = df[["type"] + sorted([c for c in df.columns if isinstance(c, datetime)])]
df.columns = [str(c.date()) if isinstance(c, datetime) else c for c in df.columns]

df.round(4).to_csv("mobility_features_google.csv")
print(df)
