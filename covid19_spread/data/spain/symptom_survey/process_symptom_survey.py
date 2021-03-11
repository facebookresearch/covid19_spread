#!/usr/bin/env python3
# Copyright (c) 2021-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import sys
import pandas as pd
import argparse
import numpy as np
from datetime import datetime

rename = {
    "Islas Canarias": "Canarias",
    "Comunidad Valenciana": "Valenciana",
    "Comunidad de Madrid": "Madrid",
    "Comunidad Foral de Navarra": "Navarra",
    "Región de Murcia": "Murcia",
    "Principado de Asturias": "Asturias",
}

parser = argparse.ArgumentParser()
parser.add_argument("-signal", default="smoothed_cli")
opt = parser.parse_args()


def get_df(signal):
    df = pd.read_csv(f"raw/{opt.signal}/survey.csv", parse_dates=["date"])
    df["region"] = df["region"].apply(lambda x: rename.get(x, x))
    print(df)
    df.dropna(axis=0, subset=["date"], inplace=True)

    df = df.pivot(index="date", columns="region", values=signal).copy()
    print(df)

    # Fill in NaNs
    df.iloc[0] = 0
    # df = df.fillna(method="ffill")
    df = df.fillna(0)
    # Normalize
    df = df.transpose()

    df["type"] = f"{signal}"
    print(df)
    return df


df = get_df(opt.signal)

df = df[["type"] + [c for c in df.columns if isinstance(c, datetime)]]
df.columns = [str(x.date()) if isinstance(x, datetime) else x for x in df.columns]

df.round(5).to_csv(f"{opt.signal}.csv", index_label="region")
