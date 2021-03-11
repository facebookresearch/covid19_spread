#!/usr/bin/env python3
# Copyright (c) 2021-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import pandas as pd
from datetime import datetime
import os

SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))


def main():
    df = pd.read_csv(
        "https://api.covidtracking.com/v1/states/daily.csv", parse_dates=["date"]
    )

    fips = pd.read_csv(
        f"{SCRIPT_DIR}/../county_fips_master.csv",
        encoding="latin1",
        dtype={"fips": str},
    )
    fips = fips.drop_duplicates("fips")

    df = df.merge(
        fips.drop_duplicates("state_abbr"), left_on="state", right_on="state_abbr"
    )
    df = df[["state_name", "negative", "positive", "date"]].set_index("date")

    df["total"] = df["positive"] + df["negative"]

    def zscore(df):
        df.iloc[:, 0:] = (
            df.iloc[:, 0:].values
            - df.iloc[:, 0:].mean(axis=1, skipna=True).values[:, None]
        ) / df.iloc[:, 0:].std(axis=1, skipna=True).values[:, None]
        df = df.fillna(0)
        return df

    def zero_one(df):
        df = df.fillna(0)
        df = df.div(df.max(axis=1), axis=0)
        # df = df / df.max()
        df = df.fillna(0)
        return df

    def fmt_features(pivot, key, func_smooth, func_normalize):
        df = pivot.transpose()
        df = func_smooth(df)
        if func_normalize is not None:
            df = func_normalize(df)
        df = df.fillna(0)
        df.index.set_names("region", inplace=True)
        df["type"] = f"testing_{key}"
        merge = df.merge(fips, left_index=True, right_on="state_name")
        merge.index = merge["county_name"] + ", " + merge["state_name"]
        return df, merge[df.columns]

    def _diff(df):
        return df.diff(axis=1).rolling(7, axis=1, min_periods=1).mean()

    state_r, county_r = fmt_features(
        df.pivot(columns="state_name", values=["positive", "total"]),
        "ratio",
        lambda _df: (_diff(_df.loc["positive"]) / _diff(_df.loc["total"])),
        None,
    )

    state_t, county_t = fmt_features(
        df.pivot(columns="state_name", values="total"), "total", _diff, zero_one,
    )

    def write_features(df, res, fout):
        df = df[["type"] + [c for c in df.columns if isinstance(c, datetime)]]
        df.columns = [
            str(x.date()) if isinstance(x, datetime) else x for x in df.columns
        ]
        df.round(3).to_csv(
            f"{SCRIPT_DIR}/{fout}_features_{res}.csv", index_label="region"
        )

    write_features(state_t, "state", "total")
    write_features(state_r, "state", "ratio")
    write_features(county_t, "county", "total")
    write_features(county_r, "county", "ratio")


if __name__ == "__main__":
    main()
