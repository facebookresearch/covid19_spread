#!/usr/bin/env python3
# Copyright (c) 2021-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import pandas
from datetime import datetime
import os
from covid19_spread.data.usa.process_cases import get_index

SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))


def main():
    index = pandas.read_csv(
        "https://storage.googleapis.com/covid19-open-data/v2/index.csv"
    )

    state_index = index[(index["key"].str.match("^US_[A-Z]+$")).fillna(False)]
    index = get_index()

    def zscore(piv):
        # z-zcore
        piv = (piv - piv.mean(skipna=True)) / piv.std(skipna=True)
        piv = piv.fillna(method="ffill").fillna(method="bfill")
        # piv = piv.fillna(0)
        return piv

    def zero_one(df):
        df = df.fillna(0)
        # df = df.div(df.max(axis=0), axis=1)
        df = df / df.max(axis=0)
        df = df.fillna(0)
        return df

    def process_df(df, columns, resolution, func_normalize):
        idx = state_index if resolution == "state" else index
        merged = df.merge(idx, on="key")
        if resolution == "state":
            exclude = {"US_MP", "US_AS", "US_GU", "US_VI", "US_PR"}
            merged = merged[~merged["key"].isin(exclude)]
            merged["region"] = merged["subregion1_name"]
        else:
            merged["region"] = merged["name"] + ", " + merged["subregion1_name"]
        piv = merged.pivot(index="date", columns="region", values=columns)
        if func_normalize is not None:
            piv = func_normalize(piv)

        dfs = []
        for k in piv.columns.get_level_values(0).unique():
            dfs.append(piv[k].transpose())
            dfs[-1]["type"] = k
        df = pandas.concat(dfs)
        df = df[["type"] + [c for c in df.columns if isinstance(c, datetime)]]
        df.columns = [
            str(c.date()) if isinstance(c, datetime) else c for c in df.columns
        ]
        return df.fillna(0)  # in case all values are NaN

    def get_df(url):
        df = pandas.read_csv(url, parse_dates=["date"])
        return df[~df["key"].isnull() & df["key"].str.startswith("US")]

    # --- Vaccination data ---
    df = get_df("https://storage.googleapis.com/covid19-open-data/v2/vaccinations.csv")
    vaccination = process_df(
        df,
        columns=["new_persons_vaccinated", "total_persons_vaccinated"],
        resolution="state",
        func_normalize=zero_one,
    )
    vaccination = vaccination.reset_index().set_index(["region", "type"])
    vaccination.to_csv(
        os.path.join(SCRIPT_DIR, "vaccination_state.csv"),
        index_label=["region", "type"],
    )

    # --- Hospitalizations ---
    df = get_df(
        "https://storage.googleapis.com/covid19-open-data/v2/hospitalizations.csv"
    )
    state_hosp = process_df(
        df,
        columns=[
            "current_hospitalized",
            "current_intensive_care",
            "current_ventilator",
        ],
        resolution="state",
        # func_normalize=lambda x: zero_one(x.clip(0, None)).rolling(7, min_periods=1).mean(),
        func_normalize=lambda x: zero_one(x.clip(0, None)),
    )
    state_hosp.round(3).to_csv(f"{SCRIPT_DIR}/hosp_features_state.csv")

    # --- Weather features ---
    df = get_df("https://storage.googleapis.com/covid19-open-data/v2/weather.csv")
    cols = [
        "average_temperature",
        "minimum_temperature",
        "maximum_temperature",
        "rainfall",
        "relative_humidity",
        "dew_point",
    ]
    weather = process_df(df, columns=cols, resolution="county", func_normalize=zscore)
    weather.round(3).to_csv(f"{SCRIPT_DIR}/weather_features_county.csv")

    weather = process_df(df, columns=cols, resolution="state", func_normalize=zscore)
    weather.round(3).to_csv(f"{SCRIPT_DIR}/weather_features_state.csv")

    # --- Epi features ---
    df = get_df("https://storage.googleapis.com/covid19-open-data/v2/epidemiology.csv")

    state_epi = process_df(
        df,
        columns=["new_confirmed"],
        resolution="state",
        func_normalize=lambda x: zero_one(x.clip(0, None)),
    )
    state_epi.round(3).to_csv(f"{SCRIPT_DIR}/epi_features_state.csv")
    epi = process_df(
        df,
        columns=["new_confirmed"],
        resolution="county",
        func_normalize=lambda x: zero_one(x.clip(0, None)),
    )
    epi.round(3).to_csv(f"{SCRIPT_DIR}/epi_features_county.csv")

    testing = process_df(
        df,
        columns=["new_tested"],
        resolution="state",
        func_normalize=lambda x: zero_one(x.clip(0, None)),
    )
    testing.round(3).to_csv(f"{SCRIPT_DIR}/tested_total_state.csv")

    df["ratio"] = df["new_confirmed"] / df["new_tested"]
    testing = process_df(
        df, columns=["ratio"], resolution="state", func_normalize=None,
    )
    testing.round(3).to_csv(f"{SCRIPT_DIR}/tested_ratio_state.csv")


if __name__ == "__main__":
    main()
