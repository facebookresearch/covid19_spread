#!/usr/bin/env python3

import sys
import os

import numpy as np
import pandas as pd
import torch as th
import requests
from collections import defaultdict as ddict
from itertools import count
import argparse
import itertools

# Data comes from https://cnecovid.isciii.es/covid19/#documentaci%C3%B3n-y-datos (see link under Data section at bottom)
# Additional context: https://github.com/CSSEGISandData/COVID-19/issues/2522


# https://en.wikipedia.org/wiki/ISO_3166-2:ES (autonomous communities)
regions = {
    "AN": "Andalucía",
    "AR": "Aragón",
    "AS": "Asturias",
    "CN": "Canarias",
    "CB": "Cantabria",
    "CL": "Castilla y León",
    "CM": "Castilla-La Mancha",
    "CT": "Cataluña",
    "CE": "Ceuta",
    "EX": "Extremadura",
    "GA": "Galicia",
    "IB": "Islas Baleares",
    "RI": "La Rioja",
    "MD": "Madrid",
    "ML": "Melilla",
    "MC": "Murcia",
    "NC": "Navarra",
    "PV": "País Vasco",
    "VC": "Valenciana",
}


def _fetch_data(metric: str = "cases"):
    assert metric in {"deaths", "cases"}
    URL = "https://cnecovid.isciii.es/covid19/resources/agregados.csv"
    df = pd.read_csv(
        URL,
        encoding="latin1",
        parse_dates=["FECHA"],
        dayfirst=True,
        error_bad_lines=False,  # Notes is source file are misformated
    )
    df = df[~df["FECHA"].isnull()]
    df["loc"] = df["CCAA"].apply(lambda x: regions[x])
    df = df.rename(
        columns={
            "FECHA": "date",
            "PCR+": "cases",
            "Fallecidos": "deaths",
        }
    )
    return df.pivot_table(values=metric, columns="loc", index="date").sort_index()


def fetch_data(metric: str = "new_confirmed"):
    rename = {
        "Islas Canarias": "Canarias",
        "Comunidad Valenciana": "Valenciana",
        "Comunidad de Madrid": "Madrid",
        "Region de Murcia": "Murcia",
    }
    index = pd.read_csv("https://storage.googleapis.com/covid19-open-data/v2/index.csv")
    index = index[index["country_code"] == "ES"]

    df = pd.read_csv(
        "https://storage.googleapis.com/covid19-open-data/v2/epidemiology.csv",
        parse_dates=["date"],
    )
    df = df.merge(index, on="key")
    df = df[df["aggregation_level"] == 1]
    df = df[~df["subregion1_name"].isnull()]
    df["region"] = df["subregion1_name"]
    df["region"] = df["region"].apply(lambda x: rename.get(x, x))

    df_piv = df.pivot_table(
        index="date", values="total_confirmed", columns="region"
    ).sort_index()
    df_piv = df_piv.dropna(axis=0)
    return df_piv


def process_time_features(df, pth, shift=0, input_resolution="county"):
    mobility = pd.read_csv(pth)
    dates = pd.to_datetime(mobility.columns[2:])
    dates = dates[np.where(dates >= df.columns.min())[0]]
    mobility = mobility[
        mobility.columns[:2].to_list()
        + list(map(lambda d: d.strftime("%Y-%m-%d"), dates))
    ]

    print(np.unique(mobility["type"]))
    n_mobility_types = len(np.unique(mobility["type"]))
    mobility_types = {r: v for (r, v) in mobility.groupby("region")}
    print(mobility.head(), len(mobility), n_mobility_types)

    mob = {}
    skipped = 0
    print(dates.min(), dates.max(), df.columns)
    start_ix = np.where(dates.min() == df.columns)[0][0]
    print(start_ix)
    start_ix += shift
    print(start_ix)
    # FIXME: check via dates between google and fb are not aligned
    # end_ix = np.where(dates.max() == df.columns)[0][0]
    end_ix = start_ix + len(dates)
    end_ix = min(end_ix, df.shape[1])
    for region in df.index:
        query = region
        if query not in mobility_types:
            # print(region)
            skipped += 1
            continue
        _m = th.zeros(df.shape[1], n_mobility_types)
        _v = mobility_types[query].iloc[:, 2:].transpose()
        _v = _v[: end_ix - start_ix]
        _m[start_ix:end_ix] = th.from_numpy(_v.values)
        assert (_m == _m).all()
        mob[region] = _m
    th.save(mob, pth.replace(".csv", ".pt"))
    print(skipped, df.shape[0])


def main(args):
    parser = argparse.ArgumentParser()
    parser.add_argument("--smooth", type=int, default=1)
    parser.add_argument("--metric", choices=["cases", "deaths"], default="cases")
    opt = parser.parse_args()

    df = fetch_data(opt.metric)
    print(df)
    df = df.transpose().cummax(axis=1)
    df.to_csv(f"data_{opt.metric}.csv", index_label="region")
    process_time_features(df, f"fb/mobility_features_fb.csv", 7)
    process_time_features(df, f"google/mobility_features_google.csv", 7)
    process_time_features(df, f"weather/weather_features.csv", 7)
    process_time_features(df, f"symptom-survey/smoothed_cli.csv", 7)
    process_time_features(df, f"symptom-survey/smoothed_mc.csv", 7)
    process_time_features(df, f"symptom-survey/smoothed_dc.csv", 7)


if __name__ == "__main__":
    main(sys.argv[1:])
