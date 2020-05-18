#!/usr/bin/env python3

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(os.path.realpath(__file__)), ".."))
from episode import mk_episode, to_h5
import h5py
import numpy as np
import pandas
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


def fetch_data():
    URL = "https://cnecovid.isciii.es/covid19/resources/agregados.csv"
    df = pandas.read_csv(URL, encoding="latin1", parse_dates=["FECHA"], dayfirst=True)
    df = df[~df["FECHA"].isnull()]
    df["loc"] = df["CCAA"].apply(lambda x: "Spain_" + regions[x])
    df = df.rename(columns={"FECHA": "date", "PCR+": "cases"})
    return df.pivot_table(values="cases", columns="loc", index="date").sort_index()


def main(args):
    parser = argparse.ArgumentParser()
    parser.add_argument("--smooth", type=int, default=1)
    opt = parser.parse_args()

    df = fetch_data()

    counter = itertools.count()
    loc_map = ddict(counter.__next__)
    episodes = mk_episode(df, df.columns, loc_map, opt.smooth)
    to_h5(df, "timeseries.h5", loc_map, [episodes])


if __name__ == "__main__":
    main(sys.argv[1:])
