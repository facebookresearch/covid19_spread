#!/usr/bin/env python3

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(os.path.realpath(__file__)), ".."))
from episode import mk_episode, to_h5
import pandas as pd
from datetime import date
from datetime import datetime
from subprocess import check_call
import enum
from glob import glob
import sys
import argparse
import itertools
import numpy as np
from collections import defaultdict


class Region(enum.Enum):
    region = "region"  # more coarse
    province = "province"  # more fine grained


def fetch_data(region: Region = Region.province):
    """
    Fetch data form https://github.com/pcm-dpc/COVID-19.

    This is CC licensed
    """
    if not os.path.exists("COVID-19"):
        check_call(["git", "clone", "https://github.com/pcm-dpc/COVID-19.git"])
    check_call(["git", "pull"], cwd="COVID-19")
    if region == Region.region:
        files = glob("COVID-19/dati-regioni/*.csv")
        df = pd.concat([pd.read_csv(f, parse_dates=["data"]) for f in files])
        df = df.rename(columns={"totale_casi": "cases", "data": "date"})
        df["location"] = "Italy_" + df["denominazione_regione"]
    else:
        files = glob("COVID-19/dati-province/*.csv")
        df = pd.concat([pd.read_csv(f, parse_dates=["data"]) for f in files])
        df = df.rename(columns={"totale_casi": "cases", "data": "date"})
        df["location"] = (
            "Italy_" + df["denominazione_regione"] + "_" + df["denominazione_provincia"]
        )
        # Some entries say "In fase di definizione/aggiornamento", which Google translate
        # says: "Being defined / updated".  Not sure what this means, dropping for now...
        df = df[df["denominazione_provincia"] != "In fase di definizione/aggiornamento"]
    return df.pivot_table(index="date", columns="location", values="cases")


def main(args):
    parser = argparse.ArgumentParser()
    parser.add_argument("--smooth", type=int, default=1)
    parser.add_argument(
        "--resolution", choices=["region", "province"], default="province"
    )
    opt = parser.parse_args()
    data = fetch_data(getattr(Region, opt.resolution))
    transposed = data.transpose()
    transposed.columns = [str(c.date()) for c in transposed.columns]
    transposed[sorted(transposed.columns)].to_csv(
        "data_cases.csv", index_label="region"
    )
    counter = itertools.count()
    loc_map = defaultdict(counter.__next__)
    episodes = mk_episode(data, data.columns, loc_map, opt.smooth)
    to_h5(data, "timeseries.h5", loc_map, [episodes])


if __name__ == "__main__":
    main(sys.argv[1:])
