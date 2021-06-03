import pandas as pd
import argparse
from datetime import datetime
import os
from covid19_spread.data.usa.process_cases import get_index

SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))


def get_df(source, signal, resolution):
    df = pd.read_csv(
        f"{SCRIPT_DIR}/{resolution}/{source}/{signal}.csv", parse_dates=["date"]
    )
    df.dropna(axis=0, subset=["date"], inplace=True)
    index = get_index()
    state_index = index.drop_duplicates("subregion1_code")

    if "state" in df.columns:
        df["state"] = df["state"].str.upper()
        merged = df.merge(state_index, left_on="state", right_on="subregion1_code")
        df = merged[["subregion1_name", "date", signal]].rename(
            columns={"subregion1_name": "loc"}
        )
    else:
        df["county"] = df["county"].astype(str).str.zfill(5)
        merged = df.merge(index, left_on="county", right_on="fips")
        merged["loc"] = merged["name"] + ", " + merged["subregion1_name"]
        df = merged[["loc", "date", signal]]

    df = df.pivot(index="date", columns="loc", values=signal).copy()

    # Fill in NaNs
    df.iloc[0] = 0
    df = df.fillna(0)
    # Normalize
    df = df.transpose() / 100

    df["type"] = f"{source}_{signal}_{resolution}"
    return df


def main(signal, resolution):
    source, signal = signal.split("/")
    df = get_df(source, signal, resolution)

    if resolution == "county":
        # Fill in missing counties with zeros
        cases = pd.read_csv(
            f"{SCRIPT_DIR}/../data_cases.csv", index_col="region"
        ).index.to_frame()
        cases["state"] = [x.split(", ")[-1] for x in cases.index]
        cases = cases.drop(columns="region")
        idx = pd.MultiIndex.from_product([cases.index, df["type"].unique()])
        type_ = df["type"].iloc[0]
        df = df.reset_index().set_index(["loc", "type"]).reindex(idx).fillna(0)
        df2 = get_df(source, signal, "state")
        df2 = df2.merge(cases[["state"]], left_index=True, right_on="state")[
            df2.columns
        ]
        df = pd.concat([df, df2.set_index("type", append=True)])

    df = df[[c for c in df.columns if isinstance(c, datetime)]]
    df.columns = [str(x.date()) if isinstance(x, datetime) else x for x in df.columns]

    df.round(3).to_csv(
        f"{SCRIPT_DIR}/{source}_{signal}-{resolution}.csv",
        index_label=["region", "type"],
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-signal", default="smoothed_hh_cmnty_cli")
    parser.add_argument("-resolution", choices=["state", "county"], default="county")
    opt = parser.parse_args()
    main(opt.signal, opt.resolution)
