import pandas as pd
from datetime import datetime
from covid19_spread.data.usa.process_cases import get_index
import os

SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))


def main():
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

    def get_county_mobility_google(fin=None):
        # Google LLC "Google COVID-19 Community Mobility Reports."
        # https://www.google.com/covid19/mobility/ Accessed: 2020-05-04.
        # unfortunately, this is only relative to mobility on a baseline date
        if fin is None:
            fin = "https://www.gstatic.com/covid19/mobility/Global_Mobility_Report.csv"
        df_Gmobility_global = pd.read_csv(
            fin, parse_dates=["date"], dtype={"census_fips_code": str}
        )
        df_Gmobility_usa = df_Gmobility_global.query("country_region_code == 'US'")
        return df_Gmobility_usa

    df = get_county_mobility_google()
    df = df[~df["census_fips_code"].isnull()]
    index = get_index()
    index["region"] = index["subregion2_name"] + ", " + index["subregion1_name"]
    df = df.merge(
        index, left_on="census_fips_code", right_on="fips", suffixes=("", "_x")
    )[list(df.columns) + ["region"]]

    df = df[cols]

    val_cols = [c for c in df.columns if c not in {"region", "date"}]
    ratio = (1 + df.set_index(["region", "date"]) / 100).reset_index()
    piv = ratio.pivot(index="date", columns="region", values=val_cols)
    piv = piv.rolling(7, min_periods=1).mean().transpose()
    piv.iloc[0] = piv.iloc[0].fillna(0)

    piv = piv.fillna(0)

    dfs = []
    for k in piv.index.get_level_values(0).unique():
        df = piv.loc[k].copy()
        df["type"] = k
        dfs.append(df)
    df = pd.concat(dfs)
    df = df[["type"] + sorted([c for c in df.columns if isinstance(c, datetime)])]
    df.columns = [str(c.date()) if isinstance(c, datetime) else c for c in df.columns]

    df.to_csv(f"{SCRIPT_DIR}/mobility_features_county_google.csv")

    state = get_county_mobility_google()
    state = state[(~state["sub_region_1"].isnull()) & (state["sub_region_2"].isnull())]
    state["region"] = state["sub_region_1"]
    state = state[cols]

    ratio = (1 + state[cols].set_index(["region", "date"]) / 100).reset_index()
    piv = ratio.pivot(index="date", columns="region", values=val_cols)
    piv = piv.rolling(7, min_periods=1).mean().transpose()
    piv.columns = [str(x.date()) for x in sorted(piv.columns)]
    piv = piv.fillna(0).reset_index(level=0).rename(columns={"level_0": "type"})
    piv.to_csv(f"{SCRIPT_DIR}/mobility_features_state_google.csv")


if __name__ == "__main__":
    main()
