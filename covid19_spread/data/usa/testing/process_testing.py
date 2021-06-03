import pandas as pd
from datetime import datetime
import os
from covid19_spread.data.usa.process_cases import get_index

SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))


def main():
    df = pd.read_csv(
        "https://api.covidtracking.com/v1/states/daily.csv", parse_dates=["date"]
    )

    index = get_index()
    states = index.drop_duplicates("subregion1_name")

    df = df.merge(states, left_on="state", right_on="subregion1_code")

    df = df[["subregion1_name", "negative", "positive", "date"]].set_index("date")
    df = df.rename(columns={"subregion1_name": "state_name"})

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
        merge = df.merge(index, left_index=True, right_on="subregion1_name")
        merge.index = merge["name"] + ", " + merge["subregion1_name"]
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
