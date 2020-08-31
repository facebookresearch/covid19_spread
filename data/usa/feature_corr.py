#!/usr/bin/env python3

import argparse
import pandas
import numpy as np
from tqdm import tqdm


parser = argparse.ArgumentParser()
parser.add_argument("-resolution", choices=["state", "county"], default="county")
parser.add_argument("-metric", choices=["cases", "deaths"], default="cases")
parser.add_argument("-threshold", type=float, default=0.2)
opt = parser.parse_args()

features = {
    "state": [
        "testing/total_features_state.csv",
        "fb/mobility_features_state_fb.csv",
        "google/mobility_features_state_google.csv",
        "google/weather_features_state.csv",
        "symptom-survey/fb-survey_smoothed_hh_cmnty_cli-state.csv",
        "symptom-survey/fb-survey_smoothed_wcli-state.csv",
        "symptom-survey/doctor-visits_smoothed_adj_cli-state.csv",
        "google/epi_features_state.csv",
        "google/hosp_features_state.csv",
    ],
    "county": [
        "symptom-survey/fb-survey_smoothed_hh_cmnty_cli-state.csv",
        "testing/ratio_features_county.csv",
        "testing/total_features_county.csv",
        "fb/mobility_features_county_fb.csv",
        "google/mobility_features_county_google.csv",
        "google/weather_features_county.csv",
        "symptom-survey/fb-survey_smoothed_wcli-state.csv",
        "symptom-survey/doctor-visits_smoothed_adj_cli-county.csv",
        "symptom-survey/doctor-visits_smoothed_adj_cli-state.csv",
    ],
}

# feats = features[opt.resolution]
df = pandas.read_csv(
    "data_cases.csv" if opt.metric == "cases" else "data_deaths.csv", index_col="region"
)
if opt.resolution == "county":
    idx = df.max(1).sort_values().iloc[-100:].index
    df = df.loc[idx]
else:
    df = df.loc[df.max(1).sort_values().index[-25:]]

df = df.transpose().diff().clip(lower=0)
df.index = pandas.to_datetime(df.index)

lags = list(range(0, 21))
result = []
for feature in tqdm(features[opt.resolution]):
    feat = pandas.read_csv(feature)
    for ty, g_feat in feat.groupby("type"):
        del g_feat["type"]
        g_feat = g_feat.set_index("region")
        if opt.resolution == "county" and feature.endswith("state.csv"):
            counties = pandas.DataFrame(df.columns)
            counties["state"] = counties["region"].apply(lambda x: x.split(", ")[1])
            g_feat = g_feat.merge(counties, left_index=True, right_on="state")
            del g_feat["state"]
            g_feat = g_feat.set_index("region")

        g_feat = g_feat.transpose()
        g_feat.index = pandas.to_datetime(g_feat.index)
        for lag in lags:
            shifted = g_feat.shift(lag).dropna()
            idx = shifted.index.intersection(df.index)
            cols = sorted(set(df.columns).intersection(shifted.columns))
            corr = df.loc[idx, cols].corrwith(shifted.loc[idx, cols]).mean()
            result.append({"feature": feature, "type": ty, "lag": lag, "corr": corr})

res = pandas.DataFrame(result)
res["abs_corr"] = res["corr"].abs()
best = res.loc[res.groupby(["feature", "type"])["abs_corr"].idxmax()]
best = best[best["corr"].abs() > opt.threshold]

# display selected features
with pandas.option_context(
    "display.max_rows",
    None,
    "display.max_columns",
    None,
    "display.expand_frame_repr",
    False,
):
    print(best.sort_values(by="abs_corr"))

dfs = []
for _, row in best.iterrows():
    df = pandas.read_csv(row.feature)
    df = df[df["type"] == row.type]
    transpose = df.drop(columns=["type"]).set_index("region").transpose()
    transpose = (
        transpose.shift(row["lag"]).fillna(method="ffill").fillna(method="bfill")
    )
    df = transpose.transpose()
    df["type"] = row["type"]
    dfs.append(df.set_index("type", append=True))

df = pandas.concat(dfs)
df = df[sorted(df.columns)]
df = df.fillna(method="ffill", axis=1).fillna(method="bfill", axis=1)
df.to_csv(f"shifted_features_{opt.resolution}.csv")
