#!/usr/bin/env python3
# Copyright (c) 2021-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import pandas as pd

rename = {"Islas Canarias": "Canarias"}
index = pd.read_csv("https://storage.googleapis.com/covid19-open-data/v2/index.csv")
index = index[index["country_code"] == "ES"]

df = pd.read_csv("https://storage.googleapis.com/covid19-open-data/v2/weather.csv")
weather = df.merge(index, on="key")
print(weather["subregion1_name"].unique())
print(weather["subregion2_name"].unique())
weather = weather[weather["aggregation_level"] == 1]
weather = weather[~weather["subregion1_name"].isnull()]
weather["loc"] = weather["subregion1_name"]
weather["loc"] = weather["loc"].apply(lambda x: rename.get(x, x))
cols = ["average_temperature", "minimum_temperature", "maximum_temperature", "rainfall"]
weather = weather.drop_duplicates(subset=["date", "loc"] + cols)
weather_piv = weather.pivot(index="date", values=cols, columns="loc")

# Transform into z-scores
weather_piv = (weather_piv - weather_piv.mean()) / weather_piv.std(skipna=True)
weather_piv.iloc[0] = weather_piv.iloc[0].fillna(0)
weather_piv = weather_piv.fillna(method="ffill")

weather_piv = weather_piv.transpose().unstack(0).transpose()
weather_piv = weather_piv.stack().unstack(0).reset_index(0)
weather_piv = weather_piv.rename(columns={"level_0": "type"})
weather_piv.round(3).to_csv("weather_features.csv", index_label="region")
