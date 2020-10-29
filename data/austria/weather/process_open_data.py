import pandas as pd

districts = pd.read_csv(
    "../polbezirke.csv",
    encoding="ISO-8859-1",
    delimiter=";",
    usecols=["Politischer Bezirk", "Bundesland"],
)

rename = {
    "Lower Austria": "Niederösterreich",
    "Vienna": "Wien",
    "Carinthia": "Kärnten",
    "Upper Austria": "Oberösterreich",
    "Styria": "Steiermark",
    "Tyrol": "Tirol",
}

index = pd.read_csv("https://storage.googleapis.com/covid19-open-data/v2/index.csv")
index = index[index["country_code"] == "AT"]

df = pd.read_csv("https://storage.googleapis.com/covid19-open-data/v2/weather.csv")
weather = df.merge(index, on="key")
print(weather["subregion1_name"].unique())
print(weather["subregion2_name"].unique())
weather = weather[weather["aggregation_level"] == 1]
weather = weather[~weather["subregion1_name"].isnull()]
weather["loc"] = weather["subregion1_name"]
weather["loc"] = weather["loc"].apply(lambda x: rename.get(x, x))
cols = [
    "average_temperature",
    "minimum_temperature",
    "maximum_temperature",
    "rainfall",
    "relative_humidity",
    "dew_point",
]
weather = weather.drop_duplicates(subset=["date", "loc"] + cols)
weather_piv = weather.pivot(index="date", values=cols, columns="loc")

# Transform into z-scores
weather_piv = (weather_piv - weather_piv.mean()) / weather_piv.std(skipna=True)
weather_piv.iloc[0] = weather_piv.iloc[0].fillna(0)
weather_piv = weather_piv.fillna(method="ffill")

weather_piv = weather_piv.transpose().unstack(0).transpose()
weather_piv = weather_piv.stack().unstack(0).reset_index(0)
weather_piv = weather_piv.rename(columns={"level_0": "type"}).reset_index()

df = pd.merge(weather_piv, districts, left_on="loc", right_on="Bundesland")
df = df.rename(columns={"Politischer Bezirk": "region"}).drop(
    columns=["Bundesland", "loc"]
)
df = df[["region", "type"] + list(df.columns[1:-1])]
df.round(3).to_csv("weather_features.csv", index=False)
print(df)
