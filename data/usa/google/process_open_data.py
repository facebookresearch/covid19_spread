import pandas


index = pandas.read_csv("https://storage.googleapis.com/covid19-open-data/v2/index.csv")
index = index[index["country_code"] == "US"]

fips = pandas.read_csv(
    "https://raw.githubusercontent.com/kjhealy/fips-codes/master/state_and_county_fips_master.csv"
)
fips["fips"] = fips["fips"].astype(str).str.zfill(5)
index = index.merge(fips, left_on="subregion2_code", right_on="fips")

# gov response is not granular enough
# df = pandas.read_csv('https://storage.googleapis.com/covid19-open-data/v2/oxford-government-response.csv')
df = pandas.read_csv("https://storage.googleapis.com/covid19-open-data/v2/weather.csv")
weather = df.merge(index, on="key")
weather = weather[~weather["subregion2_name"].isnull()]
weather["loc"] = (
    weather["name"].str.replace(" (County|Municipality|Parish)", "")
    + ", "
    + weather["subregion1_name"]
)
cols = ["average_temperature", "minimum_temperature", "maximum_temperature", "rainfall"]

weather_piv = weather.pivot(index="date", values=cols, columns="loc")

# Transform into z-scores
weather_piv = (weather_piv - weather_piv.mean()) / weather_piv.std(skipna=True)
weather_piv.iloc[0] = weather_piv.iloc[0].fillna(0)
weather_piv = weather_piv.fillna(method="ffill")

weather_piv = weather_piv.transpose().unstack(0).transpose()
weather_piv = weather_piv.stack().unstack(0).reset_index(0)
weather_piv = weather_piv.rename(columns={"level_0": "type"})
weather_piv.to_csv("weather_features.csv", index_label="region")
