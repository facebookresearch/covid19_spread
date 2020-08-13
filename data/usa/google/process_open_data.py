import pandas
import re


index = pandas.read_csv("https://storage.googleapis.com/covid19-open-data/v2/index.csv")

index = index[index["country_code"] == "US"]
state_index = index[(index["key"].str.match("^US_[A-Z]+$")).fillna(False)]

fips = pandas.read_csv(
    "https://raw.githubusercontent.com/kjhealy/fips-codes/master/state_and_county_fips_master.csv"
)
fips["fips"] = fips["fips"].astype(str).str.zfill(5)
index = index.merge(fips, left_on="subregion2_code", right_on="fips")

df = pandas.read_csv(
    "https://storage.googleapis.com/covid19-open-data/v2/epidemiology.csv",
    parse_dates=["date"],
)
epi = df.merge(state_index, on="key")

# Merge state level epi data to county level index.
epi = epi[["date", "new_confirmed", "subregion1_code"]].merge(
    index, on="subregion1_code"
)
epi["loc"] = (
    epi["name"].str.replace(" (County|Municipality|Parish)", "")
    + ", "
    + epi["subregion1_name"]
)

epi_piv = epi.pivot(index="date", values=["new_confirmed"], columns="loc")
epi_piv = (epi_piv - epi_piv.mean()) / epi_piv.std(skipna=True)
epi_piv.index = [str(x.date()) for x in epi_piv.index]
epi_piv.iloc[0] = epi_piv.iloc[0].fillna(0)
epi_piv = epi_piv.fillna(0)

epi_piv = epi_piv.transpose().unstack(0).transpose()
epi_piv = epi_piv.stack().unstack(0).reset_index(0)
epi_piv = epi_piv.rename(columns={"level_0": "type"})

epi_piv = epi_piv.set_index("type", append=True)
for ty in epi_piv.index.get_level_values(1).unique():
    values = epi_piv.loc[(slice(None), ty), :].values.reshape(-1)
    low, high = values.min(), values.max()
    epi_piv.loc[(slice(None), ty), :] = (epi_piv.loc[(slice(None), ty), :] - low) / high

epi_piv.to_csv("epi_features.csv", index_label=["region", "type"])

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

# normalize to 0-1
# for col in cols[:-1]:
#    min_temp = weather_piv[col].min()  # .values.min()
#    weather_piv[col] = weather_piv[col].values - (min_temp.values.min() - 1)
#    max_temp = weather_piv[col].max()  # .values.max()
#    weather_piv[col] = weather_piv[col].values / max_temp.values.max()
#    print(weather_piv[col])
# weather_piv["rainfall"] /= weather_piv["rainfall"].max().values.max()
# print(weather_piv["rainfall"])

# Transform into z-scores
print(weather_piv.mean())
weather_piv = (weather_piv - weather_piv.mean()) / weather_piv.std(skipna=True)
weather_piv.iloc[0] = weather_piv.iloc[0].fillna(0)
weather_piv = weather_piv.fillna(method="ffill")

weather_piv = weather_piv.transpose().unstack(0).transpose()
weather_piv = weather_piv.stack().unstack(0).reset_index(0)
weather_piv = weather_piv.rename(columns={"level_0": "type"})
weather_piv.round(3).to_csv("weather_features.csv", index_label="region")
