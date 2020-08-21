#!/usr/bin/env python3

import os
import pandas as pd
import sys
from datetime import timedelta
from delphi_epidata import Epidata
import covidcast

# Fetch data

geo_value = sys.argv[1]
source, signal = sys.argv[2].split("/")


# grab start and end date from metadata
df = covidcast.metadata().drop(columns=["time_type", "min_lag", "max_lag"])
print(source, signal, geo_value, df)
df.min_time = pd.to_datetime(df.min_time)
df.max_time = pd.to_datetime(df.max_time)
df = df.query(
    f"data_source == '{source}' and signal == '{signal}' and geo_type == '{geo_value}'"
)
print(df)
assert len(df) == 1
base_date = df.iloc[0].min_time - timedelta(1)
end_date = df.iloc[0].max_time
print(base_date, end_date)

current_date = base_date
while current_date < end_date:
    current_date = current_date + timedelta(1)
    date_str = current_date.strftime("%Y%m%d")
    fout = f"{geo_value}/{source}/{signal}-{date_str}.csv"

    # d/l only if we don't have the file already
    if os.path.exists(fout):
        continue

    res = Epidata.covidcast(source, signal, "day", geo_value, [date_str], "*")
    # res = covidcast.signal("fb-survey", signal, "day", geo_value, [date_str], "*")
    print(date_str, res["result"], res["message"])
    assert res["result"] == 1, ("CLI", res["message"], res)
    df = pd.DataFrame(res["epidata"])
    df.rename(
        columns={
            "geo_value": geo_value,
            "time_value": "date",
            "value": signal,
            "direction": f"{signal}_direction",
            "stderr": f"{signal}_stderr",
            "sample_size": f"{signal}_sample_size",
        },
        inplace=True,
    )
    df.to_csv(fout, index=False)


print(df)
