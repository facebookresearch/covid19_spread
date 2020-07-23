#!/usr/bin/env python3

import os
import pandas as pd
import sys
from datetime import timedelta
from delphi_epidata import Epidata

# Fetch data

geo_value = sys.argv[1]
signal = sys.argv[2]
base_date = pd.to_datetime("2020-04-05")
end_date = pd.to_datetime("now")
print(end_date)

current_date = base_date
while current_date < end_date:
    current_date = current_date + timedelta(1)
    date_str = current_date.strftime("%Y%m%d")
    fout = f"{geo_value}/{signal}-{date_str}.csv"

    # d/l only if we don't have the file already
    if os.path.exists(fout):
        continue

    res = Epidata.covidcast("fb-survey", signal, "day", geo_value, [date_str], "*")
    print(date_str, res["result"], res["message"])
    assert res["result"] == 1, ("CLI", res["message"])
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
