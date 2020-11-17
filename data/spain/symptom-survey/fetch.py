#!/usr/bin/env python3

import datetime
import os
import pandas as pd
import sys
from datetime import timedelta
import requests
import json

# request data from api

country = sys.argv[1]
signal = sys.argv[2]
indicator = sys.argv[3]

_request = f"https://covidmap.umd.edu/api/resources?indicator={indicator}&type=smoothed&country={country}&region=all&date={{}}"

# Fetch data
base_date = pd.to_datetime(sys.argv[4]) - timedelta(1)
end_date = datetime.datetime.now()


current_date = base_date
while current_date < end_date:
    current_date = current_date + timedelta(1)
    date_str = current_date.strftime("%Y%m%d")
    fout = f"raw/{signal}/{date_str}.csv"

    # d/l only if we don't have the file already
    if os.path.exists(fout):
        continue

    response = requests.get(_request.format(date_str)).text
    res = json.loads(response)
    assert res["status"] == "success", ("CLI", res["message"], res)
    df = pd.DataFrame(res["data"])
    if len(df) > 0:
        df = df.rename(columns={"survey_date": "date"})
        df = df[["date", "region", signal, "sample_size"]]
        df.to_csv(fout, index=False)

print(df)
