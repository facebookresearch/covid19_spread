#!/usr/bin/env python3

import requests
import pandas
from io import StringIO

FIPS = requests.get(
    "https://gist.githubusercontent.com/wavded/1250983/raw/bf7c1c08f7b1596ca10822baeb8049d7350b0a4b/fipsToState.json"
).json()
FIPS = {v: k for k, v in FIPS.items()}


def get_county(state, city):
    headers = {
        "accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.9",
        "accept-language": "en-US,en;q=0.9",
        "content-type": "application/x-www-form-urlencoded",
    }
    data = f'states={FIPS[state]}&place_name={city.replace(" ", "+")}&Submit=Submit'
    res = requests.post(
        "http://www.stats.indiana.edu/uspr/b/place_query.asp",
        headers=headers,
        data=data,
    )
    x = StringIO()
    x.write(res.text)
    x.seek(0)
    df = pandas.read_html(x)
    return df[0]["County name"].tolist()


if __name__ == "__main__":
    print(get_county("New York", "Rochester"))
    print(get_county("New York", "New York City"))
