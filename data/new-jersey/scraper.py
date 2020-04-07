#!/usr/bin/env python3

import requests
import pandas
from datetime import date, timedelta
import os

URL="https://services7.arcgis.com/Z0rixLlManVefxqY/arcgis/rest/services/DailyCaseCounts/FeatureServer/0/query?f=json&where=1%3D1&returnGeometry=false&spatialRel=esriSpatialRelIntersects&outFields=*&orderByFields=TOTAL_CASES%20desc"

yesterday = (date.today() - timedelta(days=1)).strftime('%Y%m%d')
yesterday = f'data-{yesterday}.csv'

assert os.path.exists(yesterday), "Unable to find yesterday\'s file!!!!"

df = pandas.read_csv(yesterday, index_col=0)

req = requests.get(URL)
data = req.json()

new_data = {'Date': date.today().strftime('%m/%d/%Y'), 'Start day': df['Start day'].max() + 1}

for row in data['features']:
    county = row['attributes']['COUNTY_LAB']
    new_data[' '.join(county.split()[:-1])] = row['attributes']['TOTAL_CASES']

new_data = pandas.DataFrame([new_data])

df = pandas.concat([df, new_data]).reset_index()
today = date.today().strftime('%Y%m%d')
df.to_csv(f'data-{today}.csv')

