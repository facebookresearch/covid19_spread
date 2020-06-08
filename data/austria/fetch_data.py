#!/usr/bin/env python3

import requests
import json
import pandas
from subprocess import check_call

check_call(['git', 'pull'])

newdata = pandas.read_csv("https://info.gesundheitsministerium.at/data/Bezirke.csv", sep=";")
newdata['Timestamp'] = pandas.to_datetime(newdata['Timestamp']).dt.date
current = pandas.read_csv('data.csv', index_col='region')

newest = newdata['Timestamp'].max()
if newdata['Timestamp'].max() > pandas.to_datetime(current.columns).max():
    pivot = newdata.pivot_table(index='Bezirk', values='Anzahl', columns='Timestamp')
    merged = current.merge(pivot, how='left', left_index=True, right_index=True)
    merged.to_csv('data.csv', index_label='region')
    check_call(['git', 'add', 'data.csv'])
    check_call(['git', 'commit', '-m', f'Updating Austria data: {newest}'])
    check_call(['git', 'push'])
else:
    print(f'Already updated latest data: {newest}')


