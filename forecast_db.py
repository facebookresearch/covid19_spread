#!/usr/bin/env python3

import click
from glob import glob
import numpy as np
import pandas
import os
import re
import json
import sqlite3
from subprocess import check_call
import datetime
import itertools
import tempfile
import shutil
from data.usa.process_cases import get_nyt
import requests
from xml.etree import ElementTree


DB='/checkpoint/mattle/covid19/forecast.db'


class MaxBy:
    def __init__(self):
        self.max_key = None
        self.max_val = None

    def step(self, key, value):
        if self.max_val is None or value > self.max_val:
            self.max_val = value
            self.max_key = key

    def finalize(self):
        return self.max_key


def adapt_date(x):
    if isinstance(x, datetime.datetime):
        x = x.date()
    return str(x)

def convert_date(s):
    return datetime.datetime.strptime(s.decode('utf-8'), '%Y-%M-%d').date()

sqlite3.register_adapter(datetime.date, adapt_date)
sqlite3.register_adapter(datetime.datetime, adapt_date)
sqlite3.register_converter("date", convert_date)

def regexp(expr, item):
    reg = re.compile(expr)
    return reg.search(item) is not None

def mk_db():
    conn = sqlite3.connect(DB)
    res = conn.execute("""
    CREATE TABLE IF NOT EXISTS infections(
        date date,
        loc1 text,
        loc2 text,
        loc3 text,
        counts real,
        id text,
        forecast_date date,
        UNIQUE(id, forecast_date, date, loc1, loc2, loc3) ON CONFLICT REPLACE
    );
    """)
    conn.execute("CREATE INDEX date_idx ON infections(date);")
    conn.execute("CREATE INDEX loc_1_idx ON infections(loc1);")
    conn.execute("CREATE INDEX loc_2_idx ON infections(loc2);")
    conn.execute("CREATE INDEX loc_3_idx ON infections(loc3);")
    conn.execute("CREATE INDEX id_idx ON infections(id);")
    conn.execute("CREATE INDEX forecast_date_idx ON infections(forecast_date);")
    res = conn.execute("""
    CREATE TABLE IF NOT EXISTS deaths(
        date date,
        loc1 text,
        loc2 text,
        loc3 text,
        counts real,
        id text,
        forecast_date date,
        UNIQUE(id, forecast_date, date, loc1, loc2, loc3) ON CONFLICT REPLACE
    );
    """)
    conn.execute("CREATE INDEX date_deaths_idx ON deaths(date);")
    conn.execute("CREATE INDEX loc_1_deaths_idx ON deaths(loc1);")
    conn.execute("CREATE INDEX loc_2_deaths_idx ON deaths(loc2);")
    conn.execute("CREATE INDEX loc_3_deaths_idx ON deaths(loc3);")
    conn.execute("CREATE INDEX id_deaths_idx ON deaths(id);")
    conn.execute("CREATE INDEX forecast_date_deaths_idx ON deaths(forecast_date);")

@click.group()
def cli():
    pass


LOC_MAP = {
    'new-jersey': 'New Jersey',
    'nys': 'New York'
}


def sync_max_forecasts(conn, remote_dir, local_dir):
    check_call(['scp', f'{remote_dir}/new-jersey/forecast-[0-9]*_fast.csv', '.'], cwd=f'{local_dir}/new-jersey')
    check_call(['scp', f'{remote_dir}/new-jersey/forecast-[0-9]*_slow.csv', '.'], cwd=f'{local_dir}/new-jersey')
    check_call(['scp', f'{remote_dir}/nys/forecast-[0-9]*_fast.csv', '.'], cwd=f'{local_dir}/nys')
    check_call(['scp', f'{remote_dir}/nys/forecast-[0-9]*_slow.csv', '.'], cwd=f'{local_dir}/nys')
    files = glob(f'local_dir/new-jersey/forecast-*_(fast|slow).csv')
    for state, ty in itertools.product(['new-jersey', 'nys'], ['slow', 'fast']):
        files = glob(f'{local_dir}/{state}/forecast-*_{ty}.csv')
        for f in files:
            forecast_date = re.search('forecast-(\d+)_', f).group(1)
            forecast_date = datetime.datetime.strptime(forecast_date, '%Y%m%d').date()
            res = conn.execute(f"SELECT COUNT(1) FROM infections WHERE forecast_date=? AND id=?", (forecast_date, f'{state}_{ty}'))
            if res.fetchone()[0] == 0:
                df = pandas.read_csv(f)
                df = df[df['date'].str.match('\d{2}/\d{2}')]
                df['date'] = pandas.to_datetime(df['date'] + '/2020')
                df = df.melt(id_vars=['date'], value_name='counts', var_name='location')
                df = df.rename(columns={'location': 'loc3'})
                df['forecast_date'] = forecast_date
                df['id'] = f'{state}_{ty}'
                df['loc2'] = LOC_MAP[state]
                df['loc1'] = 'United States'
                state_agg = df.groupby(['loc2', 'date', 'loc1', 'id', 'forecast_date']).counts.sum().reset_index()
                df = pandas.concat([df, state_agg])
                df.to_sql(name='infections',index=False, con=conn, if_exists='append')


def sync_nyt(conn):
    # Sync the NYTimes ground truth data
    conn.execute("DELETE FROM infections WHERE id='nyt_ground_truth'")
    conn.execute("DELETE FROM deaths WHERE id='nyt_ground_truth'")
    def dump(df, metric):
        df = df.reset_index().melt(id_vars=['date'], value_name='counts', var_name='loc2')
        df['loc3'] = df['loc2'].apply(lambda x: x.split('_')[1])
        df['loc2'] = df['loc2'].apply(lambda x: x.split('_')[0])
        df['loc1'] = 'United States'
        df['date'] = pandas.to_datetime(df['date'])
        df['id'] = 'nyt_ground_truth'
        # Aggregate to state level
        state = df.groupby(['loc1', 'loc2', 'date', 'id']).counts.sum().reset_index()
        # Aggregate to country level
        country = df.groupby(['loc1', 'date', 'id']).counts.sum().reset_index()
        df = pandas.concat([df, state, country])
        df.to_sql(name=metric, index=False, con=conn, if_exists='append')
    dump(get_nyt(metric='cases'), 'infections')
    dump(get_nyt(metric='deaths'), 'deaths')


def get_ihme_file(dir):
    csvs = glob(os.path.join(dir, '*/*.csv'))
    if len(csvs) == 1:
        return csvs[0]
    import pdb; pdb.set_trace()


def sync_ihme(conn):
    req = requests.get('https://ihmecovid19storage.blob.core.windows.net/archive?comp=list')
    req.raise_for_status()
    tree = ElementTree.fromstring(req.content)

    for elem in tree.findall('.//Blob'):
        if elem.find('Url').text.endswith('ihme-covid19.zip'):
            forecast_date = datetime.datetime.strptime(elem.find('Url').text.split('/')[-2], '%Y-%m-%d').date()
            res = conn.execute("SELECT COUNT(1) FROM deaths WHERE id='IHME' AND forecast_date=?", (forecast_date,))
            if res.fetchone()[0] > 0:
                continue
            with tempfile.TemporaryDirectory() as tdir:
                check_call(['wget', '-O', f'{tdir}/ihme-covid19.zip', elem.find('Url').text])
                shutil.unpack_archive(f'{tdir}/ihme-covid19.zip', extract_dir=tdir)
                stats_file = get_ihme_file(tdir)
                stats = pandas.read_csv(stats_file).rename(columns={'date_reported': 'date'})
                states = pandas.read_sql("SELECT DISTINCT loc2 FROM infections WHERE loc1='United States'", conn)
                df = states.merge(stats, left_on='loc2', right_on='location_name')[['loc2', 'date', 'deaths_mean']]
            df = df.dropna().rename(columns={'deaths_mean': 'counts'})
            df['loc1'] = 'United States'
            df['forecast_date'] = forecast_date
            df['id'] = 'IHME'
            df.to_sql(name='deaths', index=False, con=conn, if_exists='append')


@click.command()
def sync_forecasts():
    if not os.path.exists(DB):
        mk_db()
    conn = sqlite3.connect(DB)
    conn.create_function("REGEXP", 2, regexp)
    remote_dir = 'devfairh1:/private/home/maxn/covid19_spread/forecasts'
    local_dir = f'/checkpoint/{os.environ["USER"]}/covid19/forecasts'
    sync_max_forecasts(conn, remote_dir, local_dir)
    sync_nyt(conn)
    sync_ihme(conn)

    check_call([
        'scp', 
        f'/checkpoint/{os.environ["USER"]}/covid19/forecast.db',
        f'devfairh1:/checkpoint/{os.environ["USER"]}/covid19/forecast.db'
    ])
    

if __name__ == '__main__':
    cli.add_command(sync_forecasts)
    cli()
