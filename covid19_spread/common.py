#!/usr/bin/env python3
import numpy as np
import pandas as pd
from numpy.linalg import norm
import os
import re
from covid19_spread.lib import cluster
from subprocess import check_call
from covid19_spread import metrics
from datetime import timedelta


def mk_absolute_paths(cfg):
    if isinstance(cfg, dict):
        return {k: mk_absolute_paths(v) for k, v in cfg.items()}
    elif isinstance(cfg, list):
        return list(map(mk_absolute_paths, cfg))
    else:
        return (
            os.path.realpath(cfg)
            if isinstance(cfg, str) and os.path.exists(cfg)
            else cfg
        )


def rebase_forecast_deltas(val_in, df_forecast_deltas):
    gt = metrics.load_ground_truth(val_in)
    # Ground truth for the day before our first forecast
    prev_day = gt.loc[[df_forecast_deltas.index.min() - timedelta(days=1)]]
    # Stack the first day ground truth on top of the forecasts
    common_cols = set(df_forecast_deltas.columns).intersection(set(gt.columns))
    stacked = pd.concat([prev_day[common_cols], df_forecast_deltas[common_cols]])
    # Cumulative sum to compute total cases for the forecasts
    df_forecast = stacked.sort_index().cumsum().iloc[1:]
    return df_forecast


def update_repo(repo):
    user = cluster.USER
    match = re.search(r"([^(\/|:)]+)/([^(\/|:)]+)\.git", repo)
    name = f"{match.group(1)}_{match.group(2)}"
    data_pth = f"{cluster.FS}/{user}/covid19/data/{name}"
    if not os.path.exists(data_pth):
        check_call(["git", "clone", repo, data_pth])
    check_call(["git", "checkout", "master"], cwd=data_pth)
    check_call(["git", "pull"], cwd=data_pth)
    return data_pth


def drop_k_days_csv(dset, outfile, days):
    df = pd.read_csv(dset, index_col="region")
    if days > 0:
        df = df[sorted(df.columns)[:-days]]
    df = drop_all_zero_csv(df)
    df.to_csv(outfile)


def drop_all_zero_csv(df):
    counts = df.sum(axis=1)
    df = df[counts > 0]
    return df


def smooth_csv(indset: str, outdset: str, days: int):
    df = pd.read_csv(indset, index_col="region").transpose()
    incident_cases = df.diff()
    smooth = np.round(incident_cases.rolling(window=days, min_periods=1).mean())
    smooth.iloc[0] = df.iloc[0]
    smooth.cumsum(0).transpose().to_csv(outdset)


smooth = smooth_csv


def print_model_stats(mus, beta, S, U, V, A):
    C = A - np.diag(np.diag(A))
    print("beta =", beta)
    print(f"\nNorms      : U = {norm(U).item():.3f}, V = {norm(V):.3f}")
    print(f"Max Element: U = {np.max(U).item():.3f}, V = {np.max(V):.3f}")
    print(f"Avg Element: U = {np.mean(U).item():.3f}, V = {np.mean(V):.3f}")
    print(f"\nSelf:       max = {np.max(S):.3f}, avg = {np.mean(S):.3f}")
    print(f"Cross:      max = {np.max(C):.3f}, avg = {np.mean(C):.3f}")


def standardize_county_name(county):
    return (
        county.replace(" County", "")
        .replace(" Parish", "")
        .replace(" Municipality", "")
        .replace(" Municipality", "")
        .replace(" Borough", "")
    )
