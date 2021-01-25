#!/usr/bin/env python3
import h5py
import numpy as np
import pandas as pd
import torch as th
from numpy.linalg import norm
import itertools
from collections import defaultdict
import datetime
import os
import re
from lib import cluster
from subprocess import check_call


def update_repo(repo):
    user = os.environ["USER"]
    match = re.search(r"([^(\/|:)]+)/([^(\/|:)]+)\.git", repo)
    name = f"{match.group(1)}_{match.group(2)}"
    data_pth = f"{cluster.FS}/{user}/covid19/data/{name}"
    if not os.path.exists(data_pth):
        check_call(["git", "clone", repo, data_pth])
    check_call(["git", "checkout", "master"], cwd=data_pth)
    check_call(["git", "pull"], cwd=data_pth)
    return data_pth


def drop_k_days(dset, outfile, days):
    # We'll need to rename everything in case the number of events for a given dimension
    # drops to 0 after filtering.
    counter = itertools.count()
    namer = defaultdict(counter.__next__)
    ds_dt = h5py.special_dtype(vlen=np.dtype("int"))
    ts_dt = h5py.special_dtype(vlen=np.dtype("float32"))

    with h5py.File(dset, "r") as hin, h5py.File(outfile, "w") as hout:
        times = hin["time"][:]
        nodes = hin["node"][:]
        new_nodes = []
        new_times = []
        # Compute the new maxtime
        max_time = np.max([ts[-1] for ts in times]) - days

        processed = np.array([])
        all_times = []
        all_nodes = []

        for idx, ts in enumerate(times):
            mask = ts < max_time
            if not mask.any():
                continue
            new_nodes.append(np.array([namer[x] for x in nodes[idx][mask]]))
            new_times.append(ts[mask])

            mask = ~np.in1d(new_nodes[-1], processed)
            all_nodes.append(new_nodes[-1][mask])
            all_times.append(new_times[-1][mask])
            processed = np.concatenate([processed, np.unique(all_nodes[-1])])

        print(new_times)
        old_idxs, new_idxs = map(np.array, zip(*namer.items()))
        hout["nodes"] = hin["nodes"][:][old_idxs[sorted(new_idxs)]]
        _time = hout.create_dataset("time", (len(new_times),), dtype=ts_dt)
        _node = hout.create_dataset("node", (len(new_nodes),), dtype=ds_dt)
        _time[:] = new_times
        _node[:] = new_nodes
        hout["cascades"] = np.arange(len(new_nodes))
        if "basedate" in hin.attrs:
            date = datetime.datetime.strptime(hin.attrs["basedate"], "%Y-%m-%d")
            hout.attrs["basedate"] = str((date - datetime.timedelta(days=days)).date())
            hout.attrs["gt_basedate"] = hin.attrs.get("gt_basedate", str(date))

        if "ground_truth" in hin.keys():
            hout["ground_truth"] = hin["ground_truth"][:][old_idxs[sorted(new_idxs)]]

        all_times = np.concatenate(all_times)
        idx = all_times.argsort()
        all_times = all_times[idx]
        all_nodes = np.concatenate(all_nodes)[idx]
        hout["all_nodes"] = all_nodes
        hout["all_times"] = all_times


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


def smooth_h5(indset: str, outdset: str, smooth_days: int):
    with h5py.File(indset, "r") as hin, h5py.File(outdset, "w") as hout:
        for k, v in hin.attrs.items():
            hout.attrs[k] = v
        nodes, times, all_nodes, all_times = [], [], [], []
        nnodes = hin["nodes"].shape[0]
        hout["nodes"] = hin["nodes"][:]
        hout["cascades"] = hin["cascades"][:]
        hout["ground_truth"] = hin["ground_truth"][:]
        processed = set()
        for ns, ts in zip(hin["node"][:], hin["time"][:]):
            # Compute number of events of each type for each day
            current_nodes, current_times = [], []
            days = np.floor(ts).astype(int)
            ndays = days.max() + 1
            events = np.bincount(
                ns + (days * nnodes), minlength=ndays * nnodes
            ).reshape(-1, nnodes)
            smoothed = np.ceil(
                pd.DataFrame(events)
                .rolling(window=smooth_days, min_periods=1)
                .mean()
                .values
            ).astype(int)
            for node in range(nnodes):
                for day in range(ndays):
                    if events[day, node] <= 0:
                        continue
                    current_times.append(
                        np.random.uniform(day, day + 1, smoothed[day, node])
                    )
                    current_nodes.append(
                        np.full(current_times[-1].shape, node, dtype="int")
                    )
                    if node not in processed:
                        all_nodes.append(current_nodes[-1])
                        all_times.append(current_times[-1])
            ts_ = np.concatenate(current_times)
            idx = np.argsort(ts_)
            nodes.append(np.concatenate(current_nodes)[idx])
            times.append(ts_[idx])
        x = hout.create_dataset(
            "node", dtype=h5py.special_dtype(vlen=np.dtype("int")), shape=(len(nodes),)
        )
        x[:] = nodes
        x = hout.create_dataset(
            "time",
            dtype=h5py.special_dtype(vlen=np.dtype("float32")),
            shape=(len(nodes),),
        )
        x[:] = times
        all_times = np.concatenate(all_times)
        idx = np.argsort(all_times)
        hout["all_times"] = all_times[idx]
        hout["all_nodes"] = np.concatenate(all_nodes)[idx]


def smooth(indset: str, outdset: str, days: int):
    smoother = smooth_csv if indset.endswith(".csv") else smooth_h5
    smoother(indset, outdset, days)


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
