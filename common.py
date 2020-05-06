#!/usr/bin/env python3
import h5py
import numpy as np
import pandas as pd
import torch as th
from numpy.linalg import norm
import itertools
from collections import defaultdict
import datetime


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

        if "ground_truth" in hin.keys():
            hout['ground_truth'] = hin['ground_truth'][:][old_idxs[sorted(new_idxs)], :-days]

        all_times = np.concatenate(all_times)
        idx = all_times.argsort()
        all_times = all_times[idx]
        all_nodes = np.concatenate(all_nodes)[idx]
        hout["all_nodes"] = all_nodes
        hout["all_times"] = all_times


def drop_k_days_csv(dset, outfile, days):
    df = pd.read_csv(dset, index_col="region")
    df = df[df.columns[:-days]]
    df.to_csv(outfile)


def print_model_stats(mus, beta, S, U, V, A):
    C = A - np.diag(np.diag(A))
    print("beta =", beta)
    print(f"\nNorms      : U = {norm(U).item():.3f}, V = {norm(V):.3f}")
    print(f"Max Element: U = {np.max(U).item():.3f}, V = {np.max(V):.3f}")
    print(f"Avg Element: U = {np.mean(U).item():.3f}, V = {np.mean(V):.3f}")
    print(f"\nSelf:       max = {np.max(S):.3f}, avg = {np.mean(S):.3f}")
    print(f"Cross:      max = {np.max(C):.3f}, avg = {np.mean(C):.3f}")
