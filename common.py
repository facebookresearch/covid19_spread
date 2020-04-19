#!/usr/bin/env python3
import h5py
import numpy as np
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

    with h5py.File(dset, 'r') as hin, h5py.File(outfile, 'w') as hout:
        times = hin['time'][:]
        nodes = hin['node'][:]
        new_nodes = []
        new_times = []
        # Compute the new maxtime
        max_time = max(*[ts[-1].item() for ts in times]) - days

        processed = np.array([])
        all_times = []
        all_nodes = []

        for idx, ts in enumerate(times):
            mask = ts < max_time
            new_nodes.append(np.array([namer[x] for x in nodes[idx][mask]]))
            new_times.append(ts[mask])

            mask = ~np.in1d(new_nodes[-1], processed)
            all_nodes.append(new_nodes[-1][mask])
            all_times.append(new_times[-1][mask])
            processed = np.concatenate([processed, np.unique(all_nodes[-1])])

        old_idxs, new_idxs = map(np.array, zip(*namer.items()))
        hout['nodes'] = hin['nodes'][:][old_idxs[sorted(new_idxs)]]
        hout.create_dataset('time', data=new_times, dtype=ts_dt)
        hout.create_dataset('node', data=new_nodes, dtype=ds_dt)
        hout['cascades'] = hin['cascades'][:]
        if 'basedate' in hin.attrs:
            date = datetime.datetime.strptime(hin.attrs['basedate'], '%Y-%m-%d')
            hout.attrs['basedate'] = str((date - datetime.timedelta(days=days)).date())
        
        all_times = np.concatenate(all_times)
        idx = all_times.argsort()
        all_times = all_times[idx]
        all_nodes = np.concatenate(all_nodes)[idx]
        hout['all_nodes'] = all_nodes
        hout['all_times'] = all_times


def load_model(model_path, M):
    data = th.load(model_path)
    opt = data["opt"]
    dim, scale, timescale = opt.dim, opt.scale, opt.timescale
    mus = data["model"]["mus_.weight"].cpu()[:-1, :]
    alpha_s = data["model"]["self_A.weight"].cpu()[:-1, :]
    beta = data["model"]["beta_"]
    U = data["model"]["U.weight"].cpu()[:-1, :]
    V = data["model"]["V.weight"].cpu()[:-1, :]

    assert U.size(0) == M, (U.size(0), M)
    assert V.size(0) == M, (V.size(0), M)

    fpos = lambda x: th.nn.functional.softplus(x, beta=1 / scale)
    beta_ = fpos(beta).item()
    U_ = fpos(U).numpy()
    V_ = fpos(V).numpy()
    S_ = fpos(alpha_s).numpy().flatten()
    A = np.dot(U_, V_.T) / dim
    for i in range(M):
        A[i, i] = S_[i]
    mus_ = fpos(mus).numpy().flatten()
    return mus_, beta_, S_, U_, V_, A, scale, timescale


def load_data(data_path):
    per_district = {}
    with h5py.File(data_path, "r") as fin:
        nodes = np.array([m for m in fin["nodes"]])
        if 'all_nodes' in fin:
            ns = fin['all_nodes'][:]
            ts = fin['all_times'][:]
        else:
            ns = fin["node"][0]
            ts = fin["time"][0]
    for i, n in enumerate(nodes):
        ix = np.where(ns == i)[0]
        per_district[n] = (ts[ix], None)
    # id_to_ags = {ns[i]: ags[i] for i in range(len(ns))}
    return nodes, ns, ts, per_district


def print_model_stats(mus, beta, S, U, V, A):
    C = A - np.diag(np.diag(A))
    print("beta =", beta)
    print(f"\nNorms      : U = {norm(U).item():.3f}, V = {norm(V):.3f}")
    print(f"Max Element: U = {np.max(U).item():.3f}, V = {np.max(V):.3f}")
    print(f"Avg Element: U = {np.mean(U).item():.3f}, V = {np.mean(V):.3f}")
    print(f"\nSelf:       max = {np.max(S):.3f}, avg = {np.mean(S):.3f}")
    print(f"Cross:      max = {np.max(C):.3f}, avg = {np.mean(C):.3f}")
