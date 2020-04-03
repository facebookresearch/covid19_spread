#!/usr/bin/env python3
import h5py
import numpy as np
import torch as th
from numpy.linalg import norm


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

    fpos = lambda x: scale * th.nn.functional.softplus(x, beta=1 / scale)
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
