import h5py
import numpy as np
import pandas as pd


def load_confirmed(path, regions):
    """Loads confirmed total confirmed cases"""
    # df = pd.read_csv(path, usecols=regions)
    # cases = df.to_numpy().sum(axis=1)
    nodes, ns, ts, _ = load_data(path)
    unk = np.where(nodes == "Unknown")[0]
    if len(unk) > 0:
        ix = np.where(ns != unk[0])
        ts = ts[ix]
    cases = []
    for i in range(1, int(np.ceil(ts.max())) + 1):
        ix = np.where(ts < i)[0]
        cases.append((i, len(ix)))
    _, cases = zip(*cases)
    return np.array(cases)


def load_confirmed_by_region(path):
    nodes, ns, ts, basedate, = load_data(path)
    nodes = np.array(nodes)
    tmax = int(np.ceil(ts.max()))
    cases = np.zeros((len(nodes), tmax))
    for n in range(len(nodes)):
        ix2 = np.where(ns == n)[0]
        for i in range(1, tmax + 1):
            ix1 = np.where(ts < i)[0]
            cases[n, i - 1] = len(np.intersect1d(ix1, ix2))
    unk = np.where(nodes == "Unknown")[0]
    cases = np.delete(cases, unk, axis=0)
    nodes = np.delete(nodes, unk)
    return cases, nodes, basedate


def load_population(path, col=1):
    df = pd.read_csv(path, header=None)
    pop = df.iloc[:, col].sum()
    regions = df.iloc[:, 0].to_numpy().tolist()
    print(regions)
    return pop, regions


def load_populations_by_region(path, col=1):
    """Loads region-level populations after filtering unknown nodes

    Returns: tuple of lists with (populations, regions)
    """
    df = pd.read_csv(path, header=None)
    populations_df = df.iloc[:, [0, col]]
    populations_df.columns = ["region", "population"]
    # filter unknown regions
    populatins_df = populations_df[populations_df["region"].str.lower() != "unknown"]
    return populations_df


def filter_populations(df, nodes):
    """Removes populations and regions with unknown nodes"""
    mask = nodes == "Unknown"
    return df[mask]

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
    with h5py.File(data_path, "r") as fin:
        basedate = fin.attrs["basedate"]
        nodes = np.array([m for m in fin["nodes"]])
        if "all_nodes" in fin:
            ns = fin["all_nodes"][:]
            ts = fin["all_times"][:]
        else:
            ns = fin["node"][0]
            ts = fin["time"][0]
    return nodes, ns, ts, basedate