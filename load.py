import h5py
import numpy as np
import pandas as pd
import torch as th
from datetime import timedelta


def load_confirmed_csv(path):
    df = pd.read_csv(path)
    df.set_index("region", inplace=True)
    basedate = df.columns[-1]
    nodes = df.index.to_numpy()
    cases = df.to_numpy()
    return th.from_numpy(cases), nodes, basedate


def load_confirmed(path, regions):
    """Returns dataframe of total confirmed cases"""
    df = load_confirmed_by_region(path, regions=regions)
    return df.sum(axis=1)


def load_confirmed_by_region(path, regions=None, filter_unknown=True):
    """Loads confirmed cases from h5 or csv file.
    If regions is provided, filters cases in those regions.

    Returns: pd.DataFrame with dates along row and regions in columns
    """
    if path.endswith("csv"):
        return _load_confirmed_by_region_csv(path, regions, filter_unknown)
    elif path.endswith("h5"):
        return _load_confirmed_by_region_h5(path, regions, filter_unknown)
    raise ValueError(f"Data type for {path} not supported")


def _load_confirmed_by_region_csv(path, regions, filter_unknown):
    """Loads csv file for confirmed cases by region"""
    df = pd.read_csv(path, index_col=0, header=None)
    # transpose so dates are along rows to match h5
    df = df.T
    # set date as index
    df = df.rename(columns={"region": "date"})  
    df = df.set_index("date")
    df = df.astype(float)
    if regions is not None:
        df = df[regions]
    if filter_unknown:
        df = df.loc[:, df.columns != "Unknown"]
    return df


def _load_confirmed_by_region_h5(path, regions, filter_unknown):
    """Loads h5 files for confirmed cases by region"""
    nodes, ns, ts, end_date = load_data(path)
    nodes = np.array(nodes)
    tmax = int(np.ceil(ts.max()))
    cases = np.zeros((len(nodes), tmax))
    for n in range(len(nodes)):
        ix2 = np.where(ns == n)[0]
        for i in range(1, tmax + 1):
            ix1 = np.where(ts < i)[0]
            cases[n, i - 1] = len(np.intersect1d(ix1, ix2))
    if filter_unknown:
        cases, nodes = _filter_unknown(cases, nodes)

    df = pd.DataFrame(cases.T, columns=nodes)
    df = _set_dates(df, end_date)

    if regions:
        df = df[regions]
    return df


def _set_dates(df: pd.DataFrame, end_date: str):
    """Adds dates to predicton dataframe. 

    Args:
        df: dataframe of cases (rows are days, columns are regions)
        end_date (str): latest day with data in YYYY-MM-DD format.
    
    Returns: dataframe with dates as indices
    """
    end_date = pd.to_datetime(end_date)
    days = df.shape[0]
    dates = [end_date - timedelta(i) for i in range(days - 1, -1, -1)]
    df["date"] = dates
    df.set_index("date", inplace=True)
    return df


def _filter_unknown(cases, nodes):
    """Filters casese for unknown nodes"""
    unk = np.where(nodes == "Unknown")[0]
    cases = np.delete(cases, unk, axis=0)
    nodes = np.delete(nodes, unk)
    return cases, nodes


def load_population(path, col=1):
    df = pd.read_csv(path, header=None)
    pop = df.iloc[:, col].sum()
    regions = df.iloc[:, 0].to_numpy().tolist()
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
