#!/usr/bin/env python3


import numpy as np
from typing import List, DefaultDict, Tuple, Dict
import h5py
import pandas


def mk_episode(
    df: pandas.DataFrame,
    locations: List[str],
    location_map: DefaultDict[str, int],
    smooth: int = 1,
):
    ats = np.arange(len(df)) + 1
    timestamps = []
    nodes = []

    for county in locations:
        ws = df[county].rolling(window=max(1, smooth), min_periods=1).mean().to_numpy()
        ix = np.where(ws > 0)[0]

        if len(ix) < 1:
            continue
        ts = ats[ix]
        ws = np.diff([0] + ws[ix].tolist())
        es = []

        for i in range(len(ts)):
            w = int(ws[i])
            if w <= 0:
                continue
            tp = ts[i] - 1
            _es = sorted(np.random.uniform(tp, ts[i], w))
            if len(es) > 0:
                assert es[-1] < _es[0], (_es[0], es[-1])
            es += _es
        es = np.array(es)
        if len(es) > 0:
            timestamps.append(es)
            nodes.append(np.full(es.shape, location_map[county]))
    if len(timestamps) == 0:
        return None, None
    times = np.concatenate(timestamps)
    idx = np.argsort(times)
    return times[idx], np.concatenate(nodes)[idx]


def to_h5(
    df: pandas.DataFrame,
    outfile: str,
    location_map: Dict[str, int],
    episodes: List[Tuple[np.ndarray, np.ndarray]],
):
    """
    Create an HDF5 dataset

    Args:
        outfile: str - path to output file
        location_map: Dict[str, int] - mapping from location names to unique and contiguous ids
        episodes: List[Tuple[np.ndarray, np.ndarray]] - A list of episodes.  Each episode 
            corresponds to a pair of numpy arrays.  The first element is an array of times
            and the second element is an array of the same shape of node ids
    """
    str_dt = h5py.special_dtype(vlen=str)
    ds_dt = h5py.special_dtype(vlen=np.dtype("int"))
    ts_dt = h5py.special_dtype(vlen=np.dtype("float32"))

    times, entities = zip(*episodes)
    locations, cids = zip(*list(location_map.items()))

    with h5py.File(outfile, "w") as hf:
        hf.create_dataset(
            "nodes", data=np.array(locations, dtype="O")[np.argsort(cids)], dtype=str_dt
        )
        hf.create_dataset("cascades", data=np.arange(len(episodes)))

        x = hf.create_dataset("node", dtype=ds_dt, shape=(len(entities),))
        x[:] = entities
        x = hf.create_dataset("time", dtype=ts_dt, shape=(len(times),))
        x[:] = times
        hf.attrs["basedate"] = str(df.index.max().date())

        hf.create_dataset(
            "ground_truth",
            data=df.transpose()
            .loc[np.array(locations, dtype="O")][sorted(df.index)]
            .values,
        )

        # Group all events into a single episode
        processed = np.array([], dtype="int")
        all_times = []
        all_nodes = []
        for idx in range(len(times)):
            # Extract events corresponding to entities we haven't processed yet.
            mask = ~np.in1d(entities[idx], processed)
            all_nodes.append(entities[idx][mask])
            all_times.append(times[idx][mask])
            unique_nodes = np.unique(all_nodes[-1])
            assert np.intersect1d(processed, unique_nodes).size == 0
            processed = np.concatenate([processed, unique_nodes])

        all_times = np.concatenate(all_times)
        idx = all_times.argsort()
        all_times = all_times[idx]
        all_nodes = np.concatenate(all_nodes)[idx]
        hf.create_dataset("all_times", data=all_times, dtype="float64")
        hf.create_dataset("all_nodes", data=all_nodes)
