#!/usr/bin/env python3

import torch.multiprocessing as mp
import numpy as np
import pandas as pd
from functools import partial
from scipy.stats import ksone
from scipy import integrate
from tick.hawkes import HawkesExpKern, SimuHawkesMulti
from utils import to_tick_data
from simulation_ext import HawkesExpSimulation


def ks_critical_value(n_trials, alpha):
    """
    Critical value of Kolmogorov-Smirnov test for significance level alpha
    """
    return ksone.ppf(1 - alpha / 2, n_trials)


def resid(x, intensities, timestamps, dim):
    arrivals = timestamps[dim]
    thetas = np.zeros(len(arrivals) - 1)
    ints = intensities[dim]
    for i in range(1, len(arrivals)):
        mask = (x <= arrivals[i]) & (x >= arrivals[i - 1])
        xs = x[mask]
        ys = ints[mask]
        try:
            thetas[i - 1] = integrate.simps(ys, xs)
        except:
            thetas[i - 1] = np.nan
    return thetas


def goodness_of_fit(episode, step, mu, beta, A, nodes):
    timestamps = to_tick_data([episode], [None], nodes)
    learner = HawkesExpKern(beta)
    learner.fit(timestamps)
    learner.adjacency[:] = A[:]
    learner.baseline[:] = mu[:]
    dimension = learner.n_nodes
    intensities, x = learner.estimated_intensity(timestamps[0], step)
    residuals = [resid(x, intensities, timestamps[0], dim) for dim in range(dimension)]
    residuals = [res[np.logical_not(np.isnan(res))] for res in residuals]
    return residuals


def initialize_trial(_learner, _timestamps):
    global learner, timestamps
    learner = _learner
    timestamps = _timestamps


def run_trial(t_obs, t_max, d, M, tid):
    simu = learner._corresponding_simu()
    simu.force_simulation = True
    simu.verbose = False
    simu.track_intensity(-1)
    simu.reset()
    simu.set_timestamps(timestamps[0], t_obs)
    simu.end_time = t_max + 1
    simu.simulate()
    simc = {i: np.zeros(M) for i in range(1, d + 1)}
    for i in range(1, d + 1):
        for n in range(M):
            ix = np.where(simu.timestamps[n] <= t_obs + i)[0]
            simc[i][n] += len(ix)
    return simc


def simulate_tl_mhp(
    t_obs,
    d,
    episode,
    timescale,
    simulator,
    nodes,
    trials,
    quiet=False,
    stddev=-1,
    max_events=-1,
):
    confirmed_cases = np.bincount(episode.entities, minlength=len(nodes))
    t_max = t_obs + (d / timescale)
    d = int(np.ceil(d))
    counts = np.empty((d + 1, len(nodes), trials))
    counts[0, :, :] = confirmed_cases[:, np.newaxis]
    all_evts = simulator.simulate(
        episode.entities, episode.timestamps, t_max, trials, quiet, max_events
    )
    for trial in range(trials):
        evts = np.array(all_evts[trial])
        for i in range(1, d + 1):
            current_evts = evts[evts[:, 1] <= t_obs + (i / timescale)]
            cur_counts = np.bincount(
                current_evts[:, 0].astype(int), minlength=len(nodes)
            )
            counts[i, :, trial] = cur_counts
    std = np.clip(np.std(counts, axis=-1, keepdims=True), 0.0001, None)
    z = (counts - np.mean(counts, axis=-1, keepdims=True)) / std
    if stddev > 0:
        mask = np.abs(z) < stddev
        counts[~mask] = 0
        print(
            f"Throwing out {np.sum(~mask)} simulations out of {mask.size} ({np.sum(~mask)/mask.size})"
        )
    else:
        mask = np.ones_like(counts)
    counts = counts.sum(-1) / mask.sum(-1)
    df = pd.DataFrame(np.transpose(counts))
    df["county"] = nodes
    return df


def simulate_mhp(
    t_obs, d, episode, mus, beta, A, timescale, nodes, step, trials, episode_true=None
):
    """
    Simulate a MHP from t_obs until t_obs + d
    """
    assert episode.timestamps[0] == 0
    timestamps = to_tick_data([episode], [None], nodes)
    if episode_true is not None:
        timestamps_true = to_tick_data([episode_true], [None], nodes)
    else:
        timestamps_true = timestamps
    confirmed_cases = np.array([len(t) for t in timestamps[0]])
    confirmed_cases_true = np.array([len(t) for t in timestamps_true[0]])

    learner = HawkesExpKern(beta, max_iter=1)
    learner.fit(timestamps)
    learner.adjacency[:] = A[:]
    learner.baseline[:] = mus[:]

    t_max = (t_obs + d) / timescale
    t_obs /= timescale
    # simu = learner._corresponding_simu()
    # simu.force_simulation = True
    # simu.verbose = False
    # simu.track_intensity(-1)

    with mp.Pool(
        processes=min(trials, 40),
        initializer=initialize_trial,
        initargs=(learner, timestamps),
    ) as pool:
        ft = partial(run_trial, t_obs, t_max, d, len(nodes))
        results = pool.map(ft, range(trials))
    simc = {i: [] for i in range(1, d + 1)}

    for _simc in results:
        for k, v in _simc.items():
            simc[k].append(v)
    for k, v in simc.items():
        simc[k] = (
            np.median(np.stack(v, axis=0), axis=0)
            - confirmed_cases
            + confirmed_cases_true
        )

    # simu.set_timestamps(timestamps[0], t_obs)
    # simu.end_time = t_max
    # multi = SimuHawkesMulti(simu, n_simulations=trials, n_threads=min(trials, 40))
    # multi.simulate()
    # print("Spectral radius =", simu.spectral_radius())
    # print(sum(confirmed_cases))
    # simc = {i: np.zeros(len(nodes)) for i in range(1, d + 1)}
    # for k in range(trials):
    #     simu.reset()
    #     simu.set_timestamps(timestamps[0], t_obs)
    #     # simu.max_jumps = sum(confirmed_cases) * 2
    #     simu.end_time = t_max
    #     simu.simulate()
    #     for i in range(1, d + 1):
    #         for n in range(len(nodes)):
    #             ix = np.where(simu.timestamps[n] < t_obs + i)[0]
    #             simc[i][n] += len(ix)

    # simc = {k: v / trials for k, v in simc.items()}

    simc["county"] = nodes
    simc[0] = confirmed_cases_true
    return pd.DataFrame(simc)
