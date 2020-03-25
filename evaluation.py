#!/usr/bin/env python3

import numpy as np
import pandas as pd
from scipy.stats import ksone
from scipy import integrate
from tick.hawkes import HawkesExpKern
from utils import to_tick_data


def ks_critical_value(n_trials, alpha):
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
    timestamps = to_tick_data([episode], ["covid19_nj"], nodes)
    learner = HawkesExpKern(beta)
    learner.fit(timestamps)
    learner.adjacency[:] = A[:]
    learner.baseline[:] = mu[:]
    dimension = learner.n_nodes
    intensities, x = learner.estimated_intensity(timestamps[0], step)
    residuals = [resid(x, intensities, timestamps[0], dim) for dim in range(dimension)]
    return residuals


def simulate_mhp(t_obs, d, episode, mus, beta, A, timescale, nodes):
    timestamps = to_tick_data([episode], ["covid19_nj"], nodes)
    learner = HawkesExpKern(beta)
    learner.fit(timestamps)
    learner.adjacency[:] = A[:]
    learner.baseline[:] = mus[:]

    confirmed_cases = [len(t) for t in timestamps[0]]
    t_max = (t_obs + d) / timescale
    t_obs /= timescale
    simu = learner._corresponding_simu()
    simu.force_simulation = True
    simu.verbose = False
    simu.track_intensity(0.01)
    simu_cases = np.zeros(len(nodes))
    trials = 10
    for _ in range(trials):
        simu.reset()
        simu.set_timestamps(timestamps[0], t_obs)
        simu.end_time = t_max
        simu.simulate()
        simu_cases += np.array([len(t) for t in simu.timestamps])
        simu_times = [t[-1] for t in simu.timestamps]
    simu_cases /= trials

    df_pred = pd.DataFrame(
        {"county": nodes, "confirmed": confirmed_cases, f"MHP d{d}": simu_cases}
    )
    return df_pred
