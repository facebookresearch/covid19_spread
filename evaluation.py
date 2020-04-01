#!/usr/bin/env python3

import numpy as np
import pandas as pd
from scipy.stats import ksone
from scipy import integrate
from tick.hawkes import HawkesExpKern, SimuHawkesMulti
from utils import to_tick_data


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


def run_trial(simu, timestamps):
    pass


def simulate_mhp(t_obs, d, episode, mus, beta, A, timescale, nodes, step, trials):
    """
    Simulate a MHP from t_obs until t_obs + d
    """
    timestamps = to_tick_data([episode], [None], nodes)
    learner = HawkesExpKern(beta, max_iter=1)
    learner.fit(timestamps)
    learner.adjacency[:] = A[:]
    learner.baseline[:] = mus[:]

    confirmed_cases = [len(t) for t in timestamps[0]]
    t_max = (t_obs + d) / timescale
    t_obs /= timescale
    simu = learner._corresponding_simu()
    simu.force_simulation = True
    simu.verbose = False
    simu.track_intensity(step)
    simu.set_timestamps(timestamps[0], t_obs)
    simu.end_time = t_max
    # multi = SimuHawkesMulti(simu, n_simulations=trials, n_threads=min(trials, 40))
    # multi.simulate()
    # print("d")
    simc = {i: np.zeros(len(nodes)) for i in range(1, d + 1)}
    for k in range(trials):
        simu.reset()
        simu.set_timestamps(timestamps[0], t_obs)
        simu.end_time = t_max
        simu.simulate()
        for i in range(1, d + 1):
            for n in range(len(nodes)):
                ix = np.where(simu.timestamps[n] < t_obs + i)[0]
                simc[i][n] += len(ix)

    simc = {k: v / trials for k, v in simc.items()}
    simc["county"] = nodes
    simc[0] = confirmed_cases
    return pd.DataFrame(simc)
