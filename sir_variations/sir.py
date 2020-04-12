#!/usr/bin/env python3

import argparse
import numpy as np
import pandas as pd
import sys

# import torch as th
# from sklearn.linear_model import Ridge

import os,sys,inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir) 

from common import load_data


def load_confirmed(path, regions):
    
    nodes, ns, ts, hop = load_data(path)
    
    unk = np.where(nodes == "Unknown")[0]
    if len(unk) > 0:
        ix = np.where(ns != unk[0])
        ts = ts[ix]
    cases = []
    for i in range(1, int(np.ceil(ts.max())) + 1):
        ix = np.where(ts < i)[0]
        cases.append((i, len(ix)))
    days, cases = zip(*cases)

    return np.array(cases)


def load_population(path, col=1):
    df = pd.read_csv(path, header=None)
    pop = df.iloc[:, col].sum()
    regions = df.iloc[:, 0].to_numpy().tolist()
    return pop, regions


def convolve(x, k=1):
    # np adds padding by default, we remove the edges
    if k == 1:
        return x
    f = [1/k] * k
    return np.convolve(x, f)[k-1:1-k]


def fit_beta(beta, gamma, times, days_predict, beta_fit='exp', eps= 0.000001):
    # given beta gamma can be past few days!
    # be careful with the time axis values!
    
    _x = times 
    _y = beta - gamma + eps

    def fit(x, y):
        A = np.vstack([x, np.ones(len(x))]).T
        res = np.linalg.lstsq(A, y, rcond=None)
        slope, intercept = res[0]    
        return slope, intercept
    
    if beta_fit == 'lin':
        x, y = _x, _y 
        m, b = fit(x, y)
        beta_pred = m * days_predict + b + gamma[-1]
        return beta_pred.clip(0)
    
    elif beta_fit == 'exp':
        x, y = _x, np.log(_y)
        m, b = fit(x, y)
        beta_pred = np.exp(m * days_predict + b) + gamma[-1]
        return beta_pred
    
    elif beta_fit == 'power':
        x, y = np.log(_x), np.log(_y)
        m, b = fit(x, y)
        beta_pred = np.exp(m * np.log(days_predict) + b) + gamma[-1]
        return beta_pred
    
    
def doubling_time(i, window):
    # calculate doubling rate from seen data
    growth_rate = np.exp(np.diff(np.log(i))) - 1
    if window is not None:
        growth_rate = growth_rate[-window:]
    doubling_time = np.log(2) / growth_rate
    print(doubling_time)
    return doubling_time.mean()
    
    
def simulate(cases, s, i, r, beta, gamma, T, days, keep, dt_window=5, bc_window=1, fit_window=10, c=1, beta_fit='constant'):
    # days is the list of all days from the first case being 0
    # days_predict is the number of future days to predict
    # i[T] will be the first predicted day!
    n = s[0] + i[0] + r[0] # for normalization below

    T = len(i)
    days_given = list(range(T))
    days_predict = list(range(T, T + days))
    days_given = np.asarray(days_given)
    days_predict = np.asarray(days_predict)
    
    # fit beta
    if beta_fit == 'manual':
        # cross check values
        dt = doubling_time(i, dt_window)
        # dt = doubling_time(cases, dt_window) # calculate doubling time from cases or i
        distancing_reduction = 0.3 # copy from main sir.py
        intrinsic_growth_rate = 2.0 ** (1.0 / dt) - 1.0
        beta_val = (intrinsic_growth_rate + gamma[-1]) * (1.0 - distancing_reduction)
        beta_pred = [beta_val] * days # beta for future days
    elif beta_fit == 'constant':
        beta_val = np.mean(beta[-bc_window:])
        beta_pred = [beta_val] * days # beta for future days
    elif beta_fit in ['lin', 'exp', 'power']:
        assert fit_window > 1
        _times = days_given[-fit_window:]
        _beta = beta[-fit_window:]
        _gamma = gamma[-fit_window:]        
        beta_pred = fit_beta(_beta, _gamma, _times, days_predict, beta_fit)
    else:
        raise NotImplementedError
        
    # expand beta and gamma for past and future times
    beta = np.concatenate((beta, beta_pred))
    gamma = np.concatenate((gamma, [gamma[-1]] * days ))
    
    for t in range(days):
        # calculate sir at tau + 1 using time tau
        tau = T + t - 1

        # get beta & gamma at time tau
        beta_current = beta[tau]
        gamma_current = gamma[tau]

        # calcualte s, i, r at tau + 1
        i_next = i[-1] + i[-1] * (beta_current * (s[-1] / n)  - gamma_current)
        r_next = r[-1] + i[-1] * gamma_current / c
        s_next = s[-1] - i[-1] * beta_current * (s[-1] / n) / c

        i_next = max(0, i_next)
        s_next = max(0, s_next)

        i = np.append(i, i_next)
        r = np.append(r, r_next)
        s = np.append(s, s_next)

        if i_next == 0:
            break
            
    _i = i.astype(int) # convert predicted numbers to int
    _cases = list(cases) + ['?'] * days # for printing
    # infs = pd.DataFrame({"Day": list(range(T-1, T+keep)), f"beta_last: {beta[-1]:.2f}": i[T-1:T+keep]})
    infs = pd.DataFrame(
        {
            # "Day": list(range(keep + 1)), # days from first prediction time
            "Day": list(range(T-1, T+keep)), # days from first absolute time
            "observed": _cases[T-1:T+keep],
            beta_fit: _i[T-1:T+keep]
        }
    )
    
    ix_max = np.argmax(i)
    if ix_max == len(i) - 1:
        peak_days = f"{ix_max - T + 1}+"
    else:
        peak_days = str(ix_max - T + 1)

    mae = [abs(cases[j] - i[j]) for j in range(T, len(cases))]
    rmse = [(cases[j] - i[j]) ** 2 for j in range(T, len(cases))]
    
    meta = pd.DataFrame(
        {
            "method": [beta_fit],
            "R0": [round(beta[-1] / gamma[-1], 3)],
            "beta": [round(beta[-1], 3)],
            "gamma": [round(gamma[-1], 3)],
            "Peak days": [peak_days],
            "Peak cases": [int(i[ix_max])],
            "mae-{}-days".format(len(mae)): [np.mean(mae)],
            "rmse-{}-days".format(len(rmse)): [np.mean(rmse)],
            "dt-init": [round(doubling_time(cases, dt_window), 2)],
            "dt-final": [round(doubling_time(i, dt_window), 2)]
        }
    )
    return meta, infs


def main(args):
    parser = argparse.ArgumentParser(description="Forecasting with SIR model")
    parser.add_argument("-fdat", help="Path to confirmed cases", required=True)
    parser.add_argument("-fpop", help="Path to population data", required=True)
    parser.add_argument("-days", type=int, help="nDays to forecast", required=True)
    parser.add_argument("-keep", type=int, help="nDays to keep in CSV", required=True)
    parser.add_argument("-doubling-times", type=float, nargs="+", help="Addl d-times to simulate")
    parser.add_argument("-recovery-days", type=int, default=14, help="Recovery days")
    parser.add_argument("-fsuffix", type=str, help="prefix to store forecast and metadata")
    parser.add_argument("-dout", type=str, default=".", help="Output directory")
    parser.add_argument("-dt-window", type=int, help="window to compute doubling time")
    parser.add_argument("-bc-window", type=int, default=5, help="beta constant averaging window")
    parser.add_argument("-fit-window", type=int, default=10, help="fit using last .. days")
    parser.add_argument("-initial-state", type=int, default=0, help="days of rewinding for retrofit")
    parser.add_argument("-coeff", type=float, default=1., help="between 0 & 1, fraction of reported cases")
    
    opt = parser.parse_args(args)

    # load data
    n, regions = load_population(opt.fpop)
    cases = load_confirmed(opt.fdat, regions)
    
    # initialize s, i, r as sequences of length T
    r = cases.copy() * 0.0 # zero IC for recovered
    i = cases - r # values for the infected
    s = n - i - r
    T = len(i)
    
    # length of gamma and beta are T - 1
    gamma_constant = 1.0 / opt.recovery_days
    gamma = np.full(T - 1, gamma_constant)
    di = i[1:] - i[:-1]
    beta = (di / i[:-1] + gamma) * (n / s[:-1])
    
    f_sim = lambda beta_fit: simulate(
        cases,
        s[:len(s) - opt.initial_state],
        i[:len(i) - opt.initial_state],
        r[:len(r) - opt.initial_state],
        beta[:len(beta) - opt.initial_state],
        gamma[:len(gamma) - opt.initial_state],
        T - opt.initial_state,
        opt.days,
        opt.keep,
        opt.dt_window,
        opt.bc_window,
        opt.fit_window,
        opt.coeff,
        beta_fit
    )
    
    # simulate with constant beta average of past opt.window days
    meta, df = f_sim('manual')
    # simulate with different fits, in pessimistic order 
    for beta_fit in ['lin', 'exp', 'power', 'constant']:
        _meta, _df = f_sim(beta_fit)
        meta = meta.append(_meta, ignore_index=True)
        df = pd.merge(df, _df, on=["Day", "observed"])
    
    print('\n', meta, '\n\n', df)
    
    # add sweep through c
    
    if opt.fsuffix is not None:
        meta.to_csv(f"{opt.dout}/SIR-metadata-{opt.fsuffix}.csv")
        df.to_csv(f"{opt.dout}/SIR-forecast-{opt.fsuffix}.csv")

        
if __name__ == "__main__":
    main(sys.argv[1:])