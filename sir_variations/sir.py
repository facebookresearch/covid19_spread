#!/usr/bin/env python3

import argparse
import numpy as np
import pandas as pd
import sys
import torch as th

from sklearn.linear_model import Ridge

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



def simulate(s, i, r, beta, gamma, J, alpha_beta, days, days_predict, keep):
    # days is the list of all days from the first case being 0
    # days_predict is the number of future days to predict
    # i[T] will be the first predicted day!
    n = s[0] + i[0] + r[0] # for normalization below

    
#     # FIR filter fit for beta | not great
#     beta_head = np.asarray([beta[j - J:j] for  j in range(J, T - 1)])
#     clf_beta = Ridge(alpha=alpha_beta) # overfits at 0.001 & J=15
#     clf_beta.fit(beta_head, beta[J:T - 1])
    
    threshold = 3000 # try 3000 and 5000 as well
    k = 5 # convolution size
    i_ = i[i>threshold] # ignore first few days
    
    i_ = convolve(i_, k)
    di_ = i_[1:] - i_[:-1] # delta_i
    beta_ = di_ / i_[:-1] + gamma[-1] # gamma:1/14 | initial phase 
#     print(len(beta_))
    x_range = np.linspace(1, len(i), len(beta_))
#     print(x_range)
    x, y = np.log(x_range), np.log(beta_ - 1/14 + 0.0001)
    A = np.vstack([x, np.ones(len(x))]).T
    res = np.linalg.lstsq(A, y, rcond=None)
#     print(res[0][0], res[1])
    slope, intercept = res[0]
#     print(beta_)
    
    
    for t in range(days_predict):
        days.append(T + t)

        # predict next beta & calculate s i r
#         beta_next = clf_beta.predict(beta[-J:].reshape(1,-1))
        beta_next = np.exp(slope * np.log(T + t) + intercept) + gamma[-1]
#         print(T + t, beta_next)
        # uncomment below to cross test with sir.py
#         beta_next = 0.26523 
        
        beta_next = max(0, beta_next)
        gamma_next = gamma[-1] # constant gamma | may increase slightly

        i_next = i[-1] + i[-1] * (beta_next * (s[-1] / n)  - gamma_next)
        r_next = r[-1] + i[-1] * gamma_next
        s_next = s[-1] - i[-1] * beta_next * (s[-1] / n)

        i_next = max(0, i_next)
        s_next = max(0, s_next)

        beta = np.append(beta, beta_next)
        gamma = np.append(gamma, gamma_next)
        i = np.append(i, i_next)
        r = np.append(r, r_next)
        s = np.append(s, s_next)

        if i_next == 0:
            break
            
    i = i.astype(int) # convert predicted numbers to int
    infs = pd.DataFrame({"Day": days[T-1:T+keep], f"{beta[-1]:.2f}": i[T-1:T+keep]})
    ix_max = np.argmax(i)
    if ix_max == len(i) - 1:
        peak_days = f"{ix_max - T + 1}+" # + 1 bc indexing starts from 0, from the last day this many days to reach the peak
    else:
        peak_days = str(ix_max - T + 1)

    meta = pd.DataFrame(
        {
            "R0": [round(beta[-1] / gamma[-1], 3)],
            "beta": [round(beta[-1], 3)],
            "gamma": [round(gamma[-1], 3)],
            "Peak days": [peak_days],
            "Peak cases": [int(i[ix_max])],
        }
    )
    return meta, infs


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Forecasting with SIR model")
    parser.add_argument("-fdat", help="Path to confirmed cases", required=True)
    parser.add_argument("-fpop", help="Path to population data", required=True)
    parser.add_argument("-days", type=int, help="nDays to forecast", required=True)
    parser.add_argument("-keep", type=int, help="nDays to keep in CSV", required=True)
    parser.add_argument("-window", type=int, help="window to compute doubling time")
    parser.add_argument("-doubling-times", type=float, nargs="+", help="Addl d-times to simulate")
    parser.add_argument("-recovery-days", type=int, default=14, help="Recovery days")
    parser.add_argument("-distancing-reduction", type=float, default=0.3)
    parser.add_argument("-fsuffix", type=str, help="prefix to store forecast and metadata")
    parser.add_argument("-dout", type=str, default=".", help="Output directory")
    parser.add_argument("-firJ", type=int, default=10, help="filter number for beta")
    parser.add_argument("-alpha-beta", type=float, default=3, help="ridge reg coeff for beta")
    
    opt = parser.parse_args(sys.argv[1:])

    # load data
    n, regions = load_population(opt.fpop)
    cases = load_confirmed(opt.fdat, regions)

    T = len(cases)
    days = list(range(T))

    # HardCoded constant sq of gammas
    gamma = np.asarray([1.0 / opt.recovery_days] * T )
    
    # artificial values for the recovered
    r = cases.copy() * gamma * 0.0
    i = cases - r # values for the infected
    s = n - i - r 
    
    # calculate beta and gamma from 0 until T - 2 
    delta_i = i[1:] - i[:-1]
    delta_r = r[1:] - r[:-1]
    beta = (n / s[:-1]) * (delta_i + delta_r) / i[:-1] # consider s_n/n
    
    meta, df = simulate(s, i, r, beta, gamma, opt.firJ, opt.alpha_beta, days, opt.days, opt.keep)
    
    # calculate doubling rate from seen data
    growth_rate = np.exp(np.diff(np.log(cases))) - 1
    if opt.window is not None:
        growth_rate = growth_rate[-opt.window :]
    doubling_time = np.log(2) / growth_rate
    doubling_time = doubling_time.mean()
    
    meta.insert(0, "Doubling time", [round(doubling_time, 3)])
    
    print('\n', meta, '\n\n', df)

    if opt.fsuffix is not None:
        meta.to_csv(f"{opt.dout}/SIR-metadata-{opt.fsuffix}.csv")
        df.to_csv(f"{opt.dout}/SIR-forecast-{opt.fsuffix}.csv")
