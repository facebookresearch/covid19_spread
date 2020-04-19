#!/usr/bin/env python3

import argparse
import numpy as np
import pandas as pd
import sys
from common import load_data


def load_confirmed(path, regions):
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
    print(cases)
    days, cases = zip(*cases)
    return np.array(cases)


def load_population(path, col=1):
    df = pd.read_csv(path, header=None)
    pop = df.iloc[:, col].sum()
    regions = df.iloc[:, 0].to_numpy().tolist()
    print(regions)
    return pop, regions


def sir(s: float, i: float, r: float, beta: float, gamma: float, n: float):
    """The SIR model, one time step."""
    s_n = (-beta * s * i) + s
    i_n = (beta * s * i - gamma * i) + i
    r_n = gamma * i + r
    if s_n < 0.0:
        s_n = 0.0
    if i_n < 0.0:
        i_n = 0.0
    if r_n < 0.0:
        r_n = 0.0

    scale = n / (s_n + i_n + r_n)
    return s_n * scale, i_n * scale, r_n * scale


def gen_sir(s: float, i: float, r: float, beta: float, gamma: float, n_days: int):
    """Simulate SIR model forward in time yielding tuples."""
    s, i, r = (float(v) for v in (s, i, r))
    n = s + i + r
    for d in range(n_days + 1):
        yield d, s, i, r
        s, i, r = sir(s, i, r, beta, gamma, n)


def simulate(
    cases, population, doubling_time, recovery_days, distancing_reduction, days, keep
):
    I0 = cases[0]
    S0 = population - I0
    R0 = 0.0

    # doubling_time = 3
    # relative_contact_rate = 0.3
    intrinsic_growth_rate = 2.0 ** (1.0 / doubling_time) - 1.0
    # print(intrinsic_growth_rate, growth_rate, growth_rate.mean())

    # Contact rate, beta
    gamma = 1.0 / recovery_days
    beta = (intrinsic_growth_rate + gamma) / S0 * (1.0 - distancing_reduction)

    # compute fit quality
    res = {d: _I for d, _S, _I, _R in gen_sir(S0, I0, R0, beta, gamma, days)}
    # mae = [abs(cases[i] - res[i]) for i in range(5, len(cases))]
    # rmse = [(cases[i] - res[i]) ** 2 for i in range(5, len(cases))]

    # print(mae)
    # print(rmse)

    # simulate forward
    IN = cases[-1]
    SN = population - IN
    RN = 0
    res = {d: _I for d, _S, _I, _R in gen_sir(SN, IN, RN, beta, gamma, days)}
    # res = {d: _I for d, _S, _I, _R in gen_sir(S0 - 3772, 3773, R0, beta, gamma, days)}
    days, infected = zip(*res.items())
    infs = pd.DataFrame({"Day": days[:keep], f"{doubling_time:.2f}": infected[:keep]})
    ix_max = np.argmax(infected)
    if ix_max == len(infected) - 1:
        peak_days = f"{ix_max}+"
    else:
        peak_days = str(ix_max)
    meta = pd.DataFrame(
        {
            "Doubling time": [round(doubling_time, 3)],
            "R0": [round(beta / gamma * S0, 3)],
            "beta": [round(beta * S0, 3)],
            "gamma": [round(gamma, 3)],
            "Peak days": [peak_days],
            "Peak cases": [int(infected[ix_max])],
            # "MAE": [np.mean(mae)],
            # "RMSE": [np.mean(rmse)],
        }
    )
    return meta, infs


def estimate_growth_const(cases, window=None):
    growth_rate = np.exp(np.diff(np.log(cases))) - 1
    if window is not None:
        growth_rate = growth_rate[-window:]
    doubling_time = np.log(2) / growth_rate
    # print(doubling_time, doubling_time.mean())
    doubling_time = doubling_time.mean()
    return doubling_time


def main(args):
    parser = argparse.ArgumentParser(description="Forecasting with SIR model")
    parser.add_argument("-fdat", help="Path to confirmed cases", required=True)
    parser.add_argument("-fpop", help="Path to population data", required=True)
    parser.add_argument(
        "-days", type=int, help="Number of days to forecast", required=True
    )
    parser.add_argument(
        "-keep", type=int, help="Number of days to keep in CSV", required=True
    )
    parser.add_argument("-window", type=int, help="window to compute doubling time")
    parser.add_argument(
        "-doubling-times",
        type=float,
        nargs="+",
        help="Additional doubling times to simulate",
    )
    parser.add_argument("-recovery-days", type=int, default=14, help="Recovery days")
    parser.add_argument(
        "-distancing-reduction", type=float, default=0.2, help="Recovery days"
    )
    parser.add_argument(
        "-fsuffix", type=str, help="prefix to store forecast and metadata"
    )
    parser.add_argument("-dout", type=str, default=".", help="Output directory")
    opt = parser.parse_args(args)

    population, regions = load_population(opt.fpop)
    cases = load_confirmed(opt.fdat, regions)
    tmax = len(cases)
    t = np.arange(tmax) + 1

    doubling_time = estimate_growth_const(cases, opt.window)
    print(f"Population size = {population}")
    print(f"Inferred doubling time = {doubling_time}")

    f_sim = lambda dt: simulate(
        cases,
        population,
        dt,
        opt.recovery_days,
        opt.distancing_reduction,
        opt.days,
        opt.keep + 1,
    )
    meta, df = f_sim(doubling_time)
    for dt in opt.doubling_times:
        _meta, _df = f_sim(dt)
        meta = meta.append(_meta, ignore_index=True)
        df = pd.merge(df, _df, on="Day")
    print()
    print(meta)
    print()
    print(df)

    if opt.fsuffix is not None:
        meta.to_csv(f"{opt.dout}/SIR-metadata-{opt.fsuffix}.csv")
        df.to_csv(f"{opt.dout}/SIR-forecast-{opt.fsuffix}.csv")


if __name__ == "__main__":
    main(sys.argv[1:])
