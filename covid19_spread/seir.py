#!/usr/bin/env python3

import argparse
import numpy as np
import pandas as pd
import sys
from . import load
from .cross_val import CV
from typing import List
from datetime import timedelta


def seir(
    s: float,
    e: float,
    i: float,
    r: float,
    beta: float,
    gamma: float,
    sigma: float,
    n: float,
):
    """The SEIR model, one time step."""
    s_n = (-beta * s * i) + s
    e_n = (beta * s * i) - sigma * e + e
    i_n = (beta * s * i - gamma * i) + i
    r_n = gamma * i + r
    if s_n < 0.0:
        s_n = 0.0
    if e_n < 0.0:
        e_n = 0.0
    if i_n < 0.0:
        i_n = 0.0
    if r_n < 0.0:
        r_n = 0.0

    scale = n / (s_n + e_n + i_n + r_n)
    return s_n * scale, e_n * scale, i_n * scale, r_n * scale


def gen_seir(
    s: float,
    e: float,
    i: float,
    r: float,
    beta: float,
    gamma: float,
    sigma: float,
    n_days: int,
):
    """Simulate SEIR model forward in time yielding tuples."""
    s, e, i, r = (float(v) for v in (s, e, i, r))
    n = s + e + i + r
    for d in range(n_days + 1):
        yield d, s, e, i, r
        s, e, i, r = seir(s, e, i, r, beta, gamma, sigma, n)


def simulate(
    cases,
    population,
    doubling_time,
    recovery_days,
    distancing_reduction,
    days,
    keep,
    sigma=0.2,
    cases_to_infections_scale=1.0,
):
    # corner cases where latest cumulative case count is 0 or 1
    if doubling_time == 0 or cases[-1] == 0:
        return constant_forecast(cases[-1], keep)
    I0 = cases[0] * cases_to_infections_scale
    E0 = 20.0 * I0
    S0 = population - I0 - E0
    R0 = 0.0

    intrinsic_growth_rate = 2.0 ** (1.0 / doubling_time) - 1.0

    # Contact rate, beta
    gamma = 1.0 / recovery_days
    beta = (intrinsic_growth_rate + gamma) / S0 * (1.0 - distancing_reduction)

    # simulate forward
    IN = cases[-1] * cases_to_infections_scale
    EN = 20.0 * IN
    SN = population - IN - EN
    RN = 0
    res = {
        d: (_I, _R)
        for d, _S, _E, _I, _R in gen_seir(SN, EN, IN, RN, beta, gamma, sigma, days)
    }

    # remove day 0, since prediction parameters start at day 1
    days_kept, infected, recovered = [], [], []
    for day in range(1, keep + 1, 1):
        days_kept.append(day)
        _I, _R = res[day]
        infected.append(_I)
        recovered.append(_R)

    infs = pd.DataFrame(
        {"Day": days_kept, "infected": infected, "recovered": recovered}
    )
    ix_max = np.argmax(infected)

    if ix_max == len(infected) - 1:
        peak_days = f"{ix_max}+"
    else:
        peak_days = str(ix_max)

    peak_cases = int(infected[ix_max])

    meta = pd.DataFrame(
        {
            "Doubling time": [np.around(doubling_time, decimals=3)],
            "R0": [np.around(beta / gamma * S0, 3)],
            "beta": [np.around(beta * S0, 3)],
            "gamma": [np.around(gamma, 3)],
            "Peak days": [peak_days],
            "Peak cases": [peak_cases],
            # "MAE": [np.mean(mae)],
            # "RMSE": [np.mean(rmse)],
        }
    )
    return meta, infs


def constant_forecast(last_case_count, days):
    """Simulates infections for doubling_time = 0 by
    projecting constant of latest case count"""
    infected, recovered = [last_case_count] * days, [0] * days
    infs = pd.DataFrame(
        {"Day": range(1, days + 1, 1), "infected": infected, "recovered": recovered}
    )

    meta = pd.DataFrame({"Doubling time": [0]})
    return meta, infs


def estimate_growth_const(cases, window=None):
    """Estimates doubling time."""
    growth_rate = np.exp(np.diff(np.log(cases))) - 1
    if window is not None:
        growth_rate = growth_rate[-window:]
    doubling_times = np.log(2) / growth_rate

    # impute mean for nan or inf
    doubling_time = mean_doubling(doubling_times)
    return doubling_time


def mean_doubling(doubling_times, cap=50):
    """Returns mean accounting for inf and nans.
    Imputes the mean for nans and returns the cap if 
    doubling times contain infinity the mean is nan
    """
    doubling_time = doubling_times[np.isfinite(doubling_times)].mean()
    if np.isnan(doubling_time) or any(np.isinf(doubling_times)):
        return cap
    return doubling_time


class SEIRCV(CV):
    def run_train(self, dset, train_params, model_out):
        """Infers doubling time for SEIR model. 
        API match that of cv.py for cross validation

        Args:
            dset (str): path for confirmed cases
            train_params (dict): training parameters
            model_out (str): path for saving training checkpoints

        Returns: list of (doubling_times (np.float64), regions (list of str))
        """
        # get cases
        cases_df = load.load_confirmed_by_region(dset)
        regions = cases_df.columns
        # estimate doubling times per region
        doubling_times = []
        for region in regions:
            cases = cases_df[region].values
            doubling_time = estimate_growth_const(cases, train_params.window)
            doubling_times.append(doubling_time)

        model = [np.array(doubling_times), regions]
        # save estimate
        np.save(model_out, np.array(model))
        return model

    def run_simulate(self, dset, train_params, model, sim_params):
        """Forecasts region-level infections using
        API of cv.py for cross validation

        Args:
            dset (str): path for confirmed cases
            train_params (dict): training parameters
            model (list): [doubling times, regions]

        Returns: (pd.DataFrame) of forecasts per region
        """
        # regions are columns; dates are indices
        cases_df = load.load_confirmed_by_region(dset)
        populations_df = load.load_populations_by_region(
            train_params.fpop, regions=cases_df.columns
        )

        recovery_days, distancing_reduction, days, keep = initialize(train_params)
        doubling_times, regions = model

        region_to_prediction = dict()

        for doubling_time, region in zip(doubling_times, regions):
            # get cases and population for region
            population = populations_df[populations_df["region"] == region][
                "population"
            ].values[0]
            cases = cases_df[region].tolist()
            _, infs = simulate(
                cases,
                population,
                doubling_time,
                recovery_days,
                distancing_reduction,
                days,
                keep,
                sigma=train_params.sigma,
                cases_to_infections_scale=train_params.cases_to_infections_scale,
            )
            # prediction  = infected + recovered to match cases count
            infected = infs["infected"].values
            recovered = infs["recovered"].values
            prediction = (infected + recovered) / train_params.cases_to_infections_scale
            region_to_prediction[region] = prediction

        df = pd.DataFrame(region_to_prediction)
        # set dates
        df["date"] = _get_prediction_dates(cases_df, keep)
        df["date"] = pd.to_datetime(df["date"])
        df = df.set_index("date")

        # Return new cases per day (instead of cumulative cases)
        return pd.concat([cases_df, df]).sort_index().diff().loc[df.index]


CV_CLS = SEIRCV


def initialize(train_params):
    """Unpacks arguments needed from config"""
    recovery_days = train_params.recovery_days
    distancing_reduction = train_params.distancing_reduction
    days, keep = train_params.days, train_params.keep

    return recovery_days, distancing_reduction, days, keep


def _add_doubling_time_to_col_names(df, doubling_time):
    """Adds doubling time to infected and recovered column names"""
    df = df.rename(
        columns={
            "infected": f"infected (dt {doubling_time:.2f})",
            "recovered": f"recovered (dt {doubling_time:.2f})",
        }
    )
    return df


def _get_prediction_dates(cases_df: pd.DataFrame, days: int) -> pd.DatetimeIndex:
    """Returns dates for prediction.

    Args:
        cases_df: rows have dates and columns regions
        days: number of days forecasted
    Returns: datetime objects for prediction dates
    """
    last_confirmed_cases_date = cases_df.index.max()
    last_confirmed_cases_date = pd.to_datetime(last_confirmed_cases_date)
    prediction_end_date = last_confirmed_cases_date + timedelta(days)
    dates = pd.date_range(
        start=last_confirmed_cases_date, end=prediction_end_date, closed="right"
    )
    return dates


def parse_args(args: List):
    """Parses arguments"""
    parser = argparse.ArgumentParser(description="Forecasting with SEIR model")
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
    parser.add_argument("-sigma", type=float, default=0.2, help="Rate of progression")
    parser.add_argument(
        "-cases-to-infections-scale",
        type=float,
        default=1.0,
        help="ratio of confirmed cases to total infections",
    )
    parser.add_argument(
        "-fsuffix", type=str, help="prefix to store forecast and metadata"
    )
    parser.add_argument("-dout", type=str, default=".", help="Output directory")
    opt = parser.parse_args(args)
    return opt


def main(args):
    opt = parse_args(args)

    cases_df = load.load_confirmed_by_region(opt.fdat, None)
    regions = cases_df.columns
    # load only population data for regions with cases
    population = load.load_population(opt.fpop, regions=regions)
    cases = cases_df.sum(axis=1).tolist()
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
        sigma=opt.sigma,
        cases_to_infections_scale=opt.cases_to_infections_scale,
    )
    meta, df = f_sim(doubling_time)
    df = _add_doubling_time_to_col_names(df, doubling_time)

    for dt in opt.doubling_times:
        _meta, _df = f_sim(dt)
        meta = meta.append(_meta, ignore_index=True)

        doubling_time = float(_meta["Doubling time"].values)
        _df = _add_doubling_time_to_col_names(_df, doubling_time)

        df = pd.merge(df, _df, on="Day")

    # set prediction dates
    dates = _get_prediction_dates(cases_df, df.shape[0])
    df["dates"] = dates
    df = df.drop(columns="Day")
    df = df.set_index("dates")
    print()
    print(meta)
    print()
    print(df)

    if opt.fsuffix is not None:
        meta.to_csv(f"{opt.dout}/SEIR-metadata-{opt.fsuffix}.csv")
        df.to_csv(f"{opt.dout}/SEIR-forecast-{opt.fsuffix}.csv")


if __name__ == "__main__":
    main(sys.argv[1:])
