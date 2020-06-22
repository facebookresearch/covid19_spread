#!/usr/bin/env python3

"""
To run for US:
python sir.py -fdat data/usa/data_cases.csv -fpop data/usa/population.csv -days 21 -keep 21
"""

import argparse
import numpy as np
import pandas as pd
import sys
import load

from typing import List
from datetime import timedelta
import cv


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
    cases, population, doubling_times, recovery_days, distancing_reduction, days, keep,
):
    # begin simulation on day of first case
    start_index = cases.size - doubling_times.size
    R0 = 0.0
    I0 = cases[start_index]
    S0 = population - I0 - R0

    intrinsic_growth_rates = 2.0 ** (1.0 / doubling_times) - 1.0
    gamma = 1.0 / recovery_days
    betas = (intrinsic_growth_rates + gamma) / S0 * (1.0 - distancing_reduction)

    # simulate forward
    # I = confirmed cases - recovered
    RN = simulate_recovered(S0, I0, R0, cases, population, gamma, betas)
    IN = cases[-1] - RN
    SN = population - IN - RN

    res = {d: (_I, _R) for d, _S, _I, _R in gen_sir(SN, IN, RN, betas[-1], gamma, days)}

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
            "Doubling time": [np.around(doubling_times[-1], decimals=3)],
            "R0": [np.around(betas / gamma * S0, 3)],
            "beta": [np.around(betas * S0, 3)],
            "gamma": [np.around(gamma, 3)],
            "Peak days": [peak_days],
            "Peak cases": [peak_cases],
            # "MAE": [np.mean(mae)],
            # "RMSE": [np.mean(rmse)],
        }
    )
    return meta, infs


def simulate_recovered(S0, I0, R0, cases, population, gamma, betas):
    """Simulates recovered at the end of given time series"""
    S, I, R = S0, I0, R0
    for i, case_count in enumerate(betas):
        S, I, R = sir(S, I, R, betas[i], gamma, S + I + R)
        I = case_count - R
        S = population - I - R
    return R


def estimate_doubling_times(cases, min_window=3, cap=50.0, rolling_window=2):
    """Estimates doubling times, begining on the day of first case.

    Args:
        cases (np.array): contains array of cumulative cases.
        window (int): window to consider for estimating doubling time
        cap (int): maximum doubling time to use if no cases are present or 
            first occurrence occurs on last day.
    
    Returns: np.array of doubling times.
    """
    non_zero_indices = np.nonzero(cases)[0]
    # if there are no cases or first case occurs before window, return cap
    if non_zero_indices.size < min_window:
        return np.array([cap])

    first_non_zero_i = non_zero_indices[0]
    cases_from_first = cases[first_non_zero_i:]
    growth_rates = np.exp(np.diff(np.log(cases_from_first))) - 1

    # add epsilon to growth rate 0.0
    growth_rates[growth_rates == 0.0] = 1 / cap
    doubling_times = np.log(2) / growth_rates

    # impute max or cap when doubling time is infinite
    # occurs when the number of cases is repeated
    doubling_times = impute_max_or_cap(doubling_times, cap)
    # compute 2-day rolling mean for doubling times
    doubling_times_rolling = compute_rolling_mean(doubling_times, rolling_window)

    return doubling_times_rolling


def compute_rolling_mean(a, window):
    """Computes rolling mean over window. 
    First elements in window are unchanged.
    Example: with window 2, [1, 3, 3] -> [1, 3, 3.5]

    Returns: np.array of same length as a.
    """
    rolling_mean = pd.Series(a).rolling(window=window).mean().dropna().values
    result = np.concatenate([a[: window - 1], rolling_mean])
    return result


def impute_max_or_cap(a, cap):
    """Imputes max or cap of the array for infinite values."""
    if np.isfinite(a).any():
        imputed_value = a[np.isfinite(a)].max()
    else:
        imputed_value = cap

    a[np.isinf(a)] = imputed_value
    return a


class SIRCV(cv.CV):
    def run_train(self, dset, train_params, model_out):
        """Infers doubling time for sir model. 
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
        doubling_times_per_region = []

        for region in regions:
            cases = cases_df[region].values
            doubling_times = estimate_doubling_times(
                cases,
                min_window=train_params.min_window,
                cap=train_params.cap,
                rolling_window=train_params.rolling_window,
            )
            doubling_times_per_region.append(doubling_times)

        model = [np.array(doubling_times_per_region), regions]
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
            cases = cases_df[region].values
            _, infs = simulate(
                cases,
                population,
                doubling_time,
                recovery_days,
                distancing_reduction,
                days,
                keep,
            )
            # prediction  = infected + recovered to match cases count
            infected = infs["infected"].values
            recovered = infs["recovered"].values
            prediction = infected + recovered
            region_to_prediction[region] = prediction

        df = pd.DataFrame(region_to_prediction)
        # set dates
        df["date"] = _get_prediction_dates(cases_df, keep)
        df["date"] = pd.to_datetime(df["date"])
        df = df.set_index("date")

        # Return new cases per day (instead of cumulative cases)
        return pd.concat([cases_df, df]).sort_index().diff().loc[df.index]


CV_CLS = SIRCV


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
    parser = argparse.ArgumentParser(description="Forecasting with SIR model")
    parser.add_argument("-fdat", help="Path to confirmed cases", required=True)
    parser.add_argument("-fpop", help="Path to population data", required=True)
    parser.add_argument(
        "-days", type=int, help="Number of days to forecast", required=True
    )
    parser.add_argument(
        "-keep", type=int, help="Number of days to keep in CSV", required=True
    )
    parser.add_argument("-recovery-days", type=int, default=14, help="Recovery days")
    parser.add_argument(
        "-distancing-reduction", type=float, default=0.9, help="Recovery days"
    )
    parser.add_argument(
        "-fsuffix", type=str, help="prefix to store forecast and metadata"
    )
    parser.add_argument("-dout", type=str, default=".", help="Output directory")
    opt = parser.parse_args(args)
    return opt


class TrainParams:
    distancing_reduction = 0.9
    recovery_days = 10


def main(args):
    opt = parse_args(args)
    TrainParams.window = opt.distancing_reduction
    TrainParams.recovery_days = opt.recovery_days

    sir_model = SIRCV()

    doubling_times_per_region, regions = sir_model.run_train(
        opt.fdat, TrainParams, "/tmp/sir_model.npy"
    )

    for region, doubling_times in zip(regions, doubling_times_per_region):
        print(region)
        print(doubling_times)

    forecast = sir_model.run_simulate


if __name__ == "__main__":
    main(sys.argv[1:])
