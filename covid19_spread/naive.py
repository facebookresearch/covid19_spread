#!/usr/bin/env python3
# Copyright (c) 2021-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import pandas as pd
from .cross_val import CV
from . import load
from datetime import timedelta


def simulate(latest_count, latest_delta, latest_date, days):
    """Forecasts 7 days ahead using naive model for a single region:
    day_n+1 prediction = day_n + day_n * (day_n - day_n-1 confirmed)

    Args:
        latest_delta (int): day_n - day_n-1 confirmed
        latest_count (int): day_n confirmed
        latest_date (datetime): last date with confirmed cases
        days (int): number of days to forecast

    Returns: dataframe of predictions
    """
    forecast = {
        -1: latest_count,
        0: latest_count + latest_delta,
    }
    for day in range(1, days):
        delta = forecast[day - 1] - forecast[day - 2]
        forecast[day] = forecast[day - 1] + delta

    # remove latest confirmed from prediction
    forecast.pop(-1)
    return forecast_to_dataframe(forecast, latest_date, days)


def forecast_to_dataframe(forecast, latest_date, days):
    """Converts dictionary of forecasts into dataframe with dates.
    forcast (dict): {0: predicted case count, 1: ...}
    """
    prediction_end_date = latest_date + timedelta(days)
    dates = pd.date_range(start=latest_date, end=prediction_end_date, closed="right")
    forecast_list = [forecast[day] for day in range(days)]
    df = pd.DataFrame.from_dict(zip(dates, forecast_list))
    df.columns = ["date", "total cases"]
    df = df.set_index("date")
    return df


def train(region_cases_df):
    """Returns latest count, delta, date needed for forecasting"""
    latest_count = region_cases_df[-1]
    latest_delta = region_cases_df[-1] - region_cases_df[-2]
    latest_date = pd.to_datetime(region_cases_df.index.max())
    return latest_count, latest_delta, latest_date


def naive(data_path="data/usa/data.csv", days=7):
    """Performs region level naive forecasts"""
    cases_df = load.load_confirmed_by_region(data_path)
    regions = cases_df.columns
    forecasts = []
    for region in regions:
        latest_count, latest_delta, latest_date = train(cases_df[region])
        forecast_df = simulate(latest_count, latest_delta, latest_date, days)
        forecast_df = forecast_df.rename(columns={"total cases": region})
        forecasts.append(forecast_df)

    df = pd.concat(forecasts, axis=1)
    return df


class NaiveCV(CV):
    def run_train(self, dset, train_params, model_out):
        """Returns delta between last two days and last confirmed total.

        Args:
            dset (str): path for confirmed cases
            train_params (dict): training parameters
            model_out (str): path for saving training checkpoints

        Returns: list of (doubling_times (np.float64), regions (list of str))
        """
        pass

    def run_simulate(self, dset, train_params, model, sim_params):
        """Returns new cases count predictions"""
        days = train_params.test_on
        forecast_df = naive(data_path=dset, days=days)
        cases_df = load.load_confirmed_by_region(dset)
        new_cases_forecast_df = (
            pd.concat([cases_df, forecast_df])
            .sort_index()
            .diff()
            .loc[forecast_df.index]
        )
        return new_cases_forecast_df


CV_CLS = NaiveCV


if __name__ == "__main__":
    print(naive())
