"""
To run tests:
$ python -m pytest tests/test_sir_cv.py
"""

import os

import numpy as np
import pytest

import load
import sir


class TrainParams:
    fdat = "timeseries_filtered.h5"
    fpop = "data/population-data/US-states/new-jersey-population.csv"
    window = 14
    recovery_days = 14
    distancing_reduction = 0.2
    days = 7
    keep = 7


class TestSIRCrossValidation:

    @pytest.fixture(scope="module")
    def checkpoint_path(self):
        """Fixture to cleanup checkpoint file"""
        path = "/tmp/sir.npy"
        yield path
        try:
            os.remove(path)
        except OSError:
            pass

    def test_run_train(self, checkpoint_path):
        """Verifies doubling times are floats > 0"""
        _, regions = load.load_populations_by_region(TrainParams.fpop)
        doubling_times = sir.run_train(TrainParams, checkpoint_path)
        assert doubling_times.dtype == "float64"
        assert (doubling_times > 0).all()
        assert doubling_times.shape == (len(regions),)

    def test_run_simulate(self, checkpoint_path):
        """Verifies doubling time is a float > 0"""
        doubling_times = sir.run_train(TrainParams, checkpoint_path)
        # model is doubling_times
        predictions_df = sir.run_simulate(TrainParams, doubling_times)
        assert predictions_df.shape[0] == TrainParams.keep
