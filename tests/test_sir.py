"""
To run tests:
$ python -m pytest tests/test_sir_cv.py
"""

import os

import numpy as np
import pytest

import load
import sir


basedir = os.getcwd()


class TrainParamsNY:
    fdat = "data/nystate/data-new.csv"
    fpop = "data/population-data/US-states/new-york-population.csv"
    window = 14
    recovery_days = 14
    distancing_reduction = 0.2
    days = 7
    keep = 7


class TrainParamsUS:
    fdat = "data/usa/data_cases.csv"
    fpop = "data/usa/population.csv"
    window = 30
    recovery_days = 10
    distancing_reduction = 0.8
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

    @pytest.mark.parametrize("train_params", [TrainParamsNY, TrainParamsUS])
    def test_run_train(self, checkpoint_path, train_params):
        """Verifies doubling times are > 0 and region lengths match"""
        sir_cv = sir.SIRCV()
        model = sir_cv.run_train(train_params.fdat, train_params, checkpoint_path)
        assert isinstance(model, list)
        assert len(model) == 2

        doubling_times, regions = model
        assert (doubling_times >= 0).all()
        assert doubling_times.shape == (len(regions),)

    @pytest.mark.parametrize("train_params", [TrainParamsNY, TrainParamsUS])
    def test_run_simulate(self, checkpoint_path, train_params):
        """Verifies predictions match expected length"""
        sir_cv = sir.SIRCV()
        model = sir_cv.run_train(train_params.fdat, train_params, checkpoint_path)
        # model is doubling_times
        predictions_df = sir_cv.run_simulate(train_params.fdat, train_params, model, {})
        assert predictions_df.shape[0] == train_params.keep

    def test_estimate_doubling_times(self):
        """Verifies doubling times are correctly estimated for corner cases"""
        cap = 50.0

        doubling_all_zeros = sir.estimate_doubling_times(np.array([0, 0, 0]), cap=cap)
        assert doubling_all_zeros == np.array(np.array([cap]))

        doubling_case_last_day = sir.estimate_doubling_times(
            np.array([0, 0, 1]), cap=cap
        )
        assert doubling_case_last_day == np.array(np.array([cap]))
