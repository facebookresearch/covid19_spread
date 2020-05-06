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
    fdat = "data/nystate/timeseries.h5"
    fpop = "data/population-data/US-states/new-york-population.csv"
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
        """Verifies doubling times are > 0 and region lengths match"""
        model = sir.run_train(TrainParams.fdat, TrainParams, checkpoint_path)
        assert isinstance(model, list)
        assert len(model) == 2

        doubling_times, regions = model
        assert (doubling_times > 0).all()
        assert doubling_times.shape == (len(regions),)

    def test_run_simulate(self, checkpoint_path):
        """Verifies predictions match expected length"""
        model = sir.run_train(TrainParams.fdat, TrainParams, checkpoint_path)
        # model is doubling_times
        predictions_df = sir.run_simulate(TrainParams.fdat, TrainParams, model)
        assert predictions_df.shape[0] == TrainParams.keep
