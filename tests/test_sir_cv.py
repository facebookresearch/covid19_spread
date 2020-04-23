"""
To run tests:
$ python -m pytest tests/test_sir_cv.py
"""

import sir
import os
import pytest
import numpy as np


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

    def test_load_region_populations(self, checkpoint_path):
        """Verifies populations match regions in length"""
        path = TrainParams.fpop
        populations, regions = sir.load_population_by_region(path)
        assert len(regions) == len(populations)

    def test_cases(self, checkpoint_path):
        """Confirms cases loaded are per region"""
        populations, regions = sir.load_population_by_region(TrainParams.fpop)
        region_cases, _, _ = sir.load_confirmed_by_region(TrainParams.fdat)
        assert region_cases.shape[0] == len(regions)
        # confirm length of cases per region is correct
        cases = sir.load_confirmed(TrainParams.fdat, regions)
        assert region_cases[0].shape == cases.shape

    def test_run_train(self, checkpoint_path):
        """Verifies doubling times are floats > 0"""
        populations, regions = sir.load_population_by_region(TrainParams.fpop)
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
