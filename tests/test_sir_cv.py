"""
To run tests:
$ python -m pytest tests/test_sir_cv.py
"""

import sir
import os
import pytest
import numpy as np


TRAIN_PARAMS = {
    "fdat": "timeseries_filtered.h5",
    "fpop": "data/population-data/US-states/new-jersey-population.csv",
    "window": 14,
    "recovery_days": 14,
    "distancing_reduction": 0.2,
    "days": 7,
    "keep": 5,
}


class TestSIRCrossValidation:

    @pytest.fixture(scope='module')
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
        doubling_times = sir.run_train(TRAIN_PARAMS, checkpoint_path)    
        assert doubling_times.dtype == "float64"
        assert (doubling_times > 0).all()

    def test_run_simulate(self, checkpoint_path):
        """Verifies doubling time is a float > 0"""
        doubling_times = sir.run_train(TRAIN_PARAMS, checkpoint_path)    
        # model is doubling_times
        predictions_df = sir.run_simulate(TRAIN_PARAMS, doubling_times)
        assert predictions_df.shape[0] == TRAIN_PARAMS["keep"]

    def test_load_region_populations(self):
        """Verifies populations match regions in length"""
        path = TRAIN_PARAMS["fpop"]
        populations, regions = sir.load_region_populations(path)
        assert len(regions) == len(populations)

