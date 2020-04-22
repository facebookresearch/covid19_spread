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
        """Verifies doubling time is a float > 0"""
        doubling_time = sir.run_train(TRAIN_PARAMS, checkpoint_path)    
        assert isinstance(doubling_time, np.float64)
        assert doubling_time > 0


