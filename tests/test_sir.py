"""
To run tests:
$ python -m pytest tests/test_sir_cv.py
"""

import os

import numpy as np
import pytest

import load
import sir
import yaml
from argparse import Namespace


script_dir = os.path.dirname(os.path.realpath(__file__))
CONFIGS = {
    "us": os.path.join(script_dir, "../cv/us.yml"),
    "ny": os.path.join(script_dir, "../cv/ny.yml"),
}


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

    @pytest.mark.parametrize("cfg_pth", [CONFIGS["us"], CONFIGS["ny"]])
    def test_run_train(self, checkpoint_path, cfg_pth):
        """Verifies doubling times are > 0 and region lengths match"""
        cfg = yaml.safe_load(open(cfg_pth))
        train_params = Namespace(**cfg["sir"]["train"])
        train_params.fdat = cfg["sir"]["data"]
        sir_cv = sir.SIRCV()
        model = sir_cv.run_train(train_params.fdat, train_params, checkpoint_path)
        assert isinstance(model, list)
        assert len(model) == 2

        doubling_times, regions = model
        assert (doubling_times >= 0).all()
        assert doubling_times.shape == (len(regions),)

    @pytest.mark.parametrize("cfg_pth", [CONFIGS["us"], CONFIGS["ny"]])
    def test_run_simulate(self, checkpoint_path, cfg_pth):
        """Verifies predictions match expected length"""
        cfg = yaml.safe_load(open(cfg_pth))
        train_params = Namespace(**cfg["sir"]["train"])
        train_params.fdat = cfg["sir"]["data"]
        sir_cv = sir.SIRCV()
        model = sir_cv.run_train(train_params.fdat, train_params, checkpoint_path)
        # model is doubling_times
        predictions_df = sir_cv.run_simulate(train_params.fdat, train_params, model, {})
        assert predictions_df.shape[0] == train_params.keep
