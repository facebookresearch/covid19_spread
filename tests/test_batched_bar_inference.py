#!/usr/bin/env python3

import cv
import pytest
from bar import BARCV
import yaml
from argparse import Namespace
import torch as th


class TestBatchedInference:
    def test_batched_inference(self):
        with th.no_grad():
            mod = BARCV()
            cfg = yaml.safe_load(open("cv/ny.yml"))
            opt = Namespace(
                **{
                    k: v[0] if isinstance(v, list) else v
                    for k, v in cfg["bar"]["train"].items()
                }
            )
            opt.fdat = cfg["bar"]["data"]

            cases, regions, basedate, device = mod.initialize(opt)

            tmax = cases.size(-1)
            sim = mod.func.simulate(tmax, cases, opt.test_on, deterministic=True)
            sim_batched = mod.func.simulate(
                tmax, cases.repeat(2, 1, 1), opt.test_on, deterministic=True
            )

            assert (sim - sim_batched[0]).abs().max() == 0
