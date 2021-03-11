#!/usr/bin/env python3
# Copyright (c) 2021-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from covid19_spread.bar import BARCV
import yaml
from argparse import Namespace
import torch as th


class TestBatchedInference:
    def test_batched_inference(self):
        with th.no_grad():
            th.set_default_tensor_type(th.DoubleTensor)
            th.set_deterministic(True)
            th.manual_seed(0)
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
            cases = cases.type(th.get_default_dtype())
            tmax = cases.size(-1)

            # torch.bmm can give small precision differences on the CPU when comparing
            # batched vs. non-batched inputs.  If we do too many simulation iterations,
            # this error can compound to highly noticiable values.  Limit the number of
            # iterations to a small value.  Interestingly, on the GPU it isn't a problem...
            sim = mod.func.simulate(tmax, cases, 5, deterministic=True)
            sim_batched = mod.func.simulate(
                tmax, cases.repeat(2, 1, 1).contiguous(), 5, deterministic=True
            )
            assert (sim - sim_batched[0]).abs().max().item() < 1e-7
