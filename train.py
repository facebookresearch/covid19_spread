#!/usr/bin/env python3

# import warnings filter
from warnings import simplefilter

# ignore all future warnings
simplefilter(action="ignore", category=FutureWarning)

import numpy as np
import sys
import torch as th
from timelord.trainer import mk_parser, Trainer
import random

from model import CovidModel


class CovidTrainer(Trainer):
    def __init__(self, opt, user_control=None):
        super().__init__(opt, user_control)
        print(f"Events: {len(self.episodes[0].timestamps)}")
        print(f"T min: {self.episodes[0].timestamps.min()}")
        print(f"T max: {self.episodes[0].timestamps.max()}")

    def setup_model(self):
        print(self.opt)
        self.model = CovidModel(
            len(self.entities),
            self.opt.dim,
            self.opt.scale,
            True,
            self.opt.baseint,
            self.opt.const_beta,
        )
        self.model.initialize_weights()
        self.model = self.model.to(self.device)


def parse_opt(args):
    parser = mk_parser()
    parser.add_argument(
        "-no-baseint", action="store_false", dest="baseint", default=True
    )
    parser.add_argument("-const-beta", type=float, default=-1)
    opt = parser.parse_args(args)
    return opt


def main(args, user_control=None):
    opt = parse_opt(args)
    if opt.repro is not None:
        opt = th.load(opt.repro)["opt"]

    np.random.seed(opt.seed)
    th.manual_seed(opt.seed)
    random.seed(opt.seed)
    trainer = CovidTrainer(opt, user_control)
    trainer.train()


if __name__ == "__main__":
    main(sys.argv[1:])
