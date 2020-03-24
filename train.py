#!/usr/bin/env python3

import numpy as np
import sys
import torch as th
from timelord.trainer import parse_opt, Trainer
import random


def main(args, user_control=None):
    opt = parse_opt(args)
    if opt.repro is not None:
        opt = th.load(opt.repro)["opt"]

    np.random.seed(opt.seed)
    th.manual_seed(opt.seed)
    random.seed(opt.seed)
    trainer = Trainer(opt, user_control)
    trainer.train()


if __name__ == "__main__":
    main(sys.argv[1:])
