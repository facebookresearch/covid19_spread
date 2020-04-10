#!/usr/bin/env python3

# import warnings filter
from warnings import simplefilter

# ignore all future warnings
simplefilter(action="ignore", category=FutureWarning)

import numpy as np
import pandas as pd
import random
import sys
import torch as th
from common import load_model
from evaluation import simulate_mhp
from timelord.utils import prepare_dset
import tlc
import train


class RMSETrainer(train.CovidTrainer):
    def __init__(self, opt, n_days_back, episodes, user_control=None):
        self.device = self.get_device()
        self.last_epoch = -1
        self.all_log_msgs = []
        self.opt = opt
        self.user_control = user_control

        self.episodes, dnames, cnames, T_missing, counts, self.n_jumps = prepare_dset(
            opt.dset, -1, opt.sparse, opt.quiet, True, self.opt.timescale
        )

        assert opt.timescale == 1
        # only keep events upto maxtime
        ts, ns = self.episodes[0].timestamps, self.episodes[0].entities
        maxtime = ts.max() - n_days_back
        ix = self.episodes[0].timestamps < maxtime
        # FIXME: using sparse episodes is not correct here
        self.episodes[0] = tlc.Episode(ts[ix], ns[ix], True, len(dnames))
        self.episode_orig = tlc.Episode(ts, ns, False, len(dnames))

        self.counts = counts.to(self.device)
        self.T_missing = th.tensor(T_missing, dtype=th.double, device=self.device)
        self.entities = dnames
        self.cascade_names = cnames
        self.setup_model()
        self.setup_dataloader()
        self.setup_optimizer()
        self.setup_lr_scheduler()
        self.best_stats = {"ll": -float("inf"), "epoch": -1}

        print(f"Events: {len(self.episodes[0].timestamps)}")
        print(f"T min: {self.episodes[0].timestamps.min()}")
        print(f"T max: {self.episodes[0].timestamps.max()}")


def rmse(opt, user_control, gt, d):
    # train model
    trainer = RMSETrainer(opt, d, user_control)
    M = len(trainer.entities)
    trainer.train()

    mus, beta, S, U, V, A, scale, timescale = load_model(opt.checkpoint, M)

    # predictions
    episode = trainer.episodes[0]
    t_obs = episode.timestamps[-1].item()
    df = simulate_mhp(
        t_obs,
        d,
        episode,
        mus,
        beta,
        A,
        timescale,
        trainer.entities,
        opt.step_size,
        opt.trials,
    )[["county", d]]

    # compute rmse
    df = pd.merge(df, gt, on="county")
    # _rmse = np.sqrt(((df["ground_truth"] - df[f"MHP d{d}"]) ** 2).mean())
    _rmse = np.sqrt(((df["ground_truth"] - df[d]) ** 2).mean())
    return _rmse

def mk_parser():
    parser = train.mk_parser()
    parser.add_argument(
        "-step-size", type=float, default=0.01, help="Step size for simulation"
    )
    parser.add_argument("-trials", type=int, default=50, help="Number of trials")
    parser.add_argument("-days", type=int, default=3, help="Number of days to forecast")
    return parser

def main(args, user_control=None):
    parser = mk_parser()
    opt = parser.parse_args(args)
    if opt.repro is not None:
        opt = th.load(opt.repro)["opt"]

    np.random.seed(opt.seed)
    th.manual_seed(opt.seed)
    random.seed(opt.seed)
    _rmse = {}

    trainer = RMSETrainer(opt, 0, user_control)
    gt = {"county": [], "ground_truth": []}
    for x, name in enumerate(trainer.entities):
        gt["county"].append(name)
        gt["ground_truth"].append(len(trainer.episode_orig.occurrences_of_dim(x)) - 1)
    gt = pd.DataFrame(gt)

    for d in range(1, opt.days + 1):
        _rmse[d] = rmse(opt, user_control, gt, d)

    print("RMSE_PER_DAY:", _rmse)
    print("RMSE_AVG:", np.mean(list(_rmse.values())))


if __name__ == "__main__":
    main(sys.argv[1:])
