#!/usr/bin/env python3

import train
import numpy as np
import torch as th
import random
from load import load_data
from tlc import Episode
from evaluation import simulate_tl_mhp
import pandas
import copy
import h5py
from train import mk_parser
import cv
from metrics import load_ground_truth


class MHPCV(cv.CV):
    def run_train(self, dset, model_params, model_out):
        model_params.dset = dset

        model_params = copy.deepcopy(model_params)
        # Fill out any default parameters...
        parser = mk_parser()
        defaults = parser.parse_args([])
        for k, v in vars(defaults).items():
            if not hasattr(model_params, k):
                setattr(model_params, k, v)

        seed = getattr(model_params, "seed", 0)
        np.random.seed(seed)
        th.manual_seed(seed)
        random.seed(seed)
        model_params.checkpoint = model_out
        trainer = train.CovidTrainer(model_params)
        trainer.train()
        return model_out

    def run_simulate(self, dset, model_params, checkpoint, sim_params):
        mdl, mdl_opt = train.CovidModel.from_checkpoint(checkpoint, map_location="cpu")
        nodes, ns, ts, basedate = load_data(dset)
        episode = Episode(
            th.from_numpy(ts).double(), th.from_numpy(ns).long(), True, mdl.nnodes
        )
        basedate = pandas.Timestamp(basedate)
        simulator = mdl.mk_simulator()
        t_obs = ts[-1]

        sim = simulate_tl_mhp(
            t_obs,
            sim_params["days"],
            episode,
            mdl_opt.timescale,
            simulator,
            nodes,
            sim_params["trials"],
            max_events=sim_params["max_events"],
        )

        sim = sim.set_index("county").transpose().sort_index()
        deltas = sim.diff(axis=0).fillna(0)
        deltas.index = pandas.date_range(start=basedate, periods=len(deltas))
        return deltas[deltas.index > basedate]


CV_CLS = MHPCV
