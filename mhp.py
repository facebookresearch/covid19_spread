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


def run_train(dset, model_params, model_out):
    model_params.dset = dset

    model_params = copy.deepcopy(model_params)
    # Fill out any default parameters...
    parser = mk_parser()
    defaults = parser.parse_args([])
    for k, v in vars(defaults).items():
        if not hasattr(model_params, k):
            setattr(model_params, k, v)

    seed = getattr(model_params, 'seed', 0)
    np.random.seed(seed)
    th.manual_seed(seed)
    random.seed(seed)
    model_params.checkpoint = model_out
    trainer = train.CovidTrainer(model_params)
    trainer.train()
    return model_out

def run_simulate(dset, model_params, checkpoint, sim_params):
    mdl, mdl_opt = train.CovidModel.from_checkpoint(checkpoint, map_location='cpu')
    nodes, ns, ts, _ = load_data(dset)
    episode = Episode(th.from_numpy(ts).double(), th.from_numpy(ns).long(), True, mdl.nnodes)

    simulator = mdl.mk_simulator()
    t_obs = ts[-1]
    sim = simulate_tl_mhp(t_obs, sim_params['days'], episode, mdl_opt.timescale, simulator, 
        nodes, sim_params['trials'], max_events=sim_params['max_events'])
    sim[-1] = sim[0]

    with h5py.File(mdl_opt.dset,'r') as hf:
        assert 'basedate' in hf.attrs, '`basedate` missing from HDF5 attrs!'
        basedate = pandas.Timestamp(hf.attrs['basedate'])
        assert 'ground_truth' in hf.keys(), "`ground_truth` missing from HDF5 file!"
        ground_truth = pandas.DataFrame(hf['ground_truth'][:])
        ground_truth.columns = [str(d.date()) for d in pandas.date_range(end=basedate, periods=ground_truth.shape[1])]
        ground_truth['county'] = hf['nodes']

    mapper = {d: str((basedate + pandas.Timedelta(d, 'D')).date()) for d in sim.columns if isinstance(d, int) and d >= 0}
    merged = sim.merge(ground_truth, on='county')

    forecasts = {'location': sim['county'].values}
    for k, v in mapper.items():
        # number of cases is Day_0 + number of new cases.  If we did some kind of smoothing
        # on the data, we want to make sure that we are basing our forecasts on the actual known counts
        new_cases = merged[mapper[0]] + (merged[k] - merged[0])
        forecasts[v] = new_cases.values
    forecasts = pandas.DataFrame.from_dict(forecasts).set_index('location').transpose()
    forecasts.index.name = 'date'
    # Drop the first day, since it corresponds to the ground truth counts for the last day of
    # data we have.
    return forecasts[forecasts.index > forecasts.index.min()]
