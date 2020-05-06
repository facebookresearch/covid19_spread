import datetime
import json
import numpy as np
import os
import random
import submitit
import torch as th
from timelord import snapshot
from subprocess import check_call
import sys
from exps.compute_rmse import rmse
import train
import pandas
import itertools
import shutil
from collections import defaultdict
from itertools import product
from evaluation import simulate_mhp, goodness_of_fit
from tlc import Episode
from scipy.stats import kstest
import argparse
import subprocess
import h5py
from common import drop_k_days
from functools import partial
from exps.run_best import main as run_best
from exps.run_experiment import run_experiment


NGPUS = 1

# launch experiments
def launch(grid, exp_name, crossval, dicts, now, opt):
    exp_name = exp_name + ('_crossval' if crossval else '')
    base_dir = f'/checkpoint/{os.environ["USER"]}/exp/{exp_name}/{now}'
    folder = f'{base_dir}/%j'

    os.makedirs(base_dir)
    for i, dset in enumerate(grid['dset']):
        shutil.copy(dset, os.path.join(base_dir, os.path.basename(dset)))

    if crossval:
        for i, dset in enumerate(grid['dset']):
            outfile = os.path.join(base_dir, os.path.basename(dset) + f'.minus_{opt.days}_days')
            drop_k_days(dset, outfile, opt.days)

    executor = submitit.AutoExecutor(folder=folder)
    executor.update_parameters(
        name=exp_name,
        gpus_per_node=NGPUS,
        cpus_per_task=3,
        mem_gb=20,
        slurm_array_parallelism=200,
        timeout_min=12 * 60,
    )
    with snapshot.SnapshotManager(snapshot_dir=base_dir + "/snapshot", with_submodules=True):
        jobs = executor.map_array(partial(run_experiment, grid, opt.days, crossval, folder), dicts)
    print(folder[:-3])
    return jobs, base_dir


if __name__ == '__main__':
    random.seed(0)
    # output = subprocess.check_output("sinfo -lR | grep drng | awk '/Xid/ {print $5}'", shell=True)
    # exclude = ",".join(output.decode().splitlines())

    parser = argparse.ArgumentParser()
    parser.add_argument('-days', default=7, type=int)
    parser.add_argument('-deaths', action='store_true')
    opt = parser.parse_args()

    dset = 'deaths' if opt.deaths else 'adjacent_states'

    # construct experiment parameters
    grid = {
        'dim': [20, 30, 50],
        'lr': [0.0005, 0.001, 0.01, 0.02],
        'momentum': [0.99],
        'scale': [0.8, 1.0, 1.2],
        'optim': ['adam'],
        'weight-decay': [0, 0.1, 0.5],
        'lr-scheduler': ['cosine', 'constant'],
        'const-beta': [-1, 10, 15],
        'epochs': [100],
        'timescale': [1.0, 0.75, 0.5, 0.25],
    }
    if opt.deaths:
        grid['max-events']: [100000]
        grid['dset'] = [os.path.realpath(os.path.join(os.path.dirname(__file__), f'../data/usa/timeseries_smooth_1_days_mode_deaths.h5'))]
    else:
        grid['dset'] = [
            os.path.realpath(os.path.join(os.path.dirname(__file__), f'../data/usa/timeseries_smooth_1_days_mode_adjacent_states.h5')),
            os.path.realpath(os.path.join(os.path.dirname(__file__), f'../data/usa/timeseries_smooth_2_days_mode_adjacent_states.h5')),
            os.path.realpath(os.path.join(os.path.dirname(__file__), f'../data/usa/timeseries_smooth_3_days_mode_adjacent_states.h5')),
        ]
        
    keys = grid.keys()
    vals = list(product(*grid.values()))
    dicts = [{k: v for k, v in zip(keys, vs)} for vs in vals]

    # Postprocess the grid. 
    df = pandas.DataFrame(dicts)
    #  No need to sweep over lr/lr-scheduler if using lbfgs
    df.loc[df['optim'] == 'lbfgs', 'lr'] = 1
    df.loc[df['optim'] == 'lbfgs', 'lr-scheduler'] = 'constant'
    # Only use big LR if we do some kind of LR scheduling
    df.loc[(df['lr'] > 0.01) & (df['lr-scheduler'] == 'constant'), 'lr'] = 0.01

    df = df.drop_duplicates()
    dicts = list(df.T.to_dict().values())
    random.shuffle(dicts)
    dicts = dicts[:200]

    exp_name = 'covid19_usa' + ('_deaths' if opt.deaths else '')

    now = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    jobs, base_dir = launch(grid, exp_name, True, dicts, now, opt)
    print(f'Launched {len(jobs)} jobs in {base_dir}')

    run_best([base_dir])

