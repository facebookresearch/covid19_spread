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
import rmse
import train
import pandas
import itertools
from collections import defaultdict
from itertools import product
from evaluation import simulate_mhp, goodness_of_fit
from common import load_data, load_model
from tlc import Episode
from scipy.stats import kstest
import argparse
import subprocess
import h5py
from common import drop_k_days

output = subprocess.check_output("sinfo -lR | grep drng | awk '/Xid/ {print $5}'", shell=True)
exclude = ",".join(output.decode().splitlines())

parser = argparse.ArgumentParser()
parser.add_argument('-rmse', action='store_true')
parser.add_argument('-days', nargs='+', default=[3], type=int)
opt = parser.parse_args()


# construct experiment parameters
grid = {
    'dim': [20, 30, 50, 100],
    'lr': [0.001, 0.01, 0.1, 0.5],
    'momentum': [0.99],
    'scale': [0.7, 0.8, 1.0, 1.2],
    'optim': ['adam'],
    'weight-decay': [0, 0.1, 0.5, 1, 2],
    'lr-scheduler': ['cosine', 'constant'],
    # 'const-beta': [-1, 20, 40],
    'dset': [
        os.path.realpath(os.path.join(os.path.dirname(__file__), '../data/usa/timeseries_smooth_1_days.h5')),
        os.path.realpath(os.path.join(os.path.dirname(__file__), '../data/usa/timeseries_smooth_2_days.h5')),
        os.path.realpath(os.path.join(os.path.dirname(__file__), '../data/usa/timeseries_smooth_3_days.h5')),
    ]
}

exp_name = 'covid19_usa' + ('_rmse' if opt.rmse else '')
now = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
user = os.environ['USER']
base_dir = f'/checkpoint/{user}/exp/{exp_name}/{now}'
snapshot_dir = f'{base_dir}/snapshot'
tensorboard_dir = f'{base_dir}/tensorboard'
folder = f'{base_dir}/%j'
checkpoint = f'{folder}/model.bin'
data_dir = f'/checkpoint/{user}/data/diffusion'

executor = submitit.AutoExecutor(folder=folder)
executor.update_parameters(
    name=exp_name,
    gpus_per_node=1,
    cpus_per_task=10,
    mem_gb=20,
    slurm_array_parallelism=60,
    timeout_min=12 * 60,
    exclude=exclude,
)

if opt.rmse:
    os.makedirs(base_dir)
    for i, dset in enumerate(grid['dset']):
        for day in opt.days:
            outfile = os.path.join(base_dir, os.path.basename(dset) + f'.minus_{day}_days')
            drop_k_days(dset, outfile, day)

if opt.rmse:
    grid['day'] = opt.days

keys = grid.keys()
vals = list(product(*grid.values()))
dicts = [{k: v for k, v in zip(keys, vs)} for vs in vals]

# Postprocess the grid. 
df = pandas.DataFrame(dicts)
#  No need to sweep over lr if using lbfgs
df.loc[df['optim'] == 'lbfgs', 'lr'] = 1
# Only use big LR if we do some kind of LR scheduling
df.loc[(df['lr'] > 0.01) & (df['lr-scheduler'] == 'constant'), 'lr'] = 0.01

df = df.drop_duplicates()
dicts = list(df.T.to_dict().values())
random.shuffle(dicts)
dicts = dicts[:200]


def run_experiment(pdict, local=False, seed=42):
    if not local:
        job_env = submitit.JobEnvironment()
        job_checkpoint = checkpoint.replace('%j', submitit.JobEnvironment().job_id)
        job_dir = folder.replace('%j', submitit.JobEnvironment().job_id)
        with open(os.path.join(job_dir, 'params.json'), 'w') as fout:
            json.dump({'params': pdict, 'grid': grid}, fout)
    else:
        job_checkpoint = '/tmp/timelord_model.bin'

    if opt.rmse:
        day = pdict['day']
        job_dset = os.path.join(base_dir, os.path.basename(pdict['dset']) + f'.minus_{day}_days')
    else:
        job_dset = pdict.get('dset')

    train.main([str(x) for x in [
        '-dset', job_dset,
        '-dim', pdict.get('dim', 50),
        '-lr', pdict.get('lr'),
        '-epochs', pdict.get('epochs', 100),
        '-max-events', pdict.get('max-events', 500000),
        '-checkpoint', job_checkpoint,
        '-timescale', pdict.get('timescale', 1),
        '-scale', pdict.get('scale', 1),
        '-sparse',
        '-optim', pdict.get('optim', 'adam',),
        '-weight-decay', pdict.get('weight-decay', 0),
        '-const-beta', pdict.get('const-beta', -1),
        '-lr-scheduler', pdict.get('lr-scheduler', 'constant'),
    ]])


# launch experiments
with snapshot.SnapshotManager(snapshot_dir=base_dir + "/snapshot", with_submodules=True):
    jobs = executor.map_array(run_experiment, dicts)
print(f'Output: {folder[:-3]}')
