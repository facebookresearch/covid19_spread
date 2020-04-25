#!/usr/bin/env python3

import torch as th
import argparse
import os
import re
from functools import partial
from evaluation import simulate_tl_mhp, goodness_of_fit, ks_critical_value, resid
from model import CovidModel
from load import load_data
from tlc import Episode
import h5py
import numpy as np
import submitit
import subprocess
import pandas
import datetime
from scipy.stats import kstest
import json
from multiprocessing import cpu_count
from multiprocessing.dummy import Pool
from timelord.utils import to_tick_data
import threading
from tqdm import tqdm
import tlc


parser = argparse.ArgumentParser()
parser.add_argument('sweep_dir')
parser.add_argument('-ground-truth', '-g', required=True)
parser.add_argument('-max-events', type=int, default=2000000)
parser.add_argument('-remote', action='store_true')
parser.add_argument('-days', type=int, default=7)
opt = parser.parse_args()

def mk_cpu_model(model):
    cpu_model = tlc.SparseSoftplusEmbeddingModel(model.nnodes, model.dim, model.scale)
    cpu_model.params[0].copy_(model.mus_.weight[:-1].cpu())
    cpu_model.params[1].copy_(model.beta_.cpu())
    cpu_model.params[2].copy_(model.self_A.weight[:-1].cpu())
    cpu_model.params[3].copy_(model.U.weight[:-1].cpu())
    cpu_model.params[4].copy_(model.V.weight[:-1].cpu())
    return cpu_model

def ks_test(episode, step, model, nodes, nprocs=20):
    cpu_model = mk_cpu_model(model)
    next_dim = [0]
    result = []
    with tqdm(total=len(nodes)) as pbar:
        def _ks(tid):
            print(f'Thread {tid} starting...')
            while next_dim[0] < len(nodes):
                dim = next_dim[0]
                next_dim[0] += 1
                residuals = tlc.rescale_interarrival_times(dim, episode, cpu_model)
                ks, pval = kstest(residuals, "expon")
                result.append({'node': nodes[dim], 'ks': float(ks), 'pval': float(pval)})
                pbar.update(1)


        threads = [threading.Thread(target=_ks, args=(i,)) for i in range(cpu_count())]
        [t.start() for t in threads]
        [t.join() for t in threads]

    return pandas.DataFrame(result)


def rmse(pth, ground_truth, trials = 10):
    import torch
    print(pth)
    mdl, mdl_opt = CovidModel.from_checkpoint(pth, map_location='cpu')
    nodes, ns, ts, _ = load_data(mdl_opt.dset)
    episode = Episode(th.from_numpy(ts).double(), th.from_numpy(ns).long(), True, mdl.nnodes)

    simulator = mdl.mk_simulator()
    t_obs = ts[-1]
    sim = simulate_tl_mhp(t_obs, opt.days, episode, mdl_opt.timescale, simulator, nodes, trials, max_events=opt.max_events)
    sim[-1] = sim[0]

    with h5py.File(mdl_opt.dset,'r') as hf:
        basedate = pandas.Timestamp(hf.attrs['basedate'])
    mapper = {d: str((basedate + pandas.Timedelta(d, 'D')).date()) for d in sim.columns if isinstance(d, int) and d >= 0}
    merged = sim.merge(ground_truth, on='county')
    vals = {'pth': pth}

    ks_result = ks_test(episode, 0.001, mdl, nodes)
    ks_result.to_csv(os.path.join(os.path.dirname(pth), 'kstest.csv'), index=False)
    avg_ks = float(ks_result['ks'].mean())
    avg_pval = float(ks_result['pval'].mean())
    vals = {'pth': pth, 'ks': avg_ks, 'pval': avg_pval}
    
    forecasts = {'location': sim['county'].values}
    for k, v in mapper.items():
        # number of cases is Day_0 + number of new cases.  If we did some kind of smoothing
        # on the data, we want to make sure that we are basing our forecasts on the actual known counts
        new_cases = merged[mapper[0]] + (merged[k] - merged[0])
        forecasts[v] = new_cases.values
        if v in merged.columns:
            vals[f'day_{k}_mean'] = (new_cases - merged[v]).abs().mean()
            vals[f'day_{k}_max'] = (new_cases - merged[v]).abs().max()
            vals[f'day_{k}_median'] = (new_cases - merged[v]).abs().median()

    fout = os.path.join(os.path.dirname(pth), 'forecasts.csv')
    forecasts = pandas.DataFrame.from_dict(forecasts)
    forecasts.to_csv(fout, index=False)
    with open(os.path.join(os.path.dirname(pth), 'eval.json'), 'w') as fout:
        print(json.dumps(vals), file=fout)
    return vals



ground_truth = pandas.read_csv(opt.ground_truth)

output = subprocess.check_output("sinfo -lR | grep drng | awk '/Xid/ {print $5}'", shell=True)
exclude = ",".join(output.decode().splitlines())


user=os.environ['USER']
now = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
folder = f'/checkpoint/{user}/covid19/rmse/{now}'
executor = submitit.AutoExecutor(folder=folder)
executor.update_parameters(
    name='compute_rmse',
    gpus_per_node=0,
    cpus_per_task=10,
    mem_gb=30,
    slurm_array_parallelism=100,
    timeout_min=2 * 60,
    slurm_exclude=exclude,
)

chkpnts = []
for d in os.listdir(opt.sweep_dir):
    if re.match('\d+_\d+', d):
        chkpnt = os.path.join(opt.sweep_dir, d, 'model.bin.best')
        if os.path.exists(chkpnt):
            chkpnts.append(chkpnt)
    elif d.endswith('model.bin.best'):
        chkpnts.append(os.path.join(opt.sweep_dir, d))

mapper = executor.map_array if opt.remote else map
list(mapper(lambda x: rmse(x, ground_truth), chkpnts))
print(folder)