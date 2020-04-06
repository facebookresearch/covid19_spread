#!/usr/bin/env python3

import yaml
import json
import argparse
from datetime import date, datetime
from sir import main as sir
from rmse import main as rmse
from train import main as train
from forecast import main as forecast
import os
import contextlib
import itertools
import submitit
from timelord import snapshot


DFLT_PARAMS = [
    '-max-events', 5000, '-sparse', '-scale', '1', '-optim', 'lbfgs', '-quiet',
    '-fresh', '-epochs', 200, '-maxcor', 25
]

def forecast_train(dataset, dim, base_intensity, basedate, job_dir, days=7, trials=10, log=None):
    with contextlib.ExitStack() as stack:
        if log is not None:
            stderr = stack.enter_context(open(log + '.stderr', 'w'))
            stack.enter_context(contextlib.redirect_stderr(stderr))
            stdout = stack.enter_context(open(log + '.stdout', 'w'))
            stack.enter_context(contextlib.redirect_stdout(stdout))

        checkpoint = os.path.join(job_dir, 'model.bin')
        with_intensity = [] if base_intensity else ['-no-baseint']
        NON_DFLT = [
            '-checkpoint', checkpoint, 
            '-dset', dataset, '-dim', dim
        ] + with_intensity
        train_params = list(map(str, DFLT_PARAMS + NON_DFLT))

        rmse(train_params)
        train(train_params)

        forecast_params = [
            '-dset', dataset, 
            '-checkpoint', checkpoint,
            '-basedate', basedate,
            '-days', days,
            '-trials', trials,
            '-fout', os.path.join(job_dir, 'forecasts.csv'),
        ]
        forecast(list(map(str, forecast_params)))

def load_config(pth):
    if pth.endswith('.yml') or pth.endswith('yaml'):
        return yaml.load(open(pth), Loader=yaml.FullLoader)
    elif pth.endswiith('.json'):
        return json.load(open(pth))
    else:
        raise ValueError(f'Unrecognized grid file extension: {pth}')


def run_sir(data, population, region, base, **kwargs):
    doubling_times = kwargs.get('doubling_times', [2, 3, 4, 10])
    os.makedirs(f'{base}/sir', exist_ok=True)
    args = [
        '-fdat', data,
        '-fpop', population,
        '-dout', f'{base}/sir',
        '-days', kwargs.get('days', 60),
        '-keep', kwargs.get('keep', 7),
        '-window', kwargs.get('window', 5),
        '-fsuffix', '-'
    ] + ['-doubling-times'] + doubling_times
    sir(list(map(str, args)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('config', help='config file (json or yml)')
    parser.add_argument('-basedate', help='Base date for forecasting')
    parser.add_argument('-remote', action='store_true', help='Run jobs remotely on SLURM')
    parser.add_argument('-timeout-min', type=int, default=12*60)
    parser.add_argument('-array-parallelism', type=int, default=50)
    parser.add_argument('-mem-gb', type=int, default=20)
    opt = parser.parse_args()

    config = load_config(opt.config)
    now = datetime.now().strftime('%Y_%m_%d_%H_%M')
    user = os.environ['USER']
    base = f'/checkpoint/{user}/covid19/forecasts/{config["region"]}/{now}'

    dataset = os.path.realpath(config['data'])

    print('Running SIR model...')
    run_sir(data=dataset, base=base, region=config['region'], **config['sir'])

    keys = config['grid'].keys()
    values = list(itertools.product(*[config['grid'][k] for k in keys]))

    grid = [{k: v for k, v in zip(keys, vs)} for vs in values]

    basedate = opt.basedate or str(date.today())

    def run_job(d):
        kwargs = {**d,**config["forecast"], 'basedate': basedate}
        try:
            job_env = submitit.JobEnvironment()
            job_dir = os.path.join(base, str(submitit.JobEnvironment().job_id))
        except Exception:
            job_dir = os.path.join(base, '_'.join([f'{k}_{v}' for k, v in d.items()]))
            kwargs["log"] = f'{job_dir}/log'
        os.makedirs(job_dir, exist_ok=True)
        forecast_train(dataset=dataset, job_dir=job_dir, **kwargs)

    if opt.remote:
        executor = submitit.AutoExecutor(folder=base + '/%j')
        executor.update_parameters(
            name=f'{config["region"]}-sweep',
            gpus_per_node=1,
            cpus_per_task=3,
            mem_gb=opt.mem_gb,
            array_parallelism=opt.array_parallelism,
            timeout_min=opt.timeout_min,
        )
        with snapshot.SnapshotManager(snapshot_dir=base + '/snapshot', with_submodules=True):
            jobs = executor.map_array(run_job, grid)
        print(base)
    else:
        for d in grid:
            run_job(grid)