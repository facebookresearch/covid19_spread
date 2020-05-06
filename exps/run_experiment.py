import os
import submitit
import train
from exps.compute_rmse import rmse
import json
import pandas


def run_experiment(grid, days, crossval, folder, pdict, local=False, seed=42, chkpnt_name='model.bin', ngpus=1):
    if not local:
        checkpoint = os.path.join(folder, chkpnt_name)
        job_checkpoint = checkpoint.replace('%j', submitit.JobEnvironment().job_id)
        job_dir = folder.replace('%j', submitit.JobEnvironment().job_id)
        with open(os.path.join(job_dir, 'params.json'), 'w') as fout:
            json.dump({'params': pdict, 'grid': grid}, fout)
    else:
        job_checkpoint = f'/tmp/{chkpnt_name}'

    if crossval:
        job_dset = os.path.join(job_dir, '../', os.path.basename(pdict['dset']) + f'.minus_{days}_days')
    else:
        job_dset = os.path.join(job_dir, '../', os.path.basename(pdict['dset']))

    job_dset = os.path.realpath(job_dset)

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
        '-no-sparse-grads',
    ] + (['-data-parallel'] if ngpus > 1 else [])])

    prefix = '' if crossval else 'full_data_'
    rmse(days, job_checkpoint, prefix=prefix)
