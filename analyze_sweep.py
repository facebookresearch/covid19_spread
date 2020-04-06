import click
from glob import glob
import pandas
import os
from subprocess import check_output
import re


@click.group()
def cli():
    pass

@click.command()
@click.argument('sweep_dir')
@click.option('--verbose/--no-verbose', default=True)
def rmse(sweep_dir, verbose: bool = False):
    results = []
    for log in glob(os.path.join(sweep_dir, '**/*log.out'), recursive=True):
        rmse_per_day = check_output(f'cat {log} | grep "^RMSE_PER_DAY" | grep -o "\\{{.*\\}}"', shell=True)
        rmse_per_day = eval(rmse_per_day.decode('utf-8'))
        rmse_avg = check_output(f'cat {log} | grep "^RMSE_AVG"', shell=True)
        results.append({
            'job': os.path.dirname(log),
            'rmse_avg': float(re.search('\d*\.\d+', rmse_avg.decode('utf-8')).group(0)),
            **{f'rmse_day_{k}': v for k, v in rmse_per_day.items()}
        })
    results = pandas.DataFrame(results)
    if verbose:
        best = results.sort_values(by='rmse_avg').iloc[0]
        print(f'Best job: {best.job}')
        print('First 10 results:')
        print(results.sort_values(by='rmse_avg').iloc[:10])
    return results

if __name__ == '__main__':
    cli.add_command(rmse)
    cli()