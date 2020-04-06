import click
from glob import glob
import pandas
import os
from subprocess import check_output
import re
from typing import Optional


@click.group()
def cli():
    pass


@click.command()
@click.argument("sweep_dir")
@click.option("--verbose/--no-verbose", default=True)
def rmse(sweep_dir, verbose: bool = False):
    results = []
    for log in glob(os.path.join(sweep_dir, "**/*log.out"), recursive=True):
        rmse_per_day = check_output(
            f'cat {log} | grep "^RMSE_PER_DAY" | grep -o "\\{{.*\\}}"', shell=True
        )
        rmse_per_day = eval(rmse_per_day.decode("utf-8"))
        rmse_avg = check_output(f'cat {log} | grep "^RMSE_AVG"', shell=True)
        results.append(
            {
                "job": os.path.dirname(log),
                "rmse_avg": float(
                    re.search("\d*\.\d+", rmse_avg.decode("utf-8")).group(0)
                ),
                **{f"rmse_day_{k}": v for k, v in rmse_per_day.items()},
            }
        )
    results = pandas.DataFrame(results)
    if verbose:
        best = results.sort_values(by="rmse_avg").iloc[0]
        print(f"Best job: {best.job}")
        print("First 10 results:")
        print(results.sort_values(by="rmse_avg").iloc[:10])
    return results


@click.command()
@click.argument("sweep_dir")
@click.option("--sort-by", default=None)
@click.option("--verbose/--no-verbose", default=True)
def summary(sweep_dir, sort_by: Optional[str] = None, verbose: bool = False):
    keys = [
        "RMSE_PER_DAY",
        "RMSE_AVG",
        "beta",
        "Norms",
        "Max Element",
        "Avg Element",
        "Avg. KS",
        "Avg. pval",
    ]
    results = []
    for log in glob(os.path.join(sweep_dir, "**/*log.out"), recursive=True):
        logtxt = open(log, "r").read()
        current = {"job": os.path.dirname(log)}
        for key in keys:
            current[key] = re.search(f"{key} *(:|=) *(.*)", logtxt).group(2)
        results.append(current)

    if sort_by is not None:
        assert sort_by in results[0]
        results = sorted(results, key=lambda x: float(x[sort_by]))

    if verbose:
        for result in results:
            print(f'Job: {result["job"]}')
            for k in keys:
                print(f"{k}: {result[k]}")
            forecasts = pandas.read_csv(
                os.path.join(result["job"], "forecasts.csv"), index_col=0
            )
            print(forecasts["ALL REGIONS"].to_frame().transpose())
            print()
    return results


if __name__ == "__main__":
    cli.add_command(rmse)
    cli.add_command(summary)
    cli()
