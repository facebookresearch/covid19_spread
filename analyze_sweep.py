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
    "Report job with best RMSE"
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
    """Provide a summary of the sweep"""
    keys = [
        "RMSE_PER_DAY",
        "RMSE_AVG",
        "json_conf",
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
            match = re.search(f"{key} *(:|=) *(.*)", logtxt)
            if match is not None:
                current[key] = match.group(2)
        results.append(current)

    if sort_by is not None:
        assert sort_by in results[0]
        results = sorted(results, key=lambda x: float(x[sort_by]))

    if verbose:
        sir_forecast = glob(os.path.join(sweep_dir, "sir/SIR-forecast*"))[0]
        sir = pandas.read_csv(sir_forecast, index_col=0)
        print(sir)
        print()

        for result in results:
            f_forecast = os.path.join(result["job"], "forecasts.csv")
            print(f'Job: {result["job"]}')
            for k in keys:
                print(f"{k}: {result[k]}")
            if os.path.exists(f_forecast):
                forecasts = pandas.read_csv(f_forecast, index_col=0)
                print(forecasts["ALL REGIONS"].to_frame().transpose())
            print()
    return results


@click.command()
@click.argument("sweep_dir")
@click.option("--verbose/--no-verbose", default=True)
def sir_similarity(sweep_dir, verbose=True):
    """
    Reports the model who's forecast is most closely in line with the SIR model.
    Similarity is in terms of mean average distance of the forecasts.
    """
    sir_forecast = glob(os.path.join(sweep_dir, "sir/SIR-forecast*"))[0]
    sir = pandas.read_csv(sir_forecast, index_col=0)
    sir = sir[sir.columns[:2]]
    sir.columns = ["days", "sir"]
    results = []
    for log in glob(os.path.join(sweep_dir, "**/*log.out"), recursive=True):
        job_dir = os.path.dirname(log)
        fname = os.path.join(job_dir, "forecasts.csv")
        if not os.path.exists(fname):
            continue
        forecast = pandas.read_csv(fname, index_col=0)
        forecast = forecast.loc[(forecast.index != "KS") & (forecast.index != "pval")]
        merged = sir.set_index(forecast.index).merge(
            forecast, left_index=True, right_index=True
        )
        results.append(
            {
                "mae": (merged["sir"] - merged["ALL REGIONS"]).abs().mean(),
                "job": job_dir,
            }
        )
    df = pandas.DataFrame(results)
    if verbose:
        best = df.sort_values(by="mae").iloc[0]
        print(f"Best run = {best.job}")
        print()
        print(df.sort_values(by="mae"))
    return df


if __name__ == "__main__":
    cli.add_command(rmse)
    cli.add_command(summary)
    cli.add_command(sir_similarity)
    cli()
