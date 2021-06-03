import pandas as pd
import torch as th
import yaml
from pathlib import Path
import json
import os


def load_confirmed_csv(path):
    df = pd.read_csv(path)
    df.set_index("region", inplace=True)
    basedate = df.columns[-1]
    nodes = df.index.to_numpy()
    cases = df.to_numpy()
    return th.from_numpy(cases), nodes, basedate


def load_confirmed(path, regions):
    """Returns dataframe of total confirmed cases"""
    df = load_confirmed_by_region(path, regions=regions)
    return df.sum(axis=1)


def load_confirmed_by_region(path, regions=None, filter_unknown=True):
    """Loads csv file for confirmed cases by region"""
    df = pd.read_csv(path, index_col=0, header=None)
    # transpose so dates are along rows to match h5
    df = df.T
    # set date as index
    df = df.rename(columns={"region": "date"})
    df = df.set_index("date")
    df.index = pd.to_datetime(df.index)
    df = df.astype(float)
    if regions is not None:
        df = df[regions]
    if filter_unknown:
        df = df.loc[:, df.columns != "Unknown"]
    return df


def load_backfill(
    jobdir, model=None, indicator="model_selection.json", forecast="best_mae",
):
    """collect all forcasts from job dir"""
    forecasts = {}
    configs = []
    for path in Path(jobdir).rglob(indicator):
        date = str(path).split("/")[-2]
        assert date.startswith("sweep_"), str(path)
        jobs = [m["pth"] for m in json.load(open(path)) if m["name"] == forecast]
        assert len(jobs) == 1, jobs
        job = jobs[0]
        date = date[6:]
        forecasts[date] = os.path.join(job, "final_model_validation.csv")
        cfg = yaml.safe_load(open(os.path.join(job, "../cfg.yml")))
        cfg = yaml.safe_load(
            open(os.path.join(job, f"{model or cfg['this_module']}.yml"))
        )
        cfg = cfg["train"]
        cfg["date"] = date
        cfg["job"] = job
        configs.append(cfg)
    configs = pd.DataFrame(configs)
    configs.set_index("date", inplace=True)
    return forecasts, configs
