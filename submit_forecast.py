#!/usr/bin/env python3

from subprocess import check_call
import pandas
from datetime import timedelta, datetime
import os
import tempfile
import json
import click
from data.recurring import DB
import sqlite3
import yaml
import shutil
from lib.slack import get_client as get_slack_client
import boto3
import geopandas
from metrics import load_ground_truth
from epiweeks import Week
from common import update_repo
from io import BytesIO
from string import Template
from importlib.machinery import SourceFileLoader
from lib.context_managers import sys_path


license_txt = (
    "This work is based on publicly available third party data sources which "
    "may not necessarily agree. Facebook makes no guarantees about the "
    "reliability, accuracy, or completeness of the data. It is not intended "
    "to be a substitute for either public health guidance or professional "
    "medical advice, diagnosis, or treatment. This work is licensed under "
    "the Creative Commons Attribution-Noncommercial 4.0 International Public "
    "License (CC BY-NC 4.0). To view a copy of this license go to "
    "https://creativecommons.org/licenses/by-nc/4.0/"
)

RECIPIENTS = [
    "Matt Le <mattle@fb.com>",
]

if "__PROD__" in os.environ and os.environ["__PROD__"] == "1":
    print("Running in prod mode")
    RECIPIENTS.append("Max Nickel <maxn@fb.com>")


@click.group()
def cli():
    pass


def get_county_geometries():
    if not os.path.exists("cb_2018_us_county_20m.zip"):
        check_call(
            [
                "wget",
                "https://www2.census.gov/geo/tiger/GENZ2018/shp/cb_2018_us_county_20m.zip",
            ]
        )

    if not os.path.exists("us_counties"):
        shutil.unpack_archive("cb_2018_us_county_20m.zip", "us_counties")

    return geopandas.read_file("us_counties/cb_2018_us_county_20m.shp")


def get_gadm():
    url = "https://biogeo.ucdavis.edu/data/gadm3.6/shp/gadm36_USA_shp.zip"
    if not os.path.exists("gadm36_USA_shp.zip"):
        check_call(["wget", url])
    if not os.path.exists("gadm"):
        shutil.unpack_archive("gadm36_USA_shp.zip", "gadm")
    return geopandas.read_file("gadm/gadm36_USA_2.shp")


def format_df(df, val_name):
    df = df.melt(id_vars=["date"], value_name=val_name, var_name="location")
    df["loc3"] = df["location"].apply(lambda x: x.split(", ")[0])
    df["loc2"] = df["location"].apply(lambda x: x.split(", ")[1])
    df["loc1"] = "United States"
    return df


def get_index():
    script_dir = os.path.dirname(os.path.realpath(__file__))
    if os.path.exists(f"{script_dir}/us_counties_index.geojson"):
        index = geopandas.read_file(f"{script_dir}/us_counties_index.geojson")
        index.columns = [c.lower() for c in index.columns]
        return index
    geometries = get_county_geometries().drop(columns=["NAME"])
    index = pandas.read_csv(
        "https://storage.googleapis.com/covid19-open-data/v2/index.csv"
    )
    index = index[index["key"].str.match(r"^US_[A-Z]+_\d{5}$").fillna(False)]
    fips = pandas.read_csv(
        "https://raw.githubusercontent.com/kjhealy/fips-codes/master/state_and_county_fips_master.csv"
    )
    fips["fips"] = fips["fips"].astype(str).str.zfill(5)
    index = index.merge(fips, left_on="subregion2_code", right_on="fips")
    geom = geometries.merge(index, left_on="GEOID", right_on="subregion2_code")
    geom["name"] = geom["name"].str.replace(" (County|Municipality|Parish|Borough)", "")
    geom["geometry2"] = geom["geometry"]
    gadm = get_gadm()
    merged = geopandas.overlay(
        geopandas.GeoDataFrame(geom, geometry="geometry", crs=gadm.crs),
        gadm,
        how="intersection",
    )
    merged["area"] = merged["geometry"].area
    grouped = merged.loc[merged.groupby("fips")["area"].idxmax()].rename(
        columns={"GID_2": "GADM"}
    )
    grouped["geometry"] = grouped["geometry2"]
    del grouped["geometry2"]
    grouped.columns = [c.lower() for c in grouped.columns]
    grouped.to_file(f"{script_dir}/us_counties_index.geojson", driver="GeoJSON")
    return grouped


def get_best(pth, metric):
    model_selection = json.load(open(os.path.join(pth, "model_selection.json")))
    model_selection = {x["name"]: x["pth"] for x in model_selection}
    model_selection[metric]
    return os.path.join(pth, os.path.basename(model_selection[metric]))


def pivot_forecast(forecast, limit=True):
    index = get_index()
    forecast = format_df(forecast, "estimated_cases")
    forecast = forecast[["date", "estimated_cases", "loc1", "loc2", "loc3"]]
    dates = pandas.DataFrame.from_dict({"date": forecast["date"].unique(), "dummy": 1})
    index["dummy"] = 1
    index = index.merge(dates, on="dummy")

    merged = forecast.merge(
        index,
        left_on=["loc2", "loc3", "date"],
        right_on=["subregion1_name", "name", "date"],
        how="outer",
    )

    # Fill any missing counties with zeros.  We drop these when training the model
    merged["loc1"] = "United States"
    merged["loc2"] = merged["subregion1_name"]
    merged["loc3"] = merged["name"]
    merged["estimated_cases"] = merged["estimated_cases"].fillna(0)

    assert not merged["geometry"].isnull().any()
    merged["geometry"] = merged["geometry"].apply(lambda x: x.wkt)

    if limit:
        # limit forecasts to 41 days
        merged = merged[(merged["date"] - merged["date"].min()).dt.days < 41]

    # std = pandas.read_csv(
    #     os.path.join(job, "final_model_piv.csv"), parse_dates=["date"]
    # )
    # std = format_df(std, "std_dev")
    # merged = merged.merge(std, on=["date", "loc1", "loc2", "loc3"], how="left")
    # merged["std_dev"] = merged["std_dev"].fillna(0)
    merged["std_dev"] = 0
    merged = merged[
        [
            "date",
            "loc1",
            "loc2",
            "loc3",
            "geometry",
            "estimated_cases",
            "fips",
            "std_dev",
            "gadm",
        ]
    ].copy()
    merged["measurement_type"] = "cases"
    return merged


def prepare_forecast(path):
    forecast = pandas.read_csv(path, parse_dates=["date"])
    return pivot_forecast(forecast)


@cli.command()
@click.argument("pth")
@click.option("--metric", default="best_mae")
def submit_s3(pth, metric):
    job = get_best(pth, metric)
    df = prepare_forecast(os.path.join(job, "final_model_validation.csv"))
    client = boto3.client("s3")
    basedate = (df["date"].min() - timedelta(days=1)).date()
    object_name = f"users/mattle/h2/covid19_forecasts/forecast_{basedate}.csv"
    with tempfile.NamedTemporaryFile() as tfile:
        df.to_csv(tfile.name, index=False, sep="\t")
        client.upload_file(tfile.name, "fairusersglobal", object_name)
    with BytesIO() as fout:
        obj = {
            "fair_cluster_path": pth,
            "metric": metric,
            "basedate": str(basedate),
        }
        fout.write(json.dumps(obj).encode("utf-8"))
        fout.seek(0)
        client.upload_fileobj(
            fout,
            "fairusersglobal",
            f"users/mattle/h2/covid19_forecasts/metadata/forecast_{basedate}.json",
        )
    conn = sqlite3.connect(DB)
    vals = (pth, datetime.now().timestamp())
    conn.execute("INSERT INTO submitted (sweep_path, submitted_at) VALUES(?,?)", vals)
    conn.commit()
    client = get_slack_client()
    msg = f"*Forecast for US is in S3: {basedate}*"
    client.chat_postMessage(channel="#sweep-updates", text=msg)


def is_date(x):
    try:
        pandas.to_datetime(x)
        return True
    except Exception:
        return False


PR_TEXT = """
## Description

If you are **adding new forecasts** to an existing model, please include the following details:- 
- Team name: Facebook AI Research (FAIR)
- Model name that is being updated: Neural Relational Autoregression
---

## Checklist

- [x] Specify a proper PR title with your team name.
- [x] All validation checks ran successfully on your branch. Instructions to run the tests locally is present [here](https://github.com/reichlab/covid19-forecast-hub/wiki/Running-Checks-Locally).
"""


def get_prod_forecast_by_date(date):
    client = boto3.client("s3")
    basedate = pandas.to_datetime(date).date()
    key = f"users/mattle/h2/covid19_forecasts/metadata/forecast_{basedate}.json"
    val = client.get_object(Bucket="fairusersglobal", Key=key)
    meta = json.loads(val["Body"].read().decode("utf-8"))
    return meta["fair_cluster_path"]


@cli.command()
@click.argument("date")
def pth(date):
    print(get_prod_forecast_by_date(date))


@cli.command()
@click.argument("pth")
@click.option("-no-push", is_flag=True)
@click.option("-team", default="FAIR-NRAR")
@click.option("-nweeks", type=click.INT)
@click.option("-pivfile")
def submit_reichlab(pth, no_push, team, nweeks, pivfile):
    if not os.path.exists(pth) and is_date(pth):
        pth = get_prod_forecast_by_date(pth)
    job_pth = get_best(pth, "best_mae")
    kwargs = {"index_col": "date", "parse_dates": ["date"]}

    if os.path.exists(os.path.join(job_pth, "final_model_mean_closed_form.csv")):
        deltas = pandas.read_csv(
            os.path.join(job_pth, "final_model_mean_closed_form.csv"), **kwargs
        )
        forecast_date = deltas.index.min() - timedelta(days=1)
    else:
        forecast = pandas.read_csv(
            os.path.join(job_pth, "final_model_validation.csv"), **kwargs
        )
        gt = load_ground_truth(os.path.join(pth, "data_cases.csv"))
        forecast.loc[forecast_date] = gt.loc[forecast_date]
        deltas = forecast.sort_index().diff()
    index = get_index()

    def format_df(df, type_, quantile):
        next_week = (
            Week.fromdate(forecast_date).daydate(5)
            if forecast_date.weekday() in {0, 6}
            else Week.fromdate(forecast_date).daydate(5) + timedelta(days=7)
        )
        submission = []
        prev_date = forecast_date
        # Generate 2 epi weeks worth of data
        for i in range(1, (nweeks + 1) if nweeks is not None else 5):
            submission.append(df.loc[prev_date:next_week].sum(0).reset_index())
            submission[-1]["target"] = f"{i} wk ahead inc case"
            submission[-1]["target_end_date"] = next_week
            prev_date = next_week
            next_week += timedelta(days=7)
        submission = pandas.concat(submission).rename(
            columns={"index": "loc", 0: "value", "location": "loc"}
        )
        submission["type"] = type_
        submission["quantile"] = quantile
        submission["forecast_date"] = forecast_date
        index["loc"] = index["name"] + ", " + index["subregion1_name"]
        merged = submission.merge(index[["loc", "fips"]], on="loc").drop(
            columns=["loc"]
        )
        return merged.rename(columns={"fips": "location"})

    point_estimates = format_df(deltas, "point", "NA")

    q50 = point_estimates.copy()
    q50["quantile"] = 0.5
    q50["type"] = "quantile"

    pred_intervals = [q50]
    if os.path.exists(os.path.join(job_pth, pivfile or "final_model_piv.csv")):
        piv = pandas.read_csv(
            os.path.join(job_pth, pivfile or "final_model_piv.csv"),
            parse_dates=["date"],
        )
        for interval, group in piv.groupby("interval"):
            pivot = group.pivot(
                index="date", columns="location", values="lower"
            ).sort_index()
            quantile = round((1 - interval) / 2, 3)
            pred_intervals.append(format_df(pivot, type_="quantile", quantile=quantile))
            pivot = group.pivot(
                index="date", columns="location", values="upper"
            ).sort_index()
            quantile = round(1 - (1 - interval) / 2, 3)
            pred_intervals.append(format_df(pivot, type_="quantile", quantile=quantile))
    merged = pandas.concat(pred_intervals + [point_estimates])
    data_pth = update_repo("git@github.com:lematt1991/covid19-forecast-hub.git")
    upstream_repo = "git@github.com:reichlab/covid19-forecast-hub.git"

    # This is bad, but not sure how to add a remote if it already exists
    try:
        check_call(["git", "remote", "add", "upstream", upstream_repo], cwd=data_pth)
    except Exception:
        pass

    # Update this fork to upstream's master
    check_call(["git", "checkout", "master"], cwd=data_pth)
    check_call(["git", "fetch", "upstream"], cwd=data_pth)
    check_call(["git", "merge", "upstream/master"], cwd=data_pth)
    check_call(["git", "push"], cwd=data_pth)
    filename = str(forecast_date.date()) + f"-{team}.csv"
    outpth = os.path.join(data_pth, "data-processed", team, filename)
    merged.to_csv(outpth, index=False)

    with sys_path(os.path.join(data_pth, "code/validation")):
        module = SourceFileLoader(
            fullname=".",
            path=os.path.join(data_pth, "code/validation/test_formatting.py"),
        ).load_module()
        failed, file_errors = module.validate_forecast_file(outpth)
        if failed:
            raise RuntimeError(file_errors)
        print("Successfully validated file!")

    if not no_push:
        check_call(
            ["git", "checkout", "-b", f"forecast-{forecast_date.date()}"], cwd=data_pth
        )
        check_call(["git", "add", outpth], cwd=data_pth)
        check_call(
            [
                "git",
                "commit",
                "-m",
                f"Adding FAIR-NRAR forecast for {forecast_date.date()}",
            ],
            cwd=data_pth,
        )
        check_call(
            [
                "git",
                "push",
                "--set-upstream",
                "origin",
                f"forecast-{forecast_date.date()}",
            ],
            cwd=data_pth,
        )
        print("Create pull request by going to:")
        print("https://github.com/lematt1991/covid19-forecast-hub")
        print("-------------------------")
        print(PR_TEXT)


@cli.command()
@click.option("--dry", is_flag=True)
@click.pass_context
def check_s3_unsubmitted(ctx, dry):
    print(f"Recipients: {RECIPIENTS}")
    conn = sqlite3.connect(DB)
    res = conn.execute(
        """
    SELECT path, module
    FROM sweeps
    LEFT JOIN submitted ON sweeps.path=submitted.sweep_path
    WHERE submitted.sweep_path IS NULL AND id='us-bar'
    """
    )
    print(f"Checking for unsubmitted forecasts: {datetime.now()}")
    for path, module in res:
        fcst_pth = os.path.join(path, "forecasts/forecast_best_mae.csv")
        if not os.path.exists(fcst_pth):
            continue
        if dry:
            print(f"Dry Run: Would have submitted: {path}")
        else:
            ctx.invoke(submit_s3, pth=path, metric="best_mae")
            print(f"submitting: {path}")


@cli.command()
@click.argument("pth")
@click.option("--skip-gt", is_flag=True)
def submit_mbox(pth, skip_gt=False):
    if not os.path.exists(pth) and is_date(pth):
        pth = get_prod_forecast_by_date(pth)
    with open(
        "/checkpoint/mattle/covid19/SDX-CLI/my_commands.txt.template", "r"
    ) as fin:
        template = Template(fin.read())
    job = get_best(pth, "best_mae")
    cfg = yaml.safe_load(open(os.path.join(pth, "cfg.yml")))
    sdxpth = "/checkpoint/mattle/covid19/SDX-CLI"

    if not skip_gt:
        gt = pandas.read_csv(os.path.join(pth, "data_cases.csv"), index_col="region")
        gt = gt.transpose()
        gt.index = pandas.to_datetime(gt.index)
        gt = pivot_forecast(gt.rename_axis("date").reset_index(), limit=False)
        gt = gt.drop(columns=["std_dev", "geometry"])
        gt = gt.rename(columns={"estimated_cases": "cases"})
        with tempfile.NamedTemporaryFile() as tfile, tempfile.NamedTemporaryFile() as cmdfile:
            gt.to_csv(tfile.name, index=False)
            dest_file = f"{cfg['region']}/ground_truth.csv"
            with open(cmdfile.name, "w") as fout:
                cmd_string = template.substitute(
                    FORECAST_FILE_PATH=tfile.name, DEST_FILE_PATH=dest_file
                )
                print(cmd_string, file=fout)
            print("Uploading to mbox...")
            check_call(
                ["./CrushFTP_Transfer.sh"],
                cwd=sdxpth,
                env={**os.environ, "TUNNEL_CMDS": cmdfile.name},
            )

    # https://mboxnaprd.jnj.com/#/
    df = prepare_forecast(os.path.join(job, "final_model_validation.csv"))
    df = df.drop(columns=["geometry", "std_dev"])
    basedate = (df["date"].min() - timedelta(days=1)).date()
    with tempfile.NamedTemporaryFile() as tfile, tempfile.NamedTemporaryFile() as cmdfile:
        df.to_csv(tfile.name, index=False)
        db_file = f"{cfg['region']}/forecast_{basedate}.csv"
        with open(cmdfile.name, "w") as fout:
            cmd_string = template.substitute(
                FORECAST_FILE_PATH=tfile.name, DEST_FILE_PATH=db_file
            )
            print(cmd_string, file=fout)
        print("Uploading to mbox...")
        check_call(
            ["./CrushFTP_Transfer.sh"],
            cwd=sdxpth,
            env={**os.environ, "TUNNEL_CMDS": cmdfile.name},
        )


if __name__ == "__main__":
    cli()
