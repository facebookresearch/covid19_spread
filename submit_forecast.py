#!/usr/bin/env python3

from subprocess import check_call
import pandas
from datetime import timedelta, datetime
from nbconvert import HTMLExporter
import nbformat
import nbconvert
from nbconvert.preprocessors import ExecutePreprocessor
import os
import dropbox
import tempfile
from traitlets.config import Config
import json
import click
from data.recurring import DB, chdir, env_var
import sqlite3
from glob import glob
import re
from lib.dropbox import Uploader
import yaml
import shutil
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from nb2mail import MailExporter
import socket
from lib.slack import get_client as get_slack_client
import boto3
import geopandas
from metrics import load_ground_truth
from epiweeks import Week
from forecast_db import update_repo
from io import BytesIO
from string import Template


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
    index = index[index["key"].str.match("^US_[A-Z]+_\d{5}$").fillna(False)]
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
        # limit forecasts to 30 days
        merged = merged[(merged["date"] - merged["date"].min()).dt.days < 30]

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
@click.argument("pth")
def submit_reichlab(pth):
    if not os.path.exists(pth) and is_date(pth):
        pth = get_prod_forecast_by_date(pth)
    job_pth = get_best(pth, "best_mae")
    kwargs = {"index_col": "date", "parse_dates": ["date"]}
    forecast = pandas.read_csv(
        os.path.join(job_pth, "final_model_validation.csv"), **kwargs
    )
    gt = load_ground_truth(os.path.join(pth, "data_cases.csv"))
    forecast_date = forecast.index.min() - timedelta(days=1)
    forecast.loc[forecast_date] = gt.loc[forecast_date]
    deltas = forecast.sort_index().diff()

    next_week = (
        Week.fromdate(forecast_date).daydate(5)
        if forecast_date.weekday() in {0, 6}
        else Week.fromdate(forecast_date).daydate(5) + timedelta(days=7)
    )
    submission = []
    prev_date = forecast_date
    # Generate 2 epi weeks worth of data
    for i in range(1, 3):
        submission.append(deltas.loc[prev_date:next_week].sum(0).reset_index())
        submission[-1]["target"] = f"{i} wk ahead inc case"
        submission[-1]["target_end_date"] = next_week
        prev_date = next_week
        next_week += timedelta(days=7)
    submission = pandas.concat(submission).rename(columns={"index": "loc", 0: "value"})
    submission["type"] = "point"
    submission["quantile"] = "NA"
    submission["forecast_date"] = forecast_date
    index = get_index()
    index["loc"] = index["name"] + ", " + index["subregion1_name"]
    merged = submission.merge(index[["loc", "fips"]], on="loc").drop(columns=["loc"])
    merged = merged.rename(columns={"fips": "location"})

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
    check_call(
        ["git", "checkout", "-b", f"forecast-{forecast_date.date()}"], cwd=data_pth
    )
    team_name = "FAIR-NRAR"
    filename = str(forecast_date.date()) + f"-{team_name}.csv"
    outpth = os.path.join(data_pth, "data-processed", team_name, filename)
    merged.to_csv(outpth, index=False)
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
        ["git", "push", "--set-upstream", "origin", f"forecast-{forecast_date.date()}"],
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
            cfg = yaml.safe_load(open(os.path.join(path, "cfg.yml")))
            ctx.invoke(submit_s3, pth=path, metric="best_mae")
            vals = (path, datetime.now().timestamp())
            print(f"submitting: {path}")


@cli.command()
@click.argument("pth")
@click.option("--module", required=True)
@click.option("--region", required=True)
@click.option("--email/--no-email", default=True)
def submit(pth: str, module: str, region: str, email: bool = True):
    script_dir = os.path.dirname(os.path.realpath(__file__))
    with tempfile.TemporaryDirectory() as tdir:
        os.makedirs(tdir, exist_ok=True)
        with chdir(tdir):
            # Read and format CSV
            df = pandas.read_csv(pth, index_col="date", parse_dates=["date"])

            forecast_dir = os.path.dirname(pth)
            basedate = (df.index.min() - timedelta(days=1)).date().strftime("%Y%m%d")
            forecast = f"{forecast_dir}/forecast-{region}_{basedate}_{module}.csv"
            df = df[sorted(df.columns)].copy()
            df["ALL_REGIONS"] = df.sum(axis=1)
            df.index = [x.date() for x in df.index]
            df.loc[license_txt] = None
            df.to_csv(forecast, index_label="date", float_format="%.2f")

            if not email:
                print("Not sending email...")
                return

            try:
                client = get_slack_client()
                msg = f"*Forecast for {region} - {module} is ready: {basedate}*"
                client.chat_postMessage(channel="#sweep-updates", text=msg)
            except Exception:
                pass

            # Run the notebook for this forecast
            notebook = f"{script_dir}/notebooks/forecast_template.ipynb"
            with open(notebook, "r") as fin:
                nb = nbformat.read(fin, as_version=nbformat.current_nbformat)
            with env_var({"FORECAST_PTH": pth}):
                print("Executing notebook...")
                ep = ExecutePreprocessor(kernel_name="python3", timeout=240)
                ep.preprocess(nb, {"metadata": {"path": os.path.dirname(notebook)}})

            # Export to an email
            conf = Config()
            conf.MailExporter.exclude_input = True
            exporter = MailExporter(config=conf)

            nb["metadata"]["nb2mail"] = {
                "To": ", ".join(RECIPIENTS),
                "From": "Covid 19 Forecasting Team <mattle@devfair0222.h2.fair>",
                "Subject": f"[covid19-forecast] {region} - {module} forecast for {basedate}",
                "attachments": [forecast],
            }

            body, _ = exporter.from_notebook_node(nb)

            # Send the email...
            with open(f"msg.txt", "w") as fout:
                fout.write(body)
                hostname = socket.gethostname()
                user = os.environ["USER"]
                check_call(
                    f"sendmail -i -t -f {user}@{hostname}.h2.fair < msg.txt",
                    shell=True,
                )


@cli.command()
@click.pass_context
def check_unsubmitted(ctx):
    print(f"Recipients: {RECIPIENTS}")
    conn = sqlite3.connect(DB)
    res = conn.execute(
        """
    SELECT path, module
    FROM sweeps
    LEFT JOIN submitted ON sweeps.path=submitted.sweep_path
    WHERE submitted.sweep_path IS NULL AND module='ar'
    """
    )
    print(f"Checking for unsubmitted forecasts: {datetime.now()}")
    for path, module in res:
        fcst_pth = os.path.join(path, "forecasts/forecast_best_mae.csv")
        if not os.path.exists(fcst_pth):
            continue
        print(f"submitting: {path}")
        cfg = yaml.safe_load(open(os.path.join(path, "cfg.yml")))
        ctx.invoke(submit, pth=fcst_pth, module=module, region=cfg["region"])
        vals = (path, datetime.now().timestamp())
        conn.execute(
            "INSERT INTO submitted (sweep_path, submitted_at) VALUES(?,?)", vals
        )
        conn.commit()


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
            print(f"Uploading to mbox...")
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
        print(f"Uploading to mbox...")
        check_call(
            ["./CrushFTP_Transfer.sh"],
            cwd=sdxpth,
            env={**os.environ, "TUNNEL_CMDS": cmdfile.name},
        )


if __name__ == "__main__":
    cli()
