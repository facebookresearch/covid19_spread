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
    if os.path.exists("us_counties_index.geojson"):
        index = geopandas.read_file("us_counties_index.geojson")
        index.columns = [c.lower() for c in index.columns]
        return index
    geometries = get_county_geometries()
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
        columns={"gid_2": "GADM"}
    )
    grouped["geometry"] = grouped["geometry2"]
    del grouped["geometry2"]
    grouped.to_file("us_counties_index.geojson", driver="GeoJSON")
    return grouped


@cli.command()
@click.argument("pth")
@click.option("--metric", default="best_mae")
def submit_s3(pth, metric):
    index = get_index()
    model_selection = json.load(open(os.path.join(pth, "model_selection.json")))
    model_selection = {x["name"]: x["pth"] for x in model_selection}
    model_selection[metric]
    job = os.path.join(pth, os.path.basename(model_selection[metric]))
    forecast = pandas.read_csv(
        os.path.join(job, "final_model_validation.csv"), parse_dates=["date"]
    )
    forecast = format_df(forecast, "estimated_cases")
    forecast = forecast[["date", "estimated_cases", "loc1", "loc2", "loc3"]]
    basedate = str((forecast["date"].min() - timedelta(days=1)).date())

    merged = forecast.merge(
        index,
        left_on=["loc2", "loc3"],
        right_on=["subregion1_name", "name"],
        how="left",
    )

    assert not merged["geometry"].isnull().any()
    merged["geometry"] = merged["geometry"].apply(lambda x: x.wkt)

    merged = merged[(merged["date"] - merged["date"].min()).dt.days < 30]

    std = pandas.read_csv(
        os.path.join(job, "final_model_piv.csv"), parse_dates=["date"]
    )
    std = format_df(std, "std_dev")
    merged = merged.merge(std, on=["date", "loc1", "loc2", "loc3"])
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
    ]
    merged["measurement_type"] = "cases"
    client = boto3.client("s3")
    object_name = f"users/mattle/h2/covid19_forecasts/forecast_{basedate}.csv"
    with tempfile.NamedTemporaryFile() as tfile:
        merged.to_csv(tfile.name, index=False, sep="\t")
        client.upload_file(tfile.name, "fairusersglobal", object_name)
    client = get_slack_client()
    msg = f"*Forecast for US is in S3: {basedate}*"
    client.chat_postMessage(channel="#sweep-updates", text=msg)


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
            conn.execute(
                "INSERT INTO submitted (sweep_path, submitted_at) VALUES(?,?)", vals
            )
            conn.commit()
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
@click.option("--dry", is_flag=True)
def submit_to_dropbox(dry: bool = False):
    conn = sqlite3.connect(DB)
    res = conn.execute(
        """
    SELECT path, module
    FROM sweeps
    WHERE module IN ('ar', 'ar_daily')
    """
    )
    uploader = Uploader()
    for path, module in res:
        cfg = yaml.safe_load(open(f"{path}/cfg.yml"))
        fcsts = glob(f'{path}/forecasts/forecast-{cfg["region"]}*_{module}.csv')
        if len(fcsts) == 1:
            forecast_pth = fcsts[0]
            db_file = (
                f"/covid19_forecasts/{cfg['region']}/{os.path.basename(forecast_pth)}"
            )
            try:
                uploader.client.files_get_metadata(db_file)
                continue  # file already exists, skip it...
            except dropbox.exceptions.ApiError:
                pass
            print(f"Uploading: {db_file}")
            if not dry:
                uploader.upload(forecast_pth, db_file)


if __name__ == "__main__":
    cli()
