#!/usr/bin/env python3

from subprocess import check_call
import pandas
from datetime import timedelta, datetime
from nbconvert import HTMLExporter
import nbformat
import nbconvert
from nbconvert.preprocessors import ExecutePreprocessor
from data.recurring import env_var
import os
import tempfile
from traitlets.config import Config
import click
from data.recurring import DB, chdir
import sqlite3
import re
import yaml
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from nb2mail import MailExporter
import socket

license_txt = "This work is licensed under the Creative Commons Attribution-Noncommercial 4.0 International Public License (CC BY-NC 4.0). To view a copy of this license go to https://creativecommons.org/licenses/by-nc/4.0/. Retention of the foregoing language is sufficient for attribution."

RECIPIENTS = [
    "Matt Le <mattle@fb.com>",
]

if "__PROD__" in os.environ and os.environ["__PROD__"] == "1":
    print("Running in prod mode")
    RECIPIENTS.append("Max Nickel <maxn@fb.com>")


@click.group()
def cli():
    pass


@cli.command()
@click.argument("pth")
@click.option("--module", required=True)
@click.option("--region", required=True)
def submit(pth: str, module: str, region: str):
    script_dir = os.path.dirname(os.path.realpath(__file__))
    with tempfile.TemporaryDirectory() as tdir:
        os.makedirs(tdir, exist_ok=True)
        with chdir(tdir):
            # Read and format CSV
            df = pandas.read_csv(pth, index_col="date", parse_dates=["date"])
            basedate = (df.index.min() - timedelta(days=1)).date().strftime("%Y%m%d")
            forecast = f"forecast-{region}_{basedate}_{module}.csv"
            df = df[sorted(df.columns)].copy()
            df["ALL_REGIONS"] = df.sum(axis=1)
            df.index = [x.date() for x in df.index]
            df.loc[license_txt] = None
            df.to_csv(forecast, index_label="date", float_format="%.2f")

            # Run the notebook for this forecast
            notebook = f"{script_dir}/notebooks/forecast_template.ipynb"
            with open(notebook, "r") as fin:
                nb = nbformat.read(fin, as_version=nbformat.current_nbformat)
            with env_var({"FORECAST_PTH": pth}):
                print("Executing notebook...")
                ep = ExecutePreprocessor(kernel_name="python3")
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
    WHERE submitted.sweep_path IS NULL
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


if __name__ == "__main__":
    cli()
