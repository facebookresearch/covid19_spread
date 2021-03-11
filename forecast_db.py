#!/usr/bin/env python3

import click
from glob import glob
import pandas
import os
import re
from subprocess import check_call
import datetime
import shutil
from covid19_spread.data.usa.process_cases import get_nyt, get_google
import requests
from xml.etree import ElementTree
from datetime import timedelta
from covid19_spread.common import update_repo


@click.group()
def cli():
    pass


def to_csv(df, metric, model, forecast_date, deltas=False):
    if "loc3" not in df.columns:
        df["loc3"] = None
    if "loc2" not in df.columns:
        df["loc2"] = None
    dt = pandas.to_datetime(forecast_date if forecast_date else 0)
    basedir = f'/checkpoint/{os.environ["USER"]}/covid19/csvs'
    if forecast_date != "":
        forecast_date = pandas.to_datetime(forecast_date)
        forecast_date = "_" + str(forecast_date.date())

    df["location"] = df.apply(
        lambda x: ((x.loc3 + ", ") if x.loc3 else "") + x.loc2, axis=1
    )
    df = df.pivot_table(columns=["location"], values=["counts"], index="date")
    df.columns = df.columns.get_level_values(-1)

    if deltas:
        df = df.diff()
        if dt not in df.index:
            print(
                f"Warning: forecast_date not in forecast for {model}, {forecast_date}"
            )
    df = df[df.index > dt]

    outfile = os.path.join(basedir, metric, model, f"counts{forecast_date}.csv")
    os.makedirs(os.path.dirname(outfile), exist_ok=True)
    df.to_csv(outfile)


def sync_nyt():
    # Sync the NYTimes ground truth data
    def dump(df, metric):
        df = df.reset_index().melt(
            id_vars=["date"], value_name="counts", var_name="loc2"
        )
        df["loc3"] = df["loc2"].apply(lambda x: x.split("_")[1])
        df["loc2"] = df["loc2"].apply(lambda x: x.split("_")[0])
        df["loc1"] = "United States"
        df["date"] = pandas.to_datetime(df["date"])
        df["id"] = "nyt_ground_truth"
        to_csv(df, metric, "nyt_ground_truth", "")

    dump(get_nyt(metric="cases"), "infections")
    dump(get_nyt(metric="deaths"), "deaths")


def get_ihme_file(dir):
    """
    There's some inconsistent naming conventions for the CSV files containing the forecasts.
    This function tries to resolve these inconsistencies.
    """
    csvs = glob(os.path.join(dir, "*/*.csv"))
    if any(["Best_mask_hospitalization_all_locs.csv" in f for f in csvs]):
        csvs = [f for f in csvs if "Best_mask_hospitalization_all_locs.csv" in f]
    elif any(["Best_masks_hospitalization_all_locs.csv" in f for f in csvs]):
        csvs = [f for f in csvs if "Best_masks_hospitalization_all_locs.csv" in f]
    elif any(["best_masks_hospitalization_all_locs.csv" in f for f in csvs]):
        csvs = [f for f in csvs if "best_masks_hospitalization_all_locs.csv" in f]
    if len(csvs) > 1:
        csvs = [f for f in csvs if "hospitalization" in os.path.basename(f).lower()]
    if len(csvs) == 1:
        return csvs[0]
    if len(csvs) == 0:
        raise ValueError("No CSVs found in IHME zip!")
    else:
        raise ValueError(f"Ambiguous CSVs in IHME zip!  Found {len(csvs)} CSV files")


def sync_ihme():
    marker = None
    while True:
        url = "https://ihmecovid19storage.blob.core.windows.net/archive?comp=list"
        if marker:
            url += f"&marker={marker}"
        req = requests.get(url)
        req.raise_for_status()
        tree = ElementTree.fromstring(req.content)

        basedir = f'/checkpoint/{os.environ["USER"]}/covid19/data/ihme'
        os.makedirs(basedir, exist_ok=True)
        states = pandas.read_csv(
            "https://raw.githubusercontent.com/jasonong/List-of-US-States/master/states.csv"
        )
        states = states[["State"]].rename(columns={"State": "loc2"})
        for elem in tree.findall(".//Blob"):
            if elem.find("Url").text.endswith("ihme-covid19.zip"):
                forecast_date = datetime.datetime.strptime(
                    elem.find("Url").text.split("/")[-2], "%Y-%m-%d"
                ).date()

                os.makedirs(os.path.join(basedir, str(forecast_date)), exist_ok=True)
                zip_file = os.path.join(basedir, str(forecast_date), "ihme-covid19.zip")
                if not os.path.exists(zip_file):
                    check_call(["wget", "-O", zip_file, elem.find("Url").text])
                    shutil.unpack_archive(
                        zip_file, extract_dir=os.path.join(basedir, str(forecast_date))
                    )
                stats_file = get_ihme_file(os.path.join(basedir, str(forecast_date)))
                stats = pandas.read_csv(stats_file).rename(
                    columns={"date_reported": "date"}
                )
                stats["date"] = pandas.to_datetime(stats["date"])
                # Filter out only the US states
                df = states.merge(stats, left_on="loc2", right_on="location_name")[
                    ["loc2", "date", "totdea_mean"]
                ]
                df = df[~df["totdea_mean"].isnull()].rename(
                    columns={"totdea_mean": "counts"}
                )

                # Unfortunately, they don't explictly say what the forecast date is.  Here we try to infer it.
                if "confirmed_infections" in stats.columns:
                    # If we have a confirmed_infectiosn column.  Take the last date this is non-null for
                    forecast_date = stats[~stats["confirmed_infections"].isnull()][
                        "date"
                    ].max()
                else:
                    # continue  # not sure this is sufficient for determining forecast_date
                    # This is a pretty hacky way of determining what the actual forecast date is
                    # Find the latest date that has all whole number `totdea_mean` and at least
                    # one non-zero totdea_mean
                    temp = df.copy()
                    temp["nonzero"] = temp["counts"] > 0
                    temp["round"] = temp["counts"] % 1 == 0

                    grouped = temp.groupby("date")["round"].all().reset_index()
                    grouped = grouped.merge(
                        temp.groupby("date")["nonzero"].any().reset_index()
                    )
                    forecast_date = grouped[grouped["round"] & grouped["nonzero"]][
                        "date"
                    ].max()
                print(forecast_date)
                df["loc1"] = "United States"
                df["forecast_date"] = pandas.to_datetime(forecast_date)
                df["id"] = "IHME"
                df["loc3"] = None
                to_csv(df, "deaths", "IHME", forecast_date)
        marker = tree.find("NextMarker").text
        if marker is None:
            break


def sync_reich_forecast(name, mdl_id):
    data_dir = update_repo("https://github.com/reichlab/covid19-forecast-hub.git")
    loc_codes = pandas.read_csv(f"{data_dir}/data-locations/locations.csv")

    for pth in glob(f"{data_dir}/data-processed/{name}/*.csv"):
        value = pandas.read_csv(pth, dtype={"location": str})
        value = value[
            (value["type"] == "point")
            & (value["target"].str.match(r"\d wk ahead cum death"))
            & (value["location"].str.match(r"\d\d"))
        ].copy()
        value = value.merge(loc_codes, on="location")
        value["days"] = value["target"].str.extract(r"(\d+) wk")
        value = value[value["days"] == value["days"].max()]

        value = value.rename(
            columns={
                "target_end_date": "date",
                "location_name": "loc2",
                "value": "counts",
            }
        )
        value["loc1"] = "United States"
        value["id"] = mdl_id
        value["forecast_date"] = pandas.to_datetime(value["forecast_date"])
        value["date"] = pandas.to_datetime(value["date"])
        value = value.drop(columns=["target", "location", "type", "quantile"])
        value = value[["date", "loc1", "loc2", "counts", "id", "forecast_date"]]
        to_csv(value, "deaths", mdl_id, value["forecast_date"].iloc[0])


def sync_mit():
    sync_reich_forecast("MIT_CovidAnalytics-DELPHI", "mit-delphi")


def sync_yyg():
    sync_reich_forecast("YYG-ParamSearch", "yyg")


def sync_los_alamos():
    url = "https://covid-19.bsvgateway.org"
    req = requests.get(f"{url}/forecast/forecast_metadata.json").json()

    def fmt(df_):
        df = df_.rename(
            columns={"dates": "date", "q.50": "counts", "state": "loc2", "name": "loc2"}
        )[["date", "counts", "loc2"]]
        df["loc1"] = "United States"
        df["forecast_date"] = pandas.to_datetime(df_["fcst_date"].unique().item())
        df["date"] = pandas.to_datetime(df["date"])
        df["id"] = "los_alamos"
        return df

    for date in req["us"]["files"].keys():
        cases = fmt(
            pandas.read_csv(
                os.path.join(
                    url,
                    req["us"]["files"][date]["quantiles_confirmed_daily"].lstrip("./"),
                )
            )
        )
        deaths = fmt(
            pandas.read_csv(
                os.path.join(
                    url, req["us"]["files"][date]["quantiles_deaths_daily"].lstrip("./")
                )
            )
        )
        to_csv(deaths, "deaths", "los_alamos", deaths["forecast_date"].iloc[0])
        to_csv(cases, "infections", "los_alamos", cases["forecast_date"].iloc[0])


def sync_jhu():
    data_pth = update_repo("https://github.com/CSSEGISandData/COVID-19.git")
    col_map = {
        "Country/Region": "loc1",
        "Province/State": "loc2",
        "Last Update": "date",
        "Last_Update": "date",
        "Admin2": "loc3",
        "Province_State": "loc2",
        "Country_Region": "loc1",
    }
    for file in glob(
        f"{data_pth}/csse_covid_19_data/csse_covid_19_daily_reports/*.csv"
    ):
        print(file)
        df = pandas.read_csv(file)
        df = df.rename(columns=col_map)
        df["date"] = pandas.to_datetime(df["date"])
        df["id"] = "jhu_ground_truth"
        df["date"] = df["date"].dt.date
        df["loc3"] = None
        to_csv(
            df[~df["Confirmed"].isnull()].rename(columns={"Confirmed": "counts"}),
            "infections",
            "jhu_ground_truth",
            "",
        )
        to_csv(
            df[~df["Deaths"].isnull()].rename(columns={"Deaths": "counts"}),
            "deaths",
            "jhu_ground_truth",
            "",
        )


def sync_columbia():
    data_dir = update_repo("git@github.com:shaman-lab/COVID-19Projection.git")
    fips_dir = update_repo("git@github.com:kjhealy/fips-codes.git")
    fips = pandas.read_csv(
        f"{fips_dir}/county_fips_master.csv", encoding="latin1", dtype={"fips": str}
    )
    fips["fips"] = fips["fips"].apply(lambda x: x.zfill(5))
    fips = fips.drop_duplicates(["fips"])
    usa_facts = pandas.read_csv(
        "/checkpoint/mattle/covid19/csvs/infections/usafacts_ground_truth/counts.csv",
        parse_dates=["date"],
    )
    usa_facts = usa_facts.melt(
        id_vars=["date"], value_name="counts", var_name="location"
    )
    usa_facts["loc2"] = usa_facts["location"].apply(lambda x: x.split(", ")[1])
    usa_facts["loc3"] = usa_facts["location"].apply(lambda x: x.split(", ")[0])
    for file in glob(f"{data_dir}/Projection_*/Projection_*.csv"):
        print(file)
        df = pandas.read_csv(
            file, encoding="latin1", dtype={"fips": str}, parse_dates=["Date"]
        )
        df["fips"] = df["fips"].str.zfill(5)
        # Convert to cumulative counts
        df = df.pivot(index="Date", columns="fips", values="report_50").cumsum()
        df = df.reset_index().melt(id_vars=["Date"], value_name="report_50")

        merged = df.merge(fips[["fips", "county_name", "state_name"]], on="fips")
        merged = merged.rename(
            columns={
                "report_50": "counts",
                "state_name": "loc2",
                "county_name": "loc3",
                "Date": "date",
            }
        )
        merged = merged[["counts", "loc2", "loc3", "date"]]
        merged["loc1"] = "United States"
        merged["loc3"] = merged["loc3"].str.replace(" (County|Municipality|Parish)", "")
        merged["forecast_date"] = merged["date"].min()

        first_day = merged["date"].min()
        prev_day = usa_facts[usa_facts["date"] == first_day - timedelta(days=1)]

        with_base = merged.merge(prev_day, on=["loc2", "loc3"], suffixes=("", "_y"))
        with_base["counts"] += with_base["counts_y"]
        name = re.search("Projection_(.*).csv", os.path.basename(file)).group(1)
        with_base["id"] = f"columbia_{name}"
        with_base = with_base[
            ["loc1", "loc2", "loc3", "date", "forecast_date", "id", "counts"]
        ]
        with_base = pandas.concat([with_base, prev_day])
        with_base["forecast_date"] = first_day
        to_csv(with_base, "infections", f"columbia_{name}", first_day)


def sync_usa_facts():
    fips_dir = update_repo("git@github.com:kjhealy/fips-codes.git")
    fips = pandas.read_csv(
        f"{fips_dir}/county_fips_master.csv", encoding="latin1", dtype={"fips": str}
    )
    fips["fips"] = fips["fips"].str.zfill(5)
    for table, url in [
        (
            "infections",
            "https://usafactsstatic.blob.core.windows.net/public/data/covid-19/covid_confirmed_usafacts.csv",
        ),
        (
            "deaths",
            "https://usafactsstatic.blob.core.windows.net/public/data/covid-19/covid_deaths_usafacts.csv",
        ),
    ]:
        df = pandas.read_csv(url, dtype={"countyFIPS": str})
        df["countyFIPS"] = df["countyFIPS"].str.zfill(5)
        df = df.set_index("countyFIPS")
        df = df[[c for c in df.columns if re.match(r"\d+-\d+-\d+", c)]].transpose()
        df = (
            df.reset_index()
            .melt(id_vars=["index"])
            .rename(columns={"index": "date", "countyFIPS": "fips", "value": "counts"})
        )
        df["date"] = pandas.to_datetime(df["date"])
        df = df.merge(fips, on=["fips"])[
            ["date", "counts", "state_name", "county_name"]
        ]
        df = df.rename(columns={"state_name": "loc2", "county_name": "loc3"})
        df["loc1"] = "United States"
        df["loc3"] = df["loc3"].str.replace(" (County|Municipality|Parish)", "")
        df["id"] = "usafacts_ground_truth"
        to_csv(df, table, "usafacts_ground_truth", "")


def sync_google():
    for metric in ["cases", "deaths"]:
        df = get_google(metric=metric)
        df = df.reset_index().melt(
            id_vars=["date"], value_name="counts", var_name="loc"
        )
        df["loc1"] = "United States"
        df["loc2"] = df["loc"].apply(lambda x: x.split("_")[0])
        df["loc3"] = df["loc"].apply(lambda x: x.split("_")[1])
        df["id"] = "google_ground_truth"
        table = "infections" if metric == "cases" else metric
        to_csv(df, table, "google_ground_truth", "")


datasets = {
    "usa_facts": sync_usa_facts,
    "columbia": sync_columbia,
    "nyt": sync_nyt,
    "los_alamos": sync_los_alamos,
}


@click.command()
@click.option("--dataset", default=None, type=click.Choice(datasets.keys()))
def sync_forecasts(dataset=None):
    if dataset is not None:
        datasets[dataset]()
    else:
        for f in datasets.values():
            f()


if __name__ == "__main__":
    cli.add_command(sync_forecasts)
    cli()
