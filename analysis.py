#!/usr/bin/env python3

import numpy as np
import pandas as pd
import os
import yaml

from pathlib import Path

# bokeh
from bokeh.io import export_png, export_svgs
from bokeh.palettes import Blues3 as palette
from bokeh.models import (
    HoverTool,
    Title,
    FactorRange,
    LinearAxis,
    Legend,
    Band,
    Range1d,
    LabelSet,
    Span,
)
from bokeh.palettes import Blues
from bokeh.plotting import figure, output_file, ColumnDataSource
from bokeh.transform import factor_cmap

# lib
from metrics import _compute_metrics


def load_backfill(
    job,
    basedir="/checkpoint/maxn/covid19/forecasts",
    model="ar",
    indicator="final_model_validation*.csv",
    forecast="../forecasts/forecast_best_mae.csv",
):
    """collect all forcasts from job dir"""
    jobdir = os.path.join(basedir, job)
    forecasts = {}
    configs = []
    for path in Path(jobdir).rglob(indicator):
        date = str(path).split("/")[7]
        job = "/".join(str(path).split("/")[:-1])
        assert date.startswith("sweep_"), date
        date = date[6:]
        forecasts[date] = os.path.join(job, forecast)
        cfg = job + f"/{model}.yml"
        cfg = yaml.load(open(cfg), Loader=yaml.FullLoader)["train"]
        cfg["date"] = date
        cfg["job"] = job
        configs.append(cfg)
    configs = pd.DataFrame(configs)
    configs.set_index("date", inplace=True)
    return forecasts, configs


def plot_metric(mets, other, title, metric, height=350, weight=450):
    source = ColumnDataSource(mets)
    p = figure(
        x_axis_type="datetime",
        plot_height=height,
        plot_width=weight,
        title=f"Forecast Quality {title}",
        tools="save,hover",
        x_axis_label="Day",
        y_axis_label=metric,
    )
    p.extra_y_ranges = {
        "counts": Range1d(start=mets["counts"].min(), end=mets["counts"].max())
    }
    p.add_layout(LinearAxis(y_range_name="counts", axis_label="Deaths"), "right")

    l_ar = p.line(
        x="day",
        y="AR",
        source=source,
        line_width=3,
        color="black",
        legend_label="FAIR-AR",
    )
    l_na = p.line(
        x="day",
        y="Naive",
        source=source,
        line_width=3,
        color="#009ed7",
        legend_label="Naive",
        line_dash="dotted",
    )
    # l_ot = p.line(x='day', y=other, source=source, line_width=3, color='#009ed7', legend_label=other)
    p.line(
        x="day",
        y="counts",
        source=source,
        line_width=1,
        color="LightGray",
        line_alpha=0.2,
        y_range_name="counts",
    )
    band = Band(
        base="day",
        upper="counts",
        source=source,
        level="underlay",
        fill_alpha=0.5,
        fill_color="LightGray",
        y_range_name="counts",
    )
    p.add_layout(band)
    p.y_range.renderers = [l_ar, l_na]

    p.legend.location = "top_left"
    p.output_backend = "svg"
    p.background_fill_color = "white"
    p.border_fill_color = "white"
    p.outline_line_color = "white"
    p.title.text_font = "Montserrat"
    p.title.text_font_style = "normal"
    p.title.text_color = "#677b8c"
    return p


def plot_metric_for_dates(fs, df_gt, dates, other, model, metric="MAE", state=None):
    ps = []
    for date in dates:
        # df_other = load_predictions(f'/checkpoint/mattle/covid19/csvs/deaths/{model}/counts_{date}.csv').iloc[1:]
        df_ar = pd.read_csv(fs[date], index_col="date", parse_dates=["date"])
        if state is not None:
            # df_other = df_other[state].to_frame()
            df_ar = df_ar[state].to_frame()
            df_gt = df_gt[state].to_frame()

        # met_other = _compute_metrics(df_gt, df_other)
        met_ar = _compute_metrics(df_gt, df_ar)
        # display(met_ar)
        source = pd.DataFrame(
            {
                "Naive": met_ar.loc[f"{metric}_NAIVE"],
                # other: met_other.loc[metric],
                "AR": met_ar.loc[metric],
            }
        )
        source.index.set_names("day", inplace=True)
        source["counts"] = df_gt.loc[source.index].sum(axis=1)

        region = "US" if state is None else state
        p = plot_metric(source, other, f"{region} {date}", metric)
        ps.append(p)
    return ps


def plot_accuracy(mets, plevel, title, exclude={}):
    mets = mets[~mets["date"].isin(exclude)]
    mets["date"] = pd.to_datetime(mets["date"])
    mets = mets.sort_values(by="date")
    x = list(
        zip(map(lambda x: x.strftime("%m/%d"), mets["date"]), map(str, mets["days"]))
    )
    source = ColumnDataSource(data=dict(x=x, acc=mets["acc"].round(2)))

    p = figure(
        x_range=FactorRange(*x),
        plot_height=250,
        plot_width=700,
        title=f"Forecast Accuracy {title} - Confidence Level {plevel}",
        x_axis_label="Number of days forecasted on date",
        y_axis_label="Accuracy",
        tools="save",
        y_range=Range1d(start=0, end=1.1),
    )

    p.vbar(
        x="x",
        top="acc",
        width=0.85,
        source=source,
        line_color="black",
        fill_color=factor_cmap(
            "x", palette=palette, factors=["7", "14", "21"], start=1, end=2
        ),
    )
    labels = LabelSet(
        x="x",
        y="acc",
        text="acc",
        level="glyph",
        x_offset=-11.5,
        y_offset=0,
        source=source,
        text_font_size="0.85em",
        render_mode="canvas",
    )
    p.add_layout(labels)

    p.y_range.start = 0
    p.x_range.range_padding = 0.1
    p.xaxis.major_label_orientation = 1
    p.xgrid.grid_line_color = None
    p.output_backend = "svg"
    return p


def plot_cases(df, title, height=350, width=500, regions=None, backend="svg"):
    source = ColumnDataSource(df)
    p = figure(
        x_axis_type="datetime",
        plot_height=height,
        plot_width=width,
        title=title,
        tools="save",
        x_axis_label="Day",
        y_axis_label="Cases",
    )
    if regions is None:
        regions = df.columns
    for region in regions:
        p.line(x="date", y=region, source=source, line_width=2, color="#009ed7")
    p.output_backend = backend
    return p


def plot_prediction_interval(mean, lower, upper, df_gt, region, p, backend="svg"):
    n = len(df_gt)
    x = range(1, n + 1)
    df = pd.DataFrame({"x": x, "y": df_gt[region].values})
    source = ColumnDataSource(df)
    p.line(
        x="x",
        y="y",
        source=source,
        color="#009ed7",
        line_width=2,
        legend_label="Ground Truth",
        name=f"{region} ground truth",
    )

    y = mean[region].values
    df = pd.DataFrame(
        {"x": x, "y": y, "lower": lower[region].values, "upper": upper[region].values}
    )
    source = ColumnDataSource(df)
    p.line(
        x="x",
        y="y",
        source=source,
        line_width=2,
        color="black",
        alpha=0.5,
        line_dash="dotted",
        legend_label="Prediction",
        name=f"{region} prediction",
    )
    band = Band(
        base="x",
        lower="lower",
        upper="upper",
        source=source,
        level="underlay",
        fill_alpha=0.8,
        line_width=1,
        line_color="LightGray",
        fill_color="LightGray",
    )
    p.legend.location = "bottom_left"
    p.add_layout(band)
    p.output_backend = "svg"
    return p
