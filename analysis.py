#!/usr/bin/env python3

import numpy as np
import pandas as pd
import os
import yaml
import json
import nbformat

from pathlib import Path

# bokeh
from bokeh.io import export_png, export_svgs
from bokeh.io import show as bokeh_show
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

# Jupyter
from nbconvert import HTMLExporter
from nbconvert.preprocessors import ExecutePreprocessor
from IPython.display import Image

# lib
from metrics import _compute_metrics


def load_backfill(
    job,
    basedir=f"/checkpoint/{os.environ['USER']}/covid19/forecasts",
    model="ar",
    indicator="model_selection.json",
    forecast="best_rmse",
):
    """collect all forcasts from job dir"""
    jobdir = os.path.join(basedir, job)
    forecasts = {}
    configs = []
    for path in Path(jobdir).rglob(indicator):
        date = str(path).split("/")[-2]
        assert date.startswith("sweep_"), date
        jobs = [m["pth"] for m in json.load(open(path)) if m["name"] == forecast]
        assert len(jobs) == 1, jobs
        job = jobs[0]
        date = date[6:]
        forecasts[date] = os.path.join(job, f"../forecasts/forecast_{forecast}.csv")
        cfg = job + f"/{model}.yml"
        cfg = yaml.load(open(cfg), Loader=yaml.FullLoader)["train"]
        cfg["date"] = date
        cfg["job"] = job
        configs.append(cfg)
    configs = pd.DataFrame(configs)
    configs.set_index("date", inplace=True)
    return forecasts, configs


def export_notebook(nb_path, fout="notebook.html", no_input=False, no_prompt=False):
    os.environ["BOKEH_STATIC"] = "1"
    with open(nb_path, "r") as fin:
        nb = nbformat.read(fin, as_version=4)

    # exectute notebook
    ep = ExecutePreprocessor(timeout=600, kernel_name="python3")
    ep.preprocess(nb, {"metadata": {"path": "notebooks/"}})

    html_exporter = HTMLExporter()
    html_exporter.template_file = "basic"
    html_exporter.exclude_input = no_input
    html_exporter.exclude_input_prompt = no_prompt
    (body, resources) = html_exporter.from_notebook_node(nb)

    with open(fout, "w") as _fout:
        _fout.write(body)


def show(plot, path=None):
    if path is not None and os.environ.get("BOKEH_STATIC", 0) == "1":
        export_png(plot, filename=path)
        return Image(path)
    else:
        return bokeh_show(plot)


def plot_metric(mets, others, days, title, metric, height=350, weight=450):
    source = ColumnDataSource(mets)
    p = figure(
        x_axis_type="datetime",
        plot_height=height,
        plot_width=weight,
        title=f"Forecast Quality {title}",
        tools="save,hover",
        x_axis_label="Day",
        y_axis_label=metric,
        tooltips=[("Model", "$name"), (metric, "$y{0.0}")],
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
        legend_label="β-AR",
        name="β-AR",
    )
    l_na = p.line(
        x="day",
        y="Naive",
        source=source,
        line_width=3,
        color="#009ed7",
        legend_label="Naive",
        name="Naive",
        line_dash="dotted",
    )
    l_others = []
    for k, v in others.items():
        l_ot = p.line(
            x="day",
            y=k,
            source=source,
            line_width=3,
            color=v[1],
            line_dash=v[2],
            legend_label=k,
            name=k,
        )
    # p.line(
    #   x="day",
    #   y="counts",
    #   source=source,
    #   line_width=1,
    #   color="LightGray",
    #    line_alpha=0.2,
    #    y_range_name="counts",
    # )
    band = Band(
        base="day",
        upper="counts",
        source=source,
        level="underlay",
        fill_alpha=0.5,
        fill_color="LightGray",
        y_range_name="counts",
    )
    # p.add_layout(band)
    p.y_range.renderers = [l_ar, l_na] + l_others

    p.legend.location = "top_left"
    p.output_backend = "svg"
    p.background_fill_color = "white"
    p.border_fill_color = "white"
    p.outline_line_color = "white"
    p.title.text_font = "Montserrat"
    p.title.text_font_style = "normal"
    p.title.text_color = "#677b8c"
    return p


def plot_metric_for_dates(
    fs, df_gt, dates, metric="MAE", subregion=None, others={}, f_aggr=None
):
    ps = []
    for date in dates:
        if not os.path.exists(fs[date]):
            continue
        df_ar = pd.read_csv(fs[date], index_col="date", parse_dates=["date"])
        df_others = {
            k: pd.read_csv(v[0].format(date), index_col="date", parse_dates=["date"])
            for k, v in others.items()
        }
        if f_aggr is not None:
            df_ar = f_aggr(df_ar)
        if subregion is not None:
            df_ar = df_ar[subregion].to_frame()
            df_gt = df_gt[subregion].to_frame()
            for k, v in df_others.items():
                df_others[k] = v[subregion].to_frame()

        met_ar = _compute_metrics(df_gt, df_ar)
        days = met_ar.columns
        mets = {
            "Naive": met_ar.loc[f"{metric}_NAIVE"],
            "AR": met_ar.loc[metric],
        }
        # display(df_ar)
        for k, v in df_others.items():
            # display(v)
            # print(_compute_metrics(df_gt, v))
            mets[k] = _compute_metrics(df_gt, v).loc[metric][days]
        source = pd.DataFrame(mets)
        # display(source)
        source.index.set_names("day", inplace=True)
        source["counts"] = df_gt.loc[source.index].sum(axis=1)

        p = plot_metric(source, others, days, f"{date}", metric)
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


def plot_cases(
    df,
    title,
    height=350,
    width=500,
    color="#009ed7",
    line_width=2,
    alpha=0.5,
    regions=None,
    count_type="Cases",
    backend="svg",
    show_hover=False,
):
    source = ColumnDataSource(df)
    hover = HoverTool(
        tooltips=[("Region", "$name"), (count_type, "$y"), ("Day", "$x")],
        formatters={"$x": "datetime"},
    )
    p = figure(
        x_axis_type="datetime",
        plot_height=height,
        plot_width=width,
        title=title,
        tools=["save"],
        x_axis_label="Day",
        y_axis_label=count_type,
    )
    if show_hover:
        p.add_tools(hover)
    if regions is None:
        regions = df.columns
    for region in regions:
        p.line(
            x="date",
            y=region,
            source=source,
            line_width=line_width,
            color=color,
            alpha=alpha,
            name=region,
        )
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


def plot_error(df, gt, title, regions=None, height=400, width=600, backend="svg"):
    """
    Plot error of predictions per region over time

    Params
    ======
    - df: predictions (DataFrame date x region)
    - gt: ground truth (DataFrame date x region)
    - regions: subset of regions to plot (optional, List)
    - height: height of plot
    - width: width of plot
    - backend: bokeh plotting backend
    """
    ix = np.intersect1d(pd.to_datetime(df.index), pd.to_datetime(gt.index))
    source = ColumnDataSource(df.loc[ix] - gt.loc[ix])
    p = figure(
        x_axis_type="datetime",
        plot_height=height,
        plot_width=width,
        title=title,
        tools="save,hover",
        x_axis_label="Day",
        y_axis_label="Error",
        tooltips=[("State", "$name"), ("Error", "$y")],
    )
    if regions is None:
        regions = df.columns
    for region in regions:
        p.line(
            x="date",
            y=region,
            source=source,
            line_width=3,
            color="#009ed7",
            alpha=0.5,
            name=region,
        )
    p.output_backend = backend
    return p
