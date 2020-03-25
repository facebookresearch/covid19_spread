#!/usr/bin/env python3

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
from evaluation import ks_critical_value

# plotting imports
from bokeh.io import show, export_svgs, export_png
from bokeh.layouts import gridplot
from bokeh.models import HoverTool, GeoJSONDataSource, LinearColorMapper
from bokeh.palettes import brewer
from bokeh.plotting import figure
from bokeh.themes import built_in_themes


def qqplot(name, vks):
    ks = kstest(vks, "expon")
    # (osm, osr), (slope, intercept, r) = probplot(vks, dist='expon', fit=True)
    probplot = sm.ProbPlot(vks, stats.expon, fit=True, loc=0, scale=1)
    x = probplot.theoretical_quantiles
    y = probplot.sample_quantiles
    q25 = stats.scoreatpercentile(y, 25)
    q75 = stats.scoreatpercentile(y, 75)
    theoretical_quartiles = stats.expon.ppf([0.25, 0.75])
    m = (q75 - q25) / np.diff(theoretical_quartiles)
    b = q25 - m * theoretical_quartiles[0]
    # ax.plot(x, m*x + b, fmt)
    crit = ks_critical_value(len(vks), 0.05)
    p = figure(title=f"{name} KS={ks[0]:.3f} ({crit:.3f}) p={ks[1]:.4f}")
    # p = figure(title=f'QQ-plot {name}')
    # p.circle(osm, osr, color='blue', legend_label='Model')
    p.circle(x, y, color="blue", legend_label="Model")
    p.circle(x, m * x + b, color="red", legend_label="Exp(1)")
    # p.circle(osm, slope*osm + intercept, color='red', legend_label='Exp(1)')
    p.legend.location = "top_left"
    p.output_backend = "svg"
    return (p, ks)


def plot_district(name):
    _ts, _ws = per_district[name]
    _ts = pd.to_datetime(_ts, unit="s")
    p = figure(
        title=f"{name}, cases={len(_ts)}, tmin={_ts[0].month}/{_ts[0].day}, tmax={_ts[-1].month}/{_ts[-1].day}",
        tools="save",
        x_axis_type="datetime",
        x_axis_label="Date",
        y_axis_label="Confirmed cases (cumul.)",
        plot_height=300,
        plot_width=400,
    )
    # p.line(_ts, _ws.cumsum(), line_width=3)
    p.line(_ts, np.ones(len(_ts)).cumsum(), line_width=3)
    p.add_tools(
        HoverTool(
            tooltips=[("Date", "$x"), ("Cases", "$y")],
            formatters={"$x": "datetime"},
            mode="vline",
        )
    )
    p.xaxis.major_label_orientation = np.pi / 4
    show(p)


def plot_influence(name):
    x = np.where(nodes == name)[0][0]
    vals = [0 for _ in range(len(geoags))]
    for i, n in enumerate(nodes):
        if i != x:
            vals[geoags[n.lower()]] = A[i, x]
    vals = np.array(vals)
    palette = brewer["Blues"][8]
    palette = palette[::-1]
    geodat["vals"] = vals
    geosource = GeoJSONDataSource(geojson=geodat.to_json())
    geohome = GeoJSONDataSource(
        geojson=(geodat[geodat["COUNTY"] == name.upper()]).to_json()
    )
    color_mapper = LinearColorMapper(palette=palette, low=vals.min(), high=vals.max())
    p = figure(
        title=f"Influence of {name}",
        plot_height=800,
        plot_width=600,
        tools="hover,save",
        x_axis_location=None,
        y_axis_location=None,
    )
    p.patches(
        "xs",
        "ys",
        source=geosource,
        line_width=0.5,
        fill_color={"field": "vals", "transform": color_mapper},
    )
    p.patches("xs", "ys", source=geohome, line_width=0.5, fill_color="red")
    hover = p.select_one(HoverTool)
    hover.point_policy = "follow_mouse"
    hover.tooltips = [("District", "@COUNTY"), ("Expected Offspring", "@vals")]
    p.grid.grid_line_color = None
    return p
