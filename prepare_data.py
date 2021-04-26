#!/usr/bin/env python3
# Copyright (c) 2021-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import click
from covid19_spread.data.usa.convert import main as us_convert, SOURCES as US_SOURCES
import os

SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))


@click.group()
def cli():
    pass


@cli.command()
@click.option("--metric", default="cases", type=click.Choice(["cases", "deaths"]))
@click.option("--with-features", is_flag=True)
@click.option("--source", default="nyt", type=click.Choice(US_SOURCES.keys()))
@click.option("--resolution", default="county", type=click.Choice(["county", "state"]))
def us(metric, with_features, source, resolution):
    us_convert(metric, with_features, source, resolution)


if __name__ == "__main__":
    cli()
