#!/usr/bin/env python3
# Copyright (c) 2021-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import click
from covid19_spread.data.usa import us_recurring


@click.group()
def cli():
    pass


REGIONS = {"us": us_recurring.USARRecurring}


@cli.command()
@click.argument("region", type=click.Choice(REGIONS.keys()))
def install(region):
    mod = REGIONS[region]()
    mod.install()


@cli.command()
@click.argument("region", type=click.Choice(REGIONS.keys()))
def run(region):
    mod = REGIONS[region]()
    mod.refresh()


if __name__ == "__main__":
    cli()
