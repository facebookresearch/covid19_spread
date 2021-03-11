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
