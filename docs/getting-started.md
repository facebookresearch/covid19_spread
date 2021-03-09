# Getting Started: Modeling Covid19 Spread

## Get Code

Clone repo: https://github.com/fairinternal/covid19_spread
Update submodules: `git pull --recurse-submodules`

## Setup Environment

On FAIR machines,
`module load anaconda3 cuda`

Otherwise, make sure you have anaconda installed. 
Then run,

```
make env
conda activate covid19_spread
```

## Get Data

```
`# get new york city data
cd data/nystate && make data-nyc.csv

# ny data
make data-ny

# nj data
make data-nj

# usa data: pull New York Times and joins with NY counties
make data-us

# usa deaths: generates data/usa/data_deaths.csv
cd data/usa && python convert.py deaths
```

# Training and Validation

## Run Sweep

`sweep.py` is the script used to generated the daily forecasts. To run:
`python sweep.py grids/nyc.yml`
This generates New York City forecasts.


## Run Cross Validation

Cross validation works by specifying parameters in `yaml` files. For example, `cv/ny.yml` for example. 

* before running. cv.py, make sure you generate the appropriate data first. 

To run `cv.py:`
`python cv.py cv/ny.yml sir`

The last argument is the model you wish to run. 

## Run SIR

`python sir.py -fdat data/nystate/timeseries.h5 -fpop data/population-data/US-states/new-york-population.csv -days 8 -keep 7 -window 10 -doubling-times 14 16`

# Historical Forecasts

Forecasts are stored in a [SQLite database](https://github.com/fairinternal/covid19_spread/blob/master/docs/forecast-db.md) and equivalent CSVs in ``/checkpoint/mattle/covid19/csvs``
