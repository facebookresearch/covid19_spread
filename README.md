# Neural Relational Autoregression <br/> for High-Resolution COVID-19 Forecasting

[![CircleCI](https://circleci.com/gh/facebookresearch/covid19_spread.svg?style=shield&circle-token=c8ca107a5135df4d7544141d105031dec491d83e)](https://circleci.com/gh/facebookresearch/covid19_spread)

This library provides code to forecast COVID-19 cases.  Details regarding the methods can be found in our [paper](https://ai.facebook.com/research/publications/neural-relational-autoregression-for-high-resolution-covid-19-forecasting)

![Forecast](https://github.com/facebookresearch/covid19_spread/blob/master/img/fair_model.gif?raw=true)


## Install Dependencies

``` sh
make env  # in the covid19_spread folder
conda activate covid19_spread
```

## Install library/scripts

```sh
pip install --editable .
```

## Preparing Data

Data for each region lives in `covid19_spread/data/<region>`.  Each
dataset is built using the `prepare-data` command the gets installed
by the above pip install.  For example, to prepare the U.S. data, simply run:

```
prepare-data us --with-features
```

The `--with-features` flag indicates to also assemble the covariate time feature data

## Train bAR Model on US data
To train b-AR using the cross validation pipeline on the latest data
`python cv.py cfg/us.yml bar`.

To execute jobs in parallel on a SLURM cluster, run add the `-remote` flag,
i.e., `python cv.py cfg/us.yml bar -remote -array-parallelism 20` runs
cross-valiation on 20 SLURM nodes in parallel.

To modify the model parameters, change the corresponding field in `cv/us.yml`.

## Back-Test bAR Model on past dates
To train b-AR using the cross validation pipeline on past dates run `python
cv.py backfill cfg/us.yml bar`. The `-remote` flag executes the backfill again on
a SLURM cluster.

Dates are specified using the command line `-dates` flag.  For example:

```
python cv.py backfill cfg/us.yml bar -dates 2020-11-01 -dates 2020-12-01
```

Will run a backfill for both 11/01/2020 and 12/01/2020

## References
If you are using this code in your work, please cite
```
@techreport{le2020covid19,
     title = {{Neural Relational Autoregression for High-Resolution COVID-19 Forecasting}},
     author = {Matthew Le and Mark Ibrahim and Levent Sagun and Timothee Lacroix and Maximilian Nickel},
     url = {https://ai.facebook.com/research/publications/neural-relational-autoregression-for-high-resolution-covid-19-forecasting/},
     month = {10},
     Date-Added = {2020-10-01},
     year = {2020}
}
```

## Forecasts
We provide continuous forecasts for all counties in the United States via the [Humanitarian Data Exchange](https://data.humdata.org/dataset/fair-covid-dataset). Past forecasts of our model are also available through the [COVID-19 Forecast Hub](https://github.com/reichlab/covid19-forecast-hub).

## Tests

To run tests:

```
python -m pytest tests/
```

To exclude integration tests:
```
python -m pytest tests/ -v -m "not integration" 
```

## Lint

We have Circle CI setup which will run our test suite as well as run the `black` linter.  
Please ensure that all code pushed passes the lint check.  You can install a Git pre-commit hook with:

```
pre-commit install
```

## License

This work is licensed under CC BY-NC. See LICENSE for details
