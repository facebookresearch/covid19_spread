# Neural Relational Autoregression <br/> for High-Resolution COVID-19 Forecasting

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
