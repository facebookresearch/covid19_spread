##
# COVID-19 Spread
#
# @file
# @version 0.1

.PHONY: env

params=-max-events 5000 -sparse -scale 1 -optim lbfgs -weight-decay 0 -timescale 1 -quiet -fresh -epochs 200 -maxcor 25
DATE=$(shell date "+%Y%m%d")
LAST=1
SARGS=--sort-by RMSE_AVG

env:
	conda env create -f environment.yml

data-nj: #data/new-jersey/nj-official-$(shell date "+%m%d" -d $(DATE)).csv
	# cd data/new-jersey && make data-$(DATE).csv && make timeseries.h5 && make timeseries.h5 SMOOTH=1
	cd data/new-jersey && python3 scraper.py && make timeseries.h5 && make timeseries.h5 SMOOTH=1

data-ny:
	cd data/nystate && make raw.csv && make timeseries.h5 timeseries-nyc.h5 timeseries-nys.h5 && make timeseries.h5 timeseries-nyc.h5 timeseries-nys.h5 SMOOTH=1 && make data-new.csv

# -- United States --
data-usa:
	cd data/usa && make data_cases.csv data_states_deaths.csv

aws: latest=$(shell aws s3 ls s3://fairusersglobal/users/mattle/h2/covid19_forecasts/ | tail -1 | cut -d' ' -f 6)
aws:
	aws s3 cp s3://fairusersglobal/users/mattle/h2/covid19_forecasts/$(latest) /tmp/
	bzip2 -zv9 /tmp/$(latest)
# end
