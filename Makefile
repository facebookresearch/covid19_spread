##
# COVID-19 Spread
#
# @file
# @version 0.1

.PHONY: timelord env

params=-max-events 5000 -sparse -scale 1 -optim lbfgs -weight-decay 0 -timescale 1 -quiet -fresh -epochs 200 -maxcor 25
DATE=20200328

define forecast-train
	echo "\n--- DIM=$(1) $(if $3,NO-BASEINT)" >> $(flog)
	OMP_NUM_THREADS=1 python3 rmse.py $(params) -dset $(2) -dim $(1) $(if $3,$3)| grep "^RMSE" >> $(flog)
	OMP_NUM_THREADS=1 python3 train.py $(params) -dset $(2) -dim $(1) $(if $3,$3)
	python3 forecast.py -dset $(2) -checkpoint /tmp/timelord_model.bin -basedate $(DATE) -trials 0 | grep "=" >> $(flog)
endef

env:
	conda env create -f environment.yml

timelord:
	git submodule update --init --recursive
	cd timelord && CC="ccache gcc" python3 setup.py build -f develop

example: data/new-jersey/timeseries.h5
	OMP_NUM_THREADS=1 python3 train.py -max-events 5000 -sparse -scale 1 -optim lbfgs -weight-decay 0 -timescale 1 -quiet -fresh -dset data/new-jersey/timeseries.h5 -epochs 200 -maxcor 25 -dim 15
	python3 forecast.py -dset data/new-jersey/timeseries.h5 -checkpoint /tmp/timelord_model.bin -basedate 20200325

forecast-nj: flog="forecasts/new-jersey/forecast-$(DATE).log"
	python3 sir.py -fdat data/new-jersey/data-$(DATE).csv -fpop data/population-data/US-states/new-jersey-population.csv -fsuffix nj-$(DATE) -dout forecasts/new-jersey -days 60 -keep 7 -window 5 -doubling-times 2 3 4 10

	echo "Forecast $(DATE)" > $(flog)
	$(call forecast-train,5,"data/new-jersey/timeseries.h5")
	$(call forecast-train,10,"data/new-jersey/timeseries.h5")
	$(call forecast-train,15,"data/new-jersey/timeseries.h5")
	$(call forecast-train,20,"data/new-jersey/timeseries.h5")

	$(call forecast-train,5,"data/new-jersey/timeseries.h5",-no-baseint)
	$(call forecast-train,10,"data/new-jersey/timeseries.h5",-no-baseint)
	$(call forecast-train,15,"data/new-jersey/timeseries.h5",-no-baseint)
	$(call forecast-train,20,"data/new-jersey/timeseries.h5",-no-baseint)

# end
