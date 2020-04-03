##
# COVID-19 Spread
#
# @file
# @version 0.1

.PHONY: timelord env

params=-max-events 5000 -sparse -scale 1 -optim lbfgs -weight-decay 0 -timescale 1 -quiet -fresh -epochs 200 -maxcor 25
DATE=$(shell date "+%Y%m%d")

define forecast-train
	echo "\n--- DIM=$(1) $(if $5,NO-BASEINT)" >> $(flog)
	OMP_NUM_THREADS=1 python3 rmse.py $(params) -dset data/$(2)/timeseries.h5 -checkpoint /tmp/timelord_model_$(2).bin -dim $(1) $(if $5,$5)| grep "^RMSE" >> $(flog)
	OMP_NUM_THREADS=1 python3 train.py $(params) -dset data/$(2)/timeseries.h5 -checkpoint /tmp/timelord_model_$(2).bin -dim $(1) $(if $5,$5)
	python3 forecast.py -dset data/$(2)/timeseries.h5 -checkpoint /tmp/timelord_model_$(2).bin -basedate $(DATE) -days $(3) -trials $(4) | grep "=\|ALL REGIONS\|county" >> $(flog)
endef

env:
	conda env create -f environment.yml

timelord:
	git submodule update --init --recursive
	cd timelord && CC="ccache gcc" python3 setup.py build -f develop

example: data/new-jersey/timeseries.h5
	OMP_NUM_THREADS=1 python3 train.py -max-events 5000 -sparse -scale 1 -optim lbfgs -weight-decay 0 -timescale 1 -quiet -fresh -dset data/new-jersey/timeseries.h5 -epochs 200 -maxcor 25 -dim 15
	python3 forecast.py -dset data/new-jersey/timeseries.h5 -checkpoint /tmp/timelord_model.bin -basedate $(DATE)

forecast-nj: flog = "forecasts/new-jersey/forecast-$(DATE).log"
forecast-nj:
	python3 sir.py -fdat data/new-jersey/timeseries.h5 -fpop data/population-data/US-states/new-jersey-population.csv -fsuffix nj-$(DATE) -dout forecasts/new-jersey -days 60 -keep 7 -window 5 -doubling-times 3 4 5 10

	echo "Forecast $(DATE)" > $(flog)
	$(call forecast-train,5,new-jersey,7,50)
	$(call forecast-train,10,new-jersey,7,50)
	$(call forecast-train,15,new-jersey,7,50)
	$(call forecast-train,20,new-jersey,7,50)

	$(call forecast-train,5,new-jersey,7,50,-no-baseint)
	$(call forecast-train,10,new-jersey,7,50,-no-baseint)
	$(call forecast-train,15,new-jersey,7,50,-no-baseint)
	$(call forecast-train,20,new-jersey,7,50,-no-baseint)

forecast-nyc: params = -max-events 1000000 -sparse -scale 1 -optim lbfgs -weight-decay 0 -timescale 1 -quiet -fresh -epochs 200 -maxcor 25
forecast-nyc: flog = "forecasts/new-york-city/forecast-$(DATE).log"
forecast-nyc:
	python3 sir.py -fdat data/new-york-city/timeseries.h5 -fpop data/population-data/US-states/new-york-city.csv -fsuffix nyc-$(DATE) -dout forecasts/new-york-city -days 60 -keep 7 -window 5 -doubling-times 3 4 5 10

	echo "Forecast $(DATE)" > $(flog)
	$(call forecast-train,2,new-york-city,7,50)
	$(call forecast-train,3,new-york-city,7,50)
	$(call forecast-train,5,new-york-city,7,50)

	$(call forecast-train,2,new-york-city,7,50,-no-baseint)
	$(call forecast-train,3,new-york-city,7,50,-no-baseint)
	$(call forecast-train,5,new-york-city,7,50,-no-baseint)

forecast-nystate: params = -max-events 1000000 -sparse -scale 1 -optim lbfgs -weight-decay 0 -timescale 1 -quiet -fresh -epochs 200 -maxcor 25
forecast-nystate: flog = "forecasts/nystate/forecast-$(DATE).log"
forecast-nystate:
	python3 sir.py -fdat data/nystate/timeseries.h5 -fpop data/population-data/US-states/new-york-population.csv -fsuffix nystate-$(DATE) -dout forecasts/nystate -days 60 -keep 7 -window 5 -doubling-times 4 5 6 10

	echo "Forecast $(DATE)" > $(flog)
	$(call forecast-train,10,nystate,7,30)
	$(call forecast-train,30,nystate,7,30)
	$(call forecast-train,60,nystate,7,30)

	$(call forecast-train,10,nystate,7,30,-no-baseint)
	$(call forecast-train,30,nystate,7,30,-no-baseint)
	$(call forecast-train,60,nystate,7,30,-no-baseint)

# end
