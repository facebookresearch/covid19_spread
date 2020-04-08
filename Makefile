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
	git submodule sync
	git submodule update --init --recursive
	cd timelord && CC="ccache gcc" python3 setup.py build -f develop
	cd timelord && (rm tl_kernels.* || true)
	cd timelord && make kernels

example: data/new-jersey/timeseries.h5
	OMP_NUM_THREADS=1 python3 train.py -max-events 5000 -sparse -scale 1 -optim lbfgs -weight-decay 0 -timescale 1 -quiet -fresh -dset data/new-jersey/timeseries.h5 -epochs 200 -maxcor 25 -dim 15
	python3 forecast.py -dset data/new-jersey/timeseries.h5 -checkpoint /tmp/timelord_model.bin -basedate $(DATE)

# --- New Jersey ---

grid-nj-old: flog = "forecasts/new-jersey/forecast-$(DATE).log"
grid-nj-old:
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

grid-nj: runlog = runs/new-jersey/$(DATE).log
grid-nj:
	touch $(runlog)
	python3 sweep.py grids/new-jersey.yml -remote -ncpus 40 >> $(runlog)

forecast-nj: params = -max-events 500000 -sparse -scale 1 -optim lbfgs -weight-decay 0 -timescale 1 -quiet -fresh -epochs 200 -maxcor 25
forecast-nj: doubling-times = 4 5 16 0
forecast-nj: dset = data/new-jersey/timeseries.h5
forecast-nj:
	python3 sir.py -fdat data/new-jersey/timeseries.h5 -fpop data/population-data/US-states/new-jersey-population.csv -fsuffix nj-$(DATE) -dout forecasts/new-jersey -days 60 -keep 7 -window 5 -doubling-times $(doubling-times)
	OMP_NUM_THREADS=1 python3 train.py $(params) -dset $(dset) -checkpoint /tmp/forecast_nj.bin  $(TARGS)
	OPENBLAS_MAIN_FREE=1 python3 forecast.py -dset $(dset) -checkpoint /tmp/forecast_nj.bin -basedate $(DATE) -trials 50 -days 7 -fout forecasts/new-jersey/forecast-nj-$(DATE)$(FSUFFIX).csv


analyze-nj: sweepdir = $(shell tail -$(OFFSET) runs/new-jersey/$(DATE).log | head -n1)
analyze-nj:
	@echo "---- Summary ---------"
	-python3 analyze_sweep.py summary $(sweepdir) $(TARGS)
	@echo "\n---- SIR Similarity ----------"
	python3 analyze_sweep.py sir-similarity $(sweepdir)
	@echo "\n---- Remaining Jobs ----------"
	squeue -u $(USER)

# --- NYC ---

grid-nyc: params = -max-events 500000 -sparse -scale 1 -optim lbfgs -weight-decay 0 -timescale 1 -quiet -fresh -epochs 200 -maxcor 50
grid-nyc: flog = "forecasts/new-york-city/forecast-$(DATE).log"
grid-nyc:
	python3 sir.py -fdat data/new-york-city/timeseries.h5 -fpop data/population-data/US-states/new-york-city.csv -fsuffix nyc-$(DATE) -dout forecasts/new-york-city -days 60 -keep 7 -window 5 -doubling-times 3 4 5 10

	echo "Forecast $(DATE)" > $(flog)
	$(call forecast-train,2,new-york-city,7,50)
	$(call forecast-train,3,new-york-city,7,50)
	$(call forecast-train,4,new-york-city,7,50)

	$(call forecast-train,2,new-york-city,7,50,-no-baseint)
	$(call forecast-train,3,new-york-city,7,50,-no-baseint)
	$(call forecast-train,4,new-york-city,7,50,-no-baseint)

forecast-nyc:
	OMP_NUM_THREADS=1 python3 train.py -max-events 500000 -sparse -scale 1 -optim lbfgs -weight-decay 0 -timescale 1 -quiet -fresh -dset data/new-york-city/timeseries.h5 -epochs 200 -maxcor 50 $(TARGS)
	OPENBLAS_MAIN_FREE=1 python3 forecast.py -dset data/new-york-city/timeseries.h5 -checkpoint /tmp/timelord_model.bin -basedate $(DATE) -trials 50 -days 7 -fout forecasts/new-york-city/forecast-nyc-$(DATE)$(FSUFFIX).csv


# --- NY State ---

grid-nystate-old: params = -max-events 1000000 -sparse -scale 1 -optim lbfgs -weight-decay 0 -timescale 1 -quiet -fresh -epochs 200 -maxcor 50 -const-beta 80
grid-nystate-old: flog = "forecasts/nystate/forecast-$(DATE).log"
grid-nystate-old:
	python3 sir.py -fdat data/nystate/timeseries.h5 -fpop data/population-data/US-states/new-york-population.csv -fsuffix nystate-$(DATE) -dout forecasts/nystate -days 60 -keep 7 -window 5 -doubling-times 4 5 6 10

	echo "Forecast $(DATE)" > $(flog)
	$(call forecast-train,10,nystate,7,30)
	$(call forecast-train,30,nystate,7,30)
	$(call forecast-train,60,nystate,7,30)

	$(call forecast-train,10,nystate,7,30,-no-baseint)
	$(call forecast-train,30,nystate,7,30,-no-baseint)
	$(call forecast-train,60,nystate,7,30,-no-baseint)


forecast-nystate: params = -max-events 1000000 -sparse -scale 1 -optim lbfgs -weight-decay 0 -timescale 1 -quiet -fresh -epochs 200 -maxcor 50
forecast-nystate: doubling-times = 6 7 8 9
forecast-nystate: dset = data/nystate/timeseries.h5
forecast-nystate:
	python3 sir.py -fdat data/nystate/timeseries.h5 -fpop data/population-data/US-states/new-york-population.csv -fsuffix ny-$(DATE) -dout forecasts/nystate -days 60 -keep 7 -window 5 -doubling-times $(doubling-times)
	OMP_NUM_THREADS=1 python3 train.py $(params) -dset $(dset) $(TARGS) -checkpoint /tmp/forecast_nystate.bin
	OPENBLAS_MAIN_FREE=1 python3 forecast.py -dset $(dset) -checkpoint /tmp/forecast_nystate.bin -basedate $(DATE) -trials 50 -days 7 -fout forecasts/nystate/forecast-ny-$(DATE)$(FSUFFIX).csv


grid-nystate: runlog = runs/nystate/$(DATE).log
grid-nystate:
	touch $(runlog)
	python3 sweep.py grids/nystate.yml -remote -ncpus 40 | tail -1 >> $(runlog)
	tail -1 $(runlog)


analyze-nystate: sweepdir = $(shell tail -$(OFFSET) runs/nystate/$(DATE).log | head -n1)
analyze-nystate:
	@echo "---- Summary ---------"
	-python3 analyze_sweep.py summary $(sweepdir) $(TARGS)
	@echo "\n---- SIR Similarity ----------"
	python3 analyze_sweep.py sir-similarity $(sweepdir)
	@echo "\n---- Remaining Jobs ----------"
	squeue -u $(USER)
# end
