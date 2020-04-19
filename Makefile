##
# COVID-19 Spread
#
# @file
# @version 0.1

.PHONY: timelord env

params=-max-events 5000 -sparse -scale 1 -optim lbfgs -weight-decay 0 -timescale 1 -quiet -fresh -epochs 200 -maxcor 25
DATE=$(shell date "+%Y%m%d")
LAST=1
SARGS=--sort-by RMSE_AVG

define forecast-train
	echo "\n--- DIM=$(1) $(if $5,NO-BASEINT)" >> $(flog)
	OMP_NUM_THREADS=1 python3 rmse.py $(params) -dset data/$(2)/timeseries.h5 -checkpoint /tmp/timelord_model_$(2).bin -dim $(1) $(if $5,$5)| grep "^RMSE" >> $(flog)
	OMP_NUM_THREADS=1 python3 train.py $(params) -dset data/$(2)/timeseries.h5 -checkpoint /tmp/timelord_model_$(2).bin -dim $(1) $(if $5,$5)
	python3 forecast.py -dset data/$(2)/timeseries.h5 -checkpoint /tmp/timelord_model_$(2).bin -basedate $(DATE) -days $(3) -trials $(4) | grep "=\|ALL REGIONS\|county" >> $(flog)
endef

define _analyze_sweep
	@echo "---- Naive forecast ---------"
	python3 naive.py $(fdata).h5 $(DATE)
	python3 naive.py $(fdata)_smooth.h5 $(DATE)

	@echo "\n---- Summary ---------"
	python3 analyze_sweep.py summary $(sweepdir) --sort-by="$(if $(SORT),$(SORT),RMSE_AVG)"
	@echo "\n---- SIR Similarity ----------"
	python3 analyze_sweep.py sir-similarity $(sweepdir)
	@echo "\n---- Remaining Jobs ----------"
	squeue -u $(USER)
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

grid-nj: runlog = runs/new-jersey/$(DATE).log
grid-nj:
	touch $(runlog)
	python3 sweep.py grids/new-jersey.yml -remote -ncpus 40 -timeout-min 30 | tail -1 >> $(runlog)
	tail -1 $(runlog)

forecast-nj: params = -max-events 500000 -sparse -scale 1 -optim lbfgs -weight-decay 0 -timescale 1 -quiet -fresh -epochs 200 -maxcor 25
forecast-nj: doubling-times = 4 5 6 10
forecast-nj: dset = data/new-jersey/timeseries.h5
forecast-nj:
	python3 sir.py -fdat data/new-jersey/timeseries.h5 -fpop data/population-data/US-states/new-jersey-population.csv -fsuffix nj-$(DATE) -dout forecasts/new-jersey -days 60 -keep 7 -window 5 -doubling-times $(doubling-times)
	OMP_NUM_THREADS=1 python3 train.py $(params) -dset $(dset) -checkpoint /tmp/forecast_nj.bin  $(TARGS)
	OPENBLAS_MAIN_FREE=1 python3 forecast.py -dset $(dset) -checkpoint /tmp/forecast_nj.bin -basedate $(DATE) -trials 50 -days 7 -fout forecasts/new-jersey/forecast-nj-$(DATE)$(FSUFFIX).csv


analyze-nj: sweepdir = $(shell tail -$(LAST) runs/new-jersey/$(DATE).log | head -n1)
analyze-nj: fdata = data/new-jersey/timeseries
analyze-nj:
	$(call _analyze_sweep)

mae-nj:
	@echo "--- MAE Slow ---"
	python3 mae.py data/new-jersey/timeseries.h5 forecasts/new-jersey/forecast $(DATE) _slow
	@echo "\n--- MAE Fast ---"
	python3 mae.py data/new-jersey/timeseries.h5 forecasts/new-jersey/forecast $(DATE) _fast

data-nj: data/new-jersey/nj-official-$(shell date "+%m%d" -d $(DATE)).csv
	cd data/new-jersey && make data-$(DATE).csv && make timeseries.h5 && make timeseries.h5 SMOOTH=1

# --- NY State ---

forecast-nys: params = -max-events 1000000 -sparse -scale 1 -optim lbfgs -weight-decay 0 -timescale 1 -quiet -fresh -epochs 200
forecast-nys: doubling-times = 12 13 14 15
forecast-nys: dset = data/nystate/timeseries-nys.h5
forecast-nys:
	python3 sir.py -fdat data/nystate/timeseries.h5 -fpop data/population-data/US-states/new-york-population.csv -fsuffix ny-$(DATE) -dout forecasts/nys -days 60 -keep 7 -window 5 -doubling-times $(doubling-times)
	OMP_NUM_THREADS=1 python3 train.py $(params) -dset $(dset) $(TARGS) -checkpoint /tmp/forecast_nystate.bin
	OPENBLAS_MAIN_FREE=1 python3 forecast.py -dset $(dset) -checkpoint /tmp/forecast_nystate.bin -basedate $(DATE) -trials 50 -days 7 -fout forecasts/nystate/forecast-ny-$(DATE)$(FSUFFIX).csv

forecast-nyc: params = -max-events 1000000 -sparse -scale 1 -optim lbfgs -weight-decay 0 -timescale 1 -quiet -fresh -epochs 200
forecast-nyc: doubling-times = 12 13 14 15
forecast-nyc: dset = data/nystate/timeseries-nyc.h5
forecast-nyc:
	python3 sir.py -fdat data/nystate/timeseries.h5 -fpop data/population-data/US-states/new-york-population.csv -fsuffix ny-$(DATE) -dout forecasts/nys -days 60 -keep 7 -window 5 -doubling-times $(doubling-times)
	OMP_NUM_THREADS=1 python3 train.py $(params) -dset $(dset) $(TARGS) -checkpoint /tmp/forecast_nyc.bin
	OPENBLAS_MAIN_FREE=1 python3 forecast.py -dset $(dset) -checkpoint /tmp/forecast_nyc.bin -basedate $(DATE) -trials 50 -days 7 -fout forecasts/nyc/forecast-$(DATE)$(FSUFFIX).csv


grid-nyc: runlog = runs/nyc/$(DATE).log
grid-nyc:
	touch $(runlog)
	python3 sweep.py grids/nyc.yml -remote -ncpus 40 -timeout-min 30 | tail -1 >> $(runlog)
	tail -1 $(runlog)


grid-nys: runlog = runs/nys/$(DATE).log
grid-nys:
	touch $(runlog)
	python3 sweep.py grids/nys.yml -remote -ncpus 40 -timeout-min 120 | tail -1 >> $(runlog)
	tail -1 $(runlog)


analyze-nyc: sweepdir = $(shell tail -$(LAST) runs/nyc/$(DATE).log | head -n1)
analyze-nyc: fdata = data/nystate/timeseries-nyc
analyze-nyc:
	$(call _analyze_sweep)


analyze-nys: sweepdir = $(shell tail -$(LAST) runs/nys/$(DATE).log | head -n1)
analyze-nys: fdata = data/nystate/timeseries-nys
analyze-nys: doubling-times = 10 11 12 13
analyze-nys:
	python3 sir.py -fdat data/nystate/timeseries.h5 -fpop data/population-data/US-states/new-york-population.csv -fsuffix ny-$(DATE) -dout forecasts/nys -days 60 -keep 7 -window 5 -doubling-times $(doubling-times)
	$(call _analyze_sweep)

mae-ny:
	@echo "--- MAE Slow ---"
	python3 mae.py data/nystate/timeseries.h5 forecasts/nystate/forecast $(DATE) _slow
	@echo "\n--- MAE Fast ---"
	python3 mae.py data/nystate/timeseries.h5 forecasts/nystate/forecast $(DATE) _fast

data-ny:
	cd data/nystate && make data-nj.csv && make timeseries.h5 timeseries-nyc.h5 timeseries-nys.h5 && make timeseries.h5 timeseries-nyc.h5 timeseries-nys.h5 SMOOTH=1


select: fout = forecasts/$(REGION)/forecast-$(DATE)$(SUFFIX).csv
select: fout_sir = forecasts/$(REGION)/SIR-forecast-$(DATE).csv
select: license = "\nThis work is licensed under the Creative Commons Attribution-Noncommercial 4.0 International Public License (CC BY-NC 4.0). To view a copy of this license go to https://creativecommons.org/licenses/by-nc/4.0/. Retention of the foregoing language is sufficient for attribution."
select:
	cat $(JOB)/forecasts.csv | grep -v "^KS," | grep -v "^pval," > $(fout)
	echo $(license) >> $(fout)
	cp $(JOB)/../sir/SIR-forecast-$(REGION).csv $(fout_sir)
	echo $(license) >> $(fout_sir)
# end
