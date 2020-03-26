##
# COVID-19 Spread
#
# @file
# @version 0.1

.PHONY: timelord env

env:
	conda env create -f environment.yml

timelord:
	git submodule update --init --recursive
	cd timelord && CC="ccache gcc" python3 setup.py build -f develop

forecast-nj: data/new-jersey/timeseries.h5
	OMP_NUM_THREADS=1 python3 train.py \
		-dset data/new-jersey/timeseries.h5 \
		-dim 15 \
		-epochs 200 \
		-maxcor 25 \
		-max-events 5000 \
		-sparse \
		-scale 1 \
		-optim lbfgs \
		-weight-decay 0 \
		-timescale 1 \
		-quiet -fresh
	python3 forecast.py -dset data/new-jersey/timeseries.h5 -checkpoint /tmp/timelord_model.bin -basedate 20200325
# end
