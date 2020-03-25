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

example:
	OMP_NUM_THREADS=1 python3 train.py -dset /checkpoint/${USER}/data/covid19/nj.h5 -fresh -dim 15 -max-events 5000 -sparse -scale 1 -optim lbfgs -weight-decay 0  -timescale 1
	python3 forecast.py
# end
