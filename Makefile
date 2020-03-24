##
# COVID-19 Spread
#
# @file
# @version 0.1

example:
	OMP_NUM_THREADS=1 python3 train.py -dset /checkpoint/${USER}/data/covid19/nj.h5 -fresh -dim 15 -max-events 5000 -sparse -scale 1 -optim lbfgs -weight-decay 0  -timescale 1

# end
