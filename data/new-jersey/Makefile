##
# Project Title
#
# @file
# @version 0.1


PREV=data-$(shell date "+%Y%m%d" --date "yesterday").csv
DATE=$(shell date "+%m%d")

# PREV=data-20200416.csv
# DATE=0417

.PHONY: timeseries.h5

data-2020$(DATE).csv: $(PREV) nj-official-$(DATE).csv
	python3 merge.py $(PREV) nj-official-$(DATE).csv 2020$(DATE)

timeseries.h5: data-2020$(DATE).csv
	python3 process_cases.py $<

# end
