##
# Project Title
#
# @file
# @version 0.1

#PREV=NYU-data-with-date.csv
#DATE=0324

PREV=data-$(shell date "+%Y%m%d" --date "yesterday").csv
DATE=$(shell date "%m%d")

data-2020$(DATE).csv: $(PREV) nj-official-$(DATE).csv
	python3 merge.py $(PREV) nj-official-$(DATE).csv 2020$(DATE)

timeseries.h5: data-2020$(DATE).csv
	python3 process_cases.py $<

# end
