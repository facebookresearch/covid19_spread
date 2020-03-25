##
# Project Title
#
# @file
# @version 0.1

PREV=NYU-data-with-date.csv
DATE=0324

data-2020$(DATE).csv: $(PREV)
	python3 merge.py $(PREV) nj-official-$(DATE).csv 2020$(DATE)

timeseries.h5: data-2020$(DATE).csv
	python3 process_cases.py $<

# end
