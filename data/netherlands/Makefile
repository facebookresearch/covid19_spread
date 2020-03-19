##
# Covid-19 US counties
#
# @file
# @version 0.1

data.csv:
	curl --output $@ https://raw.githubusercontent.com/J535D165/CoronaWatchNL/master/data/rivm_corona_in_nl.csv

clean:
	rm data.csv

timeseries.h5: data.csv
	python3 process_cases.py

sync: timeseries.h5
	cp timeseries.h5 /checkpoint/maxn/data/covid19/nl.h5
# end
