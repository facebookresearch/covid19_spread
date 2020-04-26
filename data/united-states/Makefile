.PHONY: data.csv timeseries.csv

raw.csv:
	curl --output $@ https://raw.githubusercontent.com/nytimes/covid-19-data/master/us-counties.csv 

data.csv: raw.csv
	python3 convert.py

timeseries.h5: data.csv
	python3 process_cases.py

sync: timeseries.h5
	mkdir -p /checkpoint/${USER}/data/covid19
	cp timeseries.h5 /checkpoint/${USER}/data/covid19/us.h5

clean:
	rm -f timeseries.h5
	rm -f data.csv
