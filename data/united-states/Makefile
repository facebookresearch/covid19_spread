data.csv:
	curl --output $@ https://raw.githubusercontent.com/nytimes/covid-19-data/master/us-counties.csv 

timeseries.h5: data.csv
	python3 process_cases.py

sync: timeseries.h5
	mkdir -p /checkpoint/${USER}/data/covid19
	cp timeseries.h5 /checkpoint/${USER}/data/covid19/us.h5

clean:
	rm -f timeseries.h5
	rm -f data.csv
