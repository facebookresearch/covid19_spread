data.csv:
	curl --output $@ https://raw.githubusercontent.com/datadista/datasets/master/COVID%2019/ccaa_covid19_casos_long.csv

timeseries.h5: data.csv
	python3 process_cases.py

sync: timeseries.h5
	mkdir -p /checkpoint/${USER}/data/covid19
	cp timeseries.h5 /checkpoint/${USER}/data/covid19/es.h5

clean:
	rm -f timeseries.h5
	rm -f data.csv
