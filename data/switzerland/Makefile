data.csv:
	curl --output $@ https://raw.githubusercontent.com/openZH/covid_19/master/COVID19_Fallzahlen_CH_total.csv 

timeseries.h5: data.csv
	python3 process_cases.py

sync: timeseries.h5
	mkdir -p /checkpoint/${USER}/data/covid19
	cp timeseries.h5 /checkpoint/${USER}/data/covid19/es.h5

clean:
	rm -f timeseries.h5
	rm -f data.csv
