# see https://www.gov.uk/government/publications/covid-19-track-coronavirus-cases
data.csv:
	wget -O data.xlsx https://www.arcgis.com/sharing/rest/content/items/e5fd11150d274bebaaf8fe2a7a2bda11/data
	in2csv data.xlsx -f xlsx > data.csv

timeseries.h5: data.csv
	python3 process_cases.py

sync: timeseries.h5
	mkdir -p /checkpoint/${USER}/data/covid19
	cp timeseries.h5 /checkpoint/${USER}/data/covid19/gb.h5

clean:
	rm -f timeseries.h5 data.csv data.xlsx
