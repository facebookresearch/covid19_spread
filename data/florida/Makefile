##
# Covid-19 US counties
#
# @file
# @version 0.1

dout=/checkpoint/${USER}/data/covid19

data.csv:
	curl --output $@ https://opendata.arcgis.com/datasets/f5d69a918fb747019734d9a90cd602f4_0.csv

timeseries.h5: data.csv
	python3 process_cases.py

sync: timeseries.h5
	mkdir -p /checkpoint/${USER}/data/covid19
	cp timeseries.h5 /checkpoint/${USER}/data/covid19/florida.h5

clean:
	rm -f timeseries.h5
	rm -f data.csv

# end