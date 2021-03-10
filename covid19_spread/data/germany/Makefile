##
# Project Title
#
# @file
# @version 0.1

data.csv:
	curl --output $@ https://raw.githubusercontent.com/iceweasel1/COVID-19-Germany/master/germany_with_source.csv

gemeindeverzeichnis.csv:
	curl --output $@ https://github.s3.amazonaws.com/downloads/digineo/gemeindeverzeichnis/gemeindeverzeichnis.csv

timeseries.h5: data.csv gemeindeverzeichnis.csv
	python3 process_cases.py

sync: timeseries.h5 kreise.shp
	mkdir -p /checkpoint/${USER}/data/covid19
	cp timeseries.h5 /checkpoint/${USER}/data/covid19/de.h5
	cp kreise.shp /checkpoint/${USER}/data/covid19/de.shp
	cp kreise.shx /checkpoint/${USER}/data/covid19/de.shx
	cp kreise.dbf /checkpoint/${USER}/data/covid19/de.dbf
	cp kreise.prj /checkpoint/${USER}/data/covid19/de.prj

clean:
	rm timeeries.h5
	rm data.csv

# end
