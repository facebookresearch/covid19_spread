##
# Covid-19 US counties
#
# @file
# @version 0.1

dout=/checkpoint/maxn/data/covid19

data.csv:
	curl --output $@ https://raw.githubusercontent.com/J535D165/CoronaWatchNL/master/data/rivm_corona_in_nl.csv

clean:
	rm data.csv

timeseries.h5: data.csv
	python3 process_cases.py

sync: timeseries.h5
	cp timeseries.h5 $(dout)/nl.h5
	cp Gemeentegrenzen_2019.shp $(dout)/nl.shp
	cp Gemeentegrenzen_2019.shx $(dout)/nl.shx
	cp Gemeentegrenzen_2019.prj $(dout)/nl.prj
	cp Gemeentegrenzen_2019.dbf $(dout)/nl.dbf
# end
