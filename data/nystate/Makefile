##
# Project Title
#
# @file
# @version 0.1
PREV = $(shell date "+%Y%m%d" -d "yesterday")
DATE = $(shell date "+%m%d")
DATE2 = $(shell date "+%Y-%m-%d")

data-20200330.csv: data.csv
	cp $< $@
	tail -n +2 ny-official-0330.csv | sed -E "s/(.*),(.*)/2020-03-30,\1,New York,,\2,/" >> $@

data-20200331.csv: data-20200330.csv
	cp $< $@
	tail -n +2 ny-official-0331.csv | sed -E "s/(.*),(.*)/2020-03-31,\1,New York,,\2,/" >> $@

data-20200401.csv: data-20200331.csv
	cp $< $@
	tail -n +2 ny-official-0401.csv | sed -E "s/(.*),(.*)/2020-04-01,\1,New York,,\2,/" >> $@

data-2020$(DATE).csv: data-$(PREV).csv
	cp $< data-2020$(DATE).csv
	tail -n +2 ny-official-$(DATE).csv | sed -E "s/(.*),(.*)/$(DATE2),\1,New York,,\2,/" >> $@

data-with-nyc-2020$(DATE).csv: data-2020$(DATE).csv
	cat $< | grep -v ",New York City," > $@
	tail -n +2 ../new-york-city/data-2020$(DATE).csv | gawk -v FPAT='[^,]*|("[^"]*")+' -v OFS="," '{print $$1,"Manhattan,,",$$4,"\n"$$1,"Bronx,,",$$5,"\n"$$1,"Queens,,",$$6,"\n"$$1,"Brooklyn,,",$$7,"\n"$$1,"Staten Island,,",$$8,""}' >> $@

data-nj.csv:
	curl -o $@ https://health.data.ny.gov/api/views/xdss-u53e/rows.csv?accessType=DOWNLOAD

timeseries.h5: data.csv
	python3 process_cases.py data-2020$(DATE).csv
# end
