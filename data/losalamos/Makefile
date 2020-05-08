##
# Project Title
#
# @file
# @version 0.1

dates=2020-04-26 2020-04-29 2020-05-03 2020-05-06

define RAW
raw_$(1).csv:
	curl -o raw_$(1).csv https://covid-19.bsvgateway.org/forecast/us/files/$(1)/deaths/$(1)_deaths_quantiles_us_website.csv
endef

define CONVERT
predictions_$(1).csv: raw_$(1).csv
	python3 convert.py raw_$(1).csv
	rm raw_$(1).csv
endef

$(foreach _date, $(dates), $(eval $(call RAW,$(_date))))
$(foreach _date, $(dates), $(eval $(call CONVERT,$(_date))))

all: $(foreach _date, $(dates),predictions_$(_date).csv)
# end
