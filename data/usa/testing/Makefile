##
#
#
#
# @file
# @version 0.1

outputs = \
	total_features_county.csv \
	total_features_state.csv \
	ratio_features_county.csv \
	ratio_features_state.csv

.PHONY: clean process

testing_raw.csv:
	curl -o $@ https://api.covidtracking.com/v1/states/daily.csv

process: testing_raw.csv
	python3 process_testing.py

%.csv: process

all: $(outputs)

clean:
	-rm testing_raw.csv
	-rm total_features_county.csv
	-rm total_features_state.csv
	-rm ratio_features_county.csv
	-rm ratio_features_state.csv
	-rm features.pt
# end
