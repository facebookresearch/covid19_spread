##
#
#
#
# @file
# @version 0.1

.PHONY: clean

testing_raw.csv:
	curl -o $@ https://api.covidtracking.com/v1/states/daily.csv

testing_features_county.csv: testing_raw.csv
	python3 process_testing.py

testing_features_state.csv: testing_features_county.csv


clean:
	-rm testing_raw.csv
	-rm testing_features_county.csv
	-rm testing_features_state.csv
	-rm features.pt
# end
