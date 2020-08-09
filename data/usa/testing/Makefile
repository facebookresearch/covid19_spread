##
#
#
#
# @file
# @version 0.1

.PHONY: clean

testing_raw.csv:
	curl -o $@ https://api.covidtracking.com/v1/states/daily.csv

testing_features.csv: testing_raw.csv
	python3 process_testing.py

clean:
	-rm testing_raw.csv
	-rm testing_features.csv
	-rm features.pt
# end
