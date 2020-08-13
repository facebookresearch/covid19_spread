##
#
#
#
# @file
# @version 0.1

.PHONY: clean

testing_raw.csv:
	curl -o $@ https://api.covidtracking.com/v1/states/daily.csv

total_features.csv: testing_raw.csv
	python3 process_testing.py

clean:
	-rm testing_raw.csv
	-rm total_features.csv
	-rm ratio_features.csv
	-rm features.pt
# end
