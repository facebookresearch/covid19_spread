##
# Project Title
#
# @file
# @version 0.1

.PHONY: clean

google/mobility_features.csv:
	cd google && make mobility_features.csv

fb/mobility_features.csv:
	cd fb && make mobility_features.csv

symptom-survey/data-smoothed_cli-fips.csv:
	cd symptom-survey && make data-smoothed_cli-fips.csv

data.csv:
	python3 convert.py

data_deaths.csv: google/mobility_features.csv
	python3 convert.py deaths
# end
