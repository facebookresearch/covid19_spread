##
# Project Title
#
# @file
# @version 0.1

.PHONY: clean

google/mobility_features.csv:
	cd google && make mobility_features.csv

google/weather_features.csv:
	cd google && make weather_features.csv

fb/mobility_features.csv:
	cd fb && make mobility_features.csv

symptom-survey/data-smoothed_cli-fips.csv:
	cd symptom-survey && make data-smoothed_cli-fips.csv

symptom-survey/data-smoothed_cli-state.csv:
	cd symptom-survey && make data-smoothed_cli-state.csv

data.csv: google/mobility_features.csv fb/mobility_features.csv symptom-survey/data-smoothed_cli-state.csv google/weather_features.csv
	python3 convert.py -with-features

data_deaths.csv:
	python3 convert.py -metric deaths

clean:
	cd google && make clean
	cd fb && make clean
	-rm data.csv
	-rm data_deaths.csv
# end
