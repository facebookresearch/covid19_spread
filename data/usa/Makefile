##
# Project Title
#
# @file
# @version 0.1

.PHONY: clean

google/mobility_features_county.csv:
	cd google && make mobility_features_county.csv

google/weather_features_county.csv:
	cd google && make weather_features_county.csv

fb/mobility_features_county.csv:
	cd fb && make mobility_features_county.csv

testing/testing_features_county.csv:
	cd testing && make testing_features_county.csv

symptom-survey/data-smoothed_hh_cmnty_cli-county.csv:
	cd symptom-survey && make data-smoothed_hh_cmnty_cli-county.csv

symptom-survey/data-smoothed_hh_cmnty_cli-state.csv:
	cd symptom-survey && make data-smoothed_hh_cmnty_cli-state.csv

data_cases.csv: google/mobility_features_county.csv fb/mobility_features_county.csv symptom-survey/data-smoothed_hh_cmnty_cli-state.csv symptom-survey/data-smoothed_hh_cmnty_cli-county.csv google/weather_features_county.csv testing/testing_features_county.csv
	python3 convert.py -with-features -source google -metric cases
	head -1 $@ > data_cases_ny.csv
	head -1 $@ > data_cases_fl.csv
	grep "New York" $@ >> data_cases_ny.csv
	grep "Florida" $@ >> data_cases_fl.csv

data_deaths.csv:	google/mobility_features_county.csv fb/mobility_features_county.csv symptom-survey/data-smoothed_hh_cmnty_cli-state.csv symptom-survey/data-smoothed_hh_cmnty_cli-county.csv google/weather_features_county.csv testing/testing_features_county.csv
	python3 convert.py -with-features -source jhu -metric deaths

clean:
	cd google && make clean
	cd fb && make clean
	cd symptom-survey && make clean
	cd testing && make clean
	-rm data_cases.csv
	-rm data_deaths.csv
# end
