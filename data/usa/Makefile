##
# Project Title
#
# @file
# @version 0.1

.PHONY: clean symptom-survey

google/mobility_features_county_google.csv:
	cd google && make mobility_features_county_google.csv

google/weather_features_county.csv:
	cd google && make weather_features_county.csv

fb/mobility_features_county_fb.csv:
	cd fb && make mobility_features_county_fb.csv

testing/total_features_county.csv:
	cd testing && make total_features_county.csv

symptom-survey:
	cd symptom-survey && make all


data_cases.csv: \
	symptom-survey \
	google/mobility_features_county_google.csv \
	fb/mobility_features_county_fb.csv \
	google/weather_features_county.csv \
	testing/total_features_county.csv
	python3 convert.py -with-features -source google -metric cases
	head -1 $@ > data_cases_ny.csv
	head -1 $@ > data_cases_fl.csv
	grep "New York" $@ >> data_cases_ny.csv
	grep "Florida" $@ >> data_cases_fl.csv

data_deaths.csv:	google/mobility_features_county_google.csv fb/mobility_features_county_fb.csv symptom-survey/data-smoothed_hh_cmnty_cli-state.csv symptom-survey/data-smoothed_hh_cmnty_cli-county.csv google/weather_features_county.csv testing/total_features_county.csv
	python3 convert.py -with-features -source jhu -metric deaths

data_deaths_state.csv:	google/mobility_features_county_google.csv fb/mobility_features_county_fb.csv symptom-survey/data-smoothed_hh_cmnty_cli-state.csv symptom-survey/data-smoothed_hh_cmnty_cli-county.csv google/weather_features_county.csv testing/testing_features_county.csv
	python3 convert.py -with-features -source jhu -metric deaths -resolution state

clean:
	cd google && make clean
	cd fb && make clean
	cd symptom-survey && make clean
	cd testing && make clean
	-rm data_cases.csv
	-rm data_cases_fl.csv
	-rm data_cases_ny.csv
	-rm data_deaths.csv
	-rm data_deaths_state.csv
# end
