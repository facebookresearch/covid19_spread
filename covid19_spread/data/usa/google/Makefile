##
# Google Mobility data
#
# @file
# @version 0.1

mobility_raw.csv:
	curl -o $@ https://www.gstatic.com/covid19/mobility/Global_Mobility_Report.csv

mobility_features_county_google.csv: mobility_raw.csv
	python3 process_mobility.py mobility_raw.csv

mobility_features_state_google.csv: mobility_raw.csv
	python3 process_mobility.py mobility_raw.csv

weather_features_county.csv:
	python3 process_open_data.py

clean:
	-rm mobility_raw.csv
	-rm mobility_features_state_google.csv
	-rm mobility_features_state_google.pt
	-rm mobility_features_county_google.csv
	-rm mobility_features_county_google.pt
	-rm weather_features_county.csv
	-rm weather_features_county.pt
	-rm weather_features_state.csv
	-rm weather_features_state.pt
	-rm epi_features_county.csv
	-rm epi_features_county.pt
	-rm epi_features_state.csv
	-rm epi_features_state.pt
# end
