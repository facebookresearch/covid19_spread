##
# Google Mobility data
#
# @file
# @version 0.1

mobility_raw.csv:
	curl -o $@ https://www.gstatic.com/covid19/mobility/Global_Mobility_Report.csv

mobility_features.csv: mobility_raw.csv
	python3 process_mobility.py mobility_raw.csv

weather_features.csv:
	python3 process_open_data.py

clean:
	-rm mobility_raw.csv
	-rm mobility_features.csv
	-rm mobility_features.pt
# end
