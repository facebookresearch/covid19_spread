##
# Google Mobility data
#
# @file
# @version 0.1

mobility_raw.csv:
	curl -o $@ https://www.gstatic.com/covid19/mobility/Global_Mobility_Report.csv

mobility_features.csv: mobility_raw.csv
	python3 process_mobility.py mobility_raw.csv

clean:
	-rm mobility_raw.csv
	-rm mobility_features.csv
# end
