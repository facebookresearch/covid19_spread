##
# Project Title
#
# @file
# @version 0.1


.PHONY: symptom-survery fb features data.csv

symptom-survey:
	cd symptom-survey && make all

fb:
	cd fb && python3 process_mobility.py

data.csv:
	python3 fetch_data.py

features: symptom-survey fb data.csv
	python3 convert.py

# end
