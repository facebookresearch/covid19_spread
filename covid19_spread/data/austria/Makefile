##
# Project Title
#
# @file
# @version 0.1


.PHONY: symptom_survery fb features data.csv google weather

symptom_survey:
	cd symptom_survey && make all

weather:
	cd weather && python3 process_open_data.py

fb:
	cd fb && python3 process_mobility.py

google:
	cd google && python3 process_mobility.py

data.csv:
	python3 fetch_data.py

features: symptom_survey fb weather data.csv google
	python3 convert.py

# end
