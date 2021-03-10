.PHONY: all features clean symptom-survey

all: features data_cases.csv data_deaths.csv

features: \
	fb/mobility_features_fb.csv \
	google/mobility_features_google.csv \
	weather/weather_features.csv \
	testing/total_features.csv \
	symptom-survey

fb/mobility_features_fb.csv:
	cd fb && python3 process_mobility.py

google/mobility_features_google.csv:
	cd google && python3 process_mobility.py

weather/weather_features.csv:
	cd weather && python3 process_open_data.py

testing/total_features.csv:
	cd testing && python3 process_open_data.py

symptom-survey:
	cd symptom-survey && make all

data_cases.csv: 
	python3 process_cases.py --metric cases

data_deaths.csv:
	python3 process_cases.py --metric deaths

clean:
	cd symptom-survey && make clean
	rm -f fb/mobility_features_fb.csv
	rm -f google/mobility_features_google.csv
	rm -f weather/weather_features.csv
	rm -f testing/total_features.csv
	rm -f data_cases.csv
	rm -f data_deaths.csv

select:
	find $(sweepdir) -name piv_best_$(metric).csv -exec sh -c 'cp {} projection/piv_$(metric)_$$(echo {} | sed "s|.*sweep_\([^/]*\)/.*|\1|").csv' \;
	zip -9vr projections_spain.zip projection
