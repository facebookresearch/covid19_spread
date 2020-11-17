##
# Project Title
#
# @file
# @version 0.1

signals=smoothed_cli smoothed_mc smoothed_dc
country=Spain

define _fetch
	mkdir -p raw/$(signal)
	rm -f raw/$(signal)/survey.csv
	-python3 fetch.py $(country) $(signal) $(indicator) $(startdate)
	head -1 raw/$(signal)/$(startdate).csv > raw/$(signal)/survey.csv
	tail -q -n +2 raw/$(signal)/2020*.csv >> raw/$(signal)/survey.csv
endef


smoothed_cli.csv: signal=smoothed_cli
smoothed_cli.csv: indicator=covid
smoothed_cli.csv: startdate=20200503
smoothed_cli.csv:
	$(call _fetch)
	python3 process_symptom_survey.py -signal $(signal)

smoothed_mc.csv: signal=smoothed_mc
smoothed_mc.csv: indicator=mask
smoothed_mc.csv: startdate=20200503
smoothed_mc.csv:
	$(call _fetch)
	python3 process_symptom_survey.py -signal $(signal)

smoothed_dc.csv: signal=smoothed_dc
smoothed_dc.csv: indicator=contact
smoothed_dc.csv: startdate=20200503
smoothed_dc.csv:
	$(call _fetch)
	python3 process_symptom_survey.py -signal $(signal)

all: smoothed_cli.csv smoothed_mc.csv smoothed_dc.csv

clean:
	rm -f smoothed_cli.csv
	rm -f smoothed_mc.csv
	rm -f smoothed_dc.csv
# end
