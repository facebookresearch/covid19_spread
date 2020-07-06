##
# Project Title
#
# @file
# @version 0.1

signal=smoothed_cli

state/$(signal).csv:
	mkdir -p state
	-python3 fetch.py state $(signal)
	head -1 state/$(signal)-20200406.csv > $@
	tail -n +2 state/$(signal)-*.csv >> $@

county/$(signal).csv:
	mkdir -p county
	-python3 fetch.py county $(signal)
	head -1 county/$(signal)-20200406.csv > $@
	tail -n +2 county/$(signal)-*.csv >> $@

data-$(signal)-state.csv: state/$(signal).csv
	python3 convert-state.py $(signal)

data-$(signal)-fips.csv: county/$(signal).csv
	python3 convert-fips.py $(signal)
# end
# end
