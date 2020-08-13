##
# Project Title
#
# @file
# @version 0.1

#signal=smoothed_wcli
#startdate=20200406
signal=smoothed_hh_cmnty_cli
startdate=20200415

state/$(signal).csv:
	mkdir -p state
	-python3 fetch.py state $(signal)
	head -1 state/$(signal)-$(startdate).csv > $@
	tail -q -n +2 state/$(signal)-*.csv >> $@

county/$(signal).csv:
	mkdir -p county
	-python3 fetch.py county $(signal)
	head -1 county/$(signal)-$(startdate).csv > $@
	tail -q -n +2 county/$(signal)-*.csv >> $@

data-$(signal)-state.csv: state/$(signal).csv
	python3 convert-state.py $(signal)

data-$(signal)-county.csv: county/$(signal).csv
	python3 convert-fips.py $(signal)

clean:
	-rm state/$(signal).csv
	-rm county/$(signal).csv
	-rm data-$(signal)-state.csv
	-rm data-$(signal)-county.csv
# end
