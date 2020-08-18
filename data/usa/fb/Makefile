##
# FB Movement Range
#
# @file
# @version 0.1

.PHONY: mobility_features_county_fb.csv

mobility_features_county_fb.csv:
	python3 process_mobility.py mobility_raw_fb.csv

mobility_features_state_fb.csv:	mobility_features_county_fb.csv

clean:
	-rm movement-range-*.txt
	-rm mobility_raw.csv
	-rm mobility_features_county_fb.csv
	-rm mobility_features_county_fb.pt
	-rm mobility_features_state_fb.csv
	-rm mobility_features_state_fb.pt
# end
