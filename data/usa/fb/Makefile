##
# FB Movement Range
#
# @file
# @version 0.1

.PHONY: mobility_features_county.csv

mobility_features_county.csv:
	python3 process_mobility.py mobility_raw.csv

mobility_features_state.csv:	mobility_features_county.csv

clean:
	-rm movement-range-*.txt
	-rm mobility_raw.csv
	-rm mobility_features_county.csv
	-rm mobility_features_county.pt
	-rm mobility_features_state.csv
	-rm mobility_features_state.pt
# end
