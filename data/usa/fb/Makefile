##
# FB Movement Range
#
# @file
# @version 0.1

.PHONY: mobility_features.csv

mobility_features.csv:
	python3 process_mobility.py mobility_raw.csv

clean:
	-rm movement-range-*.txt
	-rm mobility_raw.csv
	-rm mobility_features.csv
	-rm mobility_features.pt
# end
