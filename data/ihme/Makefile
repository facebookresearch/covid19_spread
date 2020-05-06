##
# Project Title
#
# @file
# @version 0.1

ihme_$(DATE).zip:
	curl -o $@ https://ihmecovid19storage.blob.core.windows.net/archive/$(DATE)/ihme-covid19.zip

raw_$(DATE).csv: ihme_$(DATE).zip
	unzip -j $< "*/Hospitalization_all_locs.csv"
	mv Hospitalization_all_locs.csv raw_$(DATE).csv
	rm $<

prediction-csv: raw_$(DATE).csv
	python3 convert.py $(DATE)
	rm $<

# end
