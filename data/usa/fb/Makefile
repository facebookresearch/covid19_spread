##
# FB Movement Range
#
# @file
# @version 0.1

fb_mobility_url = https://data.humdata.org/dataset/c3429f0e-651b-4788-bb2f-4adbf222c90e/resource/31ca909c-10d9-458a-8720-88b54b3e3627/download/movement-range-data-2020-06-23.zip

fb_mobility.zip:
	curl -o fb_mobility.zip $(fb_mobility_url)

fb_mobility.csv: fb_mobility.zip
	unzip $<

# end
