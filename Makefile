##
# COVID-19 Spread
#
# @file
# @version 0.1

.PHONY: env

env:
	conda env create -f environment.yml

aws: latest=$(shell aws s3 ls s3://fairusersglobal/users/mattle/h2/covid19_forecasts/ | tail -1 | cut -d' ' -f 6)
aws:
	aws s3 cp s3://fairusersglobal/users/mattle/h2/covid19_forecasts/$(latest) /tmp/
	bzip2 -zv9 /tmp/$(latest)
# end
