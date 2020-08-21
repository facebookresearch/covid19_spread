##
# Project Title
#
# @file
# @version 0.1

signals=fb-survey/smoothed_hh_cmnty_cli fb-survey/smoothed_wcli doctor-visits/smoothed_adj_cli

define _fetch
	mkdir -p $(resolution)/$(shell dirname $(signal))
	python3 fetch.py $(resolution) $(signal)
	head -1 $(resolution)/$(signal)-$(startdate).csv > $@
	tail -q -n +2 $(resolution)/$(signal)-*.csv >> $@
endef

define _feature_name
endef

define _process
$(shell dirname $(1))_$(shell basename $(1))-state.csv: state/$(1).csv
	python3 process_symptom_survey.py -signal $(1) -resolution state

$(shell dirname $(1))_$(shell basename $(1))-county.csv: county/$(1).csv
	python3 process_symptom_survey.py -signal $(1) -resolution county
endef

$(foreach _signal, $(signals), $(eval $(call _process,$(_signal))))

state/fb-survey/smoothed_hh_cmnty_cli.csv: signal=fb-survey/smoothed_hh_cmnty_cli
state/fb-survey/smoothed_hh_cmnty_cli.csv: startdate=20200415
state/fb-survey/smoothed_hh_cmnty_cli.csv: resolution=state
state/fb-survey/smoothed_hh_cmnty_cli.csv:
	$(call _fetch)

county/fb-survey/smoothed_hh_cmnty_cli.csv: signal=fb-survey/smoothed_hh_cmnty_cli
county/fb-survey/smoothed_hh_cmnty_cli.csv: startdate=20200415
county/fb-survey/smoothed_hh_cmnty_cli.csv: resolution=county
county/fb-survey/smoothed_hh_cmnty_cli.csv:
	$(call _fetch)

state/fb-survey/smoothed_wcli.csv: signal=fb-survey/smoothed_wcli
state/fb-survey/smoothed_wcli.csv: startdate=20200406
state/fb-survey/smoothed_wcli.csv: resolution=state
state/fb-survey/smoothed_wcli.csv:
	$(call _fetch)

county/fb-survey/smoothed_wcli.csv: signal=fb-survey/smoothed_wcli
county/fb-survey/smoothed_wcli.csv: startdate=20200406
county/fb-survey/smoothed_wcli.csv: resolution=county
county/fb-survey/smoothed_wcli.csv:
	$(call _fetch)

state/doctor-visits/smoothed_adj_cli.csv: signal=doctor-visits/smoothed_adj_cli
state/doctor-visits/smoothed_adj_cli.csv: startdate=20200201
state/doctor-visits/smoothed_adj_cli.csv: resolution=state
state/doctor-visits/smoothed_adj_cli.csv:
	$(call _fetch)

county/doctor-visits/smoothed_adj_cli.csv: signal=doctor-visits/smoothed_adj_cli
county/doctor-visits/smoothed_adj_cli.csv: startdate=20200201
county/doctor-visits/smoothed_adj_cli.csv: resolution=county
county/doctor-visits/smoothed_adj_cli.csv:
	$(call _fetch)

all: \
	$(foreach _signal,$(signals),$(shell dirname $(_signal))_$(shell basename $(_signal))-state.csv) \
	$(foreach _signal,$(signals),$(shell dirname $(_signal))_$(shell basename $(_signal))-county.csv)

clean:
	$(foreach _signal, $(signals), $(shell rm -f state/$(_signal).csv))
	$(foreach _signal, $(signals), $(shell rm -f county/$(_signal).csv))
	#-rm state/$(signal).csv
	#-rm county/$(signal).csv
	#-rm data-$(signal)-state.csv
	#-rm data-$(signal)-county.csv
# end
