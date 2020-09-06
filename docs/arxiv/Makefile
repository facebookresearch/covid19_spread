##
# COVID-19 Spread Paper
#
# @file
# @version 0.1

.PHONY: clean img/us_mae/png

img/cases_example.png: img/cases_ny.svg img/cases_fl.svg
	cd img && /usr/bin/montage -density 150 -geometry '1x1<' -tile '1x2' cases_ny.svg cases_fl.svg cases_example.png


img/growth_example.png: img/growth_ny.svg img/growth_fl.svg
	cd img && /usr/bin/montage -density 150 -geometry '1x1<' -tile '1x2' growth_ny.svg growth_fl.svg growth_example.png

img/maes.png:
	cd img && /usr/bin/montage -density 150 -geometry '+0+0' -tile 3x2 mae_*d.svg maes.png

img/betas.png:
	cd img && /usr/bin/montage -density 150 -geometry '+0+20' -tile 1x3 beta_fl.svg beta_nj.svg beta_az_tx.svg betas.png

img/us_mae/png:
	mkdir -p img/us_mae/png
	rm img/us_mae/*.svg
	rm img/us_mae/png/*
	scp etfair:covid19_spread/notebooks/img/us_mae*.svg img/us_mae/
	# cd img/us_mae && /usr/bin/mogrify -format png -path png -density 150 *.svg
	cd img/us_mae && /usr/bin/montage -density 150 -geometry '1x1<' us_mae*.svg us_mae.png
	# cd img/us_mae/png && ls | cat -n | while read n f; do mv -n "$f" "us_mae_$n.png"; done

clean:
	rm -f img/maes.png
	rm -f img/cases_example.png
	rm -f img/growth_example.png
	rm -f img/betas.png
# end
