##
# COVID-19 Spread Paper
#
# @file
# @version 0.1

.PHONY: clean

img/cases_example.png: img/cases_ny.svg img/cases_fl.svg
	cd img && /usr/bin/montage -density 150 -geometry '1x1<' -tile '1x2' cases_ny.svg cases_fl.svg cases_example.png


img/growth_example.png: img/growth_ny.svg img/growth_fl.svg
	cd img && /usr/bin/montage -density 150 -geometry '1x1<' -tile '1x2' growth_ny.svg growth_fl.svg growth_example.png

img/maes.png:
	cd img && /usr/bin/montage -density 150 -geometry '+0+0' -tile 3x2 mae_*d.svg maes.png

img/betas.png:
	cd img && /usr/bin/montage -density 150 -geometry '+0+20' -tile 1x3 beta_fl.svg beta_nj.svg beta_az_tx.svg betas.png

img/us_mae/us_mae.png:
	mkdir -p img/us_mae
	rsync -v etfair:covid19_spread/notebooks/img/us_mae*.svg img/us_mae/
	rm -f img/us_mae/us_mae_granger*.svg
	cd img/us_mae && /usr/bin/montage -density 150 -geometry '1x1<' -reverse  us_mae*.svg us_mae.png

img/us_rank_mae.png:
	rsync -v etfair:covid19_spread/notebooks/img/us_rank_mae.svg img/
	cd img && /usr/bin/convert -density 150 us_rank_mae.svg us_rank_mae.png

img/us_rank_rmse.png:
	rsync -v etfair:covid19_spread/notebooks/img/us_rank_rmse.svg img/
	cd img && /usr/bin/convert -density 150 us_rank_rmse.svg us_rank_rmse.png

img/us_mae_granger_ablation/us_mae_granger_ablation.png:
	mkdir -p img/us_mae_granger_ablation
	rsync -v etfair:covid19_spread/notebooks/img/us_mae_granger_ablation*.svg img/us_mae_granger_ablation/
	cd img/us_mae_granger_ablation && /usr/bin/montage -density 150 -geometry '1x1<' -reverse us_mae_granger_ablation*.svg us_mae_granger_ablation.png

img/us_mae_loss_ablation/us_mae_loss_ablation.png:
	mkdir -p img/us_mae_loss_ablation
	rsync -v etfair:covid19_spread/notebooks/img/us_mae_loss_ablation*.svg img/us_mae_loss_ablation/
	cd img/us_mae_loss_ablation && /usr/bin/montage -density 150 -geometry '1x1<' -reverse us_mae_loss_ablation*.svg us_mae_granger_ablation.png


img/counties_bar_mae.png:
	/usr/bin/montage -density 150 -geometry '1x1<' -tile 1x2 -reverse  img/counties_bar_mae_*.svg $@

clean:
	rm -f img/maes.png
	rm -f img/cases_example.png
	rm -f img/growth_example.png
	rm -f img/betas.png
	rm -f img/us_mae/*.svg
	rm -f img/us_mae/us_mae.png
	rm -f img/us_mae_granger_ablation/*.svg
	rm -f img/us_mae_granger_ablation/us_mae_granger_ablation.png
# end
