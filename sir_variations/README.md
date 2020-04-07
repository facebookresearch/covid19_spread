 
runs similar to the initial version of sir.py  

might need to edit path...  

example:  
python3 sir.py -fdat ~/covid19_spread/data/new-jersey/timeseries.h5 -fpop ~/covid19_spread/data/population-data/US-states/new-jersey-population.csv -fsuffix nj-$(DATE) -dout ~/covid19_spread/forecasts/new-jersey -days 60 -keep 7 -dt_window 5 -doubling-times 2 3   

note there are multiple windows to consider now!  
