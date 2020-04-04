 
runs similar to the initial version of sir.py  

example:  
python3 sir.py -fdat data/new-jersey/timeseries.h5 -fpop data/population-data/US-states/new-jersey-population.csv -fsuffix nj-$(DATE) -dout forecasts/new-jersey -days 60 -keep 7 -window 5 -doubling-times 2 3 4 10  


