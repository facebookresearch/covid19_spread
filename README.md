# Modeling the spread of COVID-19

[![CircleCI](https://circleci.com/gh/fairinternal/covid19_spread.svg?style=shield&circle-token=c8ca107a5135df4d7544141d105031dec491d83e)](https://circleci.com/gh/fairinternal/covid19_spread)

<img src="https://github.com/fairinternal/covid19_spread/raw/master/img/spread.jpg" width=250 />

## How do I prepare a dataset for training?

First, set up the Anaconda environment needed to run the code:

``` sh
make env  # in the covid19_spread folder
conda activate covid19_spread
```

The above should work on FAIR Cluster (https://our.internmc.facebook.com/intern/wiki/FAIR/Platforms/FAIRClusters/FAIRClusterFAQ/#managing-the-software-en) machines after running module load anaconda3 cuda.

The datasets will live in the data folder, with a subfolder per geographical region. When adding a dataset for a new country, please implement a process_cases.py script that: (1) takes raw data files obtained  from the original source and (2) converts these files to a HDF5 file (via h5py (https://www.h5py.org/)), called timeseries.h5, that contains five datasets: 

1. A nodes dataset that contains all the names of the geographical entities (e.g., county names) in the data as strings. Henceforth, we will refer to these geographical entities as “nodes”.
2. A cascades dataset that contains a single string identifying the name of the dataset (e.g., “codiv19_nl” for The Netherlands).
3. A node dataset that contains a single Numpy array (dtype=int) with a node index for each count. The node indexes are sorted by the timestamp of the corresponding count and must be uniquely renamed to be in a contiguous range (i.e., [0, 1, ..., N-1]). We denote the length of this array by T.
4. A time dataset that contains a single Numpy array (dtype=float32) with the timestamps corresponding to the node indexes. This should have the same shape as the node dataset (T). For consistency with other datasets, we recommend you represent time as the number of days from the first outbreak. 
    MHPs can only deal with single events at a time, not with counts. If data is aggregated per day, you must split it into individual case events by sampling time stamps. Empirically, the best way to do this appears to be by sampling timestamps uniformly at random from the time stamp range of that day (example (https://github.com/fairinternal/covid19_spread/blob/master/data/new-jersey/process_cases.py#L55)).
5. An ags dataset that contains a single Numpy array (dtype=int) with the case count for the corresponding node index at the corresponding timestamp. This Numpy array will have length T, also. Please note that if you split up counts into individual events, all the “counts” here will be 1.

Examples of Python scripts that dump data in this format can be found here for Germany (https://github.com/fairinternal/covid19_spread/blob/master/data/germany/process_cases.py#L156-L170) and The Netherlands (https://github.com/fairinternal/covid19_spread/blob/master/data/netherlands/process_cases.py#L69-L83). You likely only have to make minor adaptation to the highlighted code (but will need to implement your own pre-processing code depending on what the input data file looks like).

In addition, please implement a simple Makefile that implements the following rules: (1) rules that download the raw data files so that you can easily update the data every day, (2) a timeseries.h5 rule that runs the process_cases.py script you just created, and (3) a clean rule that removes the temporary data. An example of such a Makefile can be found here (https://github.com/fairinternal/covid19_spread/blob/master/data/germany/Makefile).

Once you have prepared the data and these two scripts, please check it in the appropriate data/ subfolder, and submit a pull request with your changes to the Github repository. When sending a pull request, please do not include temporary files such as the raw data or the HDF5 data file in your commits.

For all external data that you check in to the repository, *please document where it came from*. This documentation is important for review processes that we will need to go through before we can release model results (see this SRT task (https://review.internmc.facebook.com/intern/review/research/collaboration/view/?id=846319575880871) as an example for our collaboration with NYU). 

You can generate the HDF5 data file using the scripts that you have just checked in by running:

``` sh
cd data/geographical_region  # starting in covid19_spread folder
make timeseries.h5
```

## Tests

To run tests:

```
python -m pytest tests/
```

To exclude integration tests:
```
python -m pytest tests/ -v -m "not integration" 
```
