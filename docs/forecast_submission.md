Submitting Forcasts
-------------------

## Submitting to Humanitarian Data Exchange (HDX)

When the forecasts are ready to be submitting, you should receive an email containing instructions on how to submit.  It should look something like:

On a devserver (FB infra)
```
cd ~/fbsource/fbcode/dataswarm-pipelines && \
    ./tester3 tasks/locations/covid19_forecasts/upload_forecasts.py <date>
```

## Submitting to Johnson & Johnson MBOX

On your devfair in the `covid19_spread` repo, run the following command:


```
python submit_forecast.py submit-mbox <date>
```

## Submitting to `reichlab/covid19-forecast-hub`

On your devfair in the `covid19_spread` repo, run the following command:


```
python submit_forecast.py submit-reichlab <date>
```

This will create a branch in https://github.com/lematt1991/covid19-forecast-hub.

Follow the instructions printed to stdout.  It should look something like:


```
Create pull request by going to:
https://github.com/lematt1991/covid19-forecast-hub
-------------------------
## Description

If you are **adding new forecasts** to an existing model, please include the following details:- 
- Team name: Facebook AI Research (FAIR)
- Model name that is being updated: Neural Relational Autoregression
---

## Checklist

- [x] Specify a proper PR title with your team name.
- [x] All validation checks ran successfully on your branch. Instructions to run the tests locally is present [here](https://github.com/reichlab/covid19-forecast-hub/wiki/Running-Checks-Locally).
```

Click the link (https://github.com/lematt1991/covid19-forecast-hub) to go to our fork of the reichlab repo.
Create a pull request from the newly created branch (`forecast-<date>`), and copy the markdown that was printed to stdout into the PR description.


![](reichlab_instructions.gif)