Covid-19 Forcasting pipelines
-------------------------------

## Training Pipelines

The recurring pipeline for training on US county level infection data can be found [here](https://github.com/fairinternal/covid19_spread/blob/master/data/usa/us_recurring.py).  If a sweep fails for whatever reason, remote (or rename) the directory containing the sweep (ex `/checkpoint/mattle/covid19/forecasts/us/<sweep_id>`).  

You can re-run a failed sweep by running:

```bash
cd ~/covid19_spread/data/usa && python us_recurring.py
```

This requires that no existing sweep has been run for this basedate (this is why we deleted the failed sweep above).

## Uploading Forecasts 

When the forecast is finished, run the following to upload the forecast to S3:

```bash
cd ~/covid19_spread && python submit_forecast.py submit-s3 /checkpoint/$USER/covid19/forecasts/us/<sweep_id>
```

The provided path should point to the base sweep directory (not a specific run).  It will use the `model_selection.json` to find the correct forecast to upload.

At this point, you should see the following file in S3:

```bash
aws s3 ls s3://fairusersglobal/users/mattle/h2/covid19_forecasts/forecast_<basedate>.csv
```

## Uploading S3 Forecasts to Hive

To manually run the dataswarm pipeline to upload the forecasts to Hive, run in `fbcode`:

```bash
cd ~/fbsource/fbcode/dataswarm-pipelines
./tester tasks/locations/covid19_forecasts/update_forecasts.py <basedate>  -b "" -l
```


## Publishing Forecasts to HDX

If the forecasts look reasonable with respect to previous forecasts, you can publish them to HDX with the following command:

```bash
cd ~/fbsource/fbcode/dataswarm-pipelines
./tester3 -c dataswarmadhoc tasks_adhoc/locations/covid19_forecasts/upload_forecasts.py <basedate>
```

## Updating the README.txt

First, fetch the current README from manifold:

```bash
manifold get aiplatform_tools/tree/covid19_forecasts/usa/README.txt
```

Edit the file locally, then re-upload it to manifold:

```bash
manifold put README.txt aiplatform_tools/tree/covid19_forecasts/usa/README.txt NoPredicate
```
