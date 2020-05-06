Forecast Database
-----------------

We've created an sqlite3 database for storing model forecasts and ground truth data.  This database is updated daily via a cron job and should always be up to date.

The database lives both on the H1 and H2 cluster at: `/checkpoint/mattle/covid19/forecast.db`

## Database Schema

### `infections`

This table corresponds to COVID19 confirmed cases

|Column | Type | Description |
|-------|------|-------------|
| date  | text  | The date the number of cases was reported |
| loc1  | text | Top level location (country) | 
| loc2  | text | Second level location (state in the U.S.) | 
| loc3 | text | Third level location (county in the U.S.) |
| counts | real | Number of confirmed cases | 
| id | text | ID for the model (or ground truth source) | 
| forecast_date | text | Date the forecasts were created (NULL for ground truth source)|

### `deaths`

This table corresponds to COVID19 related deaths

|Column | Type | Description |
|-------|------|-------------|
| date  | text  | The date the number of deaths was reported |
| loc1  | text | Top level location (country) | 
| loc2  | text | Second level location (state in the U.S.) | 
| loc3 | text | Third level location (county in the U.S.) |
| counts | real | Number of deaths| 
| id | text | ID for the model (or ground truth source) | 
| forecast_date | text | Date the forecasts were created (NULL for ground truth source)|

## Connecting to the database

```Python
import sqlite3

DB = '/checkpoint/mattle/covid19/forecast.db'
conn = sqlite3.connect(DB)

# Fetch IDs of all infections datapoints
res = conn.execute("SELECT DISTINCT id FROM infections;")
print([x for x, in res])
```

Should print something like:

```Python
['new-jersey_fast', 'new-jersey_slow', 'nys_fast', 'nys_slow', 'nyt_ground_truth']
```