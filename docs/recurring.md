# Recurring Jobs

The `Recurring` class in `data/recurring.py` defines a class for setting up recurring jobs that depend on fetching new data.

The class maintains an sqlite database with the following schema


| Column | Type | Description |
|--------|------|-------------|
| basedate | text | Date that the model is trained up to |
| path   | text | Path to the sweep | 
| launch_time | real | Time the sweep was launched | 
| module | text | Which cv module is used | 
| id | text | Unique identifier for this type of sweep (ex: 'new-jersey-ar') |

It works by calling the `update_data` method to check if new data is available.  If a date
exists in the new dataset that is greater than any `basedate` in the database, then it will
launch a sweep and add a new entry to the database.

### Installing

To install a cron job, use the install method.  A few existing cron jobs that  can be installed:

```
cd data/new-jeresey

# MHP using CV pipeline
python nj_recurring.py --install --kind cv

# AR model using CV pipeline
python nj_recurring.py --install --kind ar

# MHP using sweep.py
python nj_recurring.py --install --kind sweep
```

## Creating a new recurring job

You'll need to defined the following methods:

```Python
import recurring
import os

class NJRecurring(recurring.Recurring):
    script_dir = os.path.dirname(os.path.realpath(__file__))

    def get_id(self):
        return "<some unique id>"

    def command(self):
        return f"python {os.path.realpath(__file__)}"

    def update_data(self):
        # Fetch latest data
        ...        

    def latest_date(self):
        df = pandas.read_csv(...)
        return df.index.max()
```