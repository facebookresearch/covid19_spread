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