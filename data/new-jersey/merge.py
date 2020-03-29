#!/usr/bin/env python3

import numpy as np
import pandas as pd
import sys
from datetime import datetime, timedelta

base_df = pd.read_csv(sys.argv[1])
new_df = pd.read_csv(sys.argv[2])
date = pd.to_datetime(sys.argv[3])
print(f"Appending data for {date}")

assert base_df.columns[0] == "Unnamed: 0", base_df.columns
base_df = base_df[base_df.columns[1:]]

# print(new_df)

new_row = dict(zip(new_df.county, new_df.counts))
# print(new_row)

# distribute pending cases proportionally among counties
# pending = new_row.pop("Pending")
# print(f"Distributing {pending} pending cases")
# counts_all = sum(new_row.values())
# new_row = {
#    k: int(np.ceil(c + pending * float(c) / counts_all)) for k, c in new_row.items()
# }
# print(new_row)

last_date = pd.to_datetime(base_df["Date"].iloc[-1])

tdiff = date - last_date
assert tdiff.days == 1 and tdiff.seconds == 0 and tdiff.microseconds == 0

new_row["Date"] = f"{date.month}/{date.day}/{date.year}"
print(base_df["Start day"].iloc[-1])
new_row["Start day"] = base_df["Start day"].iloc[-1] + 1

new_row = pd.DataFrame([list(new_row.values())], columns=list(new_row.keys()))

old_cols = set(base_df.columns)
new_cols = set(new_row.columns)
assert old_cols == new_cols, old_cols.union(new_cols) - old_cols.intersection(new_cols)

new_df = base_df.append(new_row, sort=True, ignore_index=True)

# reorder cols after sorting
cols = list(new_df)
# move the column to head of list using index, pop and insert
cols.insert(0, cols.pop(cols.index("Start day")))
cols.insert(0, cols.pop(cols.index("Date")))
cols.append(cols.pop(cols.index("Unknown")))
new_df = new_df[cols]

print(new_df)
new_df.to_csv(f"data-{date.strftime('%Y%m%d')}.csv")
