#!/usr/bin/env python3

import numpy as np
import pandas as pd
import sys

from datetime import timedelta
from load import load_data


def load_ground_truth(path):
    nodes, ns, ts, _ = load_data(path)
    nodes = [n for n in nodes if n != "Unknown"]
    tmax = int(np.ceil(ts.max()))
    counts = {n: np.zeros(tmax) for n in nodes}
    for t in range(1, tmax + 1):
        ix2 = np.where(ts < t)[0]
        for i, n in enumerate(nodes):
            ix1 = np.where(ns == i)[0]
            counts[n][t - 1] = len(np.intersect1d(ix1, ix2))
    return nodes, counts


cols, counts = load_ground_truth(sys.argv[1])
basedate = pd.to_datetime(sys.argv[2])

ix = basedate.strftime("%m/%d")

offset = 0
cols = []
preds = []
trend = []
for d in range(0, 8):
    pdate = basedate + timedelta(d)
    cols.append(pdate.strftime("%m/%d"))
    pred_triv = {
        n: c[-(offset + 1)] + d * np.abs(c[-(offset + 1)] - c[-(offset + 2)])
        for n, c in counts.items()
    }
    preds.append(sum(pred_triv.values()))

    _trend = {n: np.abs(c[-(d + 1)] - c[-(d + 2)]) for n, c in counts.items()}
    trend.append(sum(_trend.values()))


df = pd.DataFrame([preds]).round(2)
df.columns = cols

print()
print("Trend")
print(pd.DataFrame([trend]))
print(sum(trend))
print("\nForecast")
print(df)
