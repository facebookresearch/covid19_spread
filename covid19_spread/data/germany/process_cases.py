#!/usr/bin/env python3

import h5py
import numpy as np
import pandas as pd

from collections import defaultdict as ddict
from itertools import count

fout = "timeseries.h5"

vz = pd.read_csv(
    "gemeindeverzeichnis.csv",
    delimiter=";",
    usecols=["Kreisname", "Amtl.Gemeindeschlüssel"],
)
vz.columns = ["kreis", "ags"]


dparser = lambda x: pd.to_datetime(x, format="%m/%d/%Y")
df = pd.read_csv(
    "data.csv", date_parser=dparser, parse_dates=["Date"], usecols=["Date", "District"]
)
df = df.dropna()

dest = dict(zip(vz.kreis, vz.ags))
src = df["District"].to_numpy()
kmap = {
    "Munich": "München, Landeshauptstadt",
    "Munich (district)": "München",
    "Koblenz": "Koblenz, kreisfreie Stadt",
    "Hamburg": "Hamburg, Freie und Hansestadt",
    "Giessen": "Gießen",
    "Aachen": "Städteregion Aachen",
    "Bonn": "Bonn, Stadt",
    "Bremen (state)": "Bremen, Stadt",
    "Hanover Region": "Region Hannover",
    "Lübeck": "Lübeck, Hansestadt",
    "Mönchengladbach": "Mönchengladbach, Stadt",
    "Duisburg": "Duisburg, Stadt",
    "Frankfurt": "Frankfurt am Main, Stadt",
    "Nuremberg": "Nürnberg",
    "Münster": "Münster, Stadt",
    "Essen": "Essen, Stadt",
    "Mitte": "Berlin, Stadt",
    "Cologne": "Köln, Stadt",
    "Marzahn-Hellersdorf": "Berlin, Stadt",
    "Tempelhof-Schöneberg": "Berlin, Stadt",
    "Neukölln": "Berlin, Stadt",
    "Lindau (district)": "Lindau (Bodensee)",
    "Friedrichshain-Kreuzberg": "Berlin, Stadt",
    "Düsseldorf": "Düsseldorf, Stadt",
    "Pankow": "Berlin, Stadt",
    "Bochum": "Bochum, Stadt",
    "County of Bentheim (district)": "Grafschaft Bentheim",
    "Rotenburg (district)": "Rotenburg (Wümme)",
    "Reinickendorf": "Berlin, Stadt",
    "Neustadt (Aisch)-Bad Windsheim": "Neustadt a.d.Aisch-Bad Windsheim",
    "Gelsenkirchen": "Gelsenkirchen, Stadt",
    "Braunschweig": "Braunschweig, Stadt",
    "Wilhelmshaven": "Wilhelmshaven, Stadt",
    "Delmenhorst": "Delmenhorst, Stadt",
    "Hohenlohe (district)": "Hohenlohekreis",
    "Dortmund": "Dortmund, Stadt",
    "Mainz": "Mainz, kreisfreie Stadt",
    "Charlottenburg-Wilmersdorf": "Berlin, Stadt",
    "Saarbrücken (district)": "Regionalverband Saarbrücken",
    "Bitburg-Prüm": "Eifelkreis Bitburg-Prüm",
    "Bielefeld": "Bielefeld, Stadt",
    "Remscheid": "Remscheid, Stadt",
    "Dresden": "Dresden, Stadt",
    "Steglitz-Zehlendorf": "Berlin, Stadt",
    "Pfaffenhofen (district)": "Pfaffenhofen a.d.Ilm",
    "Lichtenberg": "Berlin, Stadt",
    "Spandau": "Berlin, Stadt",
    "Hagen": "Hagen, Stadt",
    "Erfurt": "Erfurt, Stadt",
    "Wiesbaden": "Wiesbaden, Landeshauptstadt",
    "Landsberg (district)": "Landsberg am Lech",
    "Bottrop": "Bottrop, Stadt",
    "Leverkusen": "Leverkusen, Stadt",
    "Offenbach am Main": "Offenbach am Main, Stadt",
    "Neumarkt (district)": "Neumarkt i.d.OPf.",
    "Solingen": "Solingen, Stadt",
    "Halle (Saale)": "Halle (Saale), Stadt",
    "Cottbus": "Cottbus, Stadt",
    "Bremerhaven": "Bremerhaven, Stadt",
    "Neunkirchen (German district)": "Neunkirchen",
    "Chemnitz": "Chemnitz, Stadt",
}

repl = []
agss = []
for d in src:
    if d in kmap:
        d = kmap[d]
    d = d.replace(" (district)", "")
    repl.append(d)
    agss.append(int(str(dest[d])[:-3]))
assert len(repl) == len(src)
df["District"] = repl
df["AGS"] = agss
print(df)

# aggregate duplicate events
df = df.groupby(["District", "Date", "AGS"]).size().reset_index(name="Count")
print(df)

nevents = df["Count"].sum()
# nevents = len(df)
print("Number of events", nevents)

ncount = count()
kreis_ids = ddict(ncount.__next__)
_ns = []
_ts = []
_ags = []


t0 = (df["Date"].values.astype(np.int64) // 10 ** 9).min()
df_agg = df.groupby(["District", "AGS"])
for (name, aid), group in df_agg:
    print(name, aid)
    group = group.sort_values(by="Date")
    ts = group["Date"].values.astype(np.int64) // 10 ** 9
    ws = group["Count"].values.astype(np.float)
    es = []
    for i in range(len(ts)):
        w = int(ws[i])
        if i == 0:
            tp = ts[0] - w
        else:
            tp = ts[i - 1]
        _es = np.linspace(
            max(tp, int(ts[i] - w)) + 1, int(ts[i]), w, endpoint=True, dtype=np.int
        )
        es += _es.tolist()
        # es += [ts[i]] * w
    if len(es) > 0:
        kid = kreis_ids[name]
        _ts += es
        _ns += [kid] * len(es)
        _ags += [aid] * len(es)


# _ns = [kreis_ids[k] for k in df["District"]]
# _ts = df["Date"].values.astype(np.int64) // 10 ** 9
# _ags = df["AGS"]
# _ws = df["Count"]

assert len(_ts) == nevents, (len(_ts), nevents)
knames = [None for _ in range(len(kreis_ids))]
for kreis, i in kreis_ids.items():
    knames[i] = kreis

str_dt = h5py.special_dtype(vlen=str)
ds_dt = h5py.special_dtype(vlen=np.dtype("int"))
ts_dt = h5py.special_dtype(vlen=np.dtype("float32"))
with h5py.File(fout, "w") as fout:
    _dnames = fout.create_dataset("nodes", (len(knames),), dtype=str_dt)
    _dnames[:] = knames
    _cnames = fout.create_dataset("cascades", (1,), dtype=str_dt)
    _cnames[:] = ["covid19_de"]
    ix = np.argsort(_ts)
    node = fout.create_dataset("node", (1,), dtype=ds_dt)
    node[0] = np.array(_ns, dtype=np.int)[ix]
    time = fout.create_dataset("time", (1,), dtype=ts_dt)
    time[0] = np.array(_ts, dtype=np.float)[ix]
    _agss = fout.create_dataset("ags", (len(agss),), dtype=ds_dt)
    _agss[:] = np.array(_ags, dtype=np.int)[ix]
    # mark = fout.create_dataset("mark", (1,), dtype=ts_dt)
    # mark[0] = np.array(_ws, dtype=np.float)[ix]
