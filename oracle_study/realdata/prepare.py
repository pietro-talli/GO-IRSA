"""Stream the Low Carbon London smart-meter CSV (Deflate64 zip, ~8.5 GB) into a
households x half-hour matrix.

    .venv/bin/python -m oracle_study.realdata.prepare

Output (data/lcl/):
    kwh.npy     float32 (n_households, n_slots), NaN where no reading
    meta.json   household ids, t0, slot length
Source: UK Power Networks, Low Carbon London, CC-BY 4.0
https://data.london.gov.uk/dataset/smartmeter-energy-consumption-data-in-london-households-vqm0d
"""
import json
import time

import numpy as np
import pandas as pd
import zipfile_deflate64 as zipfile

ZIP = "data/lcl/LCL-FullData.zip"
OUT = "data/lcl"
T0 = pd.Timestamp("2011-11-23 00:00:00")
T1 = pd.Timestamp("2014-03-01 00:00:00")
SLOT = pd.Timedelta(minutes=30)


def main():
    n_slots = int((T1 - T0) / SLOT)
    ids, kwh = {}, None
    rows = []
    t_start = time.time()
    with zipfile.ZipFile(ZIP).open("CC_LCL-FullData.csv") as f:
        reader = pd.read_csv(f, usecols=[0, 2, 3], names=["id", "dt", "kwh"], header=0,
                             dtype={"id": "string", "dt": "string", "kwh": "string"}, chunksize=5_000_000)
        for i, ch in enumerate(reader):
            dt = pd.to_datetime(ch["dt"].str.slice(0, 19), format="%Y-%m-%d %H:%M:%S")
            val = pd.to_numeric(ch["kwh"].str.strip(), errors="coerce").astype("float32")
            slot = ((dt - T0) // SLOT).astype("int64").to_numpy()
            hid = ch["id"].map(lambda s: ids.setdefault(s, len(ids))).to_numpy(dtype=np.int64)
            ok = (slot >= 0) & (slot < n_slots) & val.notna().to_numpy()
            rows.append((hid[ok].astype(np.int32), slot[ok].astype(np.int32), val.to_numpy()[ok]))
            print(f"chunk {i}: {(i + 1) * 5} M rows, {len(ids)} households, {time.time() - t_start:.0f}s", flush=True)
    kwh = np.full((len(ids), n_slots), np.nan, dtype=np.float32)
    for h, s, v in rows:
        kwh[h, s] = v
    np.save(f"{OUT}/kwh.npy", kwh)
    with open(f"{OUT}/meta.json", "w") as fo:
        json.dump({"ids": list(ids.keys()), "t0": str(T0), "slot_minutes": 30, "n_slots": n_slots}, fo)
    print("saved", kwh.shape, f"{time.time() - t_start:.0f}s")


if __name__ == "__main__":
    main()
