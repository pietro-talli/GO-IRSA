"""NGSIM (US DOT, CC BY-SA 3.0) -> per-site lane-packed speed traces + per-site AR(1) fit.

    .venv/bin/python -m ngsim_study.prepare

Input : data/ngsim/ngsim_min.parquet (Vehicle_ID, Global_Time [ms], v_Vel [ft/s], v_Class, Location)
Output: data/ngsim/packed.npz  and  ngsim_study/results/fit.csv

* Sites: us-101 (freeway) and lankershim (arterial). i-80 is incomplete in the open export and
  peachtree has no memory at frame scale (AR coefficient 0.94 per 0.1 s), both are excluded.
* Vehicle IDs are reused across the recording periods: a trajectory is split into segments
  wherever consecutive samples are not 100 ms apart.
* Each site is split in time: first 30% -> fit AR(1) of the speed (one coefficient set per
  site), last 70% -> simulation. In the simulation part, segments are packed greedily into
  "node slots" (a slot is reused by a new vehicle once the previous one has left).
* One sample (0.1 s) = one slot. Speeds are rescaled by a global factor so that the median
  per-slot innovation std is 0.04 (the per-slot std of the paper's Wiener model).
"""
import os

import numpy as np
import pandas as pd

SITES = ("us-101", "lankershim")
TRAIN = 0.30
TARGET_SIGMA = 0.04
DT = 100  # ms


def segments(g):
    """Return list of (t_start_ms, speeds) continuous 10 Hz segments of one site."""
    g = g.sort_values(["Vehicle_ID", "Global_Time"])
    vid, tt, v = g.Vehicle_ID.values, g.Global_Time.values, g.v_Vel.values.astype(np.float64)
    brk = np.flatnonzero((vid[1:] != vid[:-1]) | (np.diff(tt) != DT)) + 1
    out = []
    for a, b in zip(np.r_[0, brk], np.r_[brk, len(v)]):
        if b - a >= 20:  # at least 2 s
            out.append((tt[a], v[a:b]))
    return out


def fit_ar1(segs):
    x0 = np.concatenate([s[:-1] for _, s in segs])
    x1 = np.concatenate([s[1:] for _, s in segs])
    A = np.vstack([x0, np.ones_like(x0)]).T
    (a, mu), *_ = np.linalg.lstsq(A, x1, rcond=None)
    if a > 1.0:  # non-stationary trend inside the training window -> random walk with drift
        a = 1.0
        mu = float(np.mean(x1 - x0))
    res = x1 - a * x0 - mu
    z = res / res.std()
    return float(a), float(mu), float(res.std()), float((z ** 4).mean() - 3)


def pack(segs, t0, T):
    """Greedy interval packing of segments into node slots over [t0, t0 + T*DT)."""
    segs = sorted(((int((ts - t0) // DT), s) for ts, s in segs), key=lambda z: z[0])
    free_at, lanes = [], []  # per lane: first free sample index
    for st, s in segs:
        if st < 0 or st >= T:
            continue
        s = s[: T - st]
        for li, f in enumerate(free_at):
            if f <= st:
                break
        else:
            li = len(free_at)
            free_at.append(0)
            lanes.append([])
        lanes[li].append((st, s))
        free_at[li] = st + len(s) + 1  # one empty sample between vehicles -> unambiguous switch
    X = np.full((len(lanes), T), np.nan, np.float32)
    SEG = np.full((len(lanes), T), -1, np.int32)
    sid = 0
    for li, L in enumerate(lanes):
        for st, s in L:
            X[li, st:st + len(s)] = s
            SEG[li, st:st + len(s)] = sid
            sid += 1
    return X, SEG


def main():
    d = pd.read_parquet("data/ngsim/ngsim_min.parquet")
    rows, packed = [], {}
    for site in SITES:
        g = d[d.Location == site]
        t0, t1 = g.Global_Time.min(), g.Global_Time.max()
        cut = t0 + TRAIN * (t1 - t0)
        tr = segments(g[g.Global_Time < cut])
        te = segments(g[g.Global_Time >= cut])
        a, mu, sig, kurt = fit_ar1(tr)
        T = int((t1 - cut) // DT) + 1
        X, SEG = pack(te, cut, T)
        occ = (SEG >= 0).sum(0)
        res_te = np.concatenate([s[1:] - a * s[:-1] - mu for _, s in te])
        rows.append(dict(site=site, a=a, mu_fts=mu, sigma_fts=sig, test_sigma_fts=res_te.std(), excess_kurtosis=kurt,
                         train_segments=len(tr), test_segments=len(te), node_slots=X.shape[0],
                         occupied_median=np.median(occ), occupied_max=occ.max(), test_minutes=T * DT / 60000))
        packed[site] = (X, SEG, a, mu, sig)
    fit = pd.DataFrame(rows)
    scale = TARGET_SIGMA / np.median(fit.sigma_fts)
    fit["scale"] = scale
    os.makedirs("ngsim_study/results", exist_ok=True)
    fit.to_csv("ngsim_study/results/fit.csv", index=False)
    print(fit.round(4).to_string())
    out = {"scale": scale, "sites": np.array(SITES)}
    for site, (X, SEG, a, mu, sig) in packed.items():
        k = site.replace("-", "")
        out[f"{k}_X"] = X * scale
        out[f"{k}_SEG"] = SEG
        out[f"{k}_par"] = np.array([a, mu * scale, sig * scale])
    np.savez_compressed("data/ngsim/packed.npz", **out)
    print("saved data/ngsim/packed.npz; scale =", scale)


if __name__ == "__main__":
    main()
