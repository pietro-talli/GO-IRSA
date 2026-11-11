"""Sweep on the LCL smart-meter traces (one sample = one frame): conservative vs estimated
GO-IRSA, optional online load calibration, pull-based reference.

    .venv/bin/python -m oracle_study.realdata.run_real --episodes 50
    .venv/bin/python -m oracle_study.realdata.run_real --episodes 50 --calib_gain 0.3 --no_pull \
        --out oracle_study/results/real_hold_episodes_calib.csv
"""
import argparse
import itertools
import multiprocessing as mp
import os
import time

import numpy as np
import pandas as pd

METHODS = ("conservative", "estimated")


def _starts(T, S, frames, episodes, seed=0):
    rng = np.random.default_rng(seed)
    return rng.integers(0, T - frames * S - 1, size=episodes)


def _worker(gpu, jobs, results, a):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    import torch
    from oracle_study.realdata.sim_real import run_batch_real as batch_fn, run_pull_real as pull_fn
    z = np.load(a.data)
    te, ar, mu, sig, scale = z["test"], z["a"], z["mu"], z["sigma"], float(z["scale"])
    while True:
        try:
            kind, S, eta = jobs.get_nowait()
        except Exception:
            return
        t0 = time.time()
        st = _starts(te.shape[1], 1, a.frames, a.episodes, seed=S)  # same windows for every method / load
        recs = []
        if kind == "pull":
            o = pull_fn(S, st, te, ar, mu, sig, "cuda:0", a.frames)
            for ep in range(a.episodes):
                recs.append(dict(method="pull", S=S, M=(S - 2) * 10, G=np.nan, eta=1.0, episode=ep,
                                 mse=o["mse"][ep].mean(), q95=o["q95"][ep].mean()))
        else:
            rows_all = list(itertools.product(METHODS, a.G, range(a.episodes)))
            for c0 in range(0, len(rows_all), a.max_rows):  # bounded GPU memory
                rows = rows_all[c0:c0 + a.max_rows]
                o = batch_fn(S, [r[0] for r in rows], [r[1] for r in rows], eta, st[[r[2] for r in rows]],
                                   te, ar, mu, sig, "cuda:0", a.frames, L=a.L, V=a.V,
                                   calib_gain=a.calib_gain)
                for i, (m, G, ep) in enumerate(rows):
                    rec = dict(method=m, S=S, M=(S - 2) * 10, G=G, eta=eta, episode=ep, calib_gain=a.calib_gain)
                    for k, v in o.items():
                        rec[k] = v[i].mean()
                    recs.append(rec)
                del o
                torch.cuda.empty_cache()
        for r in recs:
            r["mse_kwh2"] = r["mse"] / scale ** 2
            r["q95_kwh2"] = r["q95"] / scale ** 2
        results.put(recs)
        print(f"[gpu{gpu}] {kind} S={S} eta={eta} {time.time() - t0:.0f}s", flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data/lcl/ar1.npz")
    p.add_argument("--episodes", type=int, default=50)
    p.add_argument("--frames", type=int, default=200)
    p.add_argument("--S", type=int, nargs="+", default=[12, 22, 42, 62, 82])
    p.add_argument("--eta", type=float, nargs="+", default=[1.0, 0.95])
    p.add_argument("--G", type=float, nargs="+", default=[0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8])
    p.add_argument("--L", type=float, default=12.0)
    p.add_argument("--V", type=int, default=1500)
    p.add_argument("--gpus", type=int, nargs="+", default=[1, 2, 3])
    p.add_argument("--max_rows", type=int, default=350)
    p.add_argument("--calib_gain", type=float, default=0.0, help="online load calibration gain (0 = off)")
    p.add_argument("--no_pull", action="store_true")
    p.add_argument("--out", default="oracle_study/results/real_hold_episodes.csv")
    a = p.parse_args()
    ctx = mp.get_context("spawn")
    jobs, results = ctx.Queue(), ctx.Queue()
    combos = [("goirsa", S, e) for S in sorted(a.S, reverse=True) for e in a.eta] + ([] if a.no_pull else [("pull", S, 1.0) for S in a.S])
    for c in combos:
        jobs.put(c)
    procs = [ctx.Process(target=_worker, args=(g, jobs, results, a)) for g in a.gpus]
    for pr in procs:
        pr.start()
    recs, done = [], 0
    import queue
    while done < len(combos):
        try:
            recs += results.get(timeout=30)
            done += 1
        except queue.Empty:
            if not any(pr.is_alive() for pr in procs):
                raise RuntimeError(f"all workers exited, {done}/{len(combos)} jobs done")
    for pr in procs:
        pr.join()
    pd.DataFrame(recs).to_csv(a.out, index=False)
    print("saved", a.out)


if __name__ == "__main__":
    main()
