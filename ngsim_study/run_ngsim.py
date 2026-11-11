"""Sweep on NGSIM speeds (one 0.1 s sample = one slot): conservative vs estimated, with and without
online calibration, plus pull-based.

    .venv/bin/python -m ngsim_study.run_ngsim --episodes 30
Writes ngsim_study/results/episodes.csv (one row per episode).
"""
import argparse
import itertools
import multiprocessing as mp
import os
import queue
import time

import numpy as np
import pandas as pd

METHODS = ("conservative", "estimated")  # overridable with --methods


def _worker(gpu, jobs, results, a):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    import torch
    from ngsim_study.sim_ngsim import episode_starts, load_packed, run_batch, run_pull
    P = load_packed(cells=a.cells)
    while True:
        try:
            kind, S, eta, gain = jobs.get_nowait()
        except queue.Empty:
            return
        t0 = time.time()
        frames = a.duration_slots // S
        st = episode_starts(P, S, frames, a.episodes, seed=S)  # same windows for every method / load / gain
        recs = []
        if kind == "pull":
            o = run_pull(S, st, P, "cuda:0", frames)
            for ep in range(a.episodes):
                recs.append(dict(method="pull", S=S, M=(S - 2) * 10, G=np.nan, eta=1.0, calib_gain=0.0, episode=ep,
                                 mse=o["mse"][ep].mean(), q95=o["q95"][ep].mean()))
        else:
            rows_all = list(itertools.product(a.methods, a.G, range(a.episodes)))
            for c0 in range(0, len(rows_all), a.max_rows):
                rows = rows_all[c0:c0 + a.max_rows]
                o = run_batch(S, [r[0] for r in rows], [r[1] for r in rows], eta, st[[r[2] for r in rows]], P,
                              "cuda:0", frames, calib_gain=gain, ku_est=a.ku_est)
                for i, (m, G, ep) in enumerate(rows):
                    rec = dict(method=m, S=S, M=(S - 2) * 10, G=G, eta=eta, calib_gain=gain, episode=ep, ku_est=a.ku_est)
                    for k, v in o.items():
                        rec[k] = v[i].mean()
                    recs.append(rec)
                del o
                torch.cuda.empty_cache()
        results.put(recs)
        print(f"[gpu{gpu}] {kind} S={S} eta={eta} gain={gain} {time.time() - t0:.0f}s", flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--episodes", type=int, default=30)
    p.add_argument("--duration_slots", type=int, default=3600, help="episode length in samples (0.1 s each)")
    p.add_argument("--cells", type=int, default=7, help="time windows per site stacked in parallel")
    p.add_argument("--S", type=int, nargs="+", default=[12, 22, 42, 62, 82])
    p.add_argument("--eta", type=float, nargs="+", default=[1.0, 0.95])
    p.add_argument("--gains", type=float, nargs="+", default=[0.0, 0.3])
    p.add_argument("--G", type=float, nargs="+", default=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    p.add_argument("--max_rows", type=int, default=200)
    p.add_argument("--gpus", type=int, nargs="+", default=[1, 2, 3])
    p.add_argument("--methods", nargs="+", default=list(METHODS))
    p.add_argument("--ku_est", default="R", choices=["R", "RD"])
    p.add_argument("--no_pull", action="store_true")
    p.add_argument("--out", default="ngsim_study/results/episodes.csv")
    a = p.parse_args()
    ctx = mp.get_context("spawn")
    jobs, results = ctx.Queue(), ctx.Queue()
    combos = [("goirsa", S, e, g) for S in sorted(a.S, reverse=True) for e in a.eta for g in a.gains]
    combos += [] if a.no_pull else [("pull", S, 1.0, 0.0) for S in a.S]
    for c in combos:
        jobs.put(c)
    procs = [ctx.Process(target=_worker, args=(g, jobs, results, a)) for g in a.gpus]
    for pr in procs:
        pr.start()
    recs, done = [], 0
    while done < len(combos):
        try:
            recs += results.get(timeout=30)
            done += 1
        except queue.Empty:
            if not any(pr.is_alive() for pr in procs):
                raise RuntimeError(f"all workers exited, {done}/{len(combos)} jobs done")
    for pr in procs:
        pr.join()
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    pd.DataFrame(recs).to_csv(a.out, index=False)
    print("saved", a.out)


if __name__ == "__main__":
    main()
