"""GPU sweep on the Wiener model: conservative vs estimated GO-IRSA (+ pull-based reference).

    .venv/bin/python -m oracle_study.run_gpu --episodes 50 --eta 1.0 0.95
    .venv/bin/python -m oracle_study.run_gpu --episodes 50 --eta 1.0 0.95 --calib_gain 0.3 --tag _calib

Writes oracle_study/results/episodes{tag}_eta{eta}.csv, one row per episode.
"""

import argparse
import itertools
import multiprocessing as mp
import os
import queue
import time

import numpy as np
import pandas as pd


def _worker(gpu, jobs, results, args):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    from oracle_study.sim import Config
    from oracle_study.sim_gpu import run_batch

    while True:
        try:
            S, eta = jobs.get_nowait()
        except queue.Empty:
            return
        t0 = time.time()
        rows = list(itertools.product(args.methods, args.G, range(args.episodes)))
        o = run_batch(S, [r[0] for r in rows], [r[1] for r in rows], [eta] * len(rows),
                      seed=1000 * S + int(100 * eta), device="cuda:0", cfg=Config(frames=args.frames),
                      calib_gain=args.calib_gain)
        recs = []
        for i, (m, G, ep) in enumerate(rows):
            rec = dict(method=m, S=S, M=(S - 2) * 10, G=G, eta=eta, episode=ep, calib_gain=args.calib_gain)
            for k, v in o.items():
                rec[k] = v[i].mean()
            u = o["unresolved"][i] > 0  # K_u estimation error over frames with collisions
            err = np.abs(o["ku_hat"][i][u] - o["ku_true"][i][u])
            rec["ku_abs_err"] = err.mean() if u.any() else 0.0
            rec["ku_rel_err"] = (err / o["ku_true"][i][u]).mean() if u.any() else 0.0
            recs.append(rec)
        results.put(recs)
        print(f"[gpu{gpu}] S={S} eta={eta} done in {time.time() - t0:.0f}s", flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--episodes", type=int, default=50)
    p.add_argument("--frames", type=int, default=200)
    p.add_argument("--eta", type=float, nargs="+", default=[1.0])
    p.add_argument("--S", type=int, nargs="+", default=[12, 22, 42, 62, 82])
    p.add_argument("--G", type=float, nargs="+", default=[0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8])
    p.add_argument("--methods", nargs="+", default=["conservative", "estimated"])
    p.add_argument("--calib_gain", type=float, default=0.0, help="online load calibration gain (0 = off)")
    p.add_argument("--gpus", type=int, nargs="+", default=[1, 2, 3])
    p.add_argument("--out", default="oracle_study/results")
    p.add_argument("--tag", default="")
    p.add_argument("--no_pull", action="store_true")
    a = p.parse_args()

    ctx = mp.get_context("spawn")
    jobs, results = ctx.Queue(), ctx.Queue()
    combos = list(itertools.product(sorted(a.S, reverse=True), a.eta))
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

    if not a.no_pull:  # pull-based reference (round robin, cheap, CPU)
        from oracle_study.sim import Config, run_pull
        for ep in range(a.episodes):
            r = run_pull(Config(S=82, frames=a.frames), seed=ep)
            recs.append(dict(method="pull", S=82, M=800, G=0.0, eta=1.0, episode=ep,
                             mse=r["mse"].mean(), q95=r["q95"].mean()))

    df = pd.DataFrame(recs)
    os.makedirs(a.out, exist_ok=True)
    for eta in a.eta:
        sub = pd.concat([df[(df.method != "pull") & np.isclose(df.eta, eta)], df[df.method == "pull"]])
        path = os.path.join(a.out, f"episodes{a.tag}_eta{eta}.csv")
        sub.to_csv(path, index=False)
        print("saved", path)


if __name__ == "__main__":
    main()
