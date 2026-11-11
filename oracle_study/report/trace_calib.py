"""Long single-configuration runs on the real data to illustrate the calibration dynamics.

    CUDA_VISIBLE_DEVICES=1 .venv/bin/python -m oracle_study.report.trace_calib
Writes oracle_study/report/data/trace_calib.npz
"""
import os

import numpy as np

from oracle_study.realdata.sim_real import run_batch_real

S, ETA, G, FRAMES, EPISODES, GAIN = 22, 0.95, 0.7, 3000, 8, 0.3


def main():
    z = np.load("data/lcl/ar1.npz")
    te, a, mu, sig = z["test"], z["a"], z["mu"], z["sigma"]
    st = np.random.default_rng(1).integers(0, te.shape[1] - FRAMES - 1, size=EPISODES)
    res = {}
    for name, gain in (("nocalib", 0.0), ("calib", GAIN)):
        o = run_batch_real(S, ["estimated"] * EPISODES + ["conservative"] * EPISODES, [G] * (2 * EPISODES), ETA,
                           np.concatenate([st, st]), te, a, mu, sig, "cuda:0", FRAMES, calib_gain=gain)
        for k in ("load", "G_meas", "G_eff", "mse"):
            res[f"{name}_est_{k}"] = o[k][:EPISODES]
            res[f"{name}_cons_{k}"] = o[k][EPISODES:]
    os.makedirs("oracle_study/report/data", exist_ok=True)
    np.savez("oracle_study/report/data/trace_calib.npz", S=S, eta=ETA, G=G, gain=GAIN, **res)
    print("saved oracle_study/report/data/trace_calib.npz")


if __name__ == "__main__":
    main()
