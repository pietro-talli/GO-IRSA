"""Per-frame load dispersion on NGSIM: mean / std / 5-95% / RMS-to-target of K_t/M, with and without calibration.
    CUDA_VISIBLE_DEVICES=1 .venv/bin/python -m ngsim_study.load_dispersion S out.csv   (one S per GPU)
Results for all S are concatenated in ngsim_study/results/load_dispersion.csv."""
import itertools, sys
import numpy as np, pandas as pd, torch
from ngsim_study.sim_ngsim import episode_starts, load_packed, run_batch

S, out = int(sys.argv[1]), sys.argv[2]
EP, DUR, G = 30, 3600, [0.3, 0.5, 0.7, 0.9]
P = load_packed(cells=7)
frames = DUR // S
st = episode_starts(P, S, frames, EP, seed=S)
burn = frames // 10
recs = []
for eta in (1.0, 0.95):
    for gain in (0.0, 0.3):
        rows = list(itertools.product(("conservative", "estimated"), G, range(EP)))
        for c0 in range(0, len(rows), 120):
            rr = rows[c0:c0 + 120]
            o = run_batch(S, [r[0] for r in rr], [r[1] for r in rr], eta, st[[r[2] for r in rr]], P, "cuda:0",
                          frames, calib_gain=gain)
            L = o["load"][:, burn:]
            for i, (m, g, ep) in enumerate(rr):
                l = L[i]
                recs.append(dict(S=S, M=(S - 2) * 10, eta=eta, calib=gain > 0, method=m, G=g, episode=ep,
                                 mean=l.mean(), std=l.std(), rms_to_G=np.sqrt(((l - g) ** 2).mean()),
                                 p05=np.quantile(l, .05), p95=np.quantile(l, .95),
                                 over=(l > g).mean()))
            del o; torch.cuda.empty_cache()
        print(S, eta, gain, flush=True)
pd.DataFrame(recs).to_csv(out, index=False)
