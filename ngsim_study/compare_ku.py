"""Compare the K_u estimators E[K_u | R] and E[K_u | R, |D|] on NGSIM.

    .venv/bin/python -m ngsim_study.compare_ku
"""
import numpy as np
import pandas as pd

R = "ngsim_study/results/"


def main():
    a = pd.read_csv(R + "episodes.csv")
    a = a[a.method == "estimated"].assign(ku_est="R")
    b = pd.read_csv(R + "episodes_RD.csv")
    d = pd.concat([a, b], ignore_index=True)
    d["v"] = d.ku_est + np.where(d.calib_gain > 0, "+calib", "")
    s = d.groupby(["eta", "M", "v", "G"])[["mse", "q95", "load", "G_meas"]].mean()
    pd.set_option("display.width", 250)
    for eta in (0.95, 1.0):
        print(f"\n===== eta = {eta}")
        x = s.xs(0.7, level="G").loc[eta][["load", "G_meas"]].unstack("v")
        print("actual load / measured load at target 0.7:")
        print(x.round(3).to_string())
        best = s.loc[eta][["mse", "q95"]].groupby(level=[0, 1]).min().unstack("v")
        print("\nbest MSE and Q95 (each at its own best target):")
        print(best.round(4).to_string())
        for metric in ("mse", "q95"):
            bm = best[metric]
            print(f"\n{metric}: RD vs R (no calib) {100 * (1 - bm['RD'] / bm['R'])}".replace("\n", " ")
                  if False else "")
        rows = []
        for M in sorted(d.M.unique()):
            r = {"M": M}
            for v in ("R", "RD", "R+calib", "RD+calib"):
                y = s.loc[(eta, M, v)]
                ok = y.index[y.mse <= 1.05 * y.mse.min()]
                r[f"win {v}"] = f"{ok.min():.1f}-{ok.max():.1f}"
            for metric in ("mse", "q95"):
                bm = best[metric].loc[M]
                r[f"{metric} RD vs R %"] = 100 * (1 - bm["RD"] / bm["R"])
                r[f"{metric} RD+calib vs R+calib %"] = 100 * (1 - bm["RD+calib"] / bm["R+calib"])
                r[f"{metric} RD+calib vs R (no calib) %"] = 100 * (1 - bm["RD+calib"] / bm["R"])
            rows.append(r)
        print(pd.DataFrame(rows).set_index("M").round(1).to_string())


if __name__ == "__main__":
    main()
