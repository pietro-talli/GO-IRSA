"""Analyze the NGSIM sweep.  .venv/bin/python -m ngsim_study.analyze [--M 200] [--eta 0.95]"""
import argparse

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

OUT = "ngsim_study/results/"
C = {"cons": "#2a78d6", "est": "#1baf7a", "estc": "#eb6834", "pull": "#6b6a64"}
V = [("cons", "conservative [paper]", C["cons"], "-", "o"), ("est", r"estimated $\hat K_u$", C["est"], "-", "^"),
     ("estc", r"estimated $\hat K_u$ + calib.", C["estc"], "-", "D"), ("consc", "conservative + calib.", C["cons"], ":", "o")]
plt.rcParams.update({"font.size": 7.5, "font.family": "serif", "mathtext.fontset": "cm",
                     "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
                     "grid.color": "#e4e3dd", "grid.linewidth": 0.5, "axes.edgecolor": "#9a998f",
                     "lines.linewidth": 1.4, "lines.markersize": 3.2, "legend.frameon": False})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--M", type=int, default=200)
    ap.add_argument("--eta", type=float, default=0.95)
    a = ap.parse_args()
    d = pd.read_csv(OUT + "episodes.csv")
    pull = d[d.method == "pull"].groupby("M")[["mse", "q95"]].mean()
    g = d[d.method != "pull"].copy()
    g["variant"] = np.where(g.method == "estimated", "est", "cons") + np.where(g.calib_gain > 0, "c", "")
    s = g.groupby(["eta", "M", "variant", "G"])[["mse", "q95", "load", "throughput", "unresolved", "n_active"]].mean()
    print("mean occupied node slots:", round(g.n_active.mean()))

    best = s[["mse", "q95"]].groupby(level=[0, 1, 2]).min()
    Gs = s["mse"].groupby(level=[0, 1, 2]).idxmin().map(lambda t: t[3])
    tab = []
    for (eta, M), sub in best.groupby(level=[0, 1]):
        sub = sub.droplevel([0, 1])
        row = dict(eta=eta, M=M)
        for metric in ("mse", "q95"):
            ref = sub.loc["cons", metric]
            for v in ("est", "consc", "estc"):
                row[f"{metric} {v} vs cons %"] = 100 * (1 - sub.loc[v, metric] / ref)
            for v in ("cons", "est", "estc"):
                row[f"{metric} {v} vs pull %"] = 100 * (1 - sub.loc[v, metric] / pull.loc[M, metric])
        for v in ("cons", "est", "estc"):
            row[f"G* {v}"] = Gs.loc[(eta, M, v)]
        tab.append(row)
    tab = pd.DataFrame(tab).set_index(["eta", "M"])
    pd.set_option("display.width", 250)
    print("Improvement (% lower error), each variant at its best target load:")
    print(tab.round(1).to_string())
    tab.round(2).to_csv(OUT + "improvement.csv")
    print("\nabsolute best MSE:")
    print(best["mse"].unstack(2).round(4).to_string())
    print("\npull-based:")
    print(pull.round(4).to_string())
    lt = s.xs(0.7, level="G")[["load", "throughput", "unresolved"]].unstack("variant")
    print("\nload / throughput / unresolved at target 0.7:")
    print(lt.round(3).to_string())

    # figure (same layout as Fig. 2 of the report)
    eta, M = a.eta, a.M
    fig, ax = plt.subplots(1, 3, figsize=(7.0, 2.1))
    for v, lab, col, ls, mk in V:
        y = s.loc[(eta, M, v)]
        ax[0].plot(y.index, y["load"], color=col, ls=ls, marker=mk, label=lab)
        ax[1].plot(y.index, y["mse"], color=col, ls=ls, marker=mk, label=lab)
    gs = s.loc[(eta, M, "cons")].index
    ax[0].plot([gs.min(), gs.max()], [gs.min(), gs.max()], color=C["pull"], ls=":", lw=1, label=r"target $\bar G$")
    ax[0].set(xlabel=r"target load $\bar G$", ylabel="actual load $G$", title=f"(a) load, $M={M}$", ylim=(0, 2.0))
    ax[1].axhline(pull.loc[M, "mse"], color=C["pull"], ls="--", lw=1, label="pull-based")
    vals = np.concatenate([s.loc[(eta, M, v)]["mse"].values for v in ("cons", "est", "estc")])
    ax[1].set(xlabel=r"target load $\bar G$", ylabel=r"MSE $\bar E$", title=f"(b) MSE, $M={M}$",
              ylim=(0.9 * min(vals.min(), pull.loc[M, "mse"]), 1.1 * max(np.percentile(vals, 90), pull.loc[M, "mse"])))
    h0, l0 = ax[0].get_legend_handles_labels()
    h1, l1 = ax[1].get_legend_handles_labels()
    fig.legend(h0 + h1[-1:], l0 + l1[-1:], loc="upper center", ncol=6, bbox_to_anchor=(0.5, 1.0), fontsize=6.5)
    b = best["mse"].unstack(2).loc[eta]
    bq = best["q95"].unstack(2).loc[eta]
    ax[2].plot(b.index, 100 * (1 - b["est"] / b["cons"]), color=C["est"], marker="^", label="MSE, est.")
    ax[2].plot(b.index, 100 * (1 - b["estc"] / b["cons"]), color=C["estc"], marker="D", label="MSE, est. + calib.")
    ax[2].plot(bq.index, 100 * (1 - bq["estc"] / bq["cons"]), color=C["estc"], marker="D", mfc="white", ls="--",
               label="$Q_{95}$, est. + calib.")
    ax[2].plot(b.index, 100 * (1 - pull["mse"].reindex(b.index) / b["cons"]), color=C["pull"], ls="--", marker="x",
               label="MSE, pull-based")
    ax[2].axhline(0, color=C["pull"], lw=0.8)
    ax[2].set(xlabel="frame size $M$ [REs]", ylabel="reduction vs. conservative [%]", title=r"(c) gain at best $\bar G$")
    ax[2].set_xticks([100, 200, 400, 600, 800])
    ax[2].legend(fontsize=5.8)
    fig.tight_layout(pad=0.4, w_pad=1.0, rect=(0, 0, 1, 0.9))
    fig.savefig(OUT + "ngsim.pdf")
    fig.savefig(OUT + "ngsim.png", dpi=200)
    print("saved", OUT + "ngsim.pdf/.png")


if __name__ == "__main__":
    main()
