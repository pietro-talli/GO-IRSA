"""Figures, tables and in-text numbers for the LaTeX report (CPU only, reads the CSVs).

    .venv/bin/python -m oracle_study.report.make_figures
Writes oracle_study/report/figs/*.pdf, oracle_study/report/tables.tex, oracle_study/report/numbers.tex
"""
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

RES = "oracle_study/results/"
OUT = "oracle_study/report/"
C = {"cons": "#2a78d6", "est": "#1baf7a", "estc": "#eb6834", "pull": "#6b6a64"}
plt.rcParams.update({"font.size": 7.5, "font.family": "serif", "mathtext.fontset": "cm",
                     "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
                     "grid.color": "#e4e3dd", "grid.linewidth": 0.5, "axes.edgecolor": "#9a998f",
                     "lines.linewidth": 1.4, "lines.markersize": 3.2, "legend.fontsize": 6.5,
                     "legend.frameon": False, "axes.titlesize": 7.5, "axes.labelsize": 7.5,
                     "xtick.labelsize": 6.5, "ytick.labelsize": 6.5})
W, H = 7.0, 1.85  # inches: full text width, compact height
M_SHOW, ETA_SHOW = 200, 0.95  # Wiener figure
M_REAL = 400  # real-data figure: regime where uncalibrated GO-IRSA already beats pull-based


def save(fig, name):
    """PDF for the report, PNG for quick viewing."""
    fig.savefig(OUT + f"figs/{name}.pdf")
    fig.savefig(OUT + f"figs/{name}.png", dpi=200)
    plt.close(fig)


def best_gain(s, ref, var, metric):
    """% reduction of the best-over-G metric of `var` w.r.t. `ref`; s indexed by (eta, M, variant, G)."""
    b = s[metric].groupby(level=[0, 1, 2]).min().unstack(2)
    return 100 * (1 - b[var] / b[ref])


def load_synth():
    d = pd.concat([pd.read_csv(RES + f"episodes_eta{e}.csv") for e in ("1.0", "0.95")])
    dc = pd.concat([pd.read_csv(RES + f"episodes_calib_eta{e}.csv") for e in ("1.0", "0.95")])
    pull = d[d.method == "pull"].drop_duplicates(["episode"])[["mse", "q95"]].mean()
    d = d[d.method != "pull"].assign(variant=lambda x: x.method.map({"conservative": "cons", "estimated": "est"}))
    dc = dc[dc.method != "pull"].assign(variant=lambda x: x.method.map({"conservative": "consc", "estimated": "estc"}))
    return pd.concat([d, dc]), pull


def load_real():
    d = pd.read_csv(RES + "real_hold_episodes.csv")
    dc = pd.read_csv(RES + "real_hold_episodes_calib.csv")
    pull = d[d.method == "pull"].groupby("M")[["mse", "q95"]].mean()
    d = d[d.method != "pull"].assign(variant=lambda x: x.method.map({"conservative": "cons", "estimated": "est"}))
    dc = dc.assign(variant=lambda x: x.method.map({"conservative": "consc", "estimated": "estc"}))
    return pd.concat([d, dc]), pull


def curves(ax, s, eta, M, metric, variants, gmax=np.inf):
    for v, lab, col, ls, mk in variants:
        y = s.loc[(eta, M, v)][metric]
        y = y[y.index <= gmax + 1e-9]
        ax.plot(y.index, y.values, color=col, ls=ls, marker=mk, label=lab)


def fig_synth(s, pull):
    fig, ax = plt.subplots(1, 3, figsize=(W, H))
    V = [("cons", "conservative [paper]", C["cons"], "-", "o"), ("est", r"estimated $\hat K_u$", C["est"], "-", "^")]
    curves(ax[0], s, ETA_SHOW, M_SHOW, "load", V)
    ax[0].plot([0.5, 0.8], [0.5, 0.8], color=C["pull"], ls=":", lw=1, label=r"target $\bar G$")
    ax[0].set(xlabel=r"target load $\bar G$", ylabel=r"actual load $G$", title=f"(a) load, $M={M_SHOW}$")
    ax[0].legend(loc="upper left")
    curves(ax[1], s, ETA_SHOW, M_SHOW, "mse", V)
    ax[1].axhline(pull["mse"], color=C["pull"], ls="--", lw=1, label="pull-based")
    ax[1].set(xlabel=r"target load $\bar G$", ylabel=r"MSE $\bar E$", title=f"(b) MSE, $M={M_SHOW}$")
    ax[1].legend(loc="upper right")
    for eta, col, lab in ((1.0, "#86d9b6", r"perfect SIC ($\eta=1$)"), (0.95, C["est"], r"$\eta=0.95$")):
        g = best_gain(s, "cons", "est", "mse").loc[eta]
        ax[2].plot(g.index, g.values, color=col, marker="^", label=lab)
    ax[2].axhline(0, color=C["pull"], lw=0.8)
    ax[2].set(xlabel="frame size $M$ [REs]", ylabel="reduction vs. conservative [%]", title=r"(c) MSE gain at best $\bar G$")
    ax[2].set_xticks([100, 200, 400, 600, 800])
    ax[2].legend(loc="center right")
    fig.tight_layout(pad=0.4, w_pad=1.0)
    save(fig, "synthetic")


def fig_real(s, pull, gmax=0.9):
    """Real data; target range limited to the feasible region (below the saturation of the
    measured load, see report Sec. 4)."""
    fig, ax = plt.subplots(1, 3, figsize=(W, H + 0.25))
    V = [("cons", "conservative [paper]", C["cons"], "-", "o"), ("est", r"estimated $\hat K_u$", C["est"], "-", "^"),
         ("estc", r"estimated $\hat K_u$ + calib.", C["estc"], "-", "D"),
         ("consc", "conservative + calib.", C["cons"], ":", "o")]
    key = (ETA_SHOW, M_REAL)
    curves(ax[0], s, *key, "load", V, gmax)
    gs = s.loc[(*key, "cons")].index
    gs = gs[gs <= gmax + 1e-9]
    ax[0].plot([gs.min(), gs.max()], [gs.min(), gs.max()], color=C["pull"], ls=":", lw=1, label=r"target $\bar G$")
    ymax = 1.08
    ax[0].set(xlabel=r"target load $\bar G$", ylabel=r"actual load $G$", title=f"(a) load, $M={M_REAL}$",
              ylim=(0.15, ymax))
    yc = s.loc[(*key, "consc")]["load"]
    g_off = yc[yc > ymax].index.min()
    ax[0].text(0.97, 0.05, f"cons. + calib.: $G>1$ for $\\bar G\\geq{g_off:.1f}$", transform=ax[0].transAxes,
               fontsize=6, color=C["cons"], ha="right")
    curves(ax[1], s, *key, "mse", V, gmax)
    ax[1].axhline(pull.loc[M_REAL, "mse"], color=C["pull"], ls="--", lw=1, label="pull-based")
    vals = np.concatenate([s.loc[(*key, v)]["mse"][lambda y: y.index <= gmax + 1e-9].values for v in ("cons", "est", "estc")])
    lo, hi = vals.min(), max(vals.max(), pull.loc[M_REAL, "mse"])
    pad = 0.08 * (hi - lo)
    ax[1].set(xlabel=r"target load $\bar G$", ylabel=r"MSE $\bar E$", title=f"(b) MSE, $M={M_REAL}$",
              ylim=(lo - pad, hi + pad))
    h0, l0 = ax[0].get_legend_handles_labels()
    h1, l1 = ax[1].get_legend_handles_labels()
    fig.legend(h0 + h1[-1:], l0 + l1[-1:], loc="upper center", ncol=6, bbox_to_anchor=(0.5, 1.0), fontsize=6.5,
               handlelength=2.2, columnspacing=1.2)
    b = s["mse"].groupby(level=[0, 1, 2]).min().unstack(2).loc[ETA_SHOW]
    bq = s["q95"].groupby(level=[0, 1, 2]).min().unstack(2).loc[ETA_SHOW]
    ax[2].plot(b.index, 100 * (1 - b["est"] / b["cons"]), color=C["est"], marker="^", label="MSE, est.")
    ax[2].plot(b.index, 100 * (1 - b["estc"] / b["cons"]), color=C["estc"], marker="D", label="MSE, est. + calib.")
    ax[2].plot(bq.index, 100 * (1 - bq["estc"] / bq["cons"]), color=C["estc"], marker="D", mfc="white", ls="--",
               label="$Q_{95}$, est. + calib.")
    ax[2].plot(b.index, 100 * (1 - pull["mse"].reindex(b.index) / b["cons"]), color=C["pull"], ls="--", marker="x",
               label="MSE, pull-based")
    ax[2].axhline(0, color=C["pull"], lw=0.8)
    ax[2].set(xlabel="frame size $M$ [REs]", ylabel="reduction vs. conservative [%]", title=r"(c) gain at best $\bar G$")
    ax[2].set_xticks([100, 200, 400, 600, 800])
    ax[2].legend(loc="upper right", fontsize=5.8, handlelength=2.0)
    fig.tight_layout(pad=0.4, w_pad=1.0, rect=(0, 0, 1, 0.9))
    save(fig, "real")


def fig_trace():
    z = np.load(OUT + "data/trace_calib.npz")
    G, gain = float(z["G"]), float(z["gain"])
    fig, ax = plt.subplots(1, 2, figsize=(W * 0.72, H))
    day = 48
    ker = np.ones(day) / day
    t = np.arange(z["calib_est_G_meas"].shape[1])
    for key, lab, col in (("nocalib_est", "estimated, no calib.", C["est"]), ("calib_est", "estimated + calib.", C["estc"]),
                          ("calib_cons", "conservative + calib.", C["cons"])):
        g = z[f"{key}_G_meas"][0]
        ax[0].plot(t[: 10 * day], g[: 10 * day], color=col, lw=0.4, alpha=0.35)
        ax[0].plot(t[day - 1: 10 * day], np.convolve(g, ker, "valid")[: 10 * day - day + 1], color=col, lw=1.4, label=lab)
    ax[0].axhline(G, color=C["pull"], ls=":", lw=1)
    ax[0].set(xlabel="frame $k$ (half hours)", ylabel=r"measured load $\hat G(k)$", title="(a) measured load (1-day average)")
    ax[0].legend(loc="upper right", fontsize=6)
    T = np.arange(1, t.size + 1)
    for key, lab, col in (("calib_est", "estimated + calib.", C["estc"]), ("nocalib_est", "estimated, no calib.", C["est"])):
        err = np.abs(np.cumsum(z[f"{key}_G_meas"] - G, axis=1) / T).mean(0)
        ax[1].loglog(T, err, color=col, label=lab)
    geff = z["calib_est_G_eff"]
    bound = (np.abs(geff - geff[:, :1]).max(1).mean()) / (gain * T)
    ax[1].loglog(T, bound, color=C["pull"], ls="--", lw=1, label=r"bound $\Delta/(\gamma T)$")
    ax[1].set(xlabel="horizon $T$ [frames]", ylabel=r"$|\frac{1}{T}\sum_k \hat G(k) - \bar G|$", title="(b) long-run load error")
    ax[1].set_ylim(1e-4, 1)
    ax[1].legend(loc="lower left", fontsize=6)
    fig.tight_layout(pad=0.4, w_pad=1.0)
    save(fig, "trace")


def tables(ss, ps, sr, pr):
    Ms = [100, 200, 400, 800]
    lines = [r"\begin{tabular}{@{}l" + "r" * len(Ms) + "@{}}", r"\toprule",
             r" & " + " & ".join(f"$M={m}$" for m in Ms) + r" \\", r"\midrule"]

    def row(lab, series):
        lines.append(lab + " & " + " & ".join(f"{series.loc[m]:+.1f}" for m in Ms) + r" \\")

    for eta in (1.0, 0.95):
        lines.append(rf"\multicolumn{{{len(Ms) + 1}}}{{@{{}}l}}{{\emph{{Wiener model, $\eta={eta:g}$}}}} \\")
        row(r"\quad MSE, estimated", best_gain(ss, "cons", "est", "mse").loc[eta])
        row(r"\quad $Q_{95}$, estimated", best_gain(ss, "cons", "est", "q95").loc[eta])
    for eta in (1.0, 0.95):
        lines.append(rf"\multicolumn{{{len(Ms) + 1}}}{{@{{}}l}}{{\emph{{Smart-meter data, $\eta={eta:g}$}}}} \\")
        row(r"\quad MSE, estimated", best_gain(sr, "cons", "est", "mse").loc[eta])
        row(r"\quad MSE, estimated + calib.", best_gain(sr, "cons", "estc", "mse").loc[eta])
        row(r"\quad $Q_{95}$, estimated + calib.", best_gain(sr, "cons", "estc", "q95").loc[eta])
    lines += [r"\bottomrule", r"\end{tabular}"]
    open(OUT + "tables.tex", "w").write("\n".join(lines) + "\n")

    # in-text numbers
    n = {}
    L = ss["load"].xs(0.7, level="G")
    n["synLoadCons"] = L.loc[(0.95, 200, "cons")]
    n["synLoadEst"] = L.loc[(0.95, 200, "est")]
    n["synLoadConsPerfect"] = L.loc[(1.0, 200, "cons")]
    Lr = sr["load"].xs(0.7, level="G")
    for v in ("cons", "est", "estc", "consc"):
        n[f"realLoad{v.capitalize()}"] = Lr.loc[(0.95, 200, v)]
    n["realLoadEstcMin"] = Lr.xs("estc", level=2).min()
    n["realLoadEstcMax"] = Lr.xs("estc", level=2).max()
    n["realLoadConscMax"] = Lr.xs("consc", level=2).max()
    gs = best_gain(ss, "cons", "est", "mse")
    n["synGainMaxImperfect"] = gs.loc[0.95].max()
    n["synGainMaxPerfect"] = gs.loc[1.0].max()
    ge = best_gain(sr, "cons", "estc", "mse")
    n["realGainEstcMin"], n["realGainEstcMax"] = ge.min(), ge.max()
    gq = best_gain(sr, "cons", "estc", "q95")
    n["realGainQEstcMax"] = gq.max()
    ge0 = best_gain(sr, "cons", "est", "mse")
    n["realGainEstMax"] = ge0.max()
    gc = best_gain(sr, "cons", "consc", "mse")
    n["realGainConscMin"] = gc.min()
    best_estc = sr["mse"].groupby(level=[0, 1, 2]).min().xs("estc", level=2)
    vs_pull = 100 * (1 - best_estc / pr["mse"].reindex(best_estc.index.get_level_values(1)).values)
    n["realVsPullMin"], n["realVsPullMax"] = vs_pull.min(), vs_pull.max()
    # uncalibrated GO-IRSA vs pull-based (real data, M >= 400)
    for metric, key in (("mse", "Mse"), ("q95", "Q")):
        bb = sr[metric].groupby(level=[0, 1, 2]).min()
        unc = pd.concat([bb.xs(v, level=2) for v in ("cons", "est")], axis=1).min(axis=1)
        unc = unc[unc.index.get_level_values(1) >= 400]
        vp = 100 * (1 - unc / pr[metric].reindex(unc.index.get_level_values(1)).values)
        n[f"realUncalVsPull{key}Min"], n[f"realUncalVsPull{key}Max"] = vp.min(), vp.max()
        cal = bb.xs("estc", level=2)
        cal = cal[cal.index.get_level_values(1) >= 400]
        vc = 100 * (1 - cal / pr[metric].reindex(cal.index.get_level_values(1)).values)
        n[f"realCalVsPull{key}Min"], n[f"realCalVsPull{key}Max"] = vc.min(), vc.max()
    n["realLoadEstFour"] = Lr.loc[(0.95, M_REAL, "est")]
    n["realLoadEstcFour"] = Lr.loc[(0.95, M_REAL, "estc")]
    return n


def write_numbers(n):
    with open(OUT + "numbers.tex", "w") as f:
        for k, v in n.items():
            if isinstance(v, str):
                f.write(f"\\newcommand{{\\{k}}}{{{v}}}\n")
            else:
                f.write(f"\\newcommand{{\\{k}}}{{{v:.2f}}}\n" if ("Load" in k or k.startswith("ngMse")) else f"\\newcommand{{\\{k}}}{{{v:.1f}}}\n")
    print(open(OUT + "numbers.tex").read())


# ---------------------------------------------------------------------------- NGSIM (real-time data)
NGSIM = "ngsim_study/results/episodes.csv"
M_NG = 200


def load_ngsim():
    d = pd.read_csv(NGSIM)
    pull = d[d.method == "pull"].groupby("M")[["mse", "q95"]].mean()
    d = d[d.method != "pull"].copy()
    d["variant"] = np.where(d.method == "estimated", "est", "cons") + np.where(d.calib_gain > 0, "c", "")
    return d.groupby(["eta", "M", "variant", "G"])[["mse", "q95", "load"]].mean(), pull


def fig_ngsim(s, pull):
    fig, ax = plt.subplots(1, 3, figsize=(W, H + 0.25))
    V = [("cons", "conservative [paper]", C["cons"], "-", "o"), ("est", r"estimated $\hat K_u$", C["est"], "-", "^"),
         ("estc", r"estimated $\hat K_u$ + calib.", C["estc"], "-", "D"),
         ("consc", "conservative + calib.", C["cons"], ":", "o")]
    key = (ETA_SHOW, M_NG)
    curves(ax[0], s, *key, "load", V)
    gs = s.loc[(*key, "cons")].index
    ax[0].plot([gs.min(), gs.max()], [gs.min(), gs.max()], color=C["pull"], ls=":", lw=1, label=r"target $\bar G$")
    ax[0].set(xlabel=r"target load $\bar G$", ylabel=r"actual load $G$", title=f"(a) load, $M={M_NG}$", ylim=(0, 2.05))
    curves(ax[1], s, *key, "mse", V)
    ax[1].axhline(pull.loc[M_NG, "mse"], color=C["pull"], ls="--", lw=1, label="pull-based")
    ax[1].set(xlabel=r"target load $\bar G$", ylabel=r"MSE $\bar E$", title=f"(b) MSE, $M={M_NG}$")
    h0, l0 = ax[0].get_legend_handles_labels()
    h1, l1 = ax[1].get_legend_handles_labels()
    fig.legend(h0 + h1[-1:], l0 + l1[-1:], loc="upper center", ncol=6, bbox_to_anchor=(0.5, 1.0), fontsize=6.5,
               handlelength=2.2, columnspacing=1.2)
    b = s["mse"].groupby(level=[0, 1, 2]).min().unstack(2).loc[ETA_SHOW]
    bq = s["q95"].groupby(level=[0, 1, 2]).min().unstack(2).loc[ETA_SHOW]
    for v, col, mk, lab in (("cons", C["cons"], "o", "conservative"), ("est", C["est"], "^", "est."),
                            ("estc", C["estc"], "D", "est. + calib.")):
        ax[2].plot(b.index, 100 * (1 - b[v] / pull["mse"].reindex(b.index)), color=col, marker=mk, label=f"MSE, {lab}")
    ax[2].plot(bq.index, 100 * (1 - bq["estc"] / pull["q95"].reindex(bq.index)), color=C["estc"], marker="D",
               mfc="white", ls="--", label="$Q_{95}$, est. + calib.")
    ax[2].axhline(0, color=C["pull"], lw=0.8)
    ax[2].set(xlabel="frame size $M$ [REs]", ylabel="reduction vs. pull-based [%]", title=r"(c) gain over pull at best $\bar G$")
    ax[2].set_xticks([100, 200, 400, 600, 800])
    ax[2].legend(loc="upper right", fontsize=5.8, handlelength=2.0)
    fig.tight_layout(pad=0.4, w_pad=1.0, rect=(0, 0, 1, 0.9))
    save(fig, "ngsim")


def ngsim_numbers(s, pull):
    n = {}
    e = ETA_SHOW
    L = s["load"].xs(0.7, level="G")
    n["ngLoadCons"] = L.loc[(e, M_NG, "cons")]
    n["ngLoadEst"] = L.loc[(e, M_NG, "est")]
    n["ngLoadEstc"] = L.loc[(e, M_NG, "estc")]
    lc = L.xs("cons", level=2).loc[e]
    n["ngLoadConsMin"], n["ngLoadConsMax"] = lc.min(), lc.max()
    best = s[["mse", "q95"]].groupby(level=[0, 1, 2]).min()
    for metric, key in (("mse", "Mse"), ("q95", "Q")):
        bb = best[metric].unstack(2).loc[e]
        red = 100 * (1 - bb[["cons", "est", "estc"]].div(pull[metric].reindex(bb.index), axis=0))
        small = red.loc[red.index <= 400]
        n[f"ngVsPull{key}Min"], n[f"ngVsPull{key}Max"] = small.values.min(), small.values.max()
        big = red.loc[800]
        n[f"ngVsPull{key}EightMin"], n[f"ngVsPull{key}EightMax"] = big.min(), big.max()
        n[f"ngPullBetter{key}Min"], n[f"ngPullBetter{key}Max"] = -big.max(), -big.min()
        g_est = 100 * (1 - bb["est"] / bb["cons"])
        n[f"ngEstGain{key}Hundred"] = g_est.loc[100]
        g_cal = 100 * (1 - bb["estc"] / bb["cons"])
        rest = g_cal.loc[g_cal.index >= 200]
        n[f"ngCalGain{key}Min"], n[f"ngCalGain{key}Max"] = rest.min(), rest.max()
    m07 = s["mse"].xs(0.7, level="G").loc[e]
    for v, key in (("cons", "Cons"), ("est", "Est"), ("estc", "Estc")):
        y = m07.xs(v, level=1)
        n[f"ngMseSeven{key}Min"], n[f"ngMseSeven{key}Max"] = y.min(), y.max()
        # target range where the MSE stays within 5% of its best (union over M)
        lo, hi = [], []
        for M, z in s["mse"].xs((e, v), level=(0, 2)).groupby(level=0):
            z = z.droplevel(0)
            ok = z[z <= 1.05 * z.min()].index
            lo.append(ok.min()); hi.append(ok.max())
        n[f"ngWin{key}"] = f"{min(lo):.1f}--{max(hi):.1f}"
    return n


def main():
    os.makedirs(OUT + "figs", exist_ok=True)
    ds, ps = load_synth()
    ss = ds.groupby(["eta", "M", "variant", "G"])[["mse", "q95", "load"]].mean()
    dr, pr = load_real()
    sr = dr.groupby(["eta", "M", "variant", "G"])[["mse", "q95", "load"]].mean()
    fig_synth(ss, ps)
    fig_real(sr, pr)
    if os.path.exists(OUT + "data/trace_calib.npz"):
        fig_trace()
    n = tables(ss, ps, sr, pr)
    if os.path.exists(NGSIM):
        sn, pn = load_ngsim()
        fig_ngsim(sn, pn)
        n.update(ngsim_numbers(sn, pn))
    write_numbers(n)
    # K_u estimator accuracy (Wiener sweep)
    ke = ds[(ds.variant == "est") & ds.ku_rel_err.notna()].groupby(["eta", "M"])["ku_rel_err"].mean()
    print("K_u relative error:", (100 * ke).round(1).to_dict())
    print("saved figures to", OUT + "figs/")


if __name__ == "__main__":
    main()
