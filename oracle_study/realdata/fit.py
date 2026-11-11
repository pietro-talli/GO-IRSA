"""Select households, fit a per-household AR(1) model and estimate the noise parameters.

    .venv/bin/python -m oracle_study.realdata.fit

Model (one sample = one half hour = one slot):
    x_{t+1} = a x_t + mu + w_t,    w_t ~ N(0, sigma^2)
fitted by OLS on the training split. mu is the mean of the innovation (intercept),
sigma its std. Validation-split residual mean/std are reported as a check.

Output: data/lcl/ar1.npz with the scaled test traces and model parameters, and
oracle_study/results/realdata_fit.png / realdata_fit.csv with diagnostics.
"""
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

N_NODES = 2000
WINDOW = 27_400  # half-hours: 30% train, 10% val, 60% test (>= 200 frames x 82 slots)
SPLIT = (0.30, 0.10)
MAX_MISSING = 0.01
TARGET_STATIONARY_STD = 1.0  # global rescaling: median stationary error std = 1 (grid units)


def fill_linear(x):
    """Linear interpolation of NaNs along time (per row), edge values held."""
    out = x.copy()
    t = np.arange(x.shape[1])
    for i in np.flatnonzero(np.isnan(x).any(1)):
        m = ~np.isnan(x[i])
        out[i] = np.interp(t, t[m], x[i, m])
    return out


def ar1_ols(x):
    """Row-wise OLS of x[t+1] on [x[t], 1]. Returns a, mu, sigma, residuals."""
    x0, x1 = x[:, :-1], x[:, 1:]
    m0, m1 = x0.mean(1, keepdims=True), x1.mean(1, keepdims=True)
    cov = ((x0 - m0) * (x1 - m1)).mean(1)
    var = ((x0 - m0) ** 2).mean(1)
    a = cov / np.maximum(var, 1e-12)
    mu = m1[:, 0] - a * m0[:, 0]
    res = x1 - a[:, None] * x0 - mu[:, None]
    return a, mu, res.std(1), res


def main():
    kwh = np.load("data/lcl/kwh.npy", mmap_mode="r")
    meta = json.load(open("data/lcl/meta.json"))
    T = kwh.shape[1]
    # last full day with good coverage
    cov_t = (~np.isnan(kwh[:, -2000:])).mean(0)
    end = T - 2000 + int(np.flatnonzero(cov_t > 0.5).max()) + 1
    start = end - WINDOW
    win = np.asarray(kwh[:, start:end], dtype=np.float64)
    miss = np.isnan(win).mean(1)
    n_tr0 = int(SPLIT[0] * WINDOW)
    alive = np.nanstd(win[:, 1:n_tr0] - win[:, :n_tr0 - 1], axis=1) > 1e-3  # drop dead / constant meters
    ok = np.flatnonzero((miss <= MAX_MISSING) & alive)
    t0 = pd.Timestamp(meta["t0"]) + pd.Timedelta(minutes=30) * start
    print(f"window {t0} .. {t0 + pd.Timedelta(minutes=30) * WINDOW}, households with <= {MAX_MISSING:.0%} "
          f"missing: {ok.size} / {kwh.shape[0]}")
    rng = np.random.default_rng(0)
    sel = np.sort(rng.choice(ok, size=N_NODES, replace=False))
    X = fill_linear(win[sel])
    n_tr, n_va = int(SPLIT[0] * WINDOW), int(SPLIT[1] * WINDOW)
    tr, va, te = X[:, :n_tr], X[:, n_tr:n_tr + n_va], X[:, n_tr + n_va:]

    a, mu, sig, res = ar1_ols(tr)
    a = np.clip(a, -0.999, 0.999)
    va_res = va[:, 1:] - a[:, None] * va[:, :-1] - mu[:, None]
    stat_std = sig / np.sqrt(1 - a ** 2)
    scale = TARGET_STATIONARY_STD / np.median(stat_std)

    # residual diagnostics
    z = res / sig[:, None]
    kurt = (z ** 4).mean(1) - 3
    skew = (z ** 3).mean(1)
    acf1 = (z[:, 1:] * z[:, :-1]).mean(1)
    lag48 = (z[:, 48:] * z[:, :-48]).mean(1)  # one day
    df = pd.DataFrame(dict(household=np.array(meta["ids"])[sel], a=a, mu_kwh=mu, sigma_kwh=sig,
                           val_res_mean_kwh=va_res.mean(1), val_res_std_kwh=va_res.std(1),
                           stationary_std_kwh=stat_std, excess_kurtosis=kurt, skewness=skew,
                           res_acf_lag1=acf1, res_acf_lag48=lag48, missing_frac=miss[sel]))
    df.to_csv("oracle_study/results/realdata_fit.csv", index=False)
    q = df.describe(percentiles=[0.05, 0.25, 0.5, 0.75, 0.95]).T
    print(q[["mean", "5%", "25%", "50%", "75%", "95%"]].round(4).to_string())
    # one-step validation MSE: AR(1) vs persistence (random walk) vs mean
    mse_ar = (va_res ** 2).mean()
    mse_rw = ((va[:, 1:] - va[:, :-1]) ** 2).mean()
    mse_mean = ((va - tr.mean(1, keepdims=True)) ** 2).mean()
    print(f"one-step validation MSE [kWh^2]: AR(1) {mse_ar:.5f}, random walk {mse_rw:.5f}, "
          f"constant mean {mse_mean:.5f}")
    print(f"global scale factor: {scale:.3f} (1 kWh -> {scale:.3f} units); "
          f"scaled sigma median {np.median(sig) * scale:.3f}, max {sig.max() * scale:.3f}")

    np.savez("data/lcl/ar1.npz", test=(te * scale).astype(np.float32), a=a, mu=mu * scale, sigma=sig * scale,
             scale=scale, households=np.array(meta["ids"])[sel], test_start=str(t0 + pd.Timedelta(minutes=30) * (n_tr + n_va)))

    # figure
    fig, ax = plt.subplots(2, 3, figsize=(11, 6))
    ax[0, 0].hist(a, 50, color="#2a78d6"); ax[0, 0].set_title("AR coefficient $a_n$")
    ax[0, 1].hist(sig, 50, color="#2a78d6"); ax[0, 1].set_title("innovation std $\\sigma_n$ [kWh]")
    ax[0, 2].hist(mu, 50, color="#2a78d6"); ax[0, 2].set_title("innovation mean $\\mu_n$ [kWh]")
    zz = np.sort(z[rng.choice(N_NODES, 200, replace=False)].ravel())
    from scipy.stats import norm
    pp = norm.ppf((np.arange(zz.size) + 0.5) / zz.size)
    sub = np.linspace(0, zz.size - 1, 2000).astype(int)
    ax[1, 0].plot(pp[sub], zz[sub], ".", ms=2, color="#2a78d6"); ax[1, 0].plot([-4, 4], [-4, 4], color="#6b6a64", lw=1)
    ax[1, 0].set_title("QQ plot of standardized residuals"); ax[1, 0].set_xlabel("normal quantile")
    ax[1, 1].hist(np.clip(kurt, None, 60), 50, color="#2a78d6"); ax[1, 1].set_title("residual excess kurtosis (clipped 60)")
    h = sel[np.argsort(sig)[N_NODES // 2]]
    ax[1, 2].plot(np.asarray(kwh[h, start:start + 48 * 7]), color="#2a78d6", lw=1)
    ax[1, 2].set_title("one week, median-$\\sigma$ household"); ax[1, 2].set_ylabel("kWh / half hour")
    for x_ in ax.ravel():
        x_.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig("oracle_study/results/realdata_fit.png", dpi=140)
    print("saved oracle_study/results/realdata_fit.png")


if __name__ == "__main__":
    main()
