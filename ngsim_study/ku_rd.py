"""Estimator of the undecoded transmitters from (R, |D|): busy REs after SIC and decoded users.

K_u_hat(R, D) = E[K_u | R, D] from a Monte-Carlo table of IRSA alone (degrees i.i.d. from Lambda),
with the number of transmitters K uniform over load_range * M. Empty cells fall back to E[K_u | R].
Uses only what the BS already observes (no multiplicity information).
"""
import numpy as np

from oracle_study.sim import Config, _irsa_frame


def build_ku_estimator_rd(cfg: Config, n_samples=60000, load_range=(0.3, 2.5), seed=12345, min_count=5):
    rng = np.random.default_rng(seed)
    M, n_ul = cfg.M, cfg.S - 2
    rows = np.empty((n_samples, 3))
    for i in range(n_samples):
        K = int(rng.integers(int(load_range[0] * M), int(load_range[1] * M) + 1))
        deg = rng.choice(np.asarray(cfg.replicas), size=K, p=np.asarray(cfg.replicas_prob)).astype(np.int64)
        dec, _, busy = _irsa_frame(deg, n_ul, cfg.F, cfg.sic_eta, int(rng.integers(2**31)))
        nd = dec.sum()
        rows[i] = (busy, nd, K - nd)
    rows = rows[rows[:, 0] > 0]
    R, D, Ku = rows.T
    rb = np.unique(np.concatenate([np.arange(1, 16), np.geomspace(16, M + 1, 40).astype(int)])).astype(float)
    db = np.unique(np.linspace(0, M * 1.2, 30).astype(int)).astype(float)
    iR, iD = np.digitize(R, rb), np.digitize(D, db)
    nR, nD = len(rb) + 1, len(db) + 1
    s2 = np.zeros((nR, nD)); c2 = np.zeros((nR, nD))
    np.add.at(s2, (iR, iD), Ku); np.add.at(c2, (iR, iD), 1)
    s1 = s2.sum(1); c1 = c2.sum(1)
    m1 = np.where(c1 > 0, s1 / np.maximum(c1, 1), np.nan)
    # fill empty R bins of the 1-D fallback by interpolation over bin index
    ok = ~np.isnan(m1)
    m1 = np.interp(np.arange(nR), np.flatnonzero(ok), m1[ok])
    tab = np.where(c2 >= min_count, s2 / np.maximum(c2, 1), m1[:, None])

    def estimate(busy, n_dec):
        busy, n_dec = np.asarray(busy, float), np.asarray(n_dec, float)
        k = tab[np.digitize(busy, rb), np.digitize(n_dec, db)]
        return np.where(busy > 0, k, 0.0)

    return estimate
