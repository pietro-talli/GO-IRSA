"""
Shared pieces of the GO-IRSA study (Gaussian/Wiener model of the paper, Sec. III-IV).

* Config            : scenario parameters (N, F, S -> M = (S-2)F, Lambda, SIC eta, ...)
* _irsa_frame       : one IRSA frame with SIC (peeling); collision channel; optional
                      imperfect SIC (a singleton RE of original degree l is decoded with
                      probability eta^(l-1), on failure that RE is lost). Also returns the
                      number of REs still busy after SIC (observable by the BS).
* build_ku_estimator: Monte-Carlo lookup table  K_u_hat(R) = E[K_u | R]  used to estimate the
                      number of undecoded transmitters from the residual busy REs R.
* run_pull          : round-robin pull-based baseline for the Wiener model.

The GO-IRSA simulator itself (batched, GPU) is in sim_gpu.py; the real-data
simulator in realdata/sim_real.py.
"""

from dataclasses import dataclass

import numpy as np
from numba import njit


@dataclass
class Config:
    N: int = 2000
    F: int = 10
    S: int = 22  # slots per frame (M = (S-2)F)
    sigma_slot: float = 0.04  # per-slot std of the Wiener increment
    V: int = 1000  # belief grid resolution (V+1 points)
    L: float = 4.0  # grid half-width
    frames: int = 200
    target_load: float = 0.7
    sic_eta: float = 1.0  # 1.0 = perfect SIC
    replicas: tuple = (3, 8)
    replicas_prob: tuple = (0.86, 0.14)

    @property
    def M(self):
        return (self.S - 2) * self.F


# --------------------------------------------------------------------------- #
# IRSA                                                                         #
# --------------------------------------------------------------------------- #
@njit(cache=True)
def _irsa_frame(degrees, n_ul_slots, F, eta, seed):
    """Simulate one IRSA frame.

    degrees: number of replicas of each transmitting user (K,)
    returns decoded (K,) bool, first_slot (K,) in {0..n_ul_slots-1},
            busy = number of REs still non-empty after SIC
    """
    np.random.seed(seed)
    K = degrees.shape[0]
    n_re = n_ul_slots * F
    total = 0
    for i in range(K):
        total += degrees[i]
    # user -> REs (CSR)
    u_ptr = np.zeros(K + 1, np.int64)
    u_re = np.empty(total, np.int64)
    first_slot = np.empty(K, np.int64)
    perm = np.arange(n_ul_slots)
    p = 0
    for i in range(K):
        u_ptr[i] = p
        d = degrees[i]
        # partial Fisher-Yates: d distinct slots
        for a in range(d):
            b = a + np.random.randint(n_ul_slots - a)
            tmp = perm[a]
            perm[a] = perm[b]
            perm[b] = tmp
        fs = n_ul_slots
        for a in range(d):
            s = perm[a]
            if s < fs:
                fs = s
            u_re[p] = s * F + np.random.randint(F)
            p += 1
        first_slot[i] = fs
    u_ptr[K] = p
    # RE -> users (CSR)
    re_deg = np.zeros(n_re, np.int64)
    for e in range(total):
        re_deg[u_re[e]] += 1
    r_ptr = np.zeros(n_re + 1, np.int64)
    for r in range(n_re):
        r_ptr[r + 1] = r_ptr[r] + re_deg[r]
    r_usr = np.empty(total, np.int64)
    fill = r_ptr[:-1].copy()
    for i in range(K):
        for e in range(u_ptr[i], u_ptr[i + 1]):
            r = u_re[e]
            r_usr[fill[r]] = i
            fill[r] += 1
    orig_deg = re_deg.copy()
    cur_deg = re_deg.copy()
    dead = np.zeros(n_re, np.bool_)
    decoded = np.zeros(K, np.bool_)
    queue = np.empty(n_re + total + 1, np.int64)
    qh = 0
    qt = 0
    for r in range(n_re):
        if cur_deg[r] == 1:
            queue[qt] = r
            qt += 1
    while qh < qt:
        r = queue[qh]
        qh += 1
        if cur_deg[r] != 1 or dead[r]:
            continue
        u = -1
        for e in range(r_ptr[r], r_ptr[r + 1]):
            if not decoded[r_usr[e]]:
                u = r_usr[e]
                break
        if u == -1:
            continue
        if eta < 1.0 and np.random.random() >= eta ** (orig_deg[r] - 1):
            dead[r] = True  # residual interference: this RE is lost
            continue
        decoded[u] = True
        for e in range(u_ptr[u], u_ptr[u + 1]):
            s = u_re[e]
            cur_deg[s] -= 1
            if cur_deg[s] == 1:
                queue[qt] = s
                qt += 1
    busy = 0  # REs still carrying energy after SIC (observable by the BS)
    for r in range(n_re):
        if cur_deg[r] >= 1:
            busy += 1
    return decoded, first_slot, busy


# --------------------------------------------------------------------------- #
# Pull-based baseline (Wiener model)                                          #
# --------------------------------------------------------------------------- #
def _frame_noise(rng, N, S, sigma):
    """Cumulative increments at the beginning of slots 1..S+1: (N, S+1)."""
    C = np.zeros((N, S + 1))
    np.cumsum(rng.standard_normal((N, S)) * sigma, axis=1, out=C[:, 1:])
    return C


def _frame_metrics(d, C):
    E = (d[:, None] + C[:, :-1]) ** 2  # slots 1..S
    flat = E.ravel()
    q = np.partition(flat, int(0.95 * (flat.size - 1)))[int(0.95 * (flat.size - 1))]
    return flat.mean(), q


def run_pull(cfg: Config, seed: int):
    """Round-robin pull-based scheduling (optimal for identical sigmas)."""
    rng = np.random.default_rng(seed)
    N, S, F, M = cfg.N, cfg.S, cfg.F, cfg.M
    d = np.zeros(N)
    out = {k: np.zeros(cfg.frames) for k in ("mse", "q95")}
    ptr = 0
    for k in range(cfg.frames):
        sched = (ptr + np.arange(M)) % N
        ptr = (ptr + M) % N
        j_abs = 2 + np.arange(M) // F
        C = _frame_noise(rng, N, S, cfg.sigma_slot)
        out["mse"][k], out["q95"][k] = _frame_metrics(d, C)
        d_next = d + C[:, S]
        d_next[sched] = C[sched, S] - C[sched, j_abs - 1]
        d = d_next
    return out


# --------------------------------------------------------------------------- #
# Estimator of the number of undecoded transmitters from residual power       #
# --------------------------------------------------------------------------- #
def build_ku_estimator(cfg: Config, n_samples=40000, load_range=(0.3, 1.1), seed=12345):
    """Monte-Carlo lookup table  K_u_hat(R) = E[K_u | R],  R = number of REs that are still
    non-empty after SIC (energy detection at the BS). Built from IRSA alone (degrees i.i.d.
    from Lambda), with the number of transmitters K uniform over `load_range` * M.
    Returns (r_grid, ku_grid); estimate with np.interp(R, r_grid, ku_grid)."""
    rng = np.random.default_rng(seed)
    M, n_ul = cfg.M, cfg.S - 2
    ku, R = np.empty(n_samples), np.empty(n_samples)
    for i in range(n_samples):
        K = int(rng.integers(int(load_range[0] * M), int(load_range[1] * M) + 1))
        deg = rng.choice(np.asarray(cfg.replicas), size=K, p=np.asarray(cfg.replicas_prob)).astype(np.int64)
        dec, _, busy = _irsa_frame(deg, n_ul, cfg.F, cfg.sic_eta, int(rng.integers(2**31)))
        ku[i], R[i] = K - dec.sum(), busy
    m = R > 0
    ku, R = ku[m], R[m]
    # geometric bins on R: exact values for small R, wider bins for large R
    edges = np.unique(np.concatenate([np.arange(1, 16), np.geomspace(16, max(17, R.max() + 1), 40).astype(int)]))
    r_grid, ku_grid = [0.0], [0.0]
    for lo, hi in zip(edges[:-1], edges[1:]):
        sel = (R >= lo) & (R < hi)
        if sel.sum() >= 5:
            r_grid.append(R[sel].mean())
            ku_grid.append(ku[sel].mean())
    return np.asarray(r_grid), np.asarray(ku_grid)
