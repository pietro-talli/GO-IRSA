"""
Batched GPU simulator of GO-IRSA on the Wiener model of the paper (Sec. III-IV).

One call simulates R independent rows (episodes) that share the frame size S but may
differ in method, target load and SIC eta.

Methods (belief update of the nodes NOT decoded in a frame that ends with collisions):
  conservative : paper, eq. (10): keep the prior.
  estimated    : the BS estimates K_u, the number of undecoded transmitters, from the
                 number R of REs still busy after SIC (lookup table E[K_u | R], see
                 sim.build_ku_estimator) and conditions every undecoded belief on
                 "K_u of the undecoded nodes were above the threshold":
                     phi_m <- q_m phi_m 1{|d|>theta}/t_m + (1-q_m) phi_m 1{|d|<=theta}/(1-t_m),
                 t_m = prior mass above theta, q_m = P(m transmitted | K_u), computed with
                 the exponential-tilting approximation (sum_m q_m = K_u).
In frames without collisions both methods use the implicit information (eq. 6).

Online load calibration (calib_gain > 0): the threshold targets G_eff(k), updated as
    G_eff(k+1) = G_eff(k) + gain * (G_target - G_meas(k))      (no clipping; G_eff stays bounded,
    because G_eff >> N/M gives theta = 0 and G_meas >> G_target, G_eff <= 0 gives G_meas = 0),
    G_meas = |decoded| / M (conservative)   or   (|decoded| + K_u_hat) / M (estimated).

Beliefs live on the GPU as an (R, N, V+1) float32 tensor; IRSA decoding runs on CPU
with numba, in parallel over rows.
"""

import math

import numpy as np
import torch
from numba import njit, prange

from oracle_study.sim import Config, _irsa_frame, build_ku_estimator

METHODS = ("conservative", "estimated")


@njit(parallel=True, cache=True)
def _irsa_batch(deg, offs, n_ul, F, eta, seeds):
    R = offs.shape[0] - 1
    dec = np.zeros(deg.shape[0], np.bool_)
    first = np.zeros(deg.shape[0], np.int64)
    busy = np.zeros(R, np.int64)
    for r in prange(R):
        a, b = offs[r], offs[r + 1]
        if b > a:
            d_, f_, bz = _irsa_frame(deg[a:b], n_ul, F, eta[r], seeds[r])
            dec[a:b] = d_
            first[a:b] = f_
            busy[r] = bz
    return dec, first, busy


def _gauss(x, std):
    g = torch.exp(-0.5 * (x / std) ** 2)
    return g / g.sum(-1, keepdim=True)


def _tilted_marginals(t, k, iters=60):
    """Batched exponential tilting. t: (R, N) float64, k: (R,) -> q (R, N) with sum_n q = k."""
    pos = t > 1e-12
    n_pos = pos.sum(1)
    tp = t.clamp(1e-12, 1 - 1e-12)
    logit = torch.where(pos, torch.log(tp) - torch.log1p(-tp), torch.full_like(t, -torch.inf))
    lo = torch.full_like(k, -60.0)
    hi = torch.full_like(k, 60.0)
    for _ in range(iters):
        lam = 0.5 * (lo + hi)
        s = torch.sigmoid(logit + lam[:, None]).sum(1)
        big = s > k
        hi = torch.where(big, lam, hi)
        lo = torch.where(big, lo, lam)
    q = torch.sigmoid(logit + (0.5 * (lo + hi))[:, None])
    q = torch.where((k >= n_pos)[:, None] & pos, torch.ones_like(q), q)
    return torch.where((k > 0)[:, None], q, torch.zeros_like(q))


def belief_weights(p_in, dec_mask, unresolved, est, K_hat):
    """Weights of the belief mass inside / outside the threshold for every (row, node)."""
    t_out = (1 - p_in) * (~dec_mask)
    q = _tilted_marginals(t_out, torch.where(est & unresolved, K_hat, torch.zeros_like(K_hat)))
    one = torch.ones_like(p_in)
    w_in = one  # resolved frame: keep the inside part only (eq. 6)
    w_out = torch.where(unresolved[:, None], one, torch.zeros_like(p_in))  # conservative: keep prior
    orc = (est & unresolved)[:, None]
    w_in = torch.where(orc, (1 - q) / p_in.clamp_min(1e-30), w_in)
    w_out = torch.where(orc, q / (1 - p_in).clamp_min(1e-30), w_out)
    return w_in, w_out


@torch.no_grad()
def run_batch(S, methods, targets, etas, seed, device, cfg: Config = None, chunk=100, calib_gain=0.0):
    """methods/targets/etas: one entry per row. Returns a dict of (R, frames) arrays."""
    cfg = cfg or Config()
    cfg.S = S
    dev = torch.device(device)
    gen = torch.Generator(device=dev).manual_seed(seed)
    rng = np.random.default_rng(seed)
    R, N, F, M, V = len(methods), cfg.N, cfg.F, cfg.M, cfg.V
    n_ul, sig = S - 2, cfg.sigma_slot
    assert set(methods) <= set(METHODS), methods
    est = torch.tensor([m == "estimated" for m in methods], device=dev)
    eta_np = np.asarray(etas, dtype=np.float64)
    tables = {e: build_ku_estimator(Config(S=S, F=cfg.F, sic_eta=e, replicas=cfg.replicas,
                                          replicas_prob=cfg.replicas_prob))
              for e in set(eta_np.tolist())}
    G_t = torch.tensor(targets, device=dev, dtype=torch.float64)
    G_eff = G_t.clone()

    c = V // 2
    dx = 2 * cfg.L / V
    x = (torch.arange(V + 1, device=dev, dtype=torch.float64) - c) * dx  # x[c] == 0 exactly
    radii, absx = x[c:], x.abs()
    std_f = math.sqrt(S) * sig  # one-frame diffusion kernel, applied with FFT
    half = min(c, int(math.ceil(6 * std_f / dx)) + 1)
    kx = torch.arange(-half, half + 1, device=dev, dtype=torch.float64) * dx
    nfft = 1 << int(math.ceil(math.log2(V + 1 + 2 * half)))
    Kf = torch.fft.rfft(_gauss(kx, std_f).float(), n=nfft)
    # belief of a node decoded with first replica in uplink slot u (absolute slot u+2), eq. (4)
    resid = torch.stack([_gauss(x, math.sqrt(S - (u + 2) + 1) * sig) for u in range(n_ul)]).float()
    deg_vals, deg_p = np.asarray(cfg.replicas), np.asarray(cfg.replicas_prob)

    d = torch.zeros(R, N, device=dev, dtype=torch.float64)  # true error x - xhat
    B = torch.zeros(R, N, V + 1, device=dev, dtype=torch.float32)
    B[:, :, c] = 1.0
    keys = ("mse", "q95", "load", "throughput", "unresolved", "theta", "exp_load", "G_eff", "G_meas",
            "ku_true", "ku_hat")
    out = {k: np.zeros((R, cfg.frames)) for k in keys}
    kq = int(0.95 * (N * S - 1)) + 1  # kthvalue is 1-indexed
    rows = torch.arange(R, device=dev)

    for k in range(cfg.frames):
        # ---- threshold selection (eq. 11) on the effective target ----
        fold = B[:, :, c:].clone()
        fold[:, :, 1:] += torch.flip(B[:, :, :c], dims=[2])
        inside = torch.cumsum(fold, dim=2)  # (R, N, c+1): P(|d| <= radii[i])
        exp_load = (N - inside.sum(1, dtype=torch.float64)) / M
        ti = torch.argmin((exp_load - G_eff[:, None]).abs(), dim=1)
        theta = radii[ti]
        p_in = inside[rows, :, ti].double().clamp(0, 1)
        del fold, inside

        # ---- transmissions + IRSA (CPU) ----
        tx_cpu = (d.abs() > theta[:, None]).cpu().numpy()
        r_idx, n_idx = np.nonzero(tx_cpu)
        K = tx_cpu.sum(1)
        offs = np.concatenate([[0], np.cumsum(K)]).astype(np.int64)
        deg = rng.choice(deg_vals, size=r_idx.size, p=deg_p).astype(np.int64)
        dec, first, busy = _irsa_batch(deg, offs, n_ul, F, eta_np, rng.integers(2**31, size=R))
        dec_mask = torch.zeros(R, N, dtype=torch.bool, device=dev)
        first_slot = torch.zeros(R, N, dtype=torch.long, device=dev)
        rd, nd = torch.from_numpy(r_idx[dec]).to(dev), torch.from_numpy(n_idx[dec]).to(dev)
        dec_mask[rd, nd] = True
        first_slot[rd, nd] = torch.from_numpy(first[dec]).to(dev)
        n_dec = dec_mask.sum(1)
        K_t = torch.from_numpy(K).to(dev)
        K_u = (K_t - n_dec).double()
        unresolved = K_u > 0  # observable: R > 0
        kh = np.zeros(R)
        for e, (rg, kg) in tables.items():
            sel = eta_np == e
            kh[sel] = np.interp(busy[sel], rg, kg)
        K_hat = torch.from_numpy(kh).to(dev)

        # ---- online load calibration ----
        G_meas = (n_dec.double() + torch.where(est, K_hat, torch.zeros_like(K_hat))) / M
        out["G_eff"][:, k] = G_eff.cpu().numpy()
        out["G_meas"][:, k] = G_meas.cpu().numpy()
        if calib_gain > 0:
            G_eff = G_eff + calib_gain * (G_t - G_meas)  # plain integral update (no clipping)

        # ---- true process over the frame + metrics (eq. 3) ----
        Cn = torch.zeros(R, N, S + 1, device=dev, dtype=torch.float64)
        Cn[:, :, 1:] = torch.cumsum(torch.randn(R, N, S, device=dev, dtype=torch.float64, generator=gen) * sig, 2)
        E = (d[:, :, None] + Cn[:, :, :-1]).pow_(2).reshape(R, -1)
        out["mse"][:, k] = E.mean(1).cpu().numpy()
        out["q95"][:, k] = E.kthvalue(kq, dim=1).values.cpu().numpy()
        del E
        resid_err = Cn[:, :, S] - torch.gather(Cn, 2, (first_slot + 1)[:, :, None])[:, :, 0]
        d = torch.where(dec_mask, resid_err, d + Cn[:, :, S])
        del Cn

        # ---- belief update ----
        w_in, w_out = belief_weights(p_in, dec_mask, unresolved, est, K_hat)
        inmask = (absx[None, :] <= theta[:, None]).float()
        for a in range(0, R, chunk):
            b = min(R, a + chunk)
            im = inmask[a:b, None, :]
            Bn = B[a:b] * (w_in[a:b].float()[:, :, None] * im + w_out[a:b].float()[:, :, None] * (1 - im))
            s = Bn.sum(2, keepdim=True)
            Bn = torch.where(s > 0, Bn / s.clamp_min(1e-30), B[a:b])  # numerical fallback
            Y = torch.fft.irfft(torch.fft.rfft(Bn, n=nfft) * Kf, n=nfft)[:, :, half:half + V + 1]
            Y.clamp_(min=0)
            B[a:b] = Y / Y.sum(2, keepdim=True)
            del Bn, Y
        B[rd, nd] = resid[first_slot[rd, nd]]

        out["load"][:, k] = (K_t.double() / M).cpu().numpy()
        out["throughput"][:, k] = (n_dec.double() / M).cpu().numpy()
        out["unresolved"][:, k] = unresolved.cpu().numpy()
        out["theta"][:, k] = theta.cpu().numpy()
        out["exp_load"][:, k] = exp_load[rows, ti].cpu().numpy()
        out["ku_true"][:, k] = K_u.cpu().numpy()
        out["ku_hat"][:, k] = K_hat.cpu().numpy()
    return out
