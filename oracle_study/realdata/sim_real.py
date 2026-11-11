"""
GO-IRSA on real sensor traces (Low Carbon London smart meters) with a per-household
AR(1) prediction model.  One data sample = one frame (the value is held during the frame).

Model known to BS and sensors:
    x_{k+1} = a_n x_k + mu_n + w_k,     w_k ~ N(0, sigma_n^2)   (assumed)
The real residuals are whatever the data says (heavy tails, daily cycle, ...).

Error d = x_k - xhat_k.
  undecoded node: xhat <- a xhat + mu;   belief: push-forward d -> a d, then * N(0, sigma^2)
  decoded node  : xhat <- a x_k + mu;    belief: N(0, sigma^2)
Belief update in frames with collisions and online load calibration: as in sim_gpu.py.
Rows of a batch share S, eta and the household-to-node mapping (per-node kernels are
shared) and differ in method, target load and episode start time.
"""

import math

import numpy as np
import torch

from oracle_study.sim import Config, build_ku_estimator
from oracle_study.sim_gpu import METHODS, _irsa_batch, belief_weights


@torch.no_grad()
def run_batch_real(S, methods, targets, eta, starts, data, a, mu, sig, device, frames,
                   cfg: Config = None, chunk=100, L=12.0, V=1500, calib_gain=0.0):
    """Each frame k carries one data sample x_k (constant over the S slots).
    AR(1) per frame: x_{k+1} = a x_k + mu + w.  Error d = x_k - xhat_k.
    Undecoded: xhat <- a xhat + mu, belief: push-forward by a, then N(0, sigma^2).
    Decoded:   xhat <- a x_k + mu,  belief: N(0, sigma^2).
    calib_gain > 0: online load calibration (see module docstring)."""
    cfg = cfg or Config()
    cfg.S = S
    dev = torch.device(device)
    rng = np.random.default_rng(int(starts[0]) + 7 * S + 1)
    R = len(methods)
    N, F, M = data.shape[0], cfg.F, cfg.M
    n_ul = S - 2
    assert set(methods) <= set(METHODS), methods
    est = torch.tensor([m == "estimated" for m in methods], device=dev)
    G_t = torch.tensor(targets, device=dev, dtype=torch.float64)
    G_eff = G_t.clone()
    eta_np = np.full(R, eta)
    rg, kg = build_ku_estimator(Config(S=S, F=F, sic_eta=eta, replicas=cfg.replicas,
                                          replicas_prob=cfg.replicas_prob))
    data = torch.as_tensor(data, device=dev, dtype=torch.float32)
    starts = torch.as_tensor(starts, device=dev, dtype=torch.long)
    a_t, mu_t, sig_t = (torch.as_tensor(v, device=dev, dtype=torch.float64) for v in (a, mu, sig))

    c = V // 2
    dx = 2 * L / V
    x = (torch.arange(V + 1, device=dev, dtype=torch.float64) - c) * dx
    radii, absx = x[c:], x.abs()
    half = min(c, int(math.ceil(6 * sig_t.max().item() / dx)) + 1)
    kx = torch.arange(-half, half + 1, device=dev, dtype=torch.float64) * dx
    ker = torch.exp(-0.5 * (kx[None, :] / sig_t[:, None].clamp_min(1e-12)) ** 2)
    ker = (ker / ker.sum(1, keepdim=True)).float()
    nfft = 1 << int(math.ceil(math.log2(V + 1 + 2 * half)))
    Kf = torch.fft.rfft(ker, n=nfft)
    pos = (a_t[:, None] * x[None, :]) / dx + c
    i0 = pos.floor().clamp(0, V).long()
    w1 = (pos - pos.floor()).clamp(0, 1).float()
    i1 = (i0 + 1).clamp(max=V)
    g0 = torch.exp(-0.5 * (x[None, :] / sig_t[:, None].clamp_min(dx / 4)) ** 2)
    fresh = (g0 / g0.sum(1, keepdim=True)).float()  # (N, V+1) belief right after an update

    xhat = data[:, starts].T.double().clone()  # BS knows the initial state
    B = torch.zeros(R, N, V + 1, device=dev, dtype=torch.float32)
    B[:, :, c] = 1.0
    keys = ("mse", "q95", "load", "throughput", "unresolved", "exp_load", "theta", "G_eff", "G_meas")
    out = {k: np.zeros((R, frames)) for k in keys}
    kq = int(0.95 * (N - 1)) + 1
    rows = torch.arange(R, device=dev)
    deg_vals, deg_p = np.asarray(cfg.replicas), np.asarray(cfg.replicas_prob)

    for k in range(frames):
        fold = B[:, :, c:].clone()
        fold[:, :, 1:] += torch.flip(B[:, :, :c], dims=[2])
        inside = torch.cumsum(fold, dim=2)
        exp_load = (N - inside.sum(1, dtype=torch.float64)) / M
        ti = torch.argmin((exp_load - G_eff[:, None]).abs(), dim=1)
        theta = radii[ti]
        p_in = inside[rows, :, ti].double().clamp(0, 1)
        del fold, inside

        X = data[:, starts + k].T.double()  # (R, N) value held during frame k
        D = X - xhat
        E = D.pow(2)
        out["mse"][:, k] = E.mean(1).cpu().numpy()
        out["q95"][:, k] = E.kthvalue(kq, dim=1).values.cpu().numpy()

        tx_cpu = (D.abs() > theta[:, None]).cpu().numpy()
        r_idx, n_idx = np.nonzero(tx_cpu)
        K = tx_cpu.sum(1)
        offs = np.concatenate([[0], np.cumsum(K)]).astype(np.int64)
        deg = rng.choice(deg_vals, size=r_idx.size, p=deg_p).astype(np.int64)
        dec, first, busy = _irsa_batch(deg, offs, n_ul, F, eta_np, rng.integers(2**31, size=R))
        rd = torch.from_numpy(r_idx[dec]).to(dev)
        nd = torch.from_numpy(n_idx[dec]).to(dev)
        dec_mask = torch.zeros(R, N, dtype=torch.bool, device=dev)
        dec_mask[rd, nd] = True
        n_dec = dec_mask.sum(1)
        K_t = torch.from_numpy(K).to(dev)
        unresolved = (K_t - n_dec) > 0
        K_hat = torch.from_numpy(np.interp(busy, rg, kg)).to(dev)
        G_meas = (n_dec.double() + torch.where(est, K_hat, torch.zeros_like(K_hat))) / M
        out["G_eff"][:, k] = G_eff.cpu().numpy()
        out["G_meas"][:, k] = G_meas.cpu().numpy()
        if calib_gain > 0:
            G_eff = G_eff + calib_gain * (G_t - G_meas)  # plain integral update (no clipping)

        xhat = torch.where(dec_mask, a_t[None] * X + mu_t[None], a_t[None] * xhat + mu_t[None])

        w_in, w_out = belief_weights(p_in, dec_mask, unresolved, est, K_hat)
        inmask = (absx[None, :] <= theta[:, None]).float()
        for a0 in range(0, R, chunk):
            b0 = min(R, a0 + chunk)
            nb = b0 - a0
            Wi, Wo = w_in[a0:b0].float()[:, :, None], w_out[a0:b0].float()[:, :, None]
            im = inmask[a0:b0, None, :]
            Bn = B[a0:b0] * (Wi * im + Wo * (1 - im))
            s = Bn.sum(2, keepdim=True)
            Bn = torch.where(s > 0, Bn / s.clamp_min(1e-30), B[a0:b0])
            Bs = torch.zeros_like(Bn)
            Bs.scatter_add_(2, i0[None].expand(nb, -1, -1), Bn * (1 - w1)[None])
            Bs.scatter_add_(2, i1[None].expand(nb, -1, -1), Bn * w1[None])
            Y = torch.fft.irfft(torch.fft.rfft(Bs, n=nfft) * Kf[None], n=nfft)[:, :, half:half + V + 1]
            Y.clamp_(min=0)
            B[a0:b0] = Y / Y.sum(2, keepdim=True).clamp_min(1e-30)
            del Bn, Bs, Y
        B[rd, nd] = fresh[nd]

        out["load"][:, k] = (K_t.double() / M).cpu().numpy()
        out["throughput"][:, k] = (n_dec.double() / M).cpu().numpy()
        out["unresolved"][:, k] = unresolved.cpu().numpy()
        out["exp_load"][:, k] = exp_load[rows, ti].cpu().numpy()
        out["theta"][:, k] = theta.cpu().numpy()
    return out


@torch.no_grad()
def run_pull_real(S, starts, data, a, mu, sig, device, frames, F=10):
    """Pull-based, one sample per frame: poll the M nodes with the largest model error variance."""
    dev = torch.device(device)
    N = data.shape[0]
    M = (S - 2) * F
    R = len(starts)
    data = torch.as_tensor(data, device=dev, dtype=torch.float32)
    starts = torch.as_tensor(starts, device=dev, dtype=torch.long)
    a_t, mu_t, sig_t = (torch.as_tensor(v, device=dev, dtype=torch.float64) for v in (a, mu, sig))
    xhat = data[:, starts].T.double().clone()
    var = torch.zeros(R, N, device=dev, dtype=torch.float64)
    kq = int(0.95 * (N - 1)) + 1
    out = {k: np.zeros((R, frames)) for k in ("mse", "q95")}
    rows = torch.arange(R, device=dev)[:, None]
    for k in range(frames):
        X = data[:, starts + k].T.double()
        E = (X - xhat).pow(2)
        out["mse"][:, k] = E.mean(1).cpu().numpy()
        out["q95"][:, k] = E.kthvalue(kq, dim=1).values.cpu().numpy()
        sched = torch.topk(var + 1e-12 * torch.rand_like(var), min(M, N), dim=1).indices
        upd = torch.zeros(R, N, dtype=torch.bool, device=dev)
        upd[rows, sched] = True
        xhat = torch.where(upd, a_t[None] * X + mu_t[None], a_t[None] * xhat + mu_t[None])
        var = torch.where(upd, sig_t[None] ** 2, a_t[None] ** 2 * var + sig_t[None] ** 2)
    return out
