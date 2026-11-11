"""
GO-IRSA on NGSIM vehicle speeds: one 0.1 s sample = one slot (real-time regime of the paper).

Nodes are "node slots" of a site (see prepare.py), replicated over C cells per site; each
cell replays an independent time window of its site. A slot is *occupied* in frame k if one
vehicle is present over the whole frame; unoccupied slots never transmit and are excluded
from load, beliefs and metrics. When a new vehicle enters a slot, the BS learns its speed
(registration) and resets the estimate (belief = point mass at error 0).

Process model known to BS and sensors (per site): x_{t+1} = a x_t + mu + w, w ~ N(0, sigma^2),
evaluated per slot exactly as in realdata (prediction a^i x + mu (1-a^i)/(1-a), per-frame belief
push-forward by a^S followed by a Gaussian of variance sigma^2 (1-a^{2S})/(1-a^2); a decoded node
gives x(j*) at its first-replica slot j*). Belief update in frames with collisions and online
calibration: as in oracle_study/sim_gpu.py (imported).
"""
import math

import numpy as np
import torch

from oracle_study.sim import Config, build_ku_estimator
from oracle_study.sim_gpu import METHODS, _irsa_batch, belief_weights


def load_packed(path="data/ngsim/packed.npz", cells=9):
    z = np.load(path)
    sites = [str(s).replace("-", "") for s in z["sites"]]
    Xs = [z[f"{s}_X"] for s in sites]
    SEGs = [z[f"{s}_SEG"] for s in sites]
    pars = [z[f"{s}_par"] for s in sites]
    Tmax = max(x.shape[1] for x in Xs)
    lanes = sum(x.shape[0] for x in Xs)
    BX = np.full((lanes, Tmax), np.nan, np.float32)
    BS = np.full((lanes, Tmax), -1, np.int64)
    lane_site, base = [], 0
    for si, (X, SEG) in enumerate(zip(Xs, SEGs)):
        BX[base:base + X.shape[0], : X.shape[1]] = X
        BS[base:base + X.shape[0], : X.shape[1]] = SEG + si * 10_000_000  # unique ids across sites
        lane_site += [si] * X.shape[0]
        base += X.shape[0]
    lane_site = np.array(lane_site)
    # node pool: every lane replicated in `cells` cells
    node_lane = np.tile(np.arange(lanes), cells)
    node_cell = np.repeat(np.arange(cells), lanes) * len(sites) + np.tile(lane_site, cells)  # cell id = (rep, site)
    node_site = lane_site[node_lane]
    par = np.array(pars)[node_site]  # (N, 3) a, mu, sigma
    T_site = np.array([x.shape[1] for x in Xs])
    n_cells = cells * len(sites)
    cell_site = np.tile(np.arange(len(sites)), cells)
    return dict(BX=BX, BS=BS, node_lane=node_lane, node_cell=node_cell, par=par, T_site=T_site,
                cell_site=cell_site, n_cells=n_cells, sites=sites, scale=float(z["scale"]))


def episode_starts(P, S, frames, episodes, seed):
    rng = np.random.default_rng(seed)
    T = P["T_site"][P["cell_site"]]
    hi = T - frames * S - 2
    assert (hi > 0).all(), "episode longer than the test trace"
    return rng.integers(0, hi[None, :], size=(episodes, P["n_cells"]))


def _ar_tables(a, mu, sig, S, dev):
    i = torch.arange(S + 1, device=dev, dtype=torch.float64)
    a, mu, sig = (torch.as_tensor(v, device=dev, dtype=torch.float64)[:, None] for v in (a, mu, sig))
    A = a ** i
    near1 = (1 - a).abs() < 1e-9
    one = torch.ones_like(a)
    C = torch.where(near1, mu * i, mu * (1 - A) / torch.where(near1, one, 1 - a))
    V = torch.where(near1, sig ** 2 * i, sig ** 2 * (1 - A ** 2) / torch.where(near1, one, 1 - a ** 2))
    return A, C, V


class _Frames:
    """Gathers (R, N, S+1) blocks of speeds / segment ids for frame k."""

    def __init__(self, P, starts, S, dev):
        self.BX = torch.as_tensor(P["BX"], device=dev)
        self.BS = torch.as_tensor(P["BS"], device=dev)
        self.lane = torch.as_tensor(P["node_lane"], device=dev)
        self.st = torch.as_tensor(starts, device=dev)[:, torch.as_tensor(P["node_cell"], device=dev)]  # (R, N)
        self.j = torch.arange(S + 1, device=dev)
        self.S = S

    def get(self, k):
        t = self.st[:, :, None] + k * self.S + self.j[None, None, :]
        lane = self.lane[None, :, None].expand_as(t)
        X = self.BX[lane, t].double()
        SEG = self.BS[lane, t]
        act = (SEG[:, :, 0] >= 0) & (SEG == SEG[:, :, :1]).all(2)
        return torch.nan_to_num(X), SEG[:, :, 0], act


def _masked_q95(E, mask):
    """95th percentile of E over mask, per row. E, mask: (R, K)."""
    E = torch.where(mask, E, torch.full_like(E, float("inf")))
    n = mask.sum(1)
    Es, _ = torch.sort(E, dim=1)
    idx = (0.95 * (n - 1).clamp_min(0)).long()
    return Es.gather(1, idx[:, None])[:, 0]


@torch.no_grad()
def run_batch(S, methods, targets, eta, starts, P, device, frames, cfg: Config = None, chunk=60, L=4.0, V=1000,
              calib_gain=0.0, ku_est="R"):
    """ku_est: 'R' = E[K_u | R] (oracle_study default) or 'RD' = E[K_u | R, |D|] (ngsim_study.ku_rd)."""
    cfg = cfg or Config()
    cfg.S = S
    dev = torch.device(device)
    rng = np.random.default_rng(int(starts[0, 0]) + 7 * S)
    R = len(methods)
    N, F, M = len(P["node_lane"]), cfg.F, cfg.M
    n_ul = S - 2
    assert set(methods) <= set(METHODS), methods
    est = torch.tensor([m == "estimated" for m in methods], device=dev)
    G_t = torch.tensor(targets, device=dev, dtype=torch.float64)
    G_eff = G_t.clone()
    kcfg = Config(S=S, F=F, sic_eta=eta, replicas=cfg.replicas, replicas_prob=cfg.replicas_prob)
    if ku_est == "RD":
        from ngsim_study.ku_rd import build_ku_estimator_rd
        est_fn = build_ku_estimator_rd(kcfg)
    else:
        rg, kg = build_ku_estimator(kcfg)
        est_fn = lambda busy, nd: np.interp(busy, rg, kg)
    eta_np = np.full(R, eta)
    A, Cc, Vv = _ar_tables(P["par"][:, 0], P["par"][:, 1], P["par"][:, 2], S, dev)
    fr = _Frames(P, starts, S, dev)

    c = V // 2
    dx = 2 * L / V
    x = (torch.arange(V + 1, device=dev, dtype=torch.float64) - c) * dx
    radii, absx = x[c:], x.abs()
    std_f = Vv[:, S].sqrt()
    half = min(c, int(math.ceil(6 * std_f.max().item() / dx)) + 1)
    kx = torch.arange(-half, half + 1, device=dev, dtype=torch.float64) * dx
    ker = torch.exp(-0.5 * (kx[None, :] / std_f[:, None].clamp_min(1e-12)) ** 2)
    ker = (ker / ker.sum(1, keepdim=True)).float()
    nfft = 1 << int(math.ceil(math.log2(V + 1 + 2 * half)))
    Kf = torch.fft.rfft(ker, n=nfft)
    pos = (A[:, S:S + 1] * x[None, :]) / dx + c
    i0 = pos.floor().clamp(0, V).long()
    w1 = (pos - pos.floor()).clamp(0, 1).float()
    i1 = (i0 + 1).clamp(max=V)
    resid_std = Vv.sqrt()

    xhat = torch.zeros(R, N, device=dev, dtype=torch.float64)
    prev_seg = torch.full((R, N), -1, device=dev, dtype=torch.long)
    prev_act = torch.zeros(R, N, dtype=torch.bool, device=dev)
    B = torch.zeros(R, N, V + 1, device=dev, dtype=torch.float32)
    B[:, :, c] = 1.0
    keys = ("mse", "q95", "load", "throughput", "unresolved", "exp_load", "G_eff", "G_meas", "n_active")
    out = {k: np.zeros((R, frames)) for k in keys}
    rows = torch.arange(R, device=dev)
    deg_vals, deg_p = np.asarray(cfg.replicas), np.asarray(cfg.replicas_prob)

    for k in range(frames):
        X, seg0, act = fr.get(k)
        new = act & (~prev_act | (seg0 != prev_seg))
        # registration of new vehicles: estimate reset, belief = point mass
        xhat = torch.where(new, X[:, :, 0], xhat)
        if new.any():
            rn, nn = new.nonzero(as_tuple=True)
            B[rn, nn] = 0.0
            B[rn, nn, c] = 1.0
        n_act = act.sum(1)

        fold = B[:, :, c:].clone()
        fold[:, :, 1:] += torch.flip(B[:, :, :c], dims=[2])
        inside = torch.cumsum(fold, dim=2)
        exp_load = (n_act[:, None].double() - (inside * act[:, :, None]).sum(1, dtype=torch.float64)) / M
        ti = torch.argmin((exp_load - G_eff[:, None]).abs(), dim=1)
        theta = radii[ti]
        p_in = inside[rows, :, ti].double().clamp(0, 1)
        del fold, inside

        Xhat = A[None] * xhat[:, :, None] + Cc[None]
        D = X - Xhat
        E = D[:, :, :S].pow(2)
        m = act[:, :, None].expand_as(E)
        out["mse"][:, k] = ((E * m).sum((1, 2)) / m.sum((1, 2)).clamp_min(1)).cpu().numpy()
        out["q95"][:, k] = _masked_q95(E.reshape(R, -1), m.reshape(R, -1)).cpu().numpy()
        del E, m

        tx_cpu = (act & (D[:, :, 0].abs() > theta[:, None])).cpu().numpy()
        r_idx, n_idx = np.nonzero(tx_cpu)
        K = tx_cpu.sum(1)
        offs = np.concatenate([[0], np.cumsum(K)]).astype(np.int64)
        deg = rng.choice(deg_vals, size=r_idx.size, p=deg_p).astype(np.int64)
        dec, first, busy = _irsa_batch(deg, offs, n_ul, F, eta_np, rng.integers(2**31, size=R))
        rd = torch.from_numpy(r_idx[dec]).to(dev)
        nd = torch.from_numpy(n_idx[dec]).to(dev)
        jstar = torch.from_numpy(first[dec]).to(dev) + 2
        dec_mask = torch.zeros(R, N, dtype=torch.bool, device=dev)
        dec_mask[rd, nd] = True
        n_dec = dec_mask.sum(1)
        K_t = torch.from_numpy(K).to(dev)
        unresolved = (K_t - n_dec) > 0
        K_hat = torch.from_numpy(est_fn(busy, n_dec.cpu().numpy())).to(dev)
        G_meas = (n_dec.double() + torch.where(est, K_hat, torch.zeros_like(K_hat))) / M
        out["G_eff"][:, k] = G_eff.cpu().numpy()
        out["G_meas"][:, k] = G_meas.cpu().numpy()
        if calib_gain > 0:
            G_eff = G_eff + calib_gain * (G_t - G_meas)

        xhat = Xhat[:, :, S].clone()
        rr = S - jstar + 1
        xhat[rd, nd] = A[nd, rr] * X[rd, nd, jstar - 1] + Cc[nd, rr]

        w_in, w_out = belief_weights(p_in, dec_mask | ~act, unresolved, est, K_hat)
        inmask = (absx[None, :] <= theta[:, None]).float()
        for a0 in range(0, R, chunk):
            b0 = min(R, a0 + chunk)
            nb = b0 - a0
            im = inmask[a0:b0, None, :]
            Bn = B[a0:b0] * (w_in[a0:b0].float()[:, :, None] * im + w_out[a0:b0].float()[:, :, None] * (1 - im))
            s = Bn.sum(2, keepdim=True)
            Bn = torch.where(s > 0, Bn / s.clamp_min(1e-30), B[a0:b0])
            Bs = torch.zeros_like(Bn)
            Bs.scatter_add_(2, i0[None].expand(nb, -1, -1), Bn * (1 - w1)[None])
            Bs.scatter_add_(2, i1[None].expand(nb, -1, -1), Bn * w1[None])
            Y = torch.fft.irfft(torch.fft.rfft(Bs, n=nfft) * Kf[None], n=nfft)[:, :, half:half + V + 1]
            Y.clamp_(min=0)
            B[a0:b0] = Y / Y.sum(2, keepdim=True).clamp_min(1e-30)
            del Bn, Bs, Y
        for a0 in range(0, rd.numel(), 200_000):
            sl = slice(a0, a0 + 200_000)
            sd = resid_std[nd[sl], rr[sl]].clamp_min(dx / 4)
            g = torch.exp(-0.5 * (x[None, :] / sd[:, None]) ** 2)
            B[rd[sl], nd[sl]] = (g / g.sum(1, keepdim=True)).float()

        prev_seg, prev_act = seg0, act
        out["load"][:, k] = (K_t.double() / M).cpu().numpy()
        out["throughput"][:, k] = (n_dec.double() / M).cpu().numpy()
        out["unresolved"][:, k] = unresolved.cpu().numpy()
        out["exp_load"][:, k] = exp_load[rows, ti].cpu().numpy()
        out["n_active"][:, k] = n_act.cpu().numpy()
    return out


@torch.no_grad()
def run_pull(S, starts, P, device, frames, F=10):
    """Pull-based: poll the M occupied slots with the largest model error variance."""
    dev = torch.device(device)
    N = len(P["node_lane"])
    M = (S - 2) * F
    R = starts.shape[0]
    A, Cc, Vv = _ar_tables(P["par"][:, 0], P["par"][:, 1], P["par"][:, 2], S, dev)
    fr = _Frames(P, starts, S, dev)
    xhat = torch.zeros(R, N, device=dev, dtype=torch.float64)
    var = torch.zeros(R, N, device=dev, dtype=torch.float64)
    prev_seg = torch.full((R, N), -1, device=dev, dtype=torch.long)
    prev_act = torch.zeros(R, N, dtype=torch.bool, device=dev)
    out = {k: np.zeros((R, frames)) for k in ("mse", "q95")}
    j_of_rank = 2 + torch.arange(M, device=dev) // F
    rows = torch.arange(R, device=dev)[:, None]
    for k in range(frames):
        X, seg0, act = fr.get(k)
        new = act & (~prev_act | (seg0 != prev_seg))
        xhat = torch.where(new, X[:, :, 0], xhat)
        var = torch.where(new, torch.zeros_like(var), var)
        Xhat = A[None] * xhat[:, :, None] + Cc[None]
        E = (X - Xhat)[:, :, :S].pow(2)
        m = act[:, :, None].expand_as(E)
        out["mse"][:, k] = ((E * m).sum((1, 2)) / m.sum((1, 2)).clamp_min(1)).cpu().numpy()
        out["q95"][:, k] = _masked_q95(E.reshape(R, -1), m.reshape(R, -1)).cpu().numpy()
        score = torch.where(act, var + 1e-12 * torch.rand_like(var), torch.full_like(var, -1.0))
        sched = torch.topk(score, M, dim=1).indices
        ok = act.gather(1, sched)
        jj = j_of_rank[None].expand(R, -1)
        rr = S - jj + 1
        xhat = Xhat[:, :, S].clone()
        upd = A[sched, rr] * X[rows, sched, jj - 1] + Cc[sched, rr]
        xhat[rows, sched] = torch.where(ok, upd, xhat[rows, sched])
        var = A[None, :, S] ** 2 * var + Vv[None, :, S]
        var[rows, sched] = torch.where(ok, Vv[sched, rr], var[rows, sched])
        prev_seg, prev_act = seg0, act
    return out
