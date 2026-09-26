"""
bench/data.py -- windowed multi-step prediction data for the R2R architecture benchmark.

Task (identical for every architecture)
---------------------------------------
Given the last Hp samples of the state s = [T1, T2, wu, w1, wr] (true, SI units), the
interval-averaged applied torques u = [Mu, M1, Mr] and the measured roll radii
rho = [Ru, Rr] over history and horizon, predict s at the next H samples.

    history : s[0:Hp], u[0:Hp+H], rho[0:Hp+H]      (future torques/radii are known:
    target  : s[Hp:Hp+H]                              that is what an MPC evaluates)

Disturbances, faults and uncertain parameters are NOT given -- they must be inferred
from the history (or ignored, which sets an error floor).

Everything is standardised with train-split statistics; physics features
(web-speed mismatches, tension-speed products) are provided to every model.
"""
from __future__ import annotations

import glob
import json
import os
from dataclasses import dataclass

import numpy as np
import torch

from ..plants import PARAMS

FAMS_TRAIN = ["NOM", "REF", "UNC", "DIST", "NOISE", "FAULT", "EXC", "KICK", "COMBO"]
PHYS_NAMES = ["dv1", "dv2", "w1T1", "w1T2", "RuT1", "RrT2"]


def phys_feats(s, rho, R1, xp=np):
    """physics observables in SI units; works for numpy arrays and torch tensors."""
    T1, T2, wu, w1, wr = (s[..., i] for i in range(5))
    Ru, Rr = rho[..., 0], rho[..., 1]
    return xp.stack([R1 * w1 - Ru * wu, Rr * wr - R1 * w1, w1 * T1, w1 * T2, Ru * T1, Rr * T2], -1)


@dataclass
class Norm:
    s_mu: np.ndarray
    s_sd: np.ndarray
    u_mu: np.ndarray
    u_sd: np.ndarray
    r_mu: np.ndarray
    r_sd: np.ndarray
    p_mu: np.ndarray
    p_sd: np.ndarray
    ds_sd: np.ndarray          # std of one-step normalised increments (scales AR residual heads)
    sdot_sd: np.ndarray        # std of normalised state derivative (for the physics loss)
    R1: float
    dt: float

    def t(self, name):
        return torch.as_tensor(getattr(self, name), dtype=torch.float32)

    def to_json(self):
        return {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in self.__dict__.items()}

    model: str = ""


def _radii64(d, x, xd):
    """roll radii in float64 from the (float64) roll angles and each trajectory's true plant
    parameters (the stored geo channel is float32, whose 4e-9 m rounding is ~10 % of the M1
    web-speed-mismatch scale R*w).  Also returns the exact radius rates dR/dt (m/s)."""
    g = d["geo"][..., 0:2].astype(np.float64)
    if x.shape[-1] < 8 or "plant_params" not in d.files:
        rd = np.gradient(g, axis=1) / float(d["t"][1] - d["t"][0])
        return g, rd
    pn = [str(k) for k in d["param_names"]]
    P = d["plant_params"].astype(np.float64)
    Ru0, Rr0, chi = (P[:, pn.index(k)][:, None] for k in ("Ru0", "Rr0", "chi"))
    Ru = Ru0 - x[..., 5] * chi / (2 * np.pi)
    Rr = Rr0 + x[..., 7] * chi / (2 * np.pi)
    rho = np.stack([Ru, Rr], -1)
    if np.abs(rho - g).max() > 1e-6:          # safety: fall back to the stored channel
        return g, np.gradient(g, axis=1) / float(d["t"][1] - d["t"][0])
    rdot = np.stack([-xd[..., 5] * chi / (2 * np.pi), xd[..., 7] * chi / (2 * np.pi)], -1)
    return rho, rdot


def load_trajs(root, model):
    """-> dict family -> dict(s, u, rho, sdot, split, tags, D) (float64 numpy)"""
    out = {}
    for fn in sorted(glob.glob(os.path.join(root, model, f"{model}-*.npz"))):
        d = np.load(fn)
        meta = json.loads(str(d["meta"]))
        u = d["u_avg"] if "u_avg" in d.files else d["u"]
        x, xd = d["x"].astype(np.float64), d["xdot"].astype(np.float64)
        rho, rdot = _radii64(d, x, xd)
        out[meta["family"]] = dict(
            s=x[..., 0:5], sdot=xd[..., 0:5],
            u=u.astype(np.float64), ui=d["u"].astype(np.float64), rho=rho, rdot=rdot,
            D=d["D"].astype(np.float64), split=d["split"].astype(str), tags=d["tags"].astype(str),
            dt=float(d["t"][1] - d["t"][0]))
    return out


def fit_norm(trajs, model):
    R1 = PARAMS[model].R1
    S, U, Rh, P, DS, SD = [], [], [], [], [], []
    for fam, d in trajs.items():
        if fam not in FAMS_TRAIN:
            continue
        m = d["split"] == "train"
        S.append(d["s"][m].reshape(-1, 5))
        U.append(d["u"][m].reshape(-1, 3))
        Rh.append(d["rho"][m].reshape(-1, 2))
        P.append(phys_feats(d["s"][m], d["rho"][m], R1).reshape(-1, len(PHYS_NAMES)))
    S, U, Rh, P = map(np.concatenate, (S, U, Rh, P))
    s_mu, s_sd = S.mean(0), S.std(0) + 1e-9
    for fam, d in trajs.items():
        if fam not in FAMS_TRAIN:
            continue
        m = d["split"] == "train"
        sn = (d["s"][m] - s_mu) / s_sd
        DS.append(np.diff(sn, axis=1).reshape(-1, 5))
        SD.append((d["sdot"][m] / s_sd).reshape(-1, 5))
    DS, SD = np.concatenate(DS), np.concatenate(SD)
    dt = next(iter(trajs.values()))["dt"]
    return Norm(s_mu, s_sd, U.mean(0), U.std(0) + 1e-9, Rh.mean(0), Rh.std(0) + 1e-9,
                P.mean(0), P.std(0) + 1e-9, DS.std(0) + 1e-9, SD.std(0) + 1e-9, R1, dt)


def make_windows(trajs, splits, Hp, H, stride, fams=None, max_n=None, seed=0, t_min_idx=None):
    """Cut windows of length Hp+H. Returns dict of float64 arrays + family labels."""
    rng = np.random.default_rng(seed)
    S, U, UI, R, RD, SD, F, DD, TT = [], [], [], [], [], [], [], [], []
    L = Hp + H
    for fam, d in trajs.items():
        if fams is not None and fam not in fams:
            continue
        idx_b = np.where(np.isin(d["split"], splits))[0]
        N = d["s"].shape[1]
        start0 = t_min_idx if t_min_idx is not None else 0
        starts = np.arange(start0, N - L, stride)
        for b in idx_b:
            for k in starts:
                S.append(d["s"][b, k:k + L])
                U.append(d["u"][b, k:k + L])
                UI.append(d["ui"][b, k:k + L])
                R.append(d["rho"][b, k:k + L])
                RD.append(d["rdot"][b, k:k + L])
                SD.append(d["sdot"][b, k:k + L])
                DD.append(d["D"][b, k:k + L])
                F.append(fam)
                TT.append(d["tags"][b])
    W = dict(s=np.array(S), u=np.array(U), ui=np.array(UI), rho=np.array(R), rdot=np.array(RD), sdot=np.array(SD), D=np.array(DD),
             fam=np.array(F), tag=np.array(TT))
    if max_n is not None and len(W["s"]) > max_n:
        sel = rng.choice(len(W["s"]), max_n, replace=False)
        W = {k: v[sel] for k, v in W.items()}
    return W


class Batch:
    """normalised torch tensors for one set of windows"""

    F32 = ("s", "u", "ui", "r", "sdot")
    F64 = ("s64", "u64", "ui64", "r64", "rd64", "sdot64")

    def __init__(self, W, nm: Norm, Hp, device="cpu", f64=False):
        f = lambda a: torch.as_tensor(a, dtype=torch.float32, device=device)
        d = lambda a: torch.as_tensor(a, dtype=torch.float64, device=device)
        self.s = f((W["s"] - nm.s_mu) / nm.s_sd)
        self.u = f((W["u"] - nm.u_mu) / nm.u_sd)          # interval-averaged torque (drives the step)
        self.ui = f((W["ui"] - nm.u_mu) / nm.u_sd)        # instantaneous torque (pairs with sdot labels)
        self.r = f((W["rho"] - nm.r_mu) / nm.r_sd)
        self.sdot = f(W["sdot"] / nm.s_sd)
        self.has64 = f64
        if f64:   # exact copies for models that compute in double precision (same information)
            self.s64 = d((W["s"] - nm.s_mu) / nm.s_sd)
            self.u64 = d((W["u"] - nm.u_mu) / nm.u_sd)
            self.ui64 = d((W["ui"] - nm.u_mu) / nm.u_sd)
            self.r64 = d((W["rho"] - nm.r_mu) / nm.r_sd)
            self.rd64 = d(W["rdot"])                        # radius rates, m/s (derived from rho(t))
            self.sdot64 = d(W["sdot"] / nm.s_sd)
        self.fam = W["fam"]
        self.Hp = Hp
        self.n = self.s.shape[0]

    def sel(self, idx):
        b = Batch.__new__(Batch)
        for k in self.F32 + (self.F64 if self.has64 else ()) + tuple(getattr(self, "extras", ())):
            v = getattr(self, k)
            setattr(b, k, {kk: vv[idx] for kk, vv in v.items()} if isinstance(v, dict) else v[idx])
        b.extras = tuple(getattr(self, "extras", ()))
        b.has64 = self.has64
        b.fam = self.fam[idx.numpy()] if isinstance(idx, torch.Tensor) else self.fam[idx]
        b.Hp, b.n = self.Hp, len(b.s)
        return b


class Feat:
    """differentiable feature builder shared by all neural models (normalised in, normalised out)."""

    def __init__(self, nm: Norm):
        self.nm = nm
        for k in ["s_mu", "s_sd", "r_mu", "r_sd", "p_mu", "p_sd", "ds_sd"]:
            setattr(self, k, nm.t(k))

    def phys(self, sn, rn):
        s = sn * self.s_sd + self.s_mu
        r = rn * self.r_sd + self.r_mu
        p = phys_feats(s, r, self.nm.R1, xp=torch)
        return (p - self.p_mu) / self.p_sd

    def step_input(self, sn, un, rn):
        """[s, phys(s), u, rho]  (dim 5+6+3+2 = 16)"""
        return torch.cat([sn, self.phys(sn, rn), un, rn], -1)

    N_IN = 5 + len(PHYS_NAMES) + 3 + 2
