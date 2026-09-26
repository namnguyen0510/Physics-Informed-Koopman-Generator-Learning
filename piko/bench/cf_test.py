"""
bench/cf_test.py -- counterfactual-input (interventional) test set.

The closed-loop replay test feeds every model the torques the controller actually applied.  Those
torques are a function of the (noisy) state and react to disturbances and faults, so a sequence
model can partly *read the answer from the inputs*.  A model used inside MPC is queried with
candidate torque sequences that the logged controller never produced.  This test reproduces that
situation: for each test window the true history is kept, but the future torques are the logged
ones plus random piecewise-constant perturbations, and the ground truth is obtained by
re-simulating the true plant (its own parameters, disturbance and fault profiles) under those
torques with the reference RK4 integrator.

usage:  python -m r2r_nn.bench.cf_test --model M1
"""
from __future__ import annotations

import argparse
import glob
import json
import os

import numpy as np

from ..plants import BatchParams, geometry, PARAMS
from ..simulate import rk4
from .data import load_trajs, fit_norm, _radii64

FAMS = ["NOM", "REF", "UNC", "EXC", "DIST", "FAULT", "KICK", "NOISE", "COMBO", "OOD"]


def _bparams(P, names):
    bp = BatchParams.__new__(BatchParams)
    bp.B = P.shape[0]
    for k in BatchParams.NAMES:
        setattr(bp, k, P[:, names.index(k)].astype(np.float64))
    bp.ES = bp.E * bp.S
    return bp


def perturbation(rng, n, H, sd, frac):
    """random piecewise-constant torque offsets, segment length U[16, 96] samples"""
    d = np.zeros((n, H, len(sd)))
    for i in range(n):
        for j in range(len(sd)):
            k = 0
            while k < H:
                L = int(rng.integers(16, 97))
                d[i, k:k + L, j] = rng.normal(0.0, frac * sd[j])
                k += L
    return d


def build(model, root="out/datasets", Hp=32, H=256, stride=None, frac=0.25, seed=7):
    nm = fit_norm(load_trajs(root, model), model)
    stride = stride or (100 if model == "M1" else 50)
    rng = np.random.default_rng(seed)
    out = {k: [] for k in ["s", "u", "rho", "rdot", "fam", "tag"]}
    for fn in sorted(glob.glob(os.path.join(root, model, f"{model}-*.npz"))):
        d = np.load(fn)
        meta = json.loads(str(d["meta"]))
        fam = meta["family"]
        if fam not in FAMS:
            continue
        split = d["split"].astype(str)
        sel = np.where(split == ("ood" if fam == "OOD" else "test"))[0]
        if len(sel) == 0:
            continue
        x = d["x"].astype(np.float64)
        xd = d["xdot"].astype(np.float64)
        rho, rdot = _radii64(d, x, xd)
        u = d["u_avg"].astype(np.float64)
        Dd, FY, fa = d["D"].astype(np.float64), d["fy"].astype(np.float64), d["fault_a"].astype(np.float64)
        names = [str(k) for k in d["param_names"]]
        P = d["plant_params"].astype(np.float64)
        N = x.shape[1]
        n_sub = int(round(meta["dt_store"] / meta["dt_sim"]))
        dt_sim = meta["dt_sim"]
        starts = np.arange(0, N - (Hp + H), stride)
        bi, ki = np.meshgrid(sel, starts, indexing="ij")
        bi, ki = bi.ravel(), ki.ravel()
        W = len(bi)
        bp = _bparams(P[bi], names)
        uc = np.stack([u[b, k:k + Hp + H] for b, k in zip(bi, ki)])            # (W, Hp+H, 3)
        pert = perturbation(rng, W, H, nm.u_sd, frac)
        uc[:, Hp - 1:Hp - 1 + H] += pert
        uc = np.clip(uc, -bp.M_max[:, None, None], bp.M_max[:, None, None])
        xs = np.empty((W, Hp + H, 8))
        for i, (b, k) in enumerate(zip(bi, ki)):
            xs[i, :Hp] = x[b, k:k + Hp]
        xc = xs[:, Hp - 1].copy()
        for kk in range(H):
            j = Hp - 1 + kk
            idx = ki + j
            Dk, Fk = Dd[bi, idx], FY[bi, idx]
            M = uc[:, j]
            for _ in range(n_sub):
                xc = rk4(xc, M, bp, Dk, Fk, fa[bi], dt_sim)
            xs[:, j + 1] = xc
        ok = np.isfinite(xs).all(axis=(1, 2)) & (np.abs(xs[:, :, 0:2]) < 1e4).all(axis=(1, 2))
        Ru, Rr, _, _ = zip(*[geometry(xs[:, t], bp) for t in range(Hp + H)])
        rr = np.stack([np.stack(Ru, 1), np.stack(Rr, 1)], -1)
        rr[:, :Hp] = np.stack([rho[b, k:k + Hp] for b, k in zip(bi, ki)])
        rd = np.stack([-xs[..., 2] * bp.chi[:, None] / (2 * np.pi), xs[..., 4] * bp.chi[:, None] / (2 * np.pi)], -1)
        rd[:, :Hp] = np.stack([rdot[b, k:k + Hp] for b, k in zip(bi, ki)])
        tags = d["tags"].astype(str)
        out["s"].append(xs[ok, :, :5])
        out["u"].append(uc[ok])
        out["rho"].append(rr[ok])
        out["rdot"].append(rd[ok])
        out["fam"].append(np.array([fam] * int(ok.sum())))
        out["tag"].append(tags[bi[ok]])
        print(f"[{model}] CF {fam}: {int(ok.sum())}/{W} windows", flush=True)
    return {k: np.concatenate(v) for k, v in out.items()}


def as_windows(C):
    """dict in the format of data.make_windows"""
    n, L, _ = C["s"].shape
    return dict(s=C["s"], u=C["u"], ui=C["u"], rho=C["rho"], rdot=C["rdot"], sdot=np.zeros((n, L, 5)),
                D=np.zeros((n, L, 5)), fam=C["fam"], tag=C["tag"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", default="out/datasets")
    ap.add_argument("--out", default="out/bench")
    ap.add_argument("--frac", type=float, default=0.25, help="perturbation std as a fraction of the torque std")
    a = ap.parse_args()
    C = build(a.model, a.data, frac=a.frac)
    os.makedirs(os.path.join(a.out, a.model), exist_ok=True)
    np.savez_compressed(os.path.join(a.out, a.model, "cf_test.npz"), **C, frac=a.frac)
    print(f"[{a.model}] saved {len(C['s'])} counterfactual windows")


if __name__ == "__main__":
    main()
