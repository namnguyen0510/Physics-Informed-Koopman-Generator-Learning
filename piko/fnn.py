"""
fnn.py -- small feed-forward networks for the R2R control tasks (PyTorch).

Two tasks, two feature sets each:

  * controller  pi(features) -> [Mu, M1, Mr]      (imitation of the paper expert)
  * dynamics    f(state, input) -> d/dt [T1, T2, wu, w1, wr]   (black-box plant model;
                the non-physics baseline that the PINN will later be compared to)

  feature set "raw"  : measured states, radii, references (+ error integrals for the policy)
  feature set "phys" : raw + a handful of physically motivated combinations
                       (web-speed mismatches R1 w1 - Ru wu, Rr wr - R1 w1 and the
                       static torque terms Ru T1, R1 (T2 - T1), Rr T2).  Still a plain FNN;
                       this ablation measures how much structure alone buys.
"""
from __future__ import annotations

import glob
import json
import os
import time
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn

from .plants import PARAMS

torch.set_num_threads(2)

# ----------------------------------------------------------------------------
# data loading
# ----------------------------------------------------------------------------

def load_model_data(root, model):
    """Return list of per-family dicts (arrays kept as numpy float32)."""
    fams = []
    for fn in sorted(glob.glob(os.path.join(root, model, f"{model}-*.npz"))):
        d = dict(np.load(fn, allow_pickle=False))
        d["meta"] = json.loads(str(d["meta"]))
        d["family"] = d["meta"]["family"]
        fams.append(d)
    return fams


# ----------------------------------------------------------------------------
# features
# ----------------------------------------------------------------------------
CTRL_RAW = ["T1", "T2", "wu", "w1", "wr", "Ru", "Rr", "T1d", "T2d", "w1d", "dT1d", "dT2d", "dw1d",
            "eT1", "eT2", "ew1", "IeT1", "IeT2", "Iew1"]
CTRL_PHYS = CTRL_RAW + ["dv1", "dv2", "RuT1", "R1dT", "RrT2"]
DYN_RAW = ["T1", "T2", "wu", "w1", "wr", "Ru", "Rr", "Mu", "M1", "Mr"]
DYN_PHYS = DYN_RAW + ["dv1", "dv2", "RuT1", "R1dT", "RrT2", "w1T1", "w1T2"]
DYN_TARGETS = ["dT1", "dT2", "dwu", "dw1", "dwr"]


def _phys_cols(T1, T2, wu, w1, wr, Ru, Rr, R1):
    return dict(dv1=R1 * w1 - Ru * wu, dv2=Rr * wr - R1 * w1, RuT1=Ru * T1,
                R1dT=R1 * (T2 - T1), RrT2=Rr * T2, w1T1=w1 * T1, w1T2=w1 * T2)


def ctrl_features_traj(y, geo, ref, dref, dt, R1, names):
    """Offline features for one family: y (B,N,8) ... -> (B,N,F). Integrals by cumulative sum."""
    T1, T2, wu, w1, wr = (y[..., i] for i in range(5))
    Ru, Rr = geo[..., 0], geo[..., 1]
    eT1, eT2, ew1 = T1 - ref[..., 0], T2 - ref[..., 1], w1 - ref[..., 2]
    # integral uses the value *before* the current sample (as the online policy does)
    cs = lambda e: np.concatenate([np.zeros_like(e[..., :1]), np.cumsum(e, axis=-1)[..., :-1]], axis=-1) * dt
    cols = dict(T1=T1, T2=T2, wu=wu, w1=w1, wr=wr, Ru=Ru, Rr=Rr, T1d=ref[..., 0], T2d=ref[..., 1],
                w1d=ref[..., 2], dT1d=dref[..., 0], dT2d=dref[..., 1], dw1d=dref[..., 2],
                eT1=eT1, eT2=eT2, ew1=ew1, IeT1=cs(eT1), IeT2=cs(eT2), Iew1=cs(ew1))
    cols.update(_phys_cols(T1, T2, wu, w1, wr, Ru, Rr, R1))
    return np.stack([cols[k] for k in names], axis=-1).astype(np.float32)


def dyn_features(x, geo, u, R1, names):
    T1, T2, wu, w1, wr = (x[..., i] for i in range(5))
    Ru, Rr = geo[..., 0], geo[..., 1]
    cols = dict(T1=T1, T2=T2, wu=wu, w1=w1, wr=wr, Ru=Ru, Rr=Rr, Mu=u[..., 0], M1=u[..., 1], Mr=u[..., 2])
    cols.update(_phys_cols(T1, T2, wu, w1, wr, Ru, Rr, R1))
    return np.stack([cols[k] for k in names], axis=-1).astype(np.float32)


# ----------------------------------------------------------------------------
# network
# ----------------------------------------------------------------------------
class MLP(nn.Module):
    def __init__(self, n_in, n_out, width=64, depth=3, act="silu"):
        super().__init__()
        A = {"silu": nn.SiLU, "tanh": nn.Tanh, "relu": nn.ReLU}[act]
        layers, d = [], n_in
        for _ in range(depth):
            layers += [nn.Linear(d, width), A()]
            d = width
        layers.append(nn.Linear(d, n_out))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class Normalised(nn.Module):
    """MLP wrapped with input/output standardisation (stored as buffers -> saved in state_dict)."""

    def __init__(self, mlp, x_mu, x_sd, y_mu, y_sd):
        super().__init__()
        self.mlp = mlp
        for k, v in dict(x_mu=x_mu, x_sd=x_sd, y_mu=y_mu, y_sd=y_sd).items():
            self.register_buffer(k, torch.as_tensor(v, dtype=torch.float32))

    def forward_norm(self, xn):
        return self.mlp(xn)

    def forward(self, x):
        return self.mlp((x - self.x_mu) / self.x_sd) * self.y_sd + self.y_mu


def n_params(m):
    return sum(p.numel() for p in m.parameters())


@dataclass
class TrainCfg:
    width: int = 64
    depth: int = 3
    act: str = "silu"
    lr: float = 3e-3
    epochs: int = 60
    batch: int = 2048
    wd: float = 1e-6
    patience: int = 12
    seed: int = 0


def train_regressor(Xtr, Ytr, Xva, Yva, cfg: TrainCfg, log_every=0, init=None):
    """Standardised MSE regression with Adam + cosine LR + early stopping."""
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    x_mu, x_sd = Xtr.mean(0), Xtr.std(0) + 1e-8
    y_mu, y_sd = Ytr.mean(0), Ytr.std(0) + 1e-8
    if init is not None:            # keep normalisation of the network being fine-tuned
        x_mu, x_sd, y_mu, y_sd = (init.x_mu.numpy(), init.x_sd.numpy(), init.y_mu.numpy(), init.y_sd.numpy())
    Xn = torch.as_tensor((Xtr - x_mu) / x_sd, dtype=torch.float32)
    Yn = torch.as_tensor((Ytr - y_mu) / y_sd, dtype=torch.float32)
    Xvn = torch.as_tensor((Xva - x_mu) / x_sd, dtype=torch.float32)
    Yvn = torch.as_tensor((Yva - y_mu) / y_sd, dtype=torch.float32)
    if init is None:
        model = Normalised(MLP(Xtr.shape[1], Ytr.shape[1], cfg.width, cfg.depth, cfg.act), x_mu, x_sd, y_mu, y_sd)
    else:
        model = init
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.wd)
    n = Xn.shape[0]
    steps_per_epoch = max(1, n // cfg.batch)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=cfg.lr, total_steps=cfg.epochs * steps_per_epoch,
                                                pct_start=0.1, anneal_strategy="cos")
    hist = dict(epoch=[], train=[], val=[], lr=[], val_ch=[])
    best, best_state, bad = np.inf, None, 0
    for ep in range(cfg.epochs):
        model.train()
        perm = torch.randperm(n)
        tot = 0.0
        for i in range(steps_per_epoch):
            idx = perm[i * cfg.batch:(i + 1) * cfg.batch]
            loss = ((model.forward_norm(Xn[idx]) - Yn[idx]) ** 2).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            tot += loss.item()
        model.eval()
        with torch.no_grad():
            pv = model.forward_norm(Xvn)
            vch = ((pv - Yvn) ** 2).mean(0).numpy()
            vl = float(vch.mean())
        hist["epoch"].append(ep)
        hist["train"].append(tot / steps_per_epoch)
        hist["val"].append(vl)
        hist["val_ch"].append(vch.tolist())
        hist["lr"].append(sched.get_last_lr()[0])
        if log_every and ep % log_every == 0:
            print(f"   ep {ep:3d} train {tot / steps_per_epoch:.4e} val {vl:.4e}", flush=True)
        if vl < best - 1e-7:
            best, bad = vl, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= cfg.patience:
                break
    model.load_state_dict(best_state)
    model.eval()
    hist["best_val"] = best
    return model, hist


def predict(model, X, batch=65536):
    out = []
    with torch.no_grad():
        for i in range(0, X.shape[0], batch):
            out.append(model(torch.as_tensor(X[i:i + batch], dtype=torch.float32)).numpy())
    return np.concatenate(out, 0)


# ----------------------------------------------------------------------------
# closed-loop policy wrapper (called at dt_sim inside simulate_batch)
# ----------------------------------------------------------------------------
class FNNPolicy:
    def __init__(self, nets, names, model_name, dt, blend_expert=0.0):
        self.nets = nets if isinstance(nets, (list, tuple)) else [nets]
        self.names = names
        self.R1 = PARAMS[model_name].R1
        self.Mmax = PARAMS[model_name].M_max
        self.dt = dt
        self.beta = blend_expert   # DAgger mixing (needs expert command passed in)

    def reset(self, y10, ref0):
        B = y10.shape[0]
        self.I = np.zeros((B, 3))

    def features(self, y10, r):
        T1, T2, wu, w1, wr = (y10[:, i] for i in range(5))
        Ru, Rr = y10[:, 8], y10[:, 9]
        eT1, eT2, ew1 = T1 - r["Td"][:, 0], T2 - r["Td"][:, 1], w1 - r["w1d"]
        cols = dict(T1=T1, T2=T2, wu=wu, w1=w1, wr=wr, Ru=Ru, Rr=Rr, T1d=r["Td"][:, 0], T2d=r["Td"][:, 1],
                    w1d=r["w1d"], dT1d=r["dTd"][:, 0], dT2d=r["dTd"][:, 1], dw1d=r["dw1d"],
                    eT1=eT1, eT2=eT2, ew1=ew1, IeT1=self.I[:, 0], IeT2=self.I[:, 1], Iew1=self.I[:, 2])
        cols.update(_phys_cols(T1, T2, wu, w1, wr, Ru, Rr, self.R1))
        F = np.stack([cols[k] for k in self.names], axis=1)
        self.I += self.dt * np.stack([eT1, eT2, ew1], axis=1)
        return F

    def __call__(self, t, y10, r, M_exp=None):
        F = torch.as_tensor(self.features(y10, r), dtype=torch.float32)
        with torch.no_grad():
            M = np.mean([n(F).numpy() for n in self.nets], axis=0).astype(np.float64)
        if M_exp is not None and self.beta > 0:
            M = self.beta * M_exp + (1 - self.beta) * M
        return np.clip(M, -self.Mmax, self.Mmax)
