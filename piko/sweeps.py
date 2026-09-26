"""
sweeps.py -- closed-loop robustness sweeps (expert vs FNN controllers):
  (a) parametric uncertainty level 0..40 %,  (b) disturbance amplitude 0..3x.

usage: python -m r2r_nn.sweeps --model M1 --models out/models
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import torch

from .fnn import MLP, Normalised, FNNPolicy, CTRL_RAW, CTRL_PHYS
from .plants import PARAMS, UNCERTAIN_KEYS
from .scenarios import random_ref, x0_consistent, dist_random, fault_none
from .simulate import simulate_batch
from .train_fnn import cl_metrics
from .generate_datasets import DT_SIM, T_END

UNC_LEVELS = [0.0, 0.1, 0.2, 0.3, 0.4]
DIST_LEVELS = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0]
N_BASE = 6


def load_net(path_pt, n_in):
    sd = torch.load(path_pt)
    width = sd["mlp.net.0.weight"].shape[0]
    depth = sum(1 for k in sd if k.startswith("mlp.net.") and k.endswith(".weight")) - 1
    net = Normalised(MLP(n_in, 3, width, depth), sd["x_mu"], sd["x_sd"], sd["y_mu"], sd["y_sd"])
    net.load_state_dict(sd)
    net.eval()
    return net


def base_specs(model, t, seed=777):
    rng = np.random.default_rng(seed)
    out = []
    for i in range(N_BASE):
        r = random_ref(rng, t, model)
        out.append(dict(ref=r, D=np.zeros((t.size, 5)), fault=fault_none(), noise=None, dither=None,
                        rng_seed=int(rng.integers(1 << 30)), tag=f"base{i}"))
    return out


def build(model, t, kind):
    pn = PARAMS[model]
    specs, lev = [], []
    for L in (UNC_LEVELS if kind == "unc" else DIST_LEVELS):
        for b in base_specs(model, t):
            r2 = np.random.default_rng(b["rng_seed"])
            s = dict(b)
            if kind == "unc":
                # worst-case-ish: every uncertain parameter at +/-L with a random sign pattern
                signs = r2.choice([-1.0, 1.0], size=len(UNCERTAIN_KEYS[model]))
                upd = {k: getattr(pn, k) * (1 + L * sg) for k, sg in zip(UNCERTAIN_KEYS[model], signs)}
                s["plant"] = pn.__class__(**{**pn.as_dict(), **upd}, name=model)
            else:
                s["plant"] = pn
                s["D"] = dist_random(r2, t, model, L) if L > 0 else np.zeros((t.size, 5))
            s["x0"] = x0_consistent(model, s["ref"][0], s["plant"])
            s["tag"] = f"{kind}{L}-{b['tag']}"
            specs.append(s)
            lev.append(L)
    return specs, np.array(lev)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--models", default="out/models")
    ap.add_argument("--threads", type=int, default=1)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    model = a.model
    od = os.path.join(a.models, model)
    t = np.arange(0.0, T_END + 1e-12, DT_SIM)
    ctrls = {"expert": None}
    for fs, names in [("raw", CTRL_RAW), ("phys", CTRL_PHYS)]:
        ctrls[f"{fs}_dagger"] = (os.path.join(od, f"ctrl_{fs}_dagger_best.pt"), names)
    res = {}
    for kind in ["unc", "dist"]:
        specs, lev = build(model, t, kind)
        for name, c in ctrls.items():
            pol = None
            if c is not None:
                pol = FNNPolicy(load_net(c[0], len(c[1])), c[1], model, DT_SIM)
            out = simulate_batch(model, specs, t, 10, policy=pol, noise_seed=99)
            m = cl_metrics(out)
            res[f"{kind}|{name}"] = {k: v.tolist() for k, v in m.items()}
            res[f"{kind}|levels"] = lev.tolist()
            print(f"[{model}] sweep {kind} {name}: fail by level "
                  f"{[round(float(np.mean(m['fail'][lev == L])), 2) for L in np.unique(lev)]}", flush=True)
    json.dump(res, open(os.path.join(od, "sweeps.json"), "w"))


if __name__ == "__main__":
    main()
