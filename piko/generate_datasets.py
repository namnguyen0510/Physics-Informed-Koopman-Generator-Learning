"""
generate_datasets.py -- build the M1-* / M2-* closed-loop datasets.

usage:  python -m r2r_nn.generate_datasets --model M1 --out datasets [--families REF,UNC] [--n-scale 1.0]

Each family is written to <out>/<model>/<model>-<FAMILY>.npz with arrays
(B trajectories x N samples x channels, float32) and a JSON metadata string.
See DATA_CARD.md for the channel dictionary.
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np

from .scenarios import FAMILIES, make_specs
from .simulate import simulate_batch

DT_SIM = 1e-4
T_END = 4.0
STORE_EVERY = {"M1": 10, "M2": 20}          # 1 ms (M1, stiff web) / 2 ms (M2)
FAMILY_SEED = {"PAPER": 100, "NOM": 101, "REF": 102, "UNC": 103, "DIST": 104, "NOISE": 105, "FAULT": 106,
               "EXC": 107, "COMBO": 108, "OOD": 109, "KICK": 110}
MODEL_SEED = {"M1": 10_000, "M2": 20_000}

CHANNELS = {
    "x": ["T1", "T2", "wu", "w1", "wr", "phiu", "phi1", "phir"],
    "xdot": ["dT1", "dT2", "dwu", "dw1", "dwr", "dphiu", "dphi1", "dphir"],
    "y": ["T1_m", "T2_m", "wu_m", "w1_m", "wr_m", "phiu_m", "phi1_m", "phir_m"],
    "u": ["Mu", "M1", "Mr"],
    "u_avg": ["Mu_avg", "M1_avg", "Mr_avg"],
    "u_cmd": ["Mu_cmd", "M1_cmd", "Mr_cmd"],
    "ref": ["T1d", "T2d", "w1d"],
    "dref": ["dT1d", "dT2d", "dw1d"],
    "D": ["D_T1", "D_T2", "D_u", "D_1", "D_r"],
    "fy": ["y_u", "y_1", "y_r"],
    "geo": ["Ru", "Rr", "Ju", "Jr"],
    "wd": ["wu_d", "w1_d", "wr_d"],
    "est": ["est_T1", "est_T2", "est_u", "est_1", "est_r"],
}
UNITS = {"T": "N", "w": "rad/s", "phi": "rad", "M": "N m", "D_T": "N/s", "D_w": "rad/s^2",
         "R": "m", "J": "kg m^2", "t": "s"}


def split_labels(family, B, rng):
    if family in ("OOD",):
        return np.array(["ood"] * B)
    if family == "PAPER":
        return np.array(["paper"] * B)
    lab = np.array(["train"] * B, dtype=object)
    idx = rng.permutation(B)
    n_te = max(1, int(round(0.15 * B)))
    n_va = max(1, int(round(0.15 * B)))
    lab[idx[:n_te]] = "test"
    lab[idx[n_te:n_te + n_va]] = "val"
    return lab.astype(str)


def run_family(model, family, out_dir, n_scale=1.0, verbose=True, float64_states=False):
    t_sim = np.arange(0.0, T_END + 1e-12, DT_SIM)
    seed = MODEL_SEED[model] + FAMILY_SEED[family]
    specs = make_specs(model, family, seed, t_sim)
    if FAMILIES[family][0] is not None and n_scale != 1.0:
        specs = specs[:max(2, int(len(specs) * n_scale))]
    t0 = time.time()
    se = STORE_EVERY[model]
    if family == "PAPER":
        outs = [simulate_batch(model, [s], t_sim, se, expert_opts=s.get("expert_opts"), noise_seed=seed + i)
                for i, s in enumerate(specs)]
        out = {}
        for k in outs[0]:
            if k in ("t", "param_names"):
                out[k] = outs[0][k]
            elif k == "expert":
                out[k] = np.array([o[k] for o in outs])
            else:
                out[k] = np.concatenate([o[k] for o in outs], axis=0)
    else:
        out = simulate_batch(model, specs, t_sim, se, noise_seed=seed)
        out["expert"] = np.array([out["expert"]] * len(specs))
    B = len(specs)
    rng = np.random.default_rng(seed + 7)
    split = split_labels(family, B, rng)
    # expert quality flag (not a filter): steady tracking error after the first 0.5 s
    tt = out["t"]
    eT = out["x"][:, :, 0:2] - out["ref"][:, :, 0:2]
    m = tt > 0.5
    max_eT = np.abs(eT[:, m]).max(axis=(1, 2))
    rms_eT = np.sqrt((eT[:, m] ** 2).mean(axis=(1, 2)))
    keep = ~out["blown"]
    meta = dict(model=model, family=family, description=FAMILIES[family][1], seed=seed,
                dt_sim=DT_SIM, dt_store=DT_SIM * se, t_end=T_END, n_traj=int(keep.sum()),
                n_blown_removed=int((~keep).sum()), expert=sorted(set(out["expert"].tolist())),
                channels=CHANNELS, units=UNITS,
                integrator="RK4 plant, ZOH control at dt_sim; expert internal states by explicit Euler",
                generated=time.strftime("%Y-%m-%d %H:%M:%S"))
    arrays = {k: out[k][keep].astype(np.float32) for k in CHANNELS}
    if float64_states:          # M1: float32 rounding of w is amplified by E*S in f(x); keep x, xdot exact
        for k in ("x", "xdot"):
            arrays[k] = out[k][keep].astype(np.float64)
    # parameter switch (M2 PAPER S3): plant parameters after t_switch
    sw = [s.get("plant_switch") for s in specs]
    if any(v is not None for v in sw):
        from .plants import BatchParams
        arrays["t_switch"] = np.array([v[0] if v is not None else np.inf for v in sw])[keep]
        arrays["plant_params_after_switch"] = BatchParams(
            [v[1] if v is not None else s["plant"] for v, s in zip(sw, specs)]).matrix()[keep]
    arrays.update(
        t=tt.astype(np.float32),
        plant_params=out["plant_params"][keep], param_names=out["param_names"],
        fault_a=out["fault_a"][keep], fault_tf=out["fault_tf"][keep], fault_rho=out["fault_rho"][keep],
        noise_std=out["noise_std"][keep], dither_amp=out["dither_amp"][keep],
        tags=out["tags"][keep], expert=out["expert"][keep], split=split[keep],
        expert_max_eT=max_eT[keep], expert_rms_eT=rms_eT[keep],
        meta=np.array(json.dumps(meta)))
    os.makedirs(os.path.join(out_dir, model), exist_ok=True)
    fn = os.path.join(out_dir, model, f"{model}-{family}.npz")
    np.savez_compressed(fn, **arrays)
    if verbose:
        print(f"[{model}-{family}] B={B} kept={keep.sum()} {time.time() - t0:.0f}s  "
              f"expert max|eT|(t>0.5) median={np.median(max_eT[keep]):.3g} max={max_eT[keep].max():.3g}  "
              f"rms median={np.median(rms_eT[keep]):.3g}  size={os.path.getsize(fn) / 1e6:.1f} MB", flush=True)
    return fn


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=["M1", "M2"])
    ap.add_argument("--out", default="datasets")
    ap.add_argument("--families", default=",".join(FAMILIES))
    ap.add_argument("--n-scale", type=float, default=1.0)
    ap.add_argument("--float64", action="store_true", help="store x and xdot in float64 (recommended for M1 PINN work)")
    a = ap.parse_args()
    for fam in a.families.split(","):
        run_family(a.model, fam, a.out, a.n_scale, float64_states=a.float64)


if __name__ == "__main__":
    main()
