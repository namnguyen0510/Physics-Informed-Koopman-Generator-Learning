"""
train_fnn.py -- train the FNN controller (BC + DAgger) and FNN dynamics surrogates,
then evaluate the controllers in closed loop on held-out scenarios.

usage: python -m r2r_nn.train_fnn --model M1 --data out/datasets --out out/models
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import time

import numpy as np
import torch

from .fnn import (load_model_data, ctrl_features_traj, dyn_features, CTRL_RAW, CTRL_PHYS, DYN_RAW,
                  DYN_PHYS, TrainCfg, train_regressor, FNNPolicy, n_params)
from .plants import PARAMS
from .scenarios import make_specs, FAMILIES
from .simulate import simulate_batch
from .generate_datasets import MODEL_SEED, FAMILY_SEED, DT_SIM, T_END

FEATS = {"raw": CTRL_RAW, "phys": CTRL_PHYS}
DFEATS = {"raw": DYN_RAW, "phys": DYN_PHYS}
CTRL_WIDTH = 128
DYN_WIDTH = 128
TRAIN_FAMS = ["NOM", "REF", "UNC", "DIST", "NOISE", "FAULT", "EXC", "KICK", "COMBO"]


# ----------------------------------------------------------------------------
def build_ctrl_arrays(fams, model, names, splits):
    R1 = PARAMS[model].R1
    X, Y, G = [], [], []
    for d in fams:
        if d["family"] not in TRAIN_FAMS + ["OOD", "PAPER"]:
            continue
        dt = float(d["t"][1] - d["t"][0])
        F = ctrl_features_traj(d["y"], d["geo"], d["ref"], d["dref"], dt, R1, names)
        m = np.isin(d["split"], splits)
        if m.sum() == 0:
            continue
        Yc = d["u_cmd"][m]
        # drop the controller-initialisation transient (t < 30 ms) and saturated labels
        keep = (d["t"][None, :] > 0.03) & (np.abs(Yc).max(-1) < 0.98 * PARAMS[model].M_max)
        X.append(F[m][keep])
        Y.append(Yc[keep])
        G.append(np.repeat(d["family"], keep.sum()))
    return np.concatenate(X), np.concatenate(Y), np.concatenate(G)


def build_dyn_arrays(fams, model, names, splits):
    R1 = PARAMS[model].R1
    X, Y, G = [], [], []
    for d in fams:
        m = np.isin(d["split"], splits)
        if m.sum() == 0:
            continue
        F = dyn_features(d["x"][m], d["geo"][m], d["u"][m], R1, names)
        X.append(F.reshape(-1, F.shape[-1]))
        Y.append(d["xdot"][m][..., 0:5].reshape(-1, 5))
        G.append(np.repeat(np.array([d["family"]] * m.sum()), F.shape[1]))
    return np.concatenate(X), np.concatenate(Y), np.concatenate(G)


# ----------------------------------------------------------------------------
def specs_for(model, fams_data, which, t_sim, max_per_family=None, rng=None):
    """Rebuild the exact scenario specs of stored trajectories with split in `which`."""
    out = []
    for d in fams_data:
        fam = d["family"]
        specs = make_specs(model, fam, MODEL_SEED[model] + FAMILY_SEED[fam], t_sim)
        tag2spec = {s["tag"]: s for s in specs}
        sel = [i for i in range(len(d["tags"])) if d["split"][i] in which]
        if max_per_family is not None and len(sel) > max_per_family:
            sel = sorted((rng or np.random.default_rng(0)).choice(sel, max_per_family, replace=False))
        for i in sel:
            s = dict(tag2spec[str(d["tags"][i])])
            s["family"] = fam
            out.append(s)
    return out


def cl_metrics(out):
    """Paper performance indices (M1 eqs. 29-33) per trajectory + robustness flags."""
    t = out["t"]
    dt = t[1] - t[0]
    e = out["x"][:, :, 0:2] - out["ref"][:, :, 0:2]
    ew = out["x"][:, :, 3] - out["ref"][:, :, 2]
    M = out["u"]
    ISE_T = (e ** 2).sum(-1).sum(-1) * dt
    ITAE_T = (t[None, :] * np.abs(e).sum(-1)).sum(-1) * dt
    ISE_w = (ew ** 2).sum(-1) * dt
    ITAE_w = (t[None, :] * np.abs(ew)).sum(-1) * dt
    RMS_M = np.sqrt((M ** 2).mean(1)).sum(-1)
    m = t > 0.5
    maxe = np.abs(e[:, m]).max(axis=(1, 2))
    fail = out["blown"] | (maxe > 5.0) | ~np.isfinite(maxe)
    return dict(ISE_T=ISE_T, ITAE_T=ITAE_T, ISE_w1=ISE_w, ITAE_w1=ITAE_w, RMS_M=RMS_M, max_eT=maxe,
                fail=fail.astype(float))


def run_closed_loop(model, specs, t_sim, policy=None, store_every=10, seed=123):
    return simulate_batch(model, specs, t_sim, store_every, policy=policy, noise_seed=seed)


def save_sim(fn, out, specs, extra=None):
    keep = ["t", "x", "u", "u_cmd", "ref", "y", "geo", "D", "fy", "blown"]
    arr = {k: np.asarray(out[k], dtype=np.float32 if k not in ("blown",) else bool) for k in keep}
    arr["family"] = np.array([s.get("family", "?") for s in specs])
    arr["tags"] = np.array([s["tag"] for s in specs])
    if extra:
        arr.update(extra)
    np.savez_compressed(fn, **arr)


# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", default="out/datasets")
    ap.add_argument("--out", default="out/models")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--dagger", type=int, default=5)
    ap.add_argument("--reuse-bc", action="store_true")
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--skip-dyn", action="store_true")
    ap.add_argument("--skip-ctrl", action="store_true")
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    model = a.model
    od = os.path.join(a.out, model)
    os.makedirs(od, exist_ok=True)
    fams = load_model_data(a.data, model)
    summary = dict(model=model)
    t_sim = np.arange(0.0, T_END + 1e-12, DT_SIM)
    T0 = time.time()

    # ======================= dynamics surrogate =======================
    if not a.skip_dyn:
        for fs in ["raw", "phys"]:
            Xtr, Ytr, _ = build_dyn_arrays(fams, model, DFEATS[fs], ["train"])
            Xva, Yva, _ = build_dyn_arrays(fams, model, DFEATS[fs], ["val"])
            for sd in range(a.seeds):
                cfg = TrainCfg(seed=sd, epochs=80, patience=15, width=DYN_WIDTH)
                net, hist = train_regressor(Xtr, Ytr, Xva, Yva, cfg)
                torch.save(net.state_dict(), os.path.join(od, f"dyn_{fs}_s{sd}.pt"))
                json.dump(dict(hist=hist, names=DFEATS[fs], cfg=cfg.__dict__, n_params=n_params(net),
                               n_train=int(len(Xtr))), open(os.path.join(od, f"dyn_{fs}_s{sd}.json"), "w"))
                print(f"[{model}] dyn {fs} seed {sd}: best val {hist['best_val']:.4e} "
                      f"({len(hist['epoch'])} ep, {time.time() - T0:.0f}s)", flush=True)

    if a.skip_ctrl:
        return
    # ======================= controller =======================
    rng = np.random.default_rng(1)
    eval_specs = specs_for(model, [d for d in fams if d["family"] in TRAIN_FAMS], ["test"], t_sim)
    eval_specs += specs_for(model, [d for d in fams if d["family"] == "OOD"], ["ood"], t_sim,
                            max_per_family=8, rng=rng)
    eval_specs += specs_for(model, [d for d in fams if d["family"] == "PAPER"], ["paper"], t_sim)
    val_specs = specs_for(model, [d for d in fams if d["family"] in TRAIN_FAMS], ["val"], t_sim,
                          max_per_family=2, rng=rng)
    dag_specs = specs_for(model, [d for d in fams if d["family"] in TRAIN_FAMS], ["train"], t_sim,
                          max_per_family=4, rng=rng)
    print(f"[{model}] eval specs {len(eval_specs)}, val specs {len(val_specs)}, dagger specs {len(dag_specs)}", flush=True)

    def cl_eval(pol, specs, tag=None):
        out = run_closed_loop(model, specs, t_sim, pol)
        if tag is not None:
            save_sim(os.path.join(od, f"cl_{tag}.npz"), out, specs)
        return {k: v.tolist() for k, v in cl_metrics(out).items()}

    def summ(m):
        ise = np.nan_to_num(np.array(m["ISE_T"], float), nan=1e12, posinf=1e12)
        return float(np.mean(m["fail"])), float(np.median(ise))

    met = {"expert": cl_eval(None, eval_specs, "expert"), "val": {}}
    met["val"]["expert"] = cl_eval(None, val_specs)
    print(f"[{model}] expert closed loop done ({time.time() - T0:.0f}s)", flush=True)
    R1 = PARAMS[model].R1
    iE = None
    for fs in ["raw", "phys"]:
        names = FEATS[fs]
        iE = [names.index("eT1"), names.index("eT2"), names.index("ew1")]
        Xtr, Ytr, _ = build_ctrl_arrays(fams, model, names, ["train"])
        Xva, Yva, _ = build_ctrl_arrays(fams, model, names, ["val"])
        nets = []
        for sd in range(a.seeds):
            pt = os.path.join(od, f"ctrl_{fs}_bc_s{sd}.pt")
            if a.reuse_bc and os.path.exists(pt):
                from .sweeps import load_net
                nets.append(load_net(pt, len(names)))
                continue
            cfg = TrainCfg(seed=sd, epochs=80, width=CTRL_WIDTH, patience=15)
            net, hist = train_regressor(Xtr, Ytr, Xva, Yva, cfg)
            torch.save(net.state_dict(), pt)
            json.dump(dict(hist=hist, names=names, cfg=cfg.__dict__, n_params=n_params(net), n_train=int(len(Xtr))),
                      open(os.path.join(od, f"ctrl_{fs}_bc_s{sd}.json"), "w"))
            nets.append(net)
            print(f"[{model}] ctrl {fs} BC seed {sd}: best val {hist['best_val']:.4e} ({time.time() - T0:.0f}s)",
                  flush=True)
        met[f"{fs}_bc"] = cl_eval(FNNPolicy(nets[0], names, model, DT_SIM), eval_specs, f"{fs}_bc")
        prog = [dict(it=0, **dict(zip(["fail", "ISE_T"], summ(cl_eval(FNNPolicy(nets[0], names, model, DT_SIM), val_specs)))))]
        print(f"[{model}] ctrl {fs} BC: test fail {summ(met[f'{fs}_bc'])[0]:.2f}, val fail {prog[0]['fail']:.2f} "
              f"({time.time() - T0:.0f}s)", flush=True)

        # ======================= DAgger (warm-started, best iterate kept by closed-loop validation) =====
        Xagg, Yagg = Xtr.copy(), Ytr.copy()
        net = nets[0]
        best = (prog[0]["fail"], prog[0]["ISE_T"], 0, net)
        betas = [0.5, 0.25, 0.1, 0.0, 0.0, 0.0][:a.dagger]
        for k, beta in enumerate(betas, start=1):
            pol = FNNPolicy(net, names, model, DT_SIM, blend_expert=beta)
            out_d = run_closed_loop(model, dag_specs, t_sim, pol, seed=500 + k)
            ok = ~out_d["blown"]
            dtst = out_d["t"][1] - out_d["t"][0]
            F = ctrl_features_traj(out_d["y"][ok], out_d["geo"][ok], out_d["ref"][ok], out_d["dref"][ok], dtst, R1, names)
            Xn, Yn = F.reshape(-1, F.shape[-1]), out_d["u_cmd"][ok].reshape(-1, 3)
            # keep only the recoverable part of each rollout (|e_T| < 5 N, |e_w1| < 10 rad/s), unsaturated labels
            good = (np.isfinite(Xn).all(1) & np.isfinite(Yn).all(1) & (np.abs(Xn[:, iE[:2]]) < 5).all(1)
                    & (np.abs(Xn[:, iE[2]]) < 10) & (np.abs(Yn).max(1) < 0.98 * PARAMS[model].M_max))
            Xagg = np.concatenate([Xagg, Xn[good]])
            Yagg = np.concatenate([Yagg, Yn[good].astype(np.float32)])
            cfg = TrainCfg(seed=k, epochs=40, width=CTRL_WIDTH, patience=10, lr=1e-3)
            net, hist = train_regressor(Xagg, Yagg, Xva, Yva, cfg, init=copy.deepcopy(net))      # warm start
            torch.save(net.state_dict(), os.path.join(od, f"ctrl_{fs}_dagger{k}.pt"))
            json.dump(dict(hist=hist, names=names, cfg=cfg.__dict__, n_params=n_params(net), n_train=int(len(Xagg)),
                           beta=beta, n_new=int(good.sum())),
                      open(os.path.join(od, f"ctrl_{fs}_dagger{k}.json"), "w"))
            fv, iv = summ(cl_eval(FNNPolicy(net, names, model, DT_SIM), val_specs))
            prog.append(dict(it=k, fail=fv, ISE_T=iv, n_new=int(good.sum()), beta=beta))
            if (fv, iv) < best[:2]:
                best = (fv, iv, k, net)
            print(f"[{model}] ctrl {fs} DAgger{k}: +{good.sum()} samples, val fail {fv:.2f} median ISE_T {iv:.3g} "
                  f"({time.time() - T0:.0f}s)", flush=True)
        met["val"][fs] = prog
        kb, netb = best[2], best[3]
        torch.save(netb.state_dict(), os.path.join(od, f"ctrl_{fs}_dagger_best.pt"))
        json.dump(dict(best_iteration=kb, progress=prog, names=names), open(os.path.join(od, f"ctrl_{fs}_dagger_best.json"), "w"))
        met[f"{fs}_dagger"] = cl_eval(FNNPolicy(netb, names, model, DT_SIM), eval_specs, f"{fs}_dagger")
        met[f"{fs}_dagger_best_it"] = kb
        print(f"[{model}] ctrl {fs} DAgger best iterate {kb}: test fail {summ(met[f'{fs}_dagger'])[0]:.2f} "
              f"median ISE_T {summ(met[f'{fs}_dagger'])[1]:.3g} ({time.time() - T0:.0f}s)", flush=True)
    met["family"] = [s["family"] for s in eval_specs]
    met["tags"] = [s["tag"] for s in eval_specs]
    met["val_family"] = [s["family"] for s in val_specs]
    json.dump(met, open(os.path.join(od, "closed_loop_metrics.json"), "w"))
    print(f"[{model}] done in {time.time() - T0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
