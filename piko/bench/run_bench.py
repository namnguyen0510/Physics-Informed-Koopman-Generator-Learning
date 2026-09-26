"""
bench/run_bench.py -- train and evaluate every architecture on the multi-step prediction task.

usage:
  python -m r2r_nn.bench.run_bench --model M1 --archs all --out out/bench
  python -m r2r_nn.bench.run_bench --model M2 --archs PIKO,LSTM --epochs 10

Protocol (identical for all models): history Hp=32 samples, training horizon H=64,
evaluation horizon H_eval=256 (M1: 1 ms samples, M2: 2 ms), windows from the train split
for training, val split for model selection, test split + OOD + PAPER for reporting.
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import torch

from .data import load_trajs, fit_norm, make_windows, Batch, FAMS_TRAIN
from . import models as M
from .piko import PIKO

ARCHS = ["DMDc", "EDMDc-RFF", "SINDYc", "ESN", "MLP-AR", "NeuralODE", "RNN", "LSTM", "GRU", "LRU-SSM",
         "MLP-Direct", "TCN", "Transformer", "DeepKoopman", "PIKO"]
ABLATIONS = ["PIKO-core", "PIKO-core-noObs", "PIKO-core-dense", "PIKO-core-polyLPV", "PIKO-core-Euler",
             "PIKO-core-noQuad", "PIKO-core-linear", "PIKO-noObs"]
PIKO_FIT_FAMS = ["NOM", "REF"]          # nominal plant, no exogenous inputs: exact derivative identification


def build(name, nm, Hp, H):
    if name == "DMDc":
        return M.DMDc(nm)
    if name == "EDMDc-RFF":
        return M.EDMDc(nm)
    if name == "SINDYc":
        return M.SINDYc(nm)
    if name == "ESN":
        return M.ESN(nm)
    if name == "MLP-AR":
        return M.MLPAR(nm)
    if name == "NeuralODE":
        return M.NODE(nm, n_sub=1)
    if name in ("RNN", "LSTM", "GRU"):
        return M.RecurrentAR(nm, cell=name.lower())
    if name == "LRU-SSM":
        return M.LRU(nm)
    if name == "MLP-Direct":
        return M.MLPDirect(nm, Hp, H)
    if name == "TCN":
        return M.TCN(nm, H)
    if name == "Transformer":
        return M.TransformerM(nm, Hp, H)
    if name == "DeepKoopman":
        return M.DeepKoopman(nm)
    if name.startswith("PIKO"):
        kw = dict(model=nm.model)
        if "noObs" in name:
            kw["use_obs"] = False
        if "dense" in name:
            kw["struct"] = False
        if "polyLPV" in name:
            kw["sched"] = "poly"
        if "Euler" in name:
            kw["exact"] = False
        if "noQuad" in name:
            kw["quad"] = False
        if "core" in name:
            kw["innov"] = False
        if "linear" in name:
            kw["relift_every"] = 0
        if "bidir" in name:
            kw["innov_bidir"] = 32
        m = PIKO(nm, **kw)
        m.name = name
        if "noPINN" in name:
            m.no_pinn = True
        return m
    raise ValueError(name)


def n_params(m):
    return int(sum(p.numel() for p in m.parameters() if p.requires_grad))


def loss_fn(model, b, H, lam_scale=1.0):
    Hp = b.Hp
    aux = None
    if isinstance(model, PIKO):
        pred, aux = model.forward_batch(b, H, return_aux=True)
    elif isinstance(model, M.DeepKoopman):
        pred, zl = model(b.s[:, :Hp], b.u, b.r, H, return_latent=True)
    else:
        pred = model(b.s[:, :Hp], b.u, b.r, H)
    tgt = b.s[:, Hp:Hp + H]
    main = ((pred - tgt) ** 2).mean()
    extra = 0.0
    if isinstance(model, PIKO):
        lam_phys = 0.0 if getattr(model, "no_pinn", False) else 0.1
        extra = model.aux_loss(b, H, aux, lam_phys=lam_phys)
    elif isinstance(model, M.DeepKoopman):
        extra = model.aux_loss(b, H, zl)
    return main + lam_scale * extra, main


@torch.no_grad()
def predict(model, b, H, chunk=512):
    out = []
    for i in range(0, b.n, chunk):
        c = b.sel(torch.arange(i, min(i + chunk, b.n)))
        if hasattr(model, "forward_batch") and c.has64:
            out.append(model.forward_batch(c, H))
        else:
            out.append(model(c.s[:, :b.Hp], c.u, c.r, H))
    return torch.cat(out)


def train(model, btr, bva, H, epochs, lr, batch, time_cap, log):
    """Adam with a *time-based* schedule (linear warm-up over the first 5 % of the budget, cosine
    decay to 2 % of the peak at the end of the budget) and a time-based horizon curriculum
    (H grows from 8 to the full horizon over the first 40 %).  Every model gets the same wall-clock
    budget and a fully annealed learning rate, regardless of its cost per step."""
    groups = model.param_groups(lr) if hasattr(model, "param_groups") else [dict(params=list(model.parameters()), lr=lr)]
    for g_ in groups:
        g_["peak"] = g_["lr"]
    opt = torch.optim.Adam(groups, lr=lr)
    best, best_state, hist = np.inf, None, dict(epoch=[], train=[], val=[], H=[], time=[], lr=[])
    t0 = time.time()
    g = torch.Generator().manual_seed(0)

    def frac():
        return min(1.0, (time.time() - t0) / time_cap)

    def set_lr():
        f = frac()
        m = f / 0.05 if f < 0.05 else 0.02 + 0.98 * 0.5 * (1 + np.cos(np.pi * (f - 0.05) / 0.95))
        for g_ in opt.param_groups:
            g_["lr"] = g_["peak"] * max(m, 1e-3)

    ep = 0
    while ep < epochs and frac() < 1.0:
        model.train()
        f0 = frac()
        Hc = H if model.kind == "block" else int(min(H, 8 + (H - 8) * f0 / 0.4))
        perm = torch.randperm(btr.n, generator=g)
        tot, nb = 0.0, 0
        for i in range(0, btr.n - batch + 1, batch):
            set_lr()
            b = btr.sel(perm[i:i + batch])
            loss, main = loss_fn(model, b, Hc)
            if not torch.isfinite(loss):
                continue
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tot += main.item()
            nb += 1
            if frac() >= 1.0:
                break
        model.eval()
        with torch.no_grad():
            pv = predict(model, bva, H)
            vl = float(((pv - bva.s[:, bva.Hp:bva.Hp + H]) ** 2).mean())
            vl = vl if np.isfinite(vl) else 1e9
        for k, v in (("epoch", ep), ("train", tot / max(nb, 1)), ("val", vl), ("H", Hc), ("time", time.time() - t0),
                     ("lr", opt.param_groups[-1]["lr"])):
            hist[k].append(v)
        if Hc == H and vl < best:
            best, best_state = vl, {k: v.clone() for k, v in model.state_dict().items()}
        log(f"    ep {ep:2d} H={Hc:3d} train {tot / max(nb, 1):.3e} val {vl:.3e} lr {opt.param_groups[-1]['lr']:.1e} ({time.time() - t0:.0f}s)")
        ep += 1
    if best_state is None:
        best_state = {k: v.clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    hist["best_val"] = best
    hist["train_time"] = time.time() - t0
    return hist


def evaluate(model, bte, nm, H_eval, fams):
    t0 = time.time()
    pred = predict(model, bte, H_eval)
    t_pred = time.time() - t0
    tgt = bte.s[:, bte.Hp:bte.Hp + H_eval]
    err = (pred - tgt).numpy().astype(np.float64)                        # normalised units
    div = ~np.isfinite(err).all(axis=(1, 2)) | (np.abs(np.nan_to_num(err, nan=1e3)) > 20).any(axis=(1, 2))
    e = np.nan_to_num(err, nan=1e3)
    res = dict(n_windows=int(len(e)), divergence_rate=float(div.mean()))
    hs = [h for h in [1, 4, 16, 32, 64, 128, 256] if h <= H_eval]
    res["nrmse_at"] = {str(h): float(np.sqrt((e[:, h - 1] ** 2).mean())) for h in hs}
    res["nrmse_ch_at64"] = np.sqrt((e[:, 63] ** 2).mean(0)).tolist() if H_eval >= 64 else None
    res["nrmse_curve"] = np.sqrt((e ** 2).mean(axis=(0, 2))).tolist()
    res["nrmse_ch_curve"] = np.sqrt((e ** 2).mean(axis=0)).T.tolist()
    res["nrmse_win64"] = float(np.sqrt((e[:, :64] ** 2).mean()))
    res["nrmse_win256"] = float(np.sqrt((e ** 2).mean()))
    # physical units at horizon 64 (T in N, w in rad/s)
    sd = nm.s_sd
    res["rmse_phys_at64"] = (np.sqrt((e[:, 63] ** 2).mean(0)) * sd).tolist()
    fam = bte.fam
    res["by_family_win64"] = {f: float(np.sqrt((e[fam == f, :64] ** 2).mean())) for f in fams if (fam == f).any()}
    res["by_family_div"] = {f: float(div[fam == f].mean()) for f in fams if (fam == f).any()}
    res["median_window_nrmse64"] = float(np.median(np.sqrt((e[:, :64] ** 2).mean(axis=(1, 2)))))
    res["pred_seconds"] = t_pred
    return res, pred


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", default="out/datasets")
    ap.add_argument("--out", default="out/bench")
    ap.add_argument("--archs", default="all")
    ap.add_argument("--Hp", type=int, default=32)
    ap.add_argument("--H", type=int, default=64)
    ap.add_argument("--H-eval", type=int, default=256)
    ap.add_argument("--epochs", type=int, default=1000, help="upper bound; the wall-clock budget --time-cap governs")
    ap.add_argument("--max-train", type=int, default=24000)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--time-cap", type=float, default=420)
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", default="")
    ap.add_argument("--skip-done", action="store_true", help="skip architectures that already have res_<name>.json")
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    torch.manual_seed(a.seed)
    od = os.path.join(a.out, a.model)
    os.makedirs(od, exist_ok=True)
    logf = open(os.path.join(od, f"log{a.tag}.txt"), "a")

    def log(msg):
        print(msg, flush=True)
        logf.write(msg + "\n")
        logf.flush()

    trajs = load_trajs(a.data, a.model)
    nm = fit_norm(trajs, a.model)
    nm.model = a.model
    json.dump(nm.to_json(), open(os.path.join(od, "norm.json"), "w"))
    Hp, H = a.Hp, a.H
    Wtr = make_windows(trajs, ["train"], Hp, H, stride=4, fams=FAMS_TRAIN, max_n=a.max_train, seed=a.seed)
    Wva = make_windows(trajs, ["val"], Hp, H, stride=16, fams=FAMS_TRAIN, max_n=3000, seed=1)
    stride_te = 100 if a.model == "M1" else 50
    Wte = make_windows(trajs, ["test", "ood", "paper"], Hp, a.H_eval, stride=stride_te, seed=2)
    archs = ARCHS + ABLATIONS if a.archs == "all" else (ABLATIONS if a.archs == "ablations" else a.archs.split(","))
    f64 = any(n.startswith("PIKO") for n in archs)
    btr, bva, bte = Batch(Wtr, nm, Hp, f64=f64), Batch(Wva, nm, Hp, f64=f64), Batch(Wte, nm, Hp, f64=f64)
    fams = sorted(set(Wte["fam"].tolist()))
    log(f"[{a.model}] windows train {btr.n} val {bva.n} test {bte.n} | families {fams}")
    np.savez_compressed(os.path.join(od, "test_targets.npz"), s=bte.s.numpy(), fam=bte.fam, tag=Wte["tag"])
    # counterfactual-input test (see cf_test.py), if generated
    bcf, fams_cf = None, []
    cf_path = os.path.join(od, "cf_test.npz")
    if os.path.exists(cf_path):
        from .cf_test import as_windows
        Ccf = dict(np.load(cf_path))
        bcf = Batch(as_windows(Ccf), nm, Hp, f64=f64)
        fams_cf = sorted(set(bcf.fam.tolist()))
        np.savez_compressed(os.path.join(od, "cf_targets.npz"), s=bcf.s.numpy(), fam=bcf.fam, tag=Ccf["tag"])
        log(f"[{a.model}] counterfactual windows {bcf.n}")
    del trajs, Wtr, Wva
    for name in archs:
        if a.skip_done and os.path.exists(os.path.join(od, f"res_{name}.json")):
            log(f"[{a.model}] skip {name} (done)")
            continue
        torch.manual_seed(a.seed)
        model = build(name, nm, Hp, H)
        log(f"[{a.model}] === {model.name} ({model.kind}) ===")
        t0 = time.time()
        hist = {}
        if model.kind == "closed":
            if isinstance(model, M.SINDYc):   # derivative regression on windows without unmodelled inputs
                model.fit_windows(btr.sel(torch.as_tensor(np.where(np.isin(btr.fam, ["NOM", "REF", "EXC", "UNC"]))[0])))
            else:
                model.fit_windows(btr)
            hist = dict(train_time=time.time() - t0)
        else:
            if isinstance(model, PIKO):
                # derivative-based identification on nominal windows without exogenous inputs
                clean = btr.sel(torch.as_tensor(np.where(np.isin(btr.fam, PIKO_FIT_FAMS))[0]))
                model.init_gedmd(clean)
                log(f"    gEDMD init done ({time.time() - t0:.0f}s)")
                if model.innov and name != "PIKO-init":
                    model.cache_physics(btr, H)
                    model.cache_physics(bva, H)
                    log(f"    Koopman/observer predictions cached ({time.time() - t0:.0f}s)")
            lr = 1e-3 if name == "Transformer" else 2e-3
            if isinstance(model, PIKO) and not model.innov:
                hist = dict(train_time=time.time() - t0)
            else:
                hist = train(model, btr, bva, H, a.epochs, lr, a.batch, a.time_cap, log)
        res, pred = evaluate(model, bte, nm, a.H_eval, fams)
        if bcf is not None:
            if isinstance(model, PIKO) and model.innov:
                model.innov_off = True            # torques are decisions: closed-loop innovations disabled
                r_off, p_off = evaluate(model, bcf, nm, a.H_eval, fams_cf)
                model.innov_off = False
                res["cf_innov_off"] = {k: v for k, v in r_off.items() if k not in ("nrmse_curve", "nrmse_ch_curve")}
                res["cf_innov_off"]["nrmse_curve"] = r_off["nrmse_curve"]
                np.savez_compressed(os.path.join(od, f"predcf_off_{model.name}.npz"), pred=p_off.numpy().astype(np.float32))
            r_cf, p_cf = evaluate(model, bcf, nm, a.H_eval, fams_cf)
            res["cf"] = {k: v for k, v in r_cf.items() if k != "nrmse_ch_curve"}
            np.savez_compressed(os.path.join(od, f"predcf_{model.name}.npz"), pred=p_cf.numpy().astype(np.float32))
        # inference latency: one batch of 256 windows, H=64
        b1 = bte.sel(torch.arange(min(256, bte.n)))
        with torch.no_grad():
            t1 = time.time()
            if hasattr(model, "forward_batch") and b1.has64:
                model.forward_batch(b1, 64)
            else:
                model(b1.s[:, :Hp], b1.u, b1.r, 64)
            lat = (time.time() - t1) / b1.n * 1e3
        res.update(name=model.name, kind=model.kind, uses_sdot=bool(model.uses_sdot), params=n_params(model),
                   train_time=hist.get("train_time", 0.0), latency_ms_per_window=lat, hist=hist)
        if hasattr(model, "sparsity"):
            res["sindy_sparsity"] = model.sparsity
        json.dump(res, open(os.path.join(od, f"res_{model.name}.json"), "w"))
        np.savez_compressed(os.path.join(od, f"pred_{model.name}.npz"), pred=pred.numpy().astype(np.float32))
        torch.save(model.state_dict(), os.path.join(od, f"model_{model.name}.pt"))
        cfs = ""
        if "cf" in res:
            cfs = f" | CF win64 {res['cf']['nrmse_win64']:.4f} win256 {res['cf']['nrmse_win256']:.4f}"
            if "cf_innov_off" in res:
                cfs += f" (innov off: {res['cf_innov_off']['nrmse_win64']:.4f} / {res['cf_innov_off']['nrmse_win256']:.4f})"
        log(f"[{a.model}] {model.name}: NRMSE@1 {res['nrmse_at']['1']:.4f} @64 {res['nrmse_at']['64']:.4f} "
            f"@256 {res['nrmse_at'].get('256', float('nan')):.4f} win64 {res['nrmse_win64']:.4f} win256 {res['nrmse_win256']:.4f} "
            f"div {res['divergence_rate']:.3f} params {res['params']} time {res['train_time']:.0f}s" + cfs)


if __name__ == "__main__":
    main()
