"""report.py -- numeric summary tables (CSV + Markdown) for datasets, FNN controller and FNN surrogate."""
from __future__ import annotations

import json
import os

import numpy as np

from .plots_models import Ctx, r2, CTRL_KEYS
from .fnn import predict
from . import style as S
from .plots_data import FAM_ORDER, traj_metrics


def fmt(v, p=3):
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return "–"
    if abs(v) >= 1e4 or (abs(v) < 1e-3 and v != 0):
        return f"{v:.2e}"
    return f"{v:.{p}g}"


def make(model, fams, md, outdir):
    os.makedirs(outdir, exist_ok=True)
    c = Ctx(model, fams, md, os.path.join(outdir, "_tmp"))
    rep = dict(model=model)
    lines = [f"# {model}: numeric summary\n"]
    # ---------------- datasets
    lines.append("## Datasets\n\n| family | traj | samples | train/val/test/held-out | expert median ISE_T | expert median max\\|e_T\\| (t>0.5 s) |\n|---|---|---|---|---|---|")
    ds = []
    for d in c.fams:
        sp = d["split"]
        m = traj_metrics(d)
        row = dict(family=d["family"], traj=int(d["x"].shape[0]), samples=int(d["x"].shape[0] * d["x"].shape[1]),
                   train=int((sp == "train").sum()), val=int((sp == "val").sum()), test=int((sp == "test").sum()),
                   heldout=int(np.isin(sp, ["ood", "paper"]).sum()), ISE_T=float(np.median(m["ISE_T"])),
                   max_eT=float(np.median(d["expert_max_eT"])))
        ds.append(row)
        lines.append(f"| {row['family']} | {row['traj']} | {row['samples']:,} | {row['train']}/{row['val']}/{row['test']}/{row['heldout']} | "
                     f"{fmt(row['ISE_T'])} | {fmt(row['max_eT'])} |")
    rep["datasets"] = ds
    # ---------------- controller open loop
    lines.append("\n## FNN controller: open-loop imitation (test split, BC seed 0)\n\n"
                 "R² over the whole test split is dominated by the ±M_max torque spikes of KICK recoveries; "
                 "the second row per feature set excludes KICK, MAE = median absolute error.\n\n"
                 "| features | params | R² Mu | R² M1 | R² Mr | RMSE Mu | RMSE M1 | RMSE Mr | MAE Mu | MAE M1 | MAE Mr |\n|---|---|---|---|---|---|---|---|---|---|---|")
    rep["ctrl_openloop"] = {}
    for fs in ["raw", "phys"]:
        X, Y, G, _ = c.ctrl_xy(fs, ["test"], sub=2)
        P = predict(c.ctrl[f"{fs}_bc"][0], X)
        npar = sum(p.numel() for p in c.ctrl[f"{fs}_bc"][0].parameters())
        rep["ctrl_openloop"][fs] = {}
        for lab, m in [("all", np.ones(len(Y), bool)), ("excl. KICK", G != "KICK")]:
            R = r2(Y[m], P[m])
            E = np.sqrt(((P[m] - Y[m]) ** 2).mean(0))
            A = np.median(np.abs(P[m] - Y[m]), 0)
            rep["ctrl_openloop"][fs][lab] = dict(R2=R.tolist(), RMSE=E.tolist(), MAE=A.tolist(), n_params=int(npar))
            lines.append(f"| {S.FEAT_LABELS[fs]} ({lab}) | {npar:,} | " + " | ".join(fmt(v) for v in R) + " | "
                         + " | ".join(fmt(v) for v in E) + " | " + " | ".join(fmt(v) for v in A) + " |")
    # ---------------- closed loop
    if c.met is not None:
        lines.append("\n## Closed loop on held-out trajectories (test split of every family + 8 OOD + PAPER)\n\n"
                     "| controller | failure rate | median ISE_T | median ITAE_T | median ISE_ω1 | median RMS_M | median max\\|e_T\\| |\n|---|---|---|---|---|---|---|")
        rep["closed_loop"] = {}
        for k in CTRL_KEYS:
            if k not in c.met:
                continue
            m = {kk: np.nan_to_num(np.array(v, float), nan=1e12, posinf=1e12) for kk, v in c.met[k].items()}
            row = dict(fail=float(m["fail"].mean()), ISE_T=float(np.median(m["ISE_T"])), ITAE_T=float(np.median(m["ITAE_T"])),
                       ISE_w1=float(np.median(m["ISE_w1"])), RMS_M=float(np.median(m["RMS_M"])),
                       max_eT=float(np.median(m["max_eT"])))
            rep["closed_loop"][k] = row
            lines.append(f"| {S.CTRL_LABELS[k]} | {row['fail']:.2f} | {fmt(row['ISE_T'])} | {fmt(row['ITAE_T'])} | "
                         f"{fmt(row['ISE_w1'])} | {fmt(row['RMS_M'])} | {fmt(row['max_eT'])} |")
        fam = np.array(c.met["family"])
        fams = [f for f in FAM_ORDER if f in set(fam)]
        lines.append("\n### Failure rate per family\n\n| controller | " + " | ".join(fams) + " |\n|---|" + "---|" * len(fams))
        for k in CTRL_KEYS:
            if k in c.met:
                fl = np.array(c.met[k]["fail"])
                lines.append(f"| {S.CTRL_LABELS[k]} | " + " | ".join(f"{fl[fam == f].mean():.2f}" for f in fams) + " |")
    # ---------------- dynamics
    lines.append("\n## FNN dynamics surrogate (test split, seed 0)\n\n"
                 "RMSE in N/s (tension rates) and rad/s² (accelerations). On the nominal families the closed loop is smooth, so "
                 "the target variance is tiny and R² is harsh; read the RMSE.\n\n"
                 "| features | R² dT1 | R² dT2 | R² dωu | R² dω1 | R² dωr | RMSE dT1 | RMSE dT2 | RMSE dωu | RMSE dω1 | RMSE dωr |\n|---|---|---|---|---|---|---|---|---|---|---|")
    rep["dyn"] = {}
    for fs in ["raw", "phys"]:
        X, Y, G = c.dyn_xy(fs, ["test"], sub=2)
        P = predict(c.dyn[fs][0], X)
        R = r2(Y, P)
        E = np.sqrt(((P - Y) ** 2).mean(0))
        rep["dyn"][fs] = dict(R2=R.tolist(), RMSE=E.tolist())
        lines.append(f"| {S.FEAT_LABELS[fs]} (all test) | " + " | ".join(fmt(v) for v in R) + " | " + " | ".join(fmt(v) for v in E) + " |")
        # nominal-only families (no unmodelled inputs)
        Xn, Yn, _ = c.dyn_xy(fs, ["test"], fams=["NOM", "REF", "UNC"], sub=2)
        Pn = predict(c.dyn[fs][0], Xn)
        Rn = r2(Yn, Pn)
        En = np.sqrt(((Pn - Yn) ** 2).mean(0))
        rep["dyn"][fs]["R2_nominal_fams"] = Rn.tolist()
        rep["dyn"][fs]["RMSE_nominal_fams"] = En.tolist()
        lines.append(f"| {S.FEAT_LABELS[fs]} (NOM/REF/UNC) | " + " | ".join(fmt(v) for v in Rn) + " | " + " | ".join(fmt(v) for v in En) + " |")
    sw = os.path.join(md, "sweeps.json")
    if os.path.exists(sw):
        rep["sweeps"] = json.load(open(sw))
    json.dump(rep, open(os.path.join(outdir, f"summary_{model}.json"), "w"), indent=1)
    open(os.path.join(outdir, f"summary_{model}.md"), "w").write("\n".join(lines) + "\n")
    try:
        os.rmdir(os.path.join(outdir, "_tmp"))
    except OSError:
        pass
    return rep
