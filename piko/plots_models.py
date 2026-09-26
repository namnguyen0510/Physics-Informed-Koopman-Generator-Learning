"""plots_models.py -- diagnostics for the FNN controller (C01-C14) and FNN dynamics surrogate (S01-S09)."""
from __future__ import annotations

import glob
import json
import os

import numpy as np
import torch
import matplotlib.pyplot as plt

from . import style as S
from .fnn import (MLP, Normalised, ctrl_features_traj, dyn_features, CTRL_RAW, CTRL_PHYS, DYN_RAW, DYN_PHYS,
                  DYN_TARGETS, predict)
from .plants import PARAMS, BatchParams, R2RParams, xdot as plant_xdot, geometry
from .plots_data import by_family, FAM_ORDER

CF = {"raw": CTRL_RAW, "phys": CTRL_PHYS}
DF = {"raw": DYN_RAW, "phys": DYN_PHYS}
CTRL_KEYS = ["expert", "raw_dagger", "phys_dagger", "raw_bc", "phys_bc"]


def load_net(pt, n_in, n_out):
    sd = torch.load(pt)
    width = sd["mlp.net.0.weight"].shape[0]
    depth = sum(1 for k in sd if k.startswith("mlp.net.") and k.endswith(".weight")) - 1
    net = Normalised(MLP(n_in, n_out, width, depth), sd["x_mu"], sd["x_sd"], sd["y_mu"], sd["y_sd"])
    net.load_state_dict(sd)
    net.eval()
    return net


def last_dagger(md, fs):
    """number of DAgger iterations run (files ctrl_<fs>_dagger<k>.pt)"""
    ks = []
    for f in glob.glob(os.path.join(md, f"ctrl_{fs}_dagger*.pt")):
        tail = os.path.basename(f).split("dagger")[1].split(".")[0]
        if tail.isdigit():
            ks.append(int(tail))
    return max(ks) if ks else 0


def best_dagger(md, fs):
    p = os.path.join(md, f"ctrl_{fs}_dagger_best.json")
    return json.load(open(p))["best_iteration"] if os.path.exists(p) else None


def r2(y, p):
    return 1 - ((y - p) ** 2).sum(0) / ((y - y.mean(0)) ** 2).sum(0)


def plant_params_obj(names, vals):
    return R2RParams(**{k: float(v) for k, v in zip(names, vals)}, name="p")


# ============================================================================
class Ctx:
    def __init__(self, model, fams, md, outdir):
        self.model, self.md, self.out = model, md, outdir
        self.fams = by_family(fams)
        self.fd = {d["family"]: d for d in self.fams}
        self.R1 = PARAMS[model].R1
        os.makedirs(outdir, exist_ok=True)
        self.kd = {fs: last_dagger(md, fs) for fs in ["raw", "phys"]}
        self.ctrl = {}
        for fs in ["raw", "phys"]:
            self.ctrl[f"{fs}_bc"] = [load_net(os.path.join(md, f"ctrl_{fs}_bc_s{s}.pt"), len(CF[fs]), 3) for s in range(3)
                                     if os.path.exists(os.path.join(md, f"ctrl_{fs}_bc_s{s}.pt"))]
            if os.path.exists(os.path.join(md, f"ctrl_{fs}_dagger_best.pt")):
                self.ctrl[f"{fs}_dagger"] = [load_net(os.path.join(md, f"ctrl_{fs}_dagger_best.pt"), len(CF[fs]), 3)]
        self.kbest = {fs: best_dagger(md, fs) for fs in ["raw", "phys"]}
        self.dyn = {fs: [load_net(os.path.join(md, f"dyn_{fs}_s{s}.pt"), len(DF[fs]), 5) for s in range(3)
                         if os.path.exists(os.path.join(md, f"dyn_{fs}_s{s}.pt"))] for fs in ["raw", "phys"]}
        self.cl = {}
        for k, fn in [("expert", "cl_expert"), ("raw_bc", "cl_raw_bc"), ("phys_bc", "cl_phys_bc"),
                      ("raw_dagger", "cl_raw_dagger"), ("phys_dagger", "cl_phys_dagger")]:
            p = os.path.join(md, fn + ".npz")
            if os.path.exists(p):
                self.cl[k] = dict(np.load(p))
        mp = os.path.join(md, "closed_loop_metrics.json")
        self.met = json.load(open(mp)) if os.path.exists(mp) else None


    # ---------------- helpers ----------------
    def ctrl_xy(self, fs, splits, sub=1):
        X, Y, G, T = [], [], [], []
        for d in self.fams:
            m = np.isin(d["split"], splits)
            if not m.any():
                continue
            dt = float(d["t"][1] - d["t"][0])
            F = ctrl_features_traj(d["y"][m], d["geo"][m], d["ref"][m], d["dref"][m], dt, self.R1, CF[fs])[:, ::sub]
            Yc = d["u_cmd"][m][:, ::sub]
            tt = np.broadcast_to(d["t"][::sub][None, :], Yc.shape[:2])
            keep = (tt > 0.03) & (np.abs(Yc).max(-1) < 0.98 * PARAMS[self.model].M_max)
            X.append(F[keep])
            Y.append(Yc[keep])
            G.append(np.repeat(d["family"], keep.sum()))
            T.append(tt[keep])
        return np.concatenate(X), np.concatenate(Y), np.concatenate(G), np.concatenate(T)

    def dyn_xy(self, fs, splits, fams=None, sub=1):
        X, Y, G = [], [], []
        for d in self.fams:
            if fams is not None and d["family"] not in fams:
                continue
            m = np.isin(d["split"], splits)
            if not m.any():
                continue
            F = dyn_features(d["x"][m][:, ::sub], d["geo"][m][:, ::sub], d["u"][m][:, ::sub], self.R1, DF[fs])
            X.append(F.reshape(-1, F.shape[-1]))
            Y.append(d["xdot"][m][:, ::sub, 0:5].reshape(-1, 5))
            G.append(np.repeat(d["family"], F.shape[0] * F.shape[1]))
        return np.concatenate(X), np.concatenate(Y), np.concatenate(G)

    def save(self, fig, name):
        S.save(fig, os.path.join(self.out, f"{name}_{self.model}"))


# ============================================================================
#  controller figures
# ============================================================================
def c01_learning(c):
    fig, axs = plt.subplots(1, 2, figsize=(11, 3.8), sharey=True)
    for ax, fs in zip(axs, ["raw", "phys"]):
        col = S.FEAT_COLORS[fs]
        for s in range(3):
            p = os.path.join(c.md, f"ctrl_{fs}_bc_s{s}.json")
            if not os.path.exists(p):
                continue
            h = json.load(open(p))["hist"]
            ax.plot(h["epoch"], h["val"], color=col, lw=1.3 if s == 0 else 0.8, alpha=1 if s == 0 else 0.5,
                    label="BC validation (seed 0)" if s == 0 else ("BC validation (seeds 1-2)" if s == 1 else None))
            if s == 0:
                ax.plot(h["epoch"], h["train"], color=col, lw=1.0, ls=(0, (1, 1.5)), label="BC train (seed 0)")
        for k in range(1, c.kd[fs] + 1):
            h = json.load(open(os.path.join(c.md, f"ctrl_{fs}_dagger{k}.json")))["hist"]
            ax.plot(h["epoch"], h["val"], color=S.INK2, lw=0.9, alpha=0.4 + 0.6 * k / max(1, c.kd[fs]),
                    label=f"DAgger iter {k} validation")
        ax.set_yscale("log")
        ax.set_xlabel("epoch")
        ax.set_title(f"{S.FEAT_LABELS[fs]} controller")
        ax.legend(fontsize=6.8)
    axs[0].set_ylabel("MSE (standardised torque)")
    S.suptitle(fig, f"{c.model} · C01 controller learning curves")
    fig.tight_layout(rect=(0, 0, 1, fig._top))
    c.save(fig, "C01_ctrl_learning")


def c02_parity(c):
    fig, axs = plt.subplots(2, 3, figsize=(11.5, 7.2))
    for r, fs in enumerate(["raw", "phys"]):
        X, Y, G, _ = c.ctrl_xy(fs, ["test"], sub=2)
        P = predict(c.ctrl[f"{fs}_bc"][0], X)
        R = r2(Y, P)
        for j, lab in enumerate(["Mu", "M1", "Mr"]):
            ax = axs[r, j]
            lo, hi = np.percentile(Y[:, j], [0.5, 99.5])
            pad = 0.1 * (hi - lo)
            ax.hexbin(Y[:, j], P[:, j], gridsize=70, bins="log", cmap=S.CMAP_SEQ, mincnt=1, linewidths=0,
                      extent=(lo - pad, hi + pad, lo - pad, hi + pad))
            ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], color=S.INK2, lw=0.8)
            ax.set_xlim(lo - pad, hi + pad)
            ax.set_ylim(lo - pad, hi + pad)
            rm = np.sqrt(((Y[:, j] - P[:, j]) ** 2).mean())
            ax.set_title(f"{S.FEAT_LABELS[fs]} · {lab}")
            ax.text(0.03, 0.95, f"R² = {R[j]:.3f}\nRMSE = {rm:.3g} N m", transform=ax.transAxes, va="top", fontsize=7.5, color=S.INK)
            ax.set_xlabel(f"expert {lab} (N m)")
            ax.set_ylabel(f"FNN {lab} (N m)")
    S.suptitle(fig, f"{c.model} · C02 parity: FNN (BC, seed 0) vs expert torque on the test split",
               "hexbin = log sample density; diagonal = perfect imitation")
    fig.tight_layout(rect=(0, 0, 1, fig._top))
    c.save(fig, "C02_ctrl_parity")


def c03_residuals(c):
    fig, axs = plt.subplots(2, 3, figsize=(12, 6.2))
    for fs in ["raw", "phys"]:
        X, Y, G, T = c.ctrl_xy(fs, ["test"], sub=2)
        P = predict(c.ctrl[f"{fs}_bc"][0], X)
        E = P - Y
        for j, lab in enumerate(["Mu", "M1", "Mr"]):
            ax = axs[0, j]
            lim = np.percentile(np.abs(E[:, j]), 99.5)
            ax.hist(E[:, j], bins=120, range=(-lim, lim), color=S.FEAT_COLORS[fs], alpha=0.55, label=S.FEAT_LABELS[fs],
                    histtype="stepfilled", lw=0)
            ax.set_yscale("log")
            ax.set_title(f"residual FNN − expert · {lab}")
            ax.set_xlabel("N m")
            bins = np.linspace(0, T.max(), 41)
            idx = np.digitize(T, bins) - 1
            rms = np.array([np.sqrt((E[idx == k, j] ** 2).mean()) if (idx == k).any() else np.nan for k in range(40)])
            axs[1, j].plot(0.5 * (bins[1:] + bins[:-1]), rms, color=S.FEAT_COLORS[fs], lw=1.4, label=S.FEAT_LABELS[fs])
            axs[1, j].set_yscale("log")
            axs[1, j].set_title(f"RMS residual vs time · {lab}")
            axs[1, j].set_xlabel("time within trajectory (s)")
    axs[0, 0].legend()
    axs[1, 0].legend()
    S.suptitle(fig, f"{c.model} · C03 imitation residuals (BC, test split)")
    fig.tight_layout(rect=(0, 0, 1, fig._top))
    c.save(fig, "C03_ctrl_residuals")


def c04_family_error(c):
    fig, ax = plt.subplots(figsize=(11, 3.8))
    fams = [f for f in FAM_ORDER if f in c.fd]
    w = 0.38
    for i, fs in enumerate(["raw", "phys"]):
        Xtr, Ytr, _, _ = c.ctrl_xy(fs, ["train"], sub=20)
        sd = Ytr.std(0)
        X, Y, G, _ = c.ctrl_xy(fs, ["test", "ood", "paper"], sub=2)
        P = predict(c.ctrl[f"{fs}_bc"][0], X)
        vals = [np.mean(np.sqrt(((P[G == f] - Y[G == f]) ** 2).mean(0)) / sd) if (G == f).any() else np.nan for f in fams]
        ax.bar(np.arange(len(fams)) + (i - 0.5) * w, vals, width=w - 0.04, color=S.FEAT_COLORS[fs], label=S.FEAT_LABELS[fs])
    ax.set_xticks(range(len(fams)), fams)
    ax.set_ylabel("NRMSE (RMSE / train std, mean of 3 torques)")
    ax.legend()
    ax.grid(axis="x", visible=False)
    S.suptitle(fig, f"{c.model} · C04 open-loop imitation error per family (test / OOD / PAPER trajectories)")
    fig.tight_layout(rect=(0, 0, 1, fig._top))
    c.save(fig, "C04_ctrl_family_error")


def _paper_rows(c):
    e = c.cl["expert"]
    return [i for i, f in enumerate(e["family"]) if f == "PAPER"]


def c05_cl_paper(c):
    rows = _paper_rows(c)
    if not rows:
        return
    e = c.cl["expert"]
    t = e["t"]
    show = ["expert", "raw_dagger", "phys_dagger"]
    fig, axs = plt.subplots(len(rows), 4, figsize=(13, 2.3 * len(rows)), squeeze=False)
    for r, b in enumerate(rows):
        tag = str(e["tags"][b])
        g = tag.split(":")[0]
        for cc, (k, ch, rch, lab) in enumerate([("x", 0, 0, "T1 (N)"), ("x", 1, 1, "T2 (N)"), ("x", 3, 2, "ω1 (rad/s)"),
                                               ("u", 0, None, "Mu (N m)")]):
            ax = axs[r, cc]
            ref = e["ref"][b, :, rch] if rch is not None else None
            base = e[k][b, :, ch]
            lo, hi = np.nanmin(base), np.nanmax(base)
            if ref is not None:
                lo, hi = min(lo, ref.min()), max(hi, ref.max())
                ax.plot(t, ref, color=S.REFC, lw=0.9, ls=(0, (4, 2)), label="reference")
            span = max(hi - lo, {"T1 (N)": 1.0, "T2 (N)": 1.0, "ω1 (rad/s)": 2.0}.get(lab, 0.1 * max(abs(hi), abs(lo), 1e-3)))
            for key in show:
                if key not in c.cl:
                    continue
                sig = c.cl[key][k][b, :, ch]
                eT = np.abs(c.cl[key]["x"][b, :, 0:2] - c.cl[key]["ref"][b, :, 0:2]).max(1)
                bad = np.where(~np.isfinite(eT) | (eT > 5))[0]
                kend = bad[0] if (bad.size and key != "expert") else t.size
                ax.plot(t[:kend], sig[:kend], color=S.CTRL_COLORS[key], lw=1.1 if key == "expert" else 0.9,
                        label=S.CTRL_LABELS[key])
                if kend < t.size:
                    yv = np.clip(sig[max(kend - 1, 0)], lo - 0.25 * span, hi + 0.25 * span)
                    ax.scatter([t[kend]], [yv], marker="x", s=30, color=S.CTRL_COLORS[key], lw=1.4, zorder=5)
            ax.set_ylim(lo - 0.3 * span, hi + 0.3 * span)
            xl = (0, 0.2) if (c.model == "M2" and g == "S1") else (0, t[-1])
            ax.set_xlim(*xl)
            ax.set_title(f"{g} · {lab.split(' ')[0]}")
            ax.set_ylabel(lab)
        # failure annotation
        for key in show[1:]:
            if key in c.cl:
                eT = np.abs(c.cl[key]["x"][b, :, 0:2] - c.cl[key]["ref"][b, :, 0:2]).max(1)
                bad = np.where(eT > 5)[0]
                if bad.size:
                    axs[r, 2].text(0.98, 0.04 + 0.1 * (show.index(key) - 1), f"{S.CTRL_LABELS[key]} fails at t = {t[bad[0]]:.2f} s",
                                   transform=axs[r, 2].transAxes, ha="right", fontsize=6.3, color=S.INK2,
                                   bbox=dict(facecolor="white", edgecolor="none", alpha=0.8, pad=1))
    h, l = axs[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.01))
    for ax in axs[-1]:
        ax.set_xlabel("time (s)")
    S.suptitle(fig, f"{c.model} · C05 closed loop on the paper scenarios: expert vs final FNN controllers",
               "FNN traces stop at ✕ where |e_T| first exceeds 5 N (failure); y-limits follow the expert/reference envelope")
    fig.tight_layout(rect=(0, 0.03, 1, fig._top))
    c.save(fig, "C05_cl_paper")


def c06_cl_metrics(c):
    if c.met is None:
        return
    fam = np.array(c.met["family"])
    fams = [f for f in FAM_ORDER if f in set(fam)]
    keys = [k for k in CTRL_KEYS if k in c.met]
    fig, axs = plt.subplots(3, 1, figsize=(12, 8.4), sharex=True)
    w = 0.8 / len(keys)
    for i, k in enumerate(keys):
        m = c.met[k]
        fl = np.array(m["fail"], float) > 0.5
        for ax, (mk, fn) in zip(axs, [("ISE_T", np.median), ("fail", np.mean), ("RMS_M", np.median)]):
            v = np.array(m[mk], float)
            if mk == "RMS_M":      # effort is only meaningful on runs that did not fail
                vals = [np.median(v[(fam == f) & ~fl]) if ((fam == f) & ~fl).any() else np.nan for f in fams]
            else:
                vals = [fn(np.clip(np.nan_to_num(v[fam == f], nan=1e12), 0, 1e12)) for f in fams]
            ax.bar(np.arange(len(fams)) + (i - (len(keys) - 1) / 2) * w, vals, width=w * 0.9, color=S.CTRL_COLORS[k],
                   label=S.CTRL_LABELS[k])
    axs[0].set_yscale("log")
    axs[0].set_ylabel("median ISE_T (N² s)")
    axs[1].set_ylabel("failure rate\n(|e_T|>5 N or diverged)")
    axs[1].set_ylim(0, 1.05)
    axs[2].set_yscale("log")
    axs[2].set_ylabel("median RMS_M (N m)\nnon-failed runs only")
    axs[2].set_xticks(range(len(fams)), fams)
    for ax in axs:
        ax.grid(axis="x", visible=False)
    h, l = axs[0].get_legend_handles_labels()
    fig.legend(h, l, ncol=5, loc="lower center", bbox_to_anchor=(0.5, -0.005), fontsize=7.5)
    S.suptitle(fig, f"{c.model} · C06 closed-loop performance per scenario family (held-out trajectories)",
               "indices of M1 eqs. (29)-(33); ISE clipped at 1e12 for diverged runs; missing RMS_M bar = every run of that family failed")
    fig.tight_layout(rect=(0, 0.03, 1, fig._top))
    c.save(fig, "C06_cl_metrics")


def c07_cdf(c):
    if c.met is None:
        return
    fig, axs = plt.subplots(1, 2, figsize=(11, 3.8))
    for k in [k for k in CTRL_KEYS if k in c.met]:
        for ax, mk in zip(axs, ["max_eT", "ISE_T"]):
            v = np.sort(np.nan_to_num(np.array(c.met[k][mk], float), nan=1e12, posinf=1e12))
            v = np.clip(v, 1e-8, 1e12)
            ax.step(v, np.arange(1, v.size + 1) / v.size, where="post", color=S.CTRL_COLORS[k], lw=1.4, label=S.CTRL_LABELS[k])
    axs[0].axvline(5, color=S.INK2, lw=0.8)
    axs[0].text(5, 0.02, " failure threshold", fontsize=7, color=S.INK2)
    for ax, lab in zip(axs, ["max |e_T| for t>0.5 s (N)", "ISE_T (N² s)"]):
        ax.set_xscale("log")
        ax.set_xlabel(lab)
        ax.set_ylabel("fraction of trajectories ≤ x")
    axs[0].legend(fontsize=7)
    S.suptitle(fig, f"{c.model} · C07 empirical CDFs of closed-loop tension error (all held-out trajectories)")
    fig.tight_layout(rect=(0, 0, 1, fig._top))
    c.save(fig, "C07_cl_cdf")


def c08_perm_importance(c):
    fig, axs = plt.subplots(1, 2, figsize=(12, 5.4))
    rng = np.random.default_rng(0)
    for ax, fs in zip(axs, ["raw", "phys"]):
        X, Y, _, _ = c.ctrl_xy(fs, ["test"], sub=8)
        net = c.ctrl[f"{fs}_bc"][0]
        sd = Y.std(0)
        base = (((predict(net, X) - Y) / sd) ** 2).mean()
        imp = []
        for j in range(X.shape[1]):
            Xp = X.copy()
            Xp[:, j] = rng.permutation(Xp[:, j])
            imp.append((((predict(net, Xp) - Y) / sd) ** 2).mean() / base)
        imp = np.array(imp)
        o = np.argsort(imp)
        ax.barh(np.arange(len(o)), imp[o], color=S.FEAT_COLORS[fs], height=0.7)
        ax.set_yticks(range(len(o)), [CF[fs][i] for i in o], fontsize=7)
        ax.set_xscale("log")
        ax.axvline(1, color=S.INK2, lw=0.8)
        ax.set_xlabel("test MSE ratio after permuting the feature (1 = unused)")
        ax.set_title(f"{S.FEAT_LABELS[fs]} (BC seed 0)")
        ax.grid(axis="y", visible=False)
    S.suptitle(fig, f"{c.model} · C08 permutation feature importance of the FNN controller")
    fig.tight_layout(rect=(0, 0, 1, fig._top))
    c.save(fig, "C08_ctrl_perm_importance")


def c09_saliency(c):
    fig, axs = plt.subplots(1, 2, figsize=(9, 6.2), gridspec_kw=dict(width_ratios=[1, 1]))
    for ax, fs in zip(axs, ["raw", "phys"]):
        key = f"{fs}_dagger" if f"{fs}_dagger" in c.ctrl else f"{fs}_bc"
        net = c.ctrl[key][0]
        X, Y, _, _ = c.ctrl_xy(fs, ["test"], sub=20)
        Xt = torch.tensor(X, dtype=torch.float32, requires_grad=True)
        out = net(Xt)
        S_ = np.zeros((X.shape[1], 3))
        for i in range(3):
            g, = torch.autograd.grad(out[:, i].sum(), Xt, retain_graph=True)
            S_[:, i] = np.abs(g.numpy()).mean(0) * X.std(0)
        S_ = S_ / S_.max(0, keepdims=True)
        im = ax.imshow(S_, cmap=S.CMAP_SEQ, aspect="auto", vmin=0, vmax=1)
        ax.set_yticks(range(len(CF[fs])), CF[fs], fontsize=7)
        ax.set_xticks(range(3), ["Mu", "M1", "Mr"])
        ax.grid(False)
        ax.set_title(f"{S.CTRL_LABELS[key]}")
    fig.colorbar(im, ax=axs, fraction=0.03, pad=0.02, label="mean |∂M/∂f|·std(f), column-normalised")
    S.suptitle(fig, f"{c.model} · C09 gradient saliency of the final FNN controllers (test split)")
    fig.subplots_adjust(top=fig._top - 0.03, wspace=0.55, left=0.12, right=0.86)
    c.save(fig, "C09_ctrl_saliency")


def c10_robustness(c):
    p = os.path.join(c.md, "sweeps.json")
    if not os.path.exists(p):
        return
    sw = json.load(open(p))
    fig, axs = plt.subplots(2, 2, figsize=(11, 6.4))
    for col, kind, xl in [(0, "unc", "parametric uncertainty level (±, all uncertain params)"),
                          (1, "dist", "disturbance amplitude (× family scale)")]:
        lev = np.array(sw[f"{kind}|levels"])
        L = np.unique(lev)
        for k in ["expert", "raw_dagger", "phys_dagger"]:
            if f"{kind}|{k}" not in sw:
                continue
            m = sw[f"{kind}|{k}"]
            ise = np.clip(np.nan_to_num(np.array(m["ISE_T"], float), nan=1e12), 1e-10, 1e12)
            med = [np.median(ise[lev == l]) for l in L]
            q1 = [np.percentile(ise[lev == l], 25) for l in L]
            q3 = [np.percentile(ise[lev == l], 75) for l in L]
            xs = L * (100 if kind == "unc" else 1)
            axs[0, col].plot(xs, med, color=S.CTRL_COLORS[k], marker="o", ms=4, lw=1.4, label=S.CTRL_LABELS[k])
            axs[0, col].fill_between(xs, q1, q3, color=S.CTRL_COLORS[k], alpha=0.15, lw=0)
            axs[1, col].plot(xs, [np.mean(np.array(m["fail"])[lev == l]) for l in L], color=S.CTRL_COLORS[k],
                             marker="o", ms=4, lw=1.4, label=S.CTRL_LABELS[k])
        axs[0, col].set_yscale("log")
        axs[0, col].set_ylabel("ISE_T (median, IQR band)")
        axs[1, col].set_ylabel("failure rate")
        axs[1, col].set_ylim(-0.03, 1.03)
        axs[1, col].set_xlabel(xl + (" (%)" if kind == "unc" else ""))
        axs[0, col].set_title("uncertainty sweep" if kind == "unc" else "disturbance sweep")
    axs[0, 0].legend(fontsize=7)
    S.suptitle(fig, f"{c.model} · C10 closed-loop robustness sweeps (6 random references per level)")
    fig.tight_layout(rect=(0, 0, 1, fig._top))
    c.save(fig, "C10_cl_robustness")


def c11_ensemble(c):
    fig, axs = plt.subplots(1, 3, figsize=(13, 3.9))
    for fs in ["raw", "phys"]:
        nets = c.ctrl[f"{fs}_bc"]
        if len(nets) < 2:
            continue
        col = S.FEAT_COLORS[fs]
        Xtr, Ytr, _, _ = c.ctrl_xy(fs, ["train"], sub=20)
        sd = Ytr.std(0)
        for split, ls in [("test", "-"), ("ood", (0, (4, 2)))]:
            X, Y, _, _ = c.ctrl_xy(fs, [split], sub=4)
            P = np.stack([predict(n, X) for n in nets])
            std = (P.std(0) / sd).mean(1)
            err = (np.abs(P.mean(0) - Y) / sd).mean(1)
            q = np.quantile(std, np.linspace(0, 1, 11))
            idx = np.clip(np.digitize(std, q[1:-1]), 0, 9)
            axs[0].plot([std[idx == k].mean() for k in range(10)], [err[idx == k].mean() for k in range(10)],
                        color=col, ls=ls, marker="o", ms=3, lw=1.3, label=f"{S.FEAT_LABELS[fs]} · {split}")
            axs[1 if fs == "raw" else 2].hist(np.log10(std + 1e-9), bins=80, color=col if split == "test" else S.INK2,
                                              alpha=0.55, density=True, histtype="stepfilled", lw=0, label=split)
    axs[0].plot([0, axs[0].get_xlim()[1]], [0, axs[0].get_xlim()[1]], color=S.AXIS, lw=0.8)
    axs[0].set_xlabel("ensemble std (normalised), decile mean")
    axs[0].set_ylabel("|error| of ensemble mean (normalised)")
    axs[0].set_title("error vs epistemic spread (3-seed BC ensemble)")
    axs[0].legend(fontsize=6.8)
    for ax, fs in zip(axs[1:], ["raw", "phys"]):
        ax.set_xlabel("log10 ensemble std (normalised)")
        ax.set_title(f"{S.FEAT_LABELS[fs]}: spread on test vs OOD")
        ax.legend(fontsize=7)
    S.suptitle(fig, f"{c.model} · C11 ensemble disagreement as an out-of-distribution indicator")
    fig.tight_layout(rect=(0, 0, 1, fig._top))
    c.save(fig, "C11_ctrl_ensemble")


def c12_activations(c):
    key = "phys_dagger" if "phys_dagger" in c.ctrl else "phys_bc"
    net = c.ctrl[key][0]
    X, _, _, _ = c.ctrl_xy("phys", ["test"], sub=10)
    h = (torch.as_tensor(X, dtype=torch.float32) - net.x_mu) / net.x_sd
    lin = [m for m in net.mlp.net if isinstance(m, torch.nn.Linear)]
    fig, axs = plt.subplots(2, len(lin), figsize=(13, 5.4))
    with torch.no_grad():
        for i, L in enumerate(lin):
            z = L(h)
            axs[0, i].hist(z.numpy().ravel(), bins=100, color=S.BLUE, histtype="stepfilled", lw=0, alpha=0.8)
            axs[0, i].set_title(f"layer {i + 1} pre-activation" if i < len(lin) - 1 else "output (standardised)")
            axs[0, i].set_yscale("log")
            axs[1, i].hist(L.weight.numpy().ravel(), bins=60, color=S.ORANGE, histtype="stepfilled", lw=0, alpha=0.8)
            axs[1, i].set_title(f"layer {i + 1} weights (n={L.weight.numel()})")
            if i < len(lin) - 1:
                a = torch.nn.functional.silu(z)
                dead = (a.abs() < 1e-3).float().mean(0)
                axs[0, i].text(0.02, 0.95, f"units ~inactive on >90% of samples: {(dead > 0.9).sum().item()}/{dead.numel()}",
                               transform=axs[0, i].transAxes, va="top", fontsize=6.8, color=S.INK2)
                h = a
    S.suptitle(fig, f"{c.model} · C12 internal statistics of {S.CTRL_LABELS[key]} (SiLU MLP)")
    fig.tight_layout(rect=(0, 0, 1, fig._top))
    c.save(fig, "C12_ctrl_activations")


def c13_dagger(c):
    if c.met is None or "val" not in c.met:
        return
    fig, axs = plt.subplots(1, 2, figsize=(10, 3.6))
    for fs in ["raw", "phys"]:
        prog = c.met["val"].get(fs)
        if not prog:
            continue
        its = [p["it"] for p in prog]
        fail = [p["fail"] for p in prog]
        ise = [min(p["ISE_T"], 1e12) for p in prog]
        axs[0].plot(its, fail, color=S.FEAT_COLORS[fs], marker="o", ms=5, lw=1.4, label=S.FEAT_LABELS[fs])
        axs[1].plot(its, ise, color=S.FEAT_COLORS[fs], marker="o", ms=5, lw=1.4, label=S.FEAT_LABELS[fs])
        kb = c.kbest.get(fs)
        if kb is not None:
            axs[0].scatter([kb], [fail[its.index(kb)]], s=90, facecolors="none", edgecolors=S.INK, lw=1.0)
            axs[1].scatter([kb], [ise[its.index(kb)]], s=90, facecolors="none", edgecolors=S.INK, lw=1.0)
    ev = c.met["val"].get("expert")
    if ev:
        axs[0].axhline(np.mean(ev["fail"]), color=S.BLUE, lw=1.0, label="Expert")
        axs[1].axhline(np.median(ev["ISE_T"]), color=S.BLUE, lw=1.0, label="Expert")
    axs[0].set_ylabel("closed-loop failure rate (validation)")
    axs[0].set_ylim(-0.03, 1.03)
    axs[1].set_ylabel("median ISE_T (validation, N² s)")
    axs[1].set_yscale("log")
    for ax in axs:
        ax.set_xlabel("DAgger iteration (0 = behaviour cloning)")
        ax.legend(fontsize=7)
    S.suptitle(fig, f"{c.model} · C13 DAgger progress on closed-loop validation scenarios",
               "ring = iterate selected for the test evaluation; β (expert mixing) = 0.5, 0.25, 0.1, 0, 0")
    fig.tight_layout(rect=(0, 0, 1, fig._top))
    c.save(fig, "C13_ctrl_dagger")


def c14_cl_gallery(c):
    if "expert" not in c.cl:
        return
    e = c.cl["expert"]
    fam = np.array(e["family"])
    fams = [f for f in FAM_ORDER if f in set(fam)]
    n = len(fams)
    cols = 5
    rows = int(np.ceil(n / cols))
    fig, axs = plt.subplots(rows, cols, figsize=(14, 2.7 * rows), sharex=True, sharey=True, squeeze=False)
    t = e["t"]
    for ax, f in zip(axs.flat, fams):
        for key in ["expert", "phys_dagger", "raw_dagger"]:
            if key not in c.cl:
                continue
            for b in np.where(fam == f)[0]:
                err = np.abs(c.cl[key]["x"][b, :, 0] - c.cl[key]["ref"][b, :, 0])
                ax.plot(t, np.clip(err, 1e-6, 1e4), color=S.CTRL_COLORS[key], lw=0.7, alpha=0.8)
        ax.set_yscale("log")
        ax.set_title(f)
        ax.axhline(5, color=S.INK2, lw=0.6)
    for ax in axs.flat[n:]:
        ax.axis("off")
    for ax in axs[:, 0]:
        ax.set_ylabel("|e_T1| (N)")
    for ax in axs[-1]:
        ax.set_xlabel("time (s)")
    h = [plt.Line2D([], [], color=S.CTRL_COLORS[k], lw=1.4) for k in ["expert", "raw_dagger", "phys_dagger"]]
    fig.legend(h, [S.CTRL_LABELS[k] for k in ["expert", "raw_dagger", "phys_dagger"]], loc="lower center", ncol=3,
               bbox_to_anchor=(0.5, -0.02))
    S.suptitle(fig, f"{c.model} · C14 closed-loop tension-1 error over time, every held-out trajectory",
               "grey line = 5 N failure threshold; one thin line per trajectory")
    fig.tight_layout(rect=(0, 0.04, 1, fig._top))
    c.save(fig, "C14_cl_error_gallery")


# ============================================================================
#  dynamics-surrogate figures
# ============================================================================
TLAB = ["dT1/dt (N/s)", "dT2/dt (N/s)", "dωu/dt (rad/s²)", "dω1/dt (rad/s²)", "dωr/dt (rad/s²)"]


def s01_learning(c):
    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    for fs in ["raw", "phys"]:
        for s in range(3):
            p = os.path.join(c.md, f"dyn_{fs}_s{s}.json")
            if os.path.exists(p):
                h = json.load(open(p))["hist"]
                ax.plot(h["epoch"], h["val"], color=S.FEAT_COLORS[fs], lw=1.3 if s == 0 else 0.8, alpha=1 if s == 0 else 0.5,
                        label=f"{S.FEAT_LABELS[fs]} validation" if s == 0 else None)
                if s == 0:
                    ax.plot(h["epoch"], h["train"], color=S.FEAT_COLORS[fs], lw=1.0, ls=(0, (1, 1.5)),
                            label=f"{S.FEAT_LABELS[fs]} train")
    ax.set_yscale("log")
    ax.set_xlabel("epoch")
    ax.set_ylabel("MSE (standardised ẋ)")
    ax.legend(fontsize=7)
    S.suptitle(fig, f"{c.model} · S01 dynamics-surrogate learning curves (3 seeds)")
    fig.tight_layout(rect=(0, 0, 1, fig._top))
    c.save(fig, "S01_dyn_learning")


def s02_parity(c):
    fig, axs = plt.subplots(2, 5, figsize=(16, 6.4))
    for r, fs in enumerate(["raw", "phys"]):
        X, Y, _ = c.dyn_xy(fs, ["test"], sub=2)
        P = predict(c.dyn[fs][0], X)
        R = r2(Y, P)
        for j in range(5):
            ax = axs[r, j]
            lo, hi = np.percentile(Y[:, j], [0.5, 99.5])
            pad = 0.1 * (hi - lo)
            ax.hexbin(Y[:, j], P[:, j], gridsize=60, bins="log", cmap=S.CMAP_SEQ, mincnt=1, linewidths=0,
                      extent=(lo - pad, hi + pad, lo - pad, hi + pad))
            ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], color=S.INK2, lw=0.8)
            ax.set_xlim(lo - pad, hi + pad)
            ax.set_ylim(lo - pad, hi + pad)
            ax.set_title(f"{S.FEAT_LABELS[fs]} · {TLAB[j].split(' ')[0]}")
            ax.text(0.03, 0.95, f"R² = {R[j]:.3f}", transform=ax.transAxes, va="top", fontsize=7.5)
            ax.set_xlabel("true " + TLAB[j], fontsize=7)
            if j == 0:
                ax.set_ylabel("FNN prediction")
    S.suptitle(fig, f"{c.model} · S02 parity of the FNN dynamics surrogate on the test split",
               "targets are exact plant derivatives; disturbances/faults/uncertain parameters are not inputs")
    fig.tight_layout(rect=(0, 0, 1, fig._top))
    c.save(fig, "S02_dyn_parity")


def s03_family_r2(c):
    fams = [f for f in FAM_ORDER if f in c.fd]
    fig, axs = plt.subplots(1, 2, figsize=(11, 4.4))
    for ax, fs in zip(axs, ["raw", "phys"]):
        X, Y, G = c.dyn_xy(fs, ["test", "ood", "paper"], sub=2)
        P = predict(c.dyn[fs][0], X)
        M = np.array([r2(Y[G == f], P[G == f]) if (G == f).sum() > 10 else np.full(5, np.nan) for f in fams])
        im = ax.imshow(np.clip(M, 0, 1), cmap=S.CMAP_SEQ, vmin=0, vmax=1, aspect="auto")
        for i in range(M.shape[0]):
            for j in range(5):
                ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", fontsize=6.5,
                        color=S.INK if (np.nan_to_num(M[i, j]) < 0.6) else "#ffffff")
        ax.set_xticks(range(5), [l.split(" ")[0] for l in TLAB], fontsize=7)
        ax.set_yticks(range(len(fams)), fams)
        ax.grid(False)
        ax.set_title(S.FEAT_LABELS[fs])
    fig.colorbar(im, ax=axs, fraction=0.03, pad=0.02, label="R² (clipped to [0,1])")
    S.suptitle(fig, f"{c.model} · S03 surrogate accuracy per family × channel (held-out trajectories)")
    fig.subplots_adjust(top=fig._top - 0.04, wspace=0.3, right=0.86)
    c.save(fig, "S03_dyn_family_r2")


def s04_residual_vs_dist(c):
    fig, axs = plt.subplots(1, 3, figsize=(13, 3.9))
    fs = "phys"
    net = c.dyn[fs][0]
    d = c.fd["DIST"]
    m = np.isin(d["split"], ["test", "val"])
    X = dyn_features(d["x"][m], d["geo"][m], d["u"][m], c.R1, DF[fs]).reshape(-1, len(DF[fs]))
    Y = d["xdot"][m][..., 0:5].reshape(-1, 5)
    Dd = d["D"][m].reshape(-1, 5)
    Rz = Y - predict(net, X)
    ch = [(2, 2, "ωu"), (0, 0, "T1")] if c.model == "M2" else [(2, 2, "ωu"), (3, 3, "ω1")]
    for ax, (jr, jd, lab) in zip(axs[:2], ch):
        ax.hexbin(Dd[:, jd], Rz[:, jr], gridsize=60, bins="log", cmap=S.CMAP_SEQ, mincnt=1, linewidths=0)
        lo, hi = np.percentile(Dd[:, jd], [0.5, 99.5])
        ax.plot([lo, hi], [lo, hi], color=S.INK2, lw=0.8)
        cc = np.corrcoef(Dd[:, jd], Rz[:, jr])[0, 1]
        ax.set_title(f"DIST: residual on d{lab}/dt vs true D_{lab}")
        ax.text(0.03, 0.95, f"corr = {cc:.3f}", transform=ax.transAxes, va="top", fontsize=7.5)
        ax.set_xlabel(f"true disturbance D ({'N/s' if lab.startswith('T') else 'rad/s²'})")
        ax.set_ylabel("true − FNN")
    f = c.fd["FAULT"]
    m = np.isin(f["split"], ["test", "val"])
    X = dyn_features(f["x"][m], f["geo"][m], f["u"][m], c.R1, DF[fs]).reshape(-1, len(DF[fs]))
    Y = f["xdot"][m][..., 0:5].reshape(-1, 5)
    Rz = Y - predict(net, X)
    names = list(np.array(f["param_names"]).astype(str))
    J1 = f["plant_params"][m][:, names.index("J1")]
    fterm = (f["fy"][m][..., 1] * (-f["fault_a"][m][:, 1:2]) * f["u"][m][..., 1] / J1[:, None]).ravel()
    sel = np.abs(fterm) > 1e-6
    axs[2].hexbin(fterm[sel], Rz[sel, 3], gridsize=60, bins="log", cmap=S.CMAP_SEQ, mincnt=1, linewidths=0)
    if sel.any():
        lo, hi = np.percentile(fterm[sel], [0.5, 99.5])
        axs[2].plot([lo, hi], [lo, hi], color=S.INK2, lw=0.8)
    axs[2].set_title("FAULT: residual on dω1/dt vs fault term y1·g1·(−a1 M1)")
    axs[2].set_xlabel("true fault contribution (rad/s²)")
    axs[2].set_ylabel("true − FNN")
    S.suptitle(fig, f"{c.model} · S04 what the surrogate cannot see: residuals vs unmodelled inputs (FNN-phys)",
               "points on the diagonal = residual fully explained by the disturbance / fault a PINN could expose as a latent input")
    fig.tight_layout(rect=(0, 0, 1, fig._top))
    c.save(fig, "S04_dyn_residual_vs_dist")


def _rollout(net, fs, x0, geo, u, dt, R1, H):
    """Integrate the surrogate with RK4 using recorded inputs and radii (ZOH)."""
    x = x0.astype(np.float64).copy()
    out = [x.copy()]
    for k in range(H):
        def f(xx):
            F = dyn_features(xx[None], geo[k][None], u[k][None], R1, DF[fs])
            return predict(net, F)[0].astype(np.float64)
        k1 = f(x)
        k2 = f(x + 0.5 * dt * k1)
        k3 = f(x + 0.5 * dt * k2)
        k4 = f(x + dt * k3)
        x = x + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        if not np.isfinite(x).all() or np.abs(x).max() > 1e6:
            x[:] = np.nan
        out.append(x.copy())
    return np.array(out)


def _batched_rollout(net, fs, X0, GEO, U, dt, R1, H):
    """X0 (B,5), GEO (B,H,4), U (B,H,3) -> (B,H+1,5), batched over start points."""
    x = X0.astype(np.float64).copy()
    out = [x.copy()]
    for k in range(H):
        def f(xx):
            F = dyn_features(xx, GEO[:, k], U[:, k], R1, DF[fs])
            return predict(net, F).astype(np.float64)
        k1 = f(x)
        k2 = f(x + 0.5 * dt * k1)
        k3 = f(x + 0.5 * dt * k2)
        k4 = f(x + dt * k3)
        x = x + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        x[~np.isfinite(x).all(1) | (np.abs(x) > 1e6).any(1)] = np.nan
        out.append(x.copy())
    return np.stack(out, 1)


def s05_rollout(c):
    d = c.fd["REF"]
    b = int(np.where(d["split"] == "test")[0][0])
    dt = float(d["t"][1] - d["t"][0])
    H = 60 if c.model == "M1" else 150
    k0 = int(1.2 / dt)
    fig, axs = plt.subplots(1, 4, figsize=(14, 3.5))
    tt = d["t"][k0:k0 + H + 1]
    X0 = d["x"][b, k0, 0:5][None]
    GEO, U = d["geo"][b, k0:k0 + H][None], d["u"][b, k0:k0 + H][None]
    for j, (ch, lab) in enumerate([(0, "T1 (N)"), (1, "T2 (N)"), (2, "ωu (rad/s)"), (3, "ω1 (rad/s)")]):
        axs[j].plot(tt, d["x"][b, k0:k0 + H + 1, ch], color=S.INK2, lw=1.4, label="true plant")
    for fs in ["raw", "phys"]:
        R = _batched_rollout(c.dyn[fs][0], fs, X0, GEO, U, dt, c.R1, H)[0]
        for j, ch in enumerate([0, 1, 2, 3]):
            axs[j].plot(tt, R[:, ch], color=S.FEAT_COLORS[fs], lw=1.2, label=f"{S.FEAT_LABELS[fs]} rollout")
    for j, lab in enumerate(["T1 (N)", "T2 (N)", "ωu (rad/s)", "ω1 (rad/s)"]):
        truth = d["x"][b, k0:k0 + H + 1, [0, 1, 2, 3][j]]
        span = max(truth.max() - truth.min(), 1e-3 if j >= 2 else 0.5)
        axs[j].set_ylim(truth.min() - 1.5 * span, truth.max() + 1.5 * span)
        axs[j].set_title(lab)
        axs[j].set_xlabel("time (s)")
    axs[0].legend(fontsize=7)
    S.suptitle(fig, f"{c.model} · S05 open-loop multi-step rollout of the surrogate (recorded torques, RK4, {H} steps)",
               f"test trajectory {d['tags'][b]}, start t = {k0 * dt:.2f} s")
    fig.tight_layout(rect=(0, 0, 1, fig._top))
    c.save(fig, "S05_dyn_rollout")


def s06_rollout_growth(c):
    d = c.fd["REF"]
    dt = float(d["t"][1] - d["t"][0])
    H = 100 if c.model == "M1" else 250
    rng = np.random.default_rng(0)
    bs = np.where(np.isin(d["split"], ["test", "val"]))[0]
    starts = [(b, int(rng.integers(int(0.3 / dt), d["t"].size - H - 1))) for b in bs for _ in range(8)]
    X0 = np.array([d["x"][b, k, 0:5] for b, k in starts])
    GEO = np.array([d["geo"][b, k:k + H] for b, k in starts])
    U = np.array([d["u"][b, k:k + H] for b, k in starts])
    TR = np.array([d["x"][b, k:k + H + 1, 0:5] for b, k in starts])
    sd = d["x"][..., 0:5].reshape(-1, 5).std(0)
    fig, axs = plt.subplots(1, 2, figsize=(11, 3.8), sharey=True)
    steps = np.arange(H + 1)
    for fs in ["raw", "phys"]:
        R = _batched_rollout(c.dyn[fs][0], fs, X0, GEO, U, dt, c.R1, H)
        err = np.abs(R - TR) / sd
        err = np.nan_to_num(err, nan=1e3)
        for ax, chs, lab in [(axs[0], [0, 1], "tensions"), (axs[1], [2, 3, 4], "angular velocities")]:
            e = err[:, :, chs].mean(-1)
            med = np.median(e, 0)
            ax.plot(steps[1:] * dt * 1e3, med[1:], color=S.FEAT_COLORS[fs], lw=1.4, label=S.FEAT_LABELS[fs])
            ax.fill_between(steps[1:] * dt * 1e3, np.percentile(e, 25, 0)[1:], np.percentile(e, 75, 0)[1:],
                            color=S.FEAT_COLORS[fs], alpha=0.15, lw=0)
            ax.set_title(lab)
    for ax in axs:
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("prediction horizon (ms)")
    axs[0].set_ylabel("|error| / channel std (median, IQR)")
    axs[0].legend(fontsize=7)
    S.suptitle(fig, f"{c.model} · S06 compounding error of open-loop surrogate rollouts ({len(starts)} start points)")
    fig.tight_layout(rect=(0, 0, 1, fig._top))
    c.save(fig, "S06_dyn_rollout_growth")


def _true_f(d, sel_b, x5, geo_phi, u):
    """exact plant derivative (no disturbance / fault) for the trajectories' own parameters."""
    names = np.array(d["param_names"]).astype(str)
    out = []
    for b, xs, xp, us in zip(sel_b, x5, geo_phi, u):
        p = BatchParams([plant_params_obj(names, d["plant_params"][b])])
        xx = np.concatenate([xs, xp])[None].astype(np.float64)
        out.append(plant_xdot(xx, us[None].astype(np.float64), p)[0, 0:5])
    return np.array(out)


def s07_offmanifold(c):
    fams = ["REF", "UNC", "NOM"]
    rng = np.random.default_rng(3)
    pts = []
    for f in fams:
        d = c.fd[f]
        for b in np.where(np.isin(d["split"], ["test", "val"]))[0]:
            for k in rng.integers(int(0.3 / (d["t"][1] - d["t"][0])), d["t"].size, 40):
                pts.append((f, b, k))
    base_x = np.array([c.fd[f]["x"][b, k] for f, b, k in pts]).astype(np.float64)
    base_u = np.array([c.fd[f]["u"][b, k] for f, b, k in pts]).astype(np.float64)
    sd_x = np.concatenate([c.fd[f]["x"].reshape(-1, 8) for f in fams]).std(0)
    sd_u = np.concatenate([c.fd[f]["u"].reshape(-1, 3) for f in fams]).std(0)
    levels = [0.0, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0]
    fig, axs = plt.subplots(1, 2, figsize=(11, 3.8))
    for fs in ["raw", "phys"]:
        med_T, med_w, q = [], [], []
        for L in levels:
            x = base_x.copy()
            x[:, 0:5] += L * sd_x[0:5] * rng.standard_normal((len(pts), 5))
            u = base_u + L * sd_u * rng.standard_normal((len(pts), 3))
            ftrue = []
            for (f, b, k), xi, ui in zip(pts, x, u):
                d = c.fd[f]
                names = np.array(d["param_names"]).astype(str)
                p = BatchParams([plant_params_obj(names, d["plant_params"][b])])
                ftrue.append(plant_xdot(xi[None], ui[None], p)[0, 0:5])
            ftrue = np.array(ftrue)
            geo = np.stack(geometry(x, BatchParams([PARAMS[c.model]] * len(pts))), 1)
            # use the exact radii of each sample's own parameters
            geo = np.array([np.stack(geometry(xi[None], BatchParams([plant_params_obj(
                np.array(c.fd[f]["param_names"]).astype(str), c.fd[f]["plant_params"][b])])), 1)[0]
                for (f, b, k), xi in zip(pts, x)])
            P = predict(c.dyn[fs][0], dyn_features(x[:, 0:5], geo, u, c.R1, DF[fs]))
            sc = np.concatenate([c.fd[f]["xdot"][..., 0:5].reshape(-1, 5) for f in fams]).std(0)
            e = np.abs(P - ftrue) / sc
            med_T.append(np.median(e[:, 0:2]))
            med_w.append(np.median(e[:, 2:5]))
        axs[0].plot(levels, med_T, color=S.FEAT_COLORS[fs], marker="o", ms=4, lw=1.4, label=S.FEAT_LABELS[fs])
        axs[1].plot(levels, med_w, color=S.FEAT_COLORS[fs], marker="o", ms=4, lw=1.4, label=S.FEAT_LABELS[fs])
    for ax, lab in zip(axs, ["tension-rate channels", "roller-acceleration channels"]):
        ax.set_yscale("log")
        ax.set_xlabel("perturbation of (x, u) away from the data manifold (× data std)")
        ax.set_title(lab)
    axs[0].set_ylabel("median |f_FNN − f_true| / std(ẋ)")
    axs[0].legend(fontsize=7)
    S.suptitle(fig, f"{c.model} · S07 physics residual off the data manifold (collocation-style check)",
               "true f = analytic plant model with each trajectory's parameters; a PINN would penalise exactly this residual")
    fig.tight_layout(rect=(0, 0, 1, fig._top))
    c.save(fig, "S07_dyn_offmanifold")


def _jac_true(xfull, u, p):
    eps = np.array([1e-4, 1e-4, 1e-6, 1e-6, 1e-6])
    J = np.zeros((5, 5))
    for j in range(5):
        xp, xm = xfull.copy(), xfull.copy()
        xp[j] += eps[j]
        xm[j] -= eps[j]
        J[:, j] = (plant_xdot(xp[None], u[None], p)[0, 0:5] - plant_xdot(xm[None], u[None], p)[0, 0:5]) / (2 * eps[j])
    return J


def _jac_fnn(net, fs, x5, geo, u, R1):
    xs = torch.tensor(np.asarray(x5, dtype=np.float64), dtype=torch.float64, requires_grad=True)
    T1, T2, wu, w1, wr = xs[0], xs[1], xs[2], xs[3], xs[4]
    Ru, Rr = float(geo[0]), float(geo[1])
    cols = dict(T1=T1, T2=T2, wu=wu, w1=w1, wr=wr, Ru=torch.tensor(Ru, dtype=torch.float64),
                Rr=torch.tensor(Rr, dtype=torch.float64), Mu=torch.tensor(float(u[0]), dtype=torch.float64),
                M1=torch.tensor(float(u[1]), dtype=torch.float64), Mr=torch.tensor(float(u[2]), dtype=torch.float64),
                dv1=R1 * w1 - Ru * wu, dv2=Rr * wr - R1 * w1, RuT1=Ru * T1, R1dT=R1 * (T2 - T1), RrT2=Rr * T2,
                w1T1=w1 * T1, w1T2=w1 * T2)
    F = torch.stack([cols[k] for k in DF[fs]]).float()[None]
    out = net(F)[0]
    J = np.zeros((5, 5))
    for i in range(5):
        g, = torch.autograd.grad(out[i], xs, retain_graph=True)
        J[i] = g.numpy()
    return J


def s08_jacobian(c):
    d = c.fd["REF"]
    names = np.array(d["param_names"]).astype(str)
    rng = np.random.default_rng(5)
    bs = np.where(np.isin(d["split"], ["test", "val"]))[0]
    pts = [(b, int(k)) for b in bs for k in rng.integers(int(0.5 / (d["t"][1] - d["t"][0])), d["t"].size, 12)]
    ev = {"true": [], "raw": [], "phys": []}
    gains = {"true": [], "raw": [], "phys": []}
    for b, k in pts:
        p = BatchParams([plant_params_obj(names, d["plant_params"][b])])
        xf = d["x"][b, k].astype(np.float64)
        u = d["u"][b, k].astype(np.float64)
        Jt = _jac_true(xf, u, p)
        ev["true"].append(np.linalg.eigvals(Jt))
        gains["true"].append(Jt[0, 2])
        for fs in ["raw", "phys"]:
            Jf = _jac_fnn(c.dyn[fs][0], fs, xf[0:5], d["geo"][b, k], u, c.R1)
            ev[fs].append(np.linalg.eigvals(Jf))
            gains[fs].append(Jf[0, 2])
    fig, axs = plt.subplots(1, 3, figsize=(14, 4.2), gridspec_kw=dict(width_ratios=[1.2, 1.2, 1]))
    for ax, zoom in zip(axs[:2], [False, True]):
        for key, col, mk, lab in [("true", S.INK2, "o", "analytic plant"), ("raw", S.ORANGE, "x", "FNN-raw"),
                                  ("phys", S.AQUA, "+", "FNN-phys")]:
            E = np.concatenate(ev[key])
            if key == "true":
                ax.scatter(E.real, E.imag, s=26, marker="o", facecolors="none", edgecolors=col, lw=0.9, label=lab, zorder=4)
            else:
                ax.scatter(E.real, E.imag, s=18, color=col, marker=mk, lw=0.9, alpha=0.75, label=lab)
        if zoom:
            E = np.concatenate(ev["true"])
            slow = np.abs(E)[np.abs(E) < 0.2 * np.abs(E).max()]
            r = 3 * (slow.max() if slow.size else 1.0) + 20
            ax.set_xlim(-r, r)
            ax.set_ylim(-r, r)
            ax.set_title(f"zoom |λ| < {r:.0f} (slow modes)")
        else:
            ax.set_title("eigenvalues of ∂f/∂x (5 states)")
        ax.axvline(0, color=S.AXIS, lw=0.8)
        ax.set_xlabel("Re λ (1/s)")
        ax.set_ylabel("Im λ (1/s)")
    axs[0].legend(fontsize=7)
    gt = np.array(gains["true"])
    for fs in ["raw", "phys"]:
        axs[2].scatter(gt, gains[fs], s=12, color=S.FEAT_COLORS[fs], lw=0, alpha=0.8, label=S.FEAT_LABELS[fs])
    lo, hi = gt.min(), gt.max()
    axs[2].plot([lo, hi], [lo, hi], color=S.INK2, lw=0.8)
    axs[2].set_xlabel("true ∂(dT1/dt)/∂ωu = g_T1 (N/rad)")
    axs[2].set_ylabel("FNN ∂(dT1/dt)/∂ωu")
    axs[2].set_title("key coupling gain (web stiffness)")
    axs[2].ticklabel_format(axis="both", style="sci", scilimits=(0, 0))
    from matplotlib.ticker import MaxNLocator
    axs[2].xaxis.set_major_locator(MaxNLocator(4))
    axs[2].legend(fontsize=7)
    S.suptitle(fig, f"{c.model} · S08 local linearisation: surrogate Jacobian vs analytic plant ({len(pts)} operating points)")
    fig.tight_layout(rect=(0, 0, 1, fig._top))
    c.save(fig, "S08_dyn_jacobian")


def make_all(model, fams, md, outdir):
    c = Ctx(model, fams, md, outdir)
    fns = [c01_learning, c02_parity, c03_residuals, c04_family_error, c05_cl_paper, c06_cl_metrics, c07_cdf,
           c08_perm_importance, c09_saliency, c10_robustness, c11_ensemble, c12_activations, c13_dagger, c14_cl_gallery,
           s01_learning, s02_parity, s03_family_r2, s04_residual_vs_dist, s05_rollout, s06_rollout_growth,
           s07_offmanifold, s08_jacobian]
    for fn in fns:
        try:
            fn(c)
            print("  ok", fn.__name__, flush=True)
        except Exception as ex:  # keep going; report
            import traceback
            print("  FAILED", fn.__name__, repr(ex), flush=True)
            traceback.print_exc()
    return c
