"""plots_data.py -- dataset diagnostic figures (D01-D12)."""
from __future__ import annotations

import os

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import welch

from . import style as S
from .fnn import ctrl_features_traj, CTRL_RAW
from .plants import PARAMS, UNCERTAIN_KEYS, BatchParams, FG

FAM_ORDER = ["PAPER", "NOM", "REF", "UNC", "DIST", "NOISE", "FAULT", "EXC", "KICK", "COMBO", "OOD"]


def by_family(fams):
    d = {x["family"]: x for x in fams}
    return [d[f] for f in FAM_ORDER if f in d]


def traj_metrics(d):
    t = d["t"].astype(np.float64)
    dt = t[1] - t[0]
    e = d["x"][..., 0:2] - d["ref"][..., 0:2]
    ew = d["x"][..., 3] - d["ref"][..., 2]
    M = d["u"]
    return dict(ISE_T=(e ** 2).sum((1, 2)) * dt, ITAE_T=(t * np.abs(e).sum(-1)).sum(-1) * dt,
                ISE_w1=(ew ** 2).sum(-1) * dt, RMS_M=np.sqrt((M ** 2).mean(1)).sum(-1))


# ----------------------------------------------------------------------------
def fig_gallery(model, d, outdir):
    fam = d["family"]
    t = d["t"]
    n = min(3, d["x"].shape[0])
    idx = np.linspace(0, d["x"].shape[0] - 1, n).astype(int)
    fig, axs = plt.subplots(2, 3, figsize=(10.5, 5.2), sharex=True)
    specs = [("x", 0, "T1 (N)", "ref", 0), ("x", 1, "T2 (N)", "ref", 1), ("x", 3, "ω1 (rad/s)", "ref", 2),
             ("u", 0, "Mu (N m)", None, None), ("u", 1, "M1 (N m)", None, None), ("u", 2, "Mr (N m)", None, None)]
    for ax, (k, c, lab, rk, rc) in zip(axs.flat, specs):
        for j, b in enumerate(idx):
            col = S.SLOTS[j]
            ax.plot(t, d[k][b, :, c], color=col, lw=1.1, label=str(d["tags"][b])[:38])
            if rk is not None:
                ax.plot(t, d[rk][b, :, rc], color=col, lw=0.9, ls=(0, (4, 2)), alpha=0.9)
        ax.set_ylabel(lab)
        ax.set_title(lab.split(" ")[0])
    for ax in axs[1]:
        ax.set_xlabel("time (s)")
    h, l = axs[0, 0].get_legend_handles_labels()
    h.append(plt.Line2D([], [], color=S.REFC, ls=(0, (4, 2)), lw=0.9))
    l.append("reference (dashed, same colour)")
    fig.legend(h, l, loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.04))
    S.suptitle(fig, f"{model}-{fam}: example closed-loop trajectories",
               d["meta"]["description"][:140])
    fig.tight_layout(rect=(0, 0.04, 1, fig._top))
    S.save(fig, os.path.join(outdir, f"D01_gallery_{model}-{fam}"))


def fig_paper(model, d, outdir):
    t = d["t"]
    tags = [str(s) for s in d["tags"]]
    groups = {}
    for i, tg in enumerate(tags):
        groups.setdefault(tg.split(":")[0], []).append(i)
    rows = list(groups)
    fig, axs = plt.subplots(len(rows), 4, figsize=(12, 2.3 * len(rows)))
    axs = np.atleast_2d(axs)
    for r, g in enumerate(rows):
        for j, b in enumerate(groups[g]):
            col = S.SLOTS[j]
            lbl = tags[b].split(":", 1)[1].strip()
            for c, (k, ci, rci) in enumerate([("x", 0, 0), ("x", 1, 1), ("x", 3, 2)]):
                axs[r, c].plot(t, d[k][b, :, ci], color=col, lw=1.2, label=lbl if c == 0 else None)
                if j == 0:
                    axs[r, c].plot(t, d["ref"][b, :, rci], color=S.REFC, lw=0.9, ls=(0, (4, 2)),
                                   label="reference" if c == 0 else None)
            if j == 0:
                for ci, colr in zip(range(3), S.SLOTS[:3]):
                    axs[r, 3].plot(t, d["u"][b, :, ci], color=colr, lw=1.0, label=["Mu", "M1", "Mr"][ci])
        xl = (0, 0.2) if (model == "M2" and g == "S1") else (0, t[-1])
        for c, lab in enumerate(["T1 (N)", "T2 (N)", "ω1 (rad/s)", "torque (N m)"]):
            axs[r, c].set_xlim(*xl)
            axs[r, c].set_ylabel(lab)
            axs[r, c].set_title(f"{g} · {lab.split(' ')[0]}")
        axs[r, 0].legend(loc="best", fontsize=6.5)
        axs[r, 3].legend(loc="best", fontsize=6.5, ncol=3, framealpha=0.85, frameon=True, edgecolor="none")
    for ax in axs[-1]:
        ax.set_xlabel("time (s)")
    S.suptitle(fig, f"{model}-PAPER: replicas of the paper's simulation scenarios",
               "Expert = the paper's controller named in each legend" + ("; S1 zoomed to 0.2 s as in the paper" if model == "M2" else ""))
    fig.tight_layout(rect=(0, 0, 1, fig._top))
    S.save(fig, os.path.join(outdir, f"D02_paper_replicas_{model}"))


def fig_distributions(model, fams, outdir):
    chans = [("x", 0, "T1 (N)"), ("x", 1, "T2 (N)"), ("x", 2, "ωu (rad/s)"), ("x", 3, "ω1 (rad/s)"),
             ("x", 4, "ωr (rad/s)"), ("u", 0, "Mu (N m)"), ("u", 1, "M1 (N m)"), ("u", 2, "Mr (N m)")]
    fig, axs = plt.subplots(2, 4, figsize=(13, 5.6))
    names = [d["family"] for d in fams]
    for ax, (k, c, lab) in zip(axs.flat, chans):
        data = [d[k][:, ::5, c].ravel() for d in fams]
        bp = ax.boxplot(data, whis=(1, 99), showfliers=False, widths=0.6, patch_artist=True,
                        medianprops=dict(color=S.INK, lw=1.0), boxprops=dict(facecolor=S.SEQ[2], edgecolor=S.BLUE, lw=0.8),
                        whiskerprops=dict(color=S.BLUE, lw=0.8), capprops=dict(color=S.BLUE, lw=0.8))
        ax.set_xticks(range(1, len(names) + 1), names, rotation=60, fontsize=7)
        ax.set_title(lab)
        ax.grid(axis="x", visible=False)
    S.suptitle(fig, f"{model}: per-family distribution of states and torques",
               "box = IQR, whiskers = 1st-99th percentile, line = median (every 5th sample)")
    fig.tight_layout(rect=(0, 0, 1, fig._top))
    S.save(fig, os.path.join(outdir, f"D03_distributions_{model}"))


def _stack(fams, key, splits, ch, sub=4):
    out = []
    for d in fams:
        m = np.isin(d["split"], splits)
        if m.any():
            out.append(d[key][m][:, ::sub, ch].reshape(-1))
    return np.concatenate(out) if out else np.zeros(0)


def fig_coverage(model, fams, outdir):
    pairs = [(("x", 3, "ω1 (rad/s)"), ("x", 0, "T1 (N)")), (("x", 0, "T1 (N)"), ("x", 1, "T2 (N)")),
             (("x", 2, "ωu (rad/s)"), ("u", 0, "Mu (N m)")), (("x", 3, "ω1 (rad/s)"), ("u", 1, "M1 (N m)"))]
    fig, axs = plt.subplots(1, 4, figsize=(14, 3.6))
    for ax, (a, b) in zip(axs, pairs):
        xa = _stack(fams, a[0], ["train", "val"], a[1])
        ya = _stack(fams, b[0], ["train", "val"], b[1])
        hb = ax.hexbin(xa, ya, gridsize=55, bins="log", cmap=S.CMAP_SEQ, mincnt=1, linewidths=0)
        xo = _stack(fams, a[0], ["ood"], a[1], sub=40)
        yo = _stack(fams, b[0], ["ood"], b[1], sub=40)
        ax.scatter(xo, yo, s=2, color=S.ORANGE, alpha=0.5, lw=0, label="OOD (held out)")
        xp = _stack(fams, a[0], ["paper"], a[1], sub=40)
        yp = _stack(fams, b[0], ["paper"], b[1], sub=40)
        ax.scatter(xp, yp, s=2, color=S.AQUA, alpha=0.6, lw=0, label="PAPER replicas")
        ax.set_xlabel(a[2])
        ax.set_ylabel(b[2])
        ax.set_title(f"{b[2].split(' ')[0]} vs {a[2].split(' ')[0]}")
        # robust limits: train 0.2-99.8 % envelope united with the OOD cloud
        xs_ = np.concatenate([np.percentile(xa, [0.2, 99.8]), np.percentile(xo, [1, 99]) if xo.size else []])
        ys_ = np.concatenate([np.percentile(ya, [0.2, 99.8]), np.percentile(yo, [1, 99]) if yo.size else []])
        px, py = 0.08 * (xs_.max() - xs_.min()), 0.08 * (ys_.max() - ys_.min())
        ax.set_xlim(xs_.min() - px, xs_.max() + px)
        ax.set_ylim(ys_.min() - py, ys_.max() + py)
    cb = fig.colorbar(hb, ax=axs, fraction=0.015, pad=0.01)
    cb.set_label("train+val samples (log count)")
    axs[0].legend(loc="upper left", markerscale=5)
    S.suptitle(fig, f"{model}: state-action coverage (train density vs held-out sets)")
    fig.subplots_adjust(top=fig._top - 0.04, wspace=0.32)
    S.save(fig, os.path.join(outdir, f"D04_coverage_{model}"))


def fig_pca(model, fams, outdir):
    R1 = PARAMS[model].R1
    Xs, labs = [], []
    for d in fams:
        dt = float(d["t"][1] - d["t"][0])
        F = ctrl_features_traj(d["y"], d["geo"], d["ref"], d["dref"], dt, R1, CTRL_RAW)[:, ::20]
        for sp in np.unique(d["split"]):
            m = d["split"] == sp
            Xs.append(F[m].reshape(-1, F.shape[-1]))
            labs += [sp] * Xs[-1].shape[0]
    X = np.concatenate(Xs)
    labs = np.array(labs)
    tr = np.isin(labs, ["train", "val"])
    mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-9
    Z = (X - mu) / sd
    U, s, Vt = np.linalg.svd(Z[tr][::3], full_matrices=False)
    P = Z @ Vt[:3].T
    ev = s ** 2 / (s ** 2).sum()
    fig, axs = plt.subplots(1, 3, figsize=(13, 3.9), gridspec_kw=dict(width_ratios=[1, 1, 0.8]))
    for ax, (i, j) in zip(axs[:2], [(0, 1), (1, 2)]):
        ax.hexbin(P[tr, i], P[tr, j], gridsize=60, bins="log", cmap=S.CMAP_SEQ, mincnt=1, linewidths=0)
        for sp, col, lb in [("test", S.INK2, "test (in-distribution)"), ("ood", S.ORANGE, "OOD"), ("paper", S.AQUA, "PAPER")]:
            m = labs == sp
            ax.scatter(P[m, i], P[m, j], s=2, color=col, alpha=0.5, lw=0, label=lb)
        ax.set_xlabel(f"PC{i + 1} ({ev[i] * 100:.0f}%)")
        ax.set_ylabel(f"PC{j + 1} ({ev[j] * 100:.0f}%)")
        ax.set_title(f"PC{i + 1}-PC{j + 1} (train density in blue)")
    axs[0].legend(loc="best", markerscale=5)
    k = np.arange(1, len(ev) + 1)
    axs[2].bar(k, ev * 100, color=S.BLUE, width=0.7)
    axs[2].plot(k, np.cumsum(ev) * 100, color=S.INK2, lw=1.0, marker="o", ms=3)
    axs[2].set_xlabel("principal component")
    axs[2].set_ylabel("explained variance (%)")
    axs[2].set_title("scree (bars) and cumulative (line)")
    S.suptitle(fig, f"{model}: PCA of the controller feature space ({len(CTRL_RAW)} raw features)")
    fig.tight_layout(rect=(0, 0, 1, fig._top))
    S.save(fig, os.path.join(outdir, f"D05_pca_{model}"))


def fig_spectra(model, fams, outdir):
    chans = [("x", 0, "T1"), ("x", 3, "ω1"), ("u", 0, "Mu"), ("u", 1, "M1")]
    fig, axs = plt.subplots(1, 4, figsize=(14, 3.6), sharey=True)
    for ax, (k, c, lab) in zip(axs, chans):
        rows, f = [], None
        for d in fams:
            dt = float(d["t"][1] - d["t"][0])
            m = d["t"] > 0.5
            sig = d[k][:, m, c].astype(np.float64)
            sig = sig - sig.mean(1, keepdims=True)
            f, P = welch(sig, fs=1.0 / dt, nperseg=512, axis=-1)
            rows.append(np.log10(P.mean(0) + 1e-20))
        A = np.array(rows)
        im = ax.pcolormesh(f[1:], np.arange(len(fams)), A[:, 1:], cmap=S.CMAP_SEQ, shading="nearest",
                           vmin=np.percentile(A, 5), vmax=np.percentile(A, 99.5))
        ax.set_xscale("log")
        ax.set_yticks(range(len(fams)), [d["family"] for d in fams])
        ax.invert_yaxis()
        ax.set_xlabel("frequency (Hz)")
        ax.set_title(f"{lab}: log10 PSD")
        ax.grid(False)
        cb = fig.colorbar(im, ax=ax, fraction=0.05, pad=0.02)
        cb.ax.tick_params(labelsize=6.5)
    S.suptitle(fig, f"{model}: spectral content per family (Welch, t > 0.5 s, family mean)")
    fig.tight_layout(rect=(0, 0, 1, fig._top))
    S.save(fig, os.path.join(outdir, f"D06_spectra_{model}"))


def fig_correlation(model, fams, outdir):
    R1 = PARAMS[model].R1
    Xs, Ys = [], []
    for d in fams:
        m = d["split"] == "train"
        if not m.any():
            continue
        dt = float(d["t"][1] - d["t"][0])
        F = ctrl_features_traj(d["y"][m], d["geo"][m], d["ref"][m], d["dref"][m], dt, R1, CTRL_RAW)[:, ::10]
        Xs.append(F.reshape(-1, F.shape[-1]))
        Ys.append(d["u_cmd"][m][:, ::10].reshape(-1, 3))
    Z = np.concatenate([np.concatenate(Xs), np.concatenate(Ys)], 1)
    names = CTRL_RAW + ["Mu*", "M1*", "Mr*"]
    C = np.corrcoef(Z.T)
    fig, ax = plt.subplots(figsize=(7.4, 6.4))
    im = ax.imshow(C, cmap=S.CMAP_DIV, vmin=-1, vmax=1)
    ax.set_xticks(range(len(names)), names, rotation=90, fontsize=7)
    ax.set_yticks(range(len(names)), names, fontsize=7)
    ax.grid(False)
    ax.axhline(len(CTRL_RAW) - 0.5, color=S.INK, lw=0.6)
    ax.axvline(len(CTRL_RAW) - 0.5, color=S.INK, lw=0.6)
    fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02, label="Pearson r")
    S.suptitle(fig, f"{model}: feature / target correlation (train split)",
               "targets (*) = expert torque command; lines separate features from targets")
    fig.tight_layout(rect=(0, 0, 1, fig._top))
    S.save(fig, os.path.join(outdir, f"D07_correlation_{model}"))


def fig_expert_perf(model, fams, outdir):
    keys = [("ISE_T", "ISE_T (N² s)", True), ("ITAE_T", "ITAE_T (N s²)", True),
            ("ISE_w1", "ISE_ω1 (rad² s⁻¹)", True), ("RMS_M", "RMS_M (N m)", False)]
    mets = [traj_metrics(d) for d in fams]
    fig, axs = plt.subplots(1, 4, figsize=(14, 3.6))
    for ax, (k, lab, lg) in zip(axs, keys):
        for i, (d, m) in enumerate(zip(fams, mets)):
            v = m[k]
            jit = np.random.default_rng(i).uniform(-0.18, 0.18, v.size)
            ax.scatter(np.full(v.size, i) + jit, v, s=9, color=S.BLUE, alpha=0.7, lw=0)
            ax.plot([i - 0.3, i + 0.3], [np.median(v)] * 2, color=S.INK, lw=1.2)
        if lg:
            ax.set_yscale("log")
        ax.set_xticks(range(len(fams)), [d["family"] for d in fams], rotation=60, fontsize=7)
        ax.set_title(lab)
        ax.grid(axis="x", visible=False)
    S.suptitle(fig, f"{model}: expert (paper controller) performance per family",
               "dots = trajectories, bar = median; indices as in M1 eqs. (29)-(33) over 0-4 s")
    fig.tight_layout(rect=(0, 0, 1, fig._top))
    S.save(fig, os.path.join(outdir, f"D08_expert_performance_{model}"))


def fig_exogenous(model, fd, outdir):
    fig, axs = plt.subplots(2, 3, figsize=(13, 5.6))
    t = fd["REF"]["t"]
    # (a) disturbances on tension channels (M2) / roller channels
    d = fd["DIST"]
    for j, b in enumerate(range(min(3, d["D"].shape[0]))):
        ch = 0 if model == "M2" else 2
        axs[0, 0].plot(t, d["D"][b, :, ch], color=S.SLOTS[j], lw=1.0, label=f"traj {b}")
    axs[0, 0].set_title("D_T1 (N/s)" if model == "M2" else "D_u (rad/s²)")
    axs[0, 0].legend(fontsize=6.5)
    for j, ch in enumerate([2, 3, 4]):
        axs[0, 1].plot(t, d["D"][3, :, ch], color=S.SLOTS[j], lw=1.0, label=["D_u", "D_1", "D_r"][j])
    axs[0, 1].set_title("matched disturbances, one trajectory (rad/s²)")
    axs[0, 1].legend(fontsize=6.5)
    # (c) faults: effective loss a_i y_i(t)
    f = fd["FAULT"]
    for j, b in enumerate(range(min(6, f["fy"].shape[0]))):
        loss = (f["fy"][b] * f["fault_a"][b][None, :]).max(1)
        axs[0, 2].plot(t, 100 * loss, color=S.SLOTS[j], lw=1.1, label=f"traj {b}")
    axs[0, 2].set_title("actuator loss of effectiveness max_i a_i y_i(t) (%)")
    axs[0, 2].legend(fontsize=6.5, ncol=2)
    # (d) measurement noise
    n = fd["NOISE"]
    axs[1, 0].plot(t, n["y"][0, :, 0] - n["x"][0, :, 0], color=S.BLUE, lw=0.6, label="T1 meas − true (N)")
    axs[1, 0].set_title("measurement error on T1 (N)")
    axs[1, 1].plot(t, n["y"][0, :, 3] - n["x"][0, :, 3], color=S.ORANGE, lw=0.6)
    axs[1, 1].set_title("measurement error on ω1 (rad/s)")
    # (f) dither
    e = fd["EXC"]
    for j in range(3):
        axs[1, 2].plot(t[:600], (e["u"][0, :600, j] - e["u_cmd"][0, :600, j]) + 0 * j, color=S.SLOTS[j], lw=0.9,
                       label=["Mu", "M1", "Mr"][j])
    axs[1, 2].set_title("PRBS torque dither u − u_cmd (N m), first 0.6 s")
    axs[1, 2].legend(fontsize=6.5)
    for ax in axs[1]:
        ax.set_xlabel("time (s)")
    S.suptitle(fig, f"{model}: exogenous signals used to diversify the datasets")
    fig.tight_layout(rect=(0, 0, 1, fig._top))
    S.save(fig, os.path.join(outdir, f"D09_exogenous_{model}"))


def fig_params(model, fd, outdir):
    names = list(np.array(fd["REF"]["param_names"]).astype(str))
    nom = fd["NOM"]["plant_params"][0]
    keys = list(UNCERTAIN_KEYS[model])
    fig, ax = plt.subplots(figsize=(9, 3.6))
    for j, (fam, col) in enumerate([("UNC", S.BLUE), ("FAULT", S.ORANGE), ("OOD", S.AQUA)]):
        P = fd[fam]["plant_params"]
        for i, k in enumerate(keys):
            c = names.index(k)
            rel = 100 * (P[:, c] / nom[c] - 1)
            x = i + (j - 1) * 0.22 + np.random.default_rng(i + 10 * j).uniform(-0.06, 0.06, rel.size)
            ax.scatter(x, rel, s=10, color=col, lw=0, alpha=0.85, label=fam if i == 0 else None)
    ax.axhline(0, color=S.AXIS, lw=0.8)
    ax.set_xticks(range(len(keys)), keys)
    ax.set_ylabel("deviation from nominal (%)")
    ax.legend(ncol=3, loc="upper right")
    ax.grid(axis="x", visible=False)
    S.suptitle(fig, f"{model}: sampled parametric uncertainty (paper's uncertain set)")
    fig.tight_layout(rect=(0, 0, 1, fig._top))
    S.save(fig, os.path.join(outdir, f"D10_param_coverage_{model}"))


def fig_observer(model, fd, outdir):
    d = fd["PAPER"]
    t = d["t"]
    tags = [str(s) for s in d["tags"]]
    fig, axs = plt.subplots(2, 3, figsize=(13, 5.4))
    if model == "M2":
        picks = [i for i, tg in enumerate(tags) if tg.startswith("S3")] + [i for i, tg in enumerate(tags) if tg.startswith("S4")]
        for r, b in enumerate(picks[:2]):
            names = np.array(d["param_names"]).astype(str)
            P = d["plant_params"][b]
            geo = d["geo"][b]
            M = d["u"][b]
            Ju, Jr = geo[:, 2], geo[:, 3]
            J1 = P[list(names).index("J1")]
            g = np.stack([-1 / Ju, np.full_like(Ju, 1 / J1), 1 / Jr], 1)
            Ktrue = d["D"][b].copy()
            Ktrue[:, 2:5] += d["fy"][b] * g * (-d["fault_a"][b][None, :] * M)
            for c, (ch, lab) in enumerate([(0, "K_T1 (N/s)"), (2, "K_ωu (rad/s²)"), (3, "K_ω1 (rad/s²)")]):
                axs[r, c].plot(t, Ktrue[:, ch], color=S.INK2, lw=1.0, ls=(0, (4, 2)), label="true D + YC")
                axs[r, c].plot(t, d["est"][b, :, ch], color=S.BLUE, lw=1.0, label="observer estimate")
                axs[r, c].set_title(f"{tags[b].split(':')[0]} · {lab}")
        sub = "fixed-time sliding-mode observer (M2 eqs. 17-22) vs the true lumped disturbance + fault term"
    else:
        picks = [i for i, tg in enumerate(tags) if "ESO" in tg]
        pn = PARAMS[model]
        for r, b in enumerate(picks[:2]):
            names = np.array(d["param_names"]).astype(str)
            from .plants import R2RParams
            p = BatchParams([R2RParams(**{k: float(v) for k, v in zip(names, d["plant_params"][b])}, name="x")])
            F, _ = FG(d["x"][b].astype(np.float64), p)
            xi = F[:, 2:5] + d["D"][b][:, 2:5]
            for c, lab in enumerate(["ξ_u", "ξ_1", "ξ_r"]):
                axs[r, c].plot(t, xi[:, c], color=S.INK2, lw=1.0, ls=(0, (4, 2)), label="true F_ω + D_ω")
                axs[r, c].plot(t, d["est"][b, :, 2 + c], color=S.BLUE, lw=1.0, label="ESO estimate")
                axs[r, c].set_title(f"{tags[b].split(':')[0]} · {lab} (rad/s²)")
        sub = "finite-time ESO (M1 eq. 15, retuned bandwidth) vs the true lumped term ξ = F_ω + D_ω"
    axs[0, 0].legend(fontsize=7)
    for ax in axs[1]:
        ax.set_xlabel("time (s)")
    S.suptitle(fig, f"{model}: expert observer diagnostics", sub)
    fig.tight_layout(rect=(0, 0, 1, fig._top))
    S.save(fig, os.path.join(outdir, f"D11_observer_{model}"))


def table_summary(model, fams, outdir):
    rows = []
    for d in fams:
        sp = d["split"]
        m = traj_metrics(d)
        rows.append([d["family"], d["x"].shape[0], d["x"].shape[0] * d["x"].shape[1], f"{d['meta']['dt_store'] * 1e3:.0f} ms",
                     int((sp == "train").sum()), int((sp == "val").sum()), int((sp == "test").sum()),
                     int(np.isin(sp, ["ood", "paper"]).sum()), f"{np.median(m['ISE_T']):.2e}",
                     d["meta"]["description"]])
    hdr = ["family", "traj", "samples", "Δt", "train", "val", "test", "held-out", "expert ISE_T (med.)", "description"]
    with open(os.path.join(outdir, f"D12_summary_{model}.csv"), "w") as f:
        f.write(",".join(hdr) + "\n")
        for r in rows:
            f.write(",".join(str(x).replace(",", ";") for x in r) + "\n")
    fig, ax = plt.subplots(figsize=(14, 0.45 * len(rows) + 1.2))
    ax.axis("off")
    cell = [[str(x) if i < 9 else str(x)[:78] for i, x in enumerate(r)] for r in rows]
    tb = ax.table(cellText=cell, colLabels=hdr, loc="center", cellLoc="left", colLoc="left")
    tb.auto_set_font_size(False)
    tb.set_fontsize(7)
    tb.auto_set_column_width(list(range(len(hdr))))
    for (i, j), c in tb.get_celld().items():
        c.set_edgecolor(S.GRID)
        c.set_linewidth(0.5)
        if i == 0:
            c.set_text_props(weight="bold", color=S.INK)
    S.suptitle(fig, f"{model}: dataset summary")
    S.save(fig, os.path.join(outdir, f"D12_summary_{model}"))


def make_all(model, fams, outdir):
    os.makedirs(outdir, exist_ok=True)
    fams = by_family(fams)
    fd = {d["family"]: d for d in fams}
    for d in fams:
        if d["family"] == "PAPER":
            fig_paper(model, d, outdir)
        else:
            fig_gallery(model, d, outdir)
    fig_distributions(model, fams, outdir)
    fig_coverage(model, fams, outdir)
    fig_pca(model, fams, outdir)
    fig_spectra(model, fams, outdir)
    fig_correlation(model, fams, outdir)
    fig_expert_perf(model, fams, outdir)
    fig_exogenous(model, fd, outdir)
    fig_params(model, fd, outdir)
    fig_observer(model, fd, outdir)
    table_summary(model, fams, outdir)
