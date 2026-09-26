"""
bench/report.py -- leaderboards and figures for the architecture benchmark.

Robust metrics are recomputed from the saved predictions (normalised units):
  * errors are NaN->20 and clipped to |e| <= 20 (a diverged window counts as "20 std everywhere"
    instead of dominating the average with 1e3), divergence = any |e| > 20 or non-finite;
  * NRMSE over the first 64 / all 256 predicted steps, for
      ID    closed-loop replay, in-distribution test families (test split)
      OOD   closed-loop replay, held-out extrapolation family
      PAPER the paper scenarios
      CF    counterfactual inputs (re-simulated plant, perturbed future torques), ID and OOD windows
  * mean rank over the six headline metrics (ID/OOD/CF x 64/256).

usage:  python -m r2r_nn.bench.report --root out/bench_final
"""
from __future__ import annotations

import argparse
import glob
import json
import os

import numpy as np

from .. import style as S

ORDER = ["DMDc", "EDMDc-RFF", "SINDYc", "ESN", "MLP-AR", "NeuralODE", "RNN", "LSTM", "GRU", "LRU-SSM",
         "MLP-Direct", "TCN", "Transformer", "DeepKoopman", "PIKO"]
ABL = ["PIKO", "PIKO-bidir", "PIKO-noObs", "PIKO-core", "PIKO-core-noObs", "PIKO-core-dense", "PIKO-core-polyLPV",
       "PIKO-core-Euler", "PIKO-core-noQuad", "PIKO-core-linear"]
CLASS = {"DMDc": "linear / Koopman", "EDMDc-RFF": "linear / Koopman", "DeepKoopman": "linear / Koopman",
         "SINDYc": "sparse physics", "ESN": "reservoir", "MLP-AR": "feed-forward AR", "NeuralODE": "feed-forward AR",
         "RNN": "recurrent", "LSTM": "recurrent", "GRU": "recurrent", "LRU-SSM": "state-space",
         "MLP-Direct": "sequence-to-sequence", "TCN": "sequence-to-sequence", "Transformer": "sequence-to-sequence",
         "PIKO": "PIKO (ours)"}
CLIP = 20.0
HEAD = [("ID", 64), ("ID", 256), ("OOD", 64), ("OOD", 256), ("CF", 64), ("CF", 256)]


def errs(pred, tgt):
    e = np.nan_to_num(pred.astype(np.float64) - tgt, nan=CLIP, posinf=CLIP, neginf=-CLIP)
    div = (np.abs(e) >= CLIP).any(axis=(1, 2)) | ~np.isfinite(pred).all(axis=(1, 2))
    return np.clip(e, -CLIP, CLIP), div


def nrmse(e, m, h):
    return float(np.sqrt((e[m, :h] ** 2).mean())) if m.any() else float("nan")


def load_plant(root, model):
    od = os.path.join(root, model)
    T = np.load(os.path.join(od, "test_targets.npz"))
    s, fam = T["s"][:, 32:], T["fam"].astype(str)
    C = np.load(os.path.join(od, "cf_targets.npz")) if os.path.exists(os.path.join(od, "cf_targets.npz")) else None
    out = {}
    for fn in sorted(glob.glob(os.path.join(od, "res_*.json"))):
        r = json.load(open(fn))
        name = r["name"]
        p = np.load(os.path.join(od, f"pred_{name}.npz"))["pred"]
        e, div = errs(p, s)
        idm, oodm, pam = ~np.isin(fam, ["OOD", "PAPER"]), fam == "OOD", fam == "PAPER"
        row = dict(name=name, params=r["params"], train_time=r["train_time"], latency=r["latency_ms_per_window"],
                   div=float(div.mean()), div_id=float(div[idm].mean()), e=e, fam=fam)
        for tag, m in (("ID", idm), ("OOD", oodm), ("PAPER", pam)):
            for h in (64, 256):
                row[f"{tag}{h}"] = nrmse(e, m, h)
        row["curve_ID"] = np.sqrt((e[idm] ** 2).mean(axis=(0, 2)))
        row["curve_OOD"] = np.sqrt((e[oodm] ** 2).mean(axis=(0, 2)))
        row["fam64"] = {f: nrmse(e, fam == f, 64) for f in sorted(set(fam))}
        row["fam256"] = {f: nrmse(e, fam == f, 256) for f in sorted(set(fam))}
        row["ch_ID64"] = np.sqrt((e[idm, :64] ** 2).mean(axis=(0, 1)))
        if C is not None:
            sc, fc = C["s"][:, 32:], C["fam"].astype(str)
            key = f"predcf_off_{name}.npz" if os.path.exists(os.path.join(od, f"predcf_off_{name}.npz")) else f"predcf_{name}.npz"
            if os.path.exists(os.path.join(od, key)):
                ec, dc = errs(np.load(os.path.join(od, key))["pred"], sc)
                cid, cood = fc != "OOD", fc == "OOD"
                row["CF64"], row["CF256"] = nrmse(ec, cid, 64), nrmse(ec, cid, 256)
                row["CFOOD64"], row["CFOOD256"] = nrmse(ec, cood, 64), nrmse(ec, cood, 256)
                row["div_cf"] = float(dc.mean())
                row["curve_CF"] = np.sqrt((ec[cid] ** 2).mean(axis=(0, 2)))
                row["cf_fam64"] = {f: nrmse(ec, fc == f, 64) for f in sorted(set(fc))}
                row["ecf"], row["fcf"] = ec, fc
                row["cf_mode"] = "innovation off" if "off" in key else "as trained"
                if "off" in key:
                    e2, _ = errs(np.load(os.path.join(od, f"predcf_{name}.npz"))["pred"], sc)
                    row["CF64_on"], row["CF256_on"] = nrmse(e2, cid, 64), nrmse(e2, cid, 256)
        out[name] = row
    return out, s, fam, C


def ranks(rows, names):
    R = {n: [] for n in names}
    for tag, h in HEAD:
        k = f"{tag}{h}"
        vals = [(rows[n].get(k, np.nan), n) for n in names]
        vals = sorted(vals, key=lambda v: (np.inf if not np.isfinite(v[0]) else v[0]))
        for i, (_, n) in enumerate(vals):
            R[n].append(i + 1)
    return {n: float(np.mean(v)) for n, v in R.items()}


def fmt(v, best=False):
    if v is None or not np.isfinite(v):
        return "–"
    s = f"{v:.3f}" if v < 10 else f"{v:.1f}"
    return f"**{s}**" if best else s


def leaderboard_md(rows, model, names):
    names = [n for n in names if n in rows]
    rk = ranks(rows, names)
    names = sorted(names, key=lambda n: rk[n])
    cols = [f"{t}{h}" for t, h in HEAD]
    best = {c: min(rows[n].get(c, np.inf) for n in names) for c in cols}
    hdr = ("| # | model | class | ID@64 | ID@256 | OOD@64 | OOD@256 | CF@64 | CF@256 | diverged (ID/all) | params | train s | mean rank |\n"
           "|---|---|---|---|---|---|---|---|---|---|---|---|---|\n")
    lines = []
    for i, n in enumerate(names):
        r = rows[n]
        cells = [fmt(r.get(c, np.nan), abs(r.get(c, np.inf) - best[c]) < 1e-12) for c in cols]
        lines.append(f"| {i + 1} | {'**' + n + '**' if n == 'PIKO' else n} | {CLASS.get(n, 'PIKO ablation')} | " + " | ".join(cells)
                     + f" | {100 * r['div_id']:.1f}% / {100 * r['div']:.1f}% | {r['params']:,} | {r['train_time']:.0f} | {rk[n]:.2f} |")
    return f"### {model}\n\n" + hdr + "\n".join(lines) + "\n", rk


# ============================================================================ figures
def fig_dotplot(rowsM, out):
    """small multiples: rows = models (sorted by mean rank over both plants), columns = test regime;
    open dot = @64, filled dot = @256; PIKO emphasised, baselines in muted ink."""
    import matplotlib.pyplot as plt
    S.apply()
    plants = list(rowsM)
    names = [n for n in ORDER if all(n in rowsM[p] for p in plants)]
    mr = {n: np.mean([ranks(rowsM[p], names)[n] for p in plants]) for n in names}
    names = sorted(names, key=lambda n: -mr[n])
    fig, axes = plt.subplots(len(plants), 3, figsize=(10.5, 0.24 * len(names) * len(plants) + 1.4), sharey=True)
    axes = np.atleast_2d(axes)
    for pi, p in enumerate(plants):
        for ci, tag in enumerate(["ID", "OOD", "CF"]):
            ax = axes[pi, ci]
            for yi, n in enumerate(names):
                r = rowsM[p][n]
                c = S.BLUE if n == "PIKO" else S.MUTED
                v64, v256 = r.get(f"{tag}64", np.nan), r.get(f"{tag}256", np.nan)
                ax.plot([v64, v256], [yi, yi], color=S.GRID if n != "PIKO" else S.SEQ[3], lw=1.2, zorder=1)
                ax.scatter([v64], [yi], s=26, facecolor=S.SURF, edgecolor=c, lw=1.3, zorder=3)
                ax.scatter([v256], [yi], s=26, color=c, zorder=3, edgecolor=S.SURF, lw=0.6)
            vals = [rowsM[p][n].get(f"{tag}{h}", np.nan) for n in names for h in (64, 256)]
            if np.isfinite(vals).any():
                ax.set_xscale("log")
            ax.set_yticks(range(len(names)))
            ax.set_yticklabels(names)
            for tl in ax.get_yticklabels():
                if tl.get_text() == "PIKO":
                    tl.set_color(S.INK)
                    tl.set_fontweight("bold")
            ax.set_title({"ID": "Closed-loop replay, in-distribution", "OOD": "Closed-loop replay, OOD",
                          "CF": "Counterfactual torques (MPC-relevant)"}[tag] + f" — {p}", fontsize=8.5)
            ax.set_xlabel("NRMSE (standardised units, log)")
            ax.grid(axis="y", visible=False)
    h1 = axes[0, 0].scatter([], [], s=26, facecolor=S.SURF, edgecolor=S.INK2)
    h2 = axes[0, 0].scatter([], [], s=26, color=S.INK2)
    top = S.suptitle(fig, "Multi-step prediction error by test regime", "Models sorted by mean rank over both plants (best at top); diverged windows count as error 20")
    fig.legend([h1, h2], ["first 64 steps", "all 256 steps"], loc="upper right", ncol=2, fontsize=7.5,
               bbox_to_anchor=(0.995, 1 - 0.12 / fig.get_figheight()))
    fig.tight_layout(rect=(0, 0, 1, top))
    S.save(fig, os.path.join(out, "bench_dotplot"))


def fig_horizon(rowsM, out, key="curve_ID", title="In-distribution closed-loop replay"):
    import matplotlib.pyplot as plt
    S.apply()
    plants = [p for p in rowsM if any(key in r for r in rowsM[p].values())]
    if not plants:
        return
    fig, axes = plt.subplots(1, len(plants), figsize=(5.2 * len(plants), 3.4))
    axes = np.atleast_1d(axes)
    for ax, p in zip(axes, plants):
        rows = rowsM[p]
        base = [n for n in ORDER if n in rows and n != "PIKO" and key in rows[n]]
        best = sorted(base, key=lambda n: rows[n].get("ID256" if key != "curve_CF" else "CF256", np.inf))[:3]
        hi = {best[0]: S.ORANGE, best[1]: S.AQUA, best[2]: S.YELLOW} if len(best) >= 3 else {}
        h = np.arange(1, 257)
        for n in base:
            if n not in hi:
                ax.plot(h, rows[n][key], color=S.CONTEXT, lw=0.9, zorder=1)
        for n, c in hi.items():
            ax.plot(h, rows[n][key], color=c, lw=1.6, label=n, zorder=2)
        if "PIKO" in rows and key in rows["PIKO"]:
            ax.plot(h, rows["PIKO"][key], color=S.BLUE, lw=2.0, label="PIKO", zorder=3)
        ax.set_yscale("log")
        ax.set_xlabel("prediction step")
        ax.set_ylabel("NRMSE (standardised)")
        ax.set_title(p)
        ax.legend(loc="lower right")
        ax.set_xlim(1, 256)
    top = S.suptitle(fig, f"Error growth with horizon — {title}", "PIKO vs the three best baselines at 256 steps; other models in grey")
    fig.tight_layout(rect=(0, 0, 1, top))
    S.save(fig, os.path.join(out, f"bench_horizon_{key.split('_')[1]}"))


def fig_family_heat(rowsM, out, cf=False):
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm
    S.apply()
    k = "cf_fam64" if cf else "fam64"
    plants = [p for p in rowsM if any(k in r for r in rowsM[p].values())]
    if not plants:
        return
    fig, axes = plt.subplots(1, len(plants), figsize=(6.2 * len(plants), 5.2))
    axes = np.atleast_1d(axes)
    for ax, p in zip(axes, plants):
        rows = rowsM[p]
        names = [n for n in ORDER if n in rows and k in rows[n]]
        fams = sorted(rows[names[0]][k])
        M = np.array([[rows[n][k][f] for f in fams] for n in names])
        im = ax.imshow(np.clip(M, 1e-3, 20), cmap=S.CMAP_SEQ, norm=LogNorm(1e-3, 20), aspect="auto")
        ax.set_xticks(range(len(fams)))
        ax.set_xticklabels(fams, rotation=45, ha="right")
        ax.set_yticks(range(len(names)))
        ax.set_yticklabels(names)
        colbest = np.nanargmin(M, axis=0)
        for i in range(len(names)):
            for j in range(len(fams)):
                v = M[i, j]
                ax.text(j, i, f"{v:.3f}" if v < 1 else f"{v:.1f}", ha="center", va="center", fontsize=5.6,
                        color=S.SURF if v > 0.3 else S.INK, fontweight="bold" if colbest[j] == i else "normal")
        ax.grid(False)
        ax.set_title(p)
        fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02, label="NRMSE, first 64 steps")
    top = S.suptitle(fig, "Error by scenario family — " + ("counterfactual torques" if cf else "closed-loop replay"),
                     "bold = best model for that family")
    fig.tight_layout(rect=(0, 0, 1, top))
    S.save(fig, os.path.join(out, "bench_families_cf" if cf else "bench_families"))


def fig_ablation(rowsM, out):
    import matplotlib.pyplot as plt
    from matplotlib.ticker import LogLocator, NullFormatter
    S.apply()
    plants = list(rowsM)
    regs = [("ID", 64), ("ID", 256), ("OOD", 64), ("CF", 64)]
    fig, axes = plt.subplots(len(plants), len(regs), figsize=(12, 3.2 * len(plants)))
    axes = np.atleast_2d(axes)
    for pi, p in enumerate(plants):
        rows = rowsM[p]
        names = [n for n in ABL if n in rows]
        for ci, (t, h) in enumerate(regs):
            ax = axes[pi, ci]
            v = np.array([rows[n].get(f"{t}{h}", np.nan) for n in names])
            y = np.arange(len(names))[::-1]
            for yi, vi, n in zip(y, v, names):
                c = S.BLUE if n.startswith("PIKO") and n in ("PIKO", "PIKO-bidir") else (S.SEQ[5] if n == "PIKO-core" else S.MUTED)
                ax.plot([0, vi], [yi, yi], color=S.GRID, lw=1.0, zorder=1)
                ax.scatter([vi], [yi], s=30, color=c, zorder=3, edgecolor=S.SURF, lw=0.6)
                if np.isfinite(vi):
                    ax.annotate(f"{vi:.3f}" if vi < 10 else f"{vi:.0f}", (vi, yi), xytext=(5, -3),
                                textcoords="offset points", fontsize=6.5, color=S.INK2)
            if np.isfinite(v).any():
                ax.set_xscale("log")
                ax.set_xlim(np.nanmin(v) / 1.6, np.nanmax(v) * 3)
                ax.xaxis.set_major_locator(LogLocator(base=10, numticks=6))
                ax.xaxis.set_minor_formatter(NullFormatter())
            ax.set_yticks(y)
            ax.set_yticklabels(names if ci == 0 else [""] * len(names))
            ax.set_title(f"{p} — {t}@{h}")
            ax.grid(axis="y", visible=False)
    top = S.suptitle(fig, "PIKO ablations", "core = Koopman + observer only (no innovation network); NRMSE, log scale")
    fig.tight_layout(rect=(0, 0, 1, top))
    S.save(fig, os.path.join(out, "bench_ablation"))


def fig_efficiency(rowsM, out):
    import matplotlib.pyplot as plt
    S.apply()
    plants = list(rowsM)
    fig, axes = plt.subplots(1, len(plants), figsize=(5.4 * len(plants), 3.6))
    axes = np.atleast_1d(axes)
    for ax, p in zip(axes, plants):
        rows = rowsM[p]
        for n in ORDER:
            if n not in rows:
                continue
            r = rows[n]
            c = S.BLUE if n == "PIKO" else S.MUTED
            ax.scatter(max(r["latency"], 1e-3), r["ID256"], s=34 if n == "PIKO" else 20, color=c, zorder=3,
                       edgecolor=S.SURF, lw=0.6)
            ax.annotate(n, (max(r["latency"], 1e-3), r["ID256"]), xytext=(4, 2), textcoords="offset points",
                        fontsize=6.5, color=S.INK if n == "PIKO" else S.INK2)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("inference latency per window, 64 steps (ms, batch 256, 1 CPU thread)")
        ax.set_ylabel("ID NRMSE @256")
        ax.set_title(p)
    top = S.suptitle(fig, "Accuracy versus inference cost")
    fig.tight_layout(rect=(0, 0, 1, top))
    S.save(fig, os.path.join(out, "bench_efficiency"))


def fig_examples(root, model, rows, s, fam, out, names=("PIKO", "GRU", "SINDYc"), cf=False, C=None):
    """one window per family: T1 / T2 / w1 truth vs models (physical units)"""
    import matplotlib.pyplot as plt
    S.apply()
    nm = json.load(open(os.path.join(root, model, "norm.json")))
    mu, sd = np.array(nm["s_mu"]), np.array(nm["s_sd"])
    od = os.path.join(root, model)
    if cf:
        tgt, fm = C["s"], C["fam"].astype(str)
        fams = ["NOM", "UNC", "DIST", "FAULT"]
    else:
        tgt, fm = np.load(os.path.join(od, "test_targets.npz"))["s"], fam
        fams = ["REF", "UNC", "DIST", "FAULT"]
    preds = {}
    for n in names:
        if n not in rows:
            continue
        fn = (f"predcf_off_{n}.npz" if os.path.exists(os.path.join(od, f"predcf_off_{n}.npz")) else f"predcf_{n}.npz") if cf else f"pred_{n}.npz"
        preds[n] = np.load(os.path.join(od, fn))["pred"]
    colors = {"PIKO": S.BLUE, "GRU": S.ORANGE, "SINDYc": S.AQUA, "LRU-SSM": S.YELLOW, "TCN": S.MAGENTA}
    ch = [(0, "T1 (N)"), (1, "T2 (N)"), (3, "w1 (rad/s)")]
    fig, axes = plt.subplots(len(ch), len(fams), figsize=(3.1 * len(fams), 6.2), sharex=True)
    rng = np.random.default_rng(3)
    dt = 1.0 if model == "M1" else 2.0
    for j, f in enumerate(fams):
        idx = np.where(fm == f)[0]
        if not len(idx):
            continue
        ref = preds.get("PIKO")
        # pick a window with visible dynamics (upper quartile of truth variation)
        var = tgt[idx, 32:, 0].std(1)
        i = idx[np.argsort(var)[int(0.8 * len(idx))]]
        t = (np.arange(288) - 31) * dt
        for k, (c, lab) in enumerate(ch):
            ax = axes[k, j]
            for n, p in preds.items():
                ax.plot(t[32:], p[i, :, c] * sd[c] + mu[c], color=colors.get(n, S.MUTED), lw=1.5 if n == "PIKO" else 1.2,
                        label=n, alpha=0.95)
            ax.plot(t, tgt[i, :, c] * sd[c] + mu[c], color=S.INK, lw=0.9, label="truth", ls=(0, (3, 1.5)), zorder=5)
            ax.axvline(0, color=S.AXIS, lw=0.8)
            tr_ = tgt[i, :, c] * sd[c] + mu[c]
            span = max(tr_.max() - tr_.min(), 0.02 * max(abs(tr_).max(), 1e-3))
            ax.set_ylim(tr_.min() - 1.2 * span, tr_.max() + 1.2 * span)
            if j == 0:
                ax.set_ylabel(lab)
            if k == 0:
                ax.set_title(f"{f}")
            if k == len(ch) - 1:
                ax.set_xlabel("time (ms), 0 = forecast start")
    axes[0, 0].legend(loc="best", fontsize=6.5)
    top = S.suptitle(fig, f"{model} — example forecasts, " + ("counterfactual torques" if cf else "closed-loop replay"),
                     "32-sample history (left of 0), 256-step forecast; physical units")
    fig.tight_layout(rect=(0, 0, 1, top))
    S.save(fig, os.path.join(out, f"bench_examples_{model}" + ("_cf" if cf else "")))


def fig_mpc(root, model, out):
    import matplotlib.pyplot as plt
    od = os.path.join(root, model, "mpc")
    if not os.path.exists(os.path.join(od, "mpc_summary.json")):
        return
    S.apply()
    summ = json.load(open(os.path.join(od, "mpc_summary.json")))
    keys = list(summ)
    fig, axes = plt.subplots(4, len(keys), figsize=(5.2 * len(keys), 8.4), sharex="col")
    axes = np.atleast_2d(axes.T).T if len(keys) == 1 else axes
    for j, kk in enumerate(keys):
        for ctrl, c, lab in (("paper", S.ORANGE, "paper SMC (10 kHz)"), ("piko_mpc", S.BLUE, "PIKO-MPC (sample rate)")):
            fn = os.path.join(od, f"{kk}_{ctrl}.npz")
            if not os.path.exists(fn):
                continue
            d = np.load(fn)
            t, x, u, ref = d["t"], d["x"], d["u"], d["ref"]
            axes[0, j].plot(t, x[:, 0] - ref[:, 0], color=c, lw=1.1, label=lab)
            axes[1, j].plot(t, x[:, 1] - ref[:, 1], color=c, lw=1.1)
            axes[2, j].plot(t, x[:, 3] - ref[:, 2], color=c, lw=1.1)
            axes[3, j].plot(t, u[:, 0], color=c, lw=0.9)
        axes[0, j].set_title(summ[kk]["tag"], fontsize=8)
        for i, lab in enumerate(["T1 − T1d (N)", "T2 − T2d (N)", "w1 − w1d (rad/s)", "Mu (N m)"]):
            axes[i, j].set_ylabel(lab)
            if i < 3:
                axes[i, j].set_yscale("symlog", linthresh=1e-3)
        axes[3, j].set_xlabel("time (s)")
    axes[0, 0].legend(loc="upper right", fontsize=7)
    top = S.suptitle(fig, f"{model} — closed loop: Koopman-MPC with the PIKO core vs the paper controller",
                     "tracking errors (symlog) and unwinder torque")
    fig.tight_layout(rect=(0, 0, 1, top))
    S.save(fig, os.path.join(out, f"mpc_{model}"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="out/bench_final")
    ap.add_argument("--models", default="M1,M2")
    a = ap.parse_args()
    figd = os.path.join(a.root, "figures")
    os.makedirs(figd, exist_ok=True)
    rowsM, md, csv = {}, [], ["plant,model,class,ID64,ID256,OOD64,OOD256,PAPER64,PAPER256,CF64,CF256,CFOOD64,CFOOD256,div_id,div_all,div_cf,params,train_s,latency_ms,mean_rank"]
    extra = {}
    for model in a.models.split(","):
        rows, s, fam, C = load_plant(a.root, model)
        rowsM[model] = rows
        extra[model] = (s, fam, C)
        t, rk = leaderboard_md(rows, model, ORDER)
        md.append(t)
        ta, _ = leaderboard_md(rows, model + " — PIKO ablations", ABL)
        md.append(ta)
        for n, r in rows.items():
            csv.append(",".join([model, n, CLASS.get(n, "PIKO ablation")] + [f"{r.get(k, np.nan):.5g}" for k in
                       ["ID64", "ID256", "OOD64", "OOD256", "PAPER64", "PAPER256", "CF64", "CF256", "CFOOD64", "CFOOD256",
                        "div_id", "div", "div_cf", "params", "train_time", "latency"]] + [f"{rk.get(n, np.nan):.3g}"]))
    open(os.path.join(a.root, "leaderboard.md"), "w").write("\n".join(md))
    open(os.path.join(a.root, "leaderboard.csv"), "w").write("\n".join(csv) + "\n")
    json.dump({m: {n: {k: v for k, v in r.items() if isinstance(v, (int, float, str, dict))} for n, r in rows.items()}
               for m, rows in rowsM.items()}, open(os.path.join(a.root, "leaderboard.json"), "w"), indent=1, default=float)
    fig_dotplot(rowsM, figd)
    fig_horizon(rowsM, figd, "curve_ID", "in-distribution closed-loop replay")
    fig_horizon(rowsM, figd, "curve_OOD", "out-of-distribution closed-loop replay")
    fig_horizon(rowsM, figd, "curve_CF", "counterfactual torques")
    fig_family_heat(rowsM, figd)
    fig_family_heat(rowsM, figd, cf=True)
    fig_ablation(rowsM, figd)
    fig_efficiency(rowsM, figd)
    for model in rowsM:
        s, fam, C = extra[model]
        fig_examples(a.root, model, rowsM[model], s, fam, figd)
        if C is not None:
            fig_examples(a.root, model, rowsM[model], s, fam, figd, cf=True, C=C)
        fig_mpc(a.root, model, figd)
    print("\n".join(md))


if __name__ == "__main__":
    main()
