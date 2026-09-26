"""
bench/mpc_demo.py -- closed-loop Koopman-MPC with the PIKO core versus the paper controllers.

The PIKO core (gEDMD-identified LPV Koopman generator in kinematic coordinates + unknown-input
observer; no closed-loop innovation network, because in MPC the torques are decisions) is used as
the prediction model of a linear time-varying MPC that runs at the data sampling rate
(M1: 1 kHz, M2: 500 Hz) with zero-order hold, whereas the paper controllers run at 10 kHz.

At every sampling instant:
  1. q0 = kinematic coordinates of the measured state; every `obs_every` samples the observer
     re-estimates the lumped input w and the actuator effectiveness theta on the last 32 samples;
  2. the Koopman step q+ = A q + Gm quad(q) + Gu u~ + c is linearised about q0 (exact matrix
     exponential of the scheduled generator at the current radii);
  3. condensed LQ tracking over N steps (tensions and line speed), torque-rate penalty, solved in
     closed form and clipped to the actuator limits; the first move is applied.

usage:  python -m r2r_nn.bench.mpc_demo --model M1 --scen 0,3
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import torch

from ..plants import PARAMS
from ..scenarios import make_specs
from ..simulate import simulate_batch
from ..generate_datasets import T_END, DT_SIM, STORE_EVERY, MODEL_SEED, FAMILY_SEED
from .data import load_trajs, fit_norm, make_windows, Batch
from .piko import PIKO

F64 = torch.float64


class PIKOMPC:
    def __init__(self, pk: PIKO, model, every, N=20, Wq=(1.0, 1.0, 10.0, 0.0, 0.0), lam_du=3e-3, lam_u=1e-5,
                 obs_every=16, Hp=32, use_obs=True, meas_std=None, tau_meas=0.0, dt_sim=1e-4, kf_q=0.1):
        self.pk, self.nm, self.every, self.N = pk, pk.nm, every, N
        self.Wq = torch.as_tensor(Wq, dtype=F64)
        self.lam_du, self.lam_u, self.obs_every, self.Hp, self.use_obs = lam_du, lam_u, obs_every, Hp, use_obs
        self.Mmax = PARAMS[model].M_max
        f = lambda a: torch.as_tensor(np.asarray(a), dtype=F64)
        self.s_mu, self.s_sd, self.u_mu, self.u_sd = f(self.nm.s_mu), f(self.nm.s_sd), f(self.nm.u_mu), f(self.nm.u_sd)
        self.r_mu, self.r_sd = f(self.nm.r_mu), f(self.nm.r_sd)
        self.solve_time = []
        # extended Kalman filter on the Koopman core (needed with noisy speed sensors: on M1 the web-speed
        # mismatch v1 is ~1e-6 of the speeds and cannot be computed from noisy speed measurements)
        self.kf = meas_std is not None
        if self.kf:
            m = np.asarray(meas_std, float)[:5]
            al = dt_sim / (tau_meas + dt_sim)
            m = m * np.sqrt(al / (2 - al))                   # std after the first-order sensor filter
            R1, Ru, Rr = PARAMS[model].R1, PARAMS[model].Ru0, PARAMS[model].Rr0
            sq = [m[0], m[1], m[3], np.hypot(R1 * m[3], Ru * m[2]), np.hypot(Rr * m[4], R1 * m[3])]
            qsd = pk.q_sd.numpy()
            self.Rk = torch.diag(torch.as_tensor((np.array(sq) / qsd) ** 2 + 1e-12, dtype=F64))
            self.Qk = torch.diag((kf_q * getattr(pk, "dq_sd_mpc", torch.ones(5, dtype=F64))) ** 2)

    def reset(self, y10, ref0):
        B = y10.shape[0]
        self.n = 0
        self.S, self.R, self.U = [], [], []
        self.M = np.zeros((B, 3))
        self.u_prev = torch.zeros(B, 3, dtype=F64)
        self.w = torch.zeros(B, 5, dtype=F64)
        self.th = torch.zeros(B, 3, dtype=F64)
        self.qh, self.P, self.last = None, None, None

    def __call__(self, t, y10, r_n, M_exp):
        if self.n % self.every == 0:
            self.sample(y10, r_n)
        self.n += 1
        return self.M

    # ------------------------------------------------------------------
    def kf_step(self, qm):
        pk = self.pk
        if self.qh is None or self.last is None:
            self.qh, self.P = qm.clone(), self.Rk.expand(qm.shape[0], -1, -1).clone()
            return self.qh
        A, Gm, Gu, c = self.last
        q = self.qh
        quad = q[:, pk.iu0] * q[:, pk.iu1]
        ue = self.u_prev + self.th * (self.u_prev + pk.u_off)
        qp = torch.bmm(A, q.unsqueeze(-1)).squeeze(-1) + torch.bmm(Gm, quad.unsqueeze(-1)).squeeze(-1) \
            + torch.bmm(Gu, ue.unsqueeze(-1)).squeeze(-1) + c
        F = A + torch.bmm(Gm, self.jac(q))
        Pp = torch.bmm(torch.bmm(F, self.P), F.transpose(1, 2)) + self.Qk
        K = torch.linalg.solve((Pp + self.Rk).transpose(1, 2), Pp.transpose(1, 2)).transpose(1, 2)
        self.qh = qp + torch.bmm(K, (qm - qp).unsqueeze(-1)).squeeze(-1)
        self.P = torch.bmm(torch.eye(5, dtype=F64) - K, Pp)
        return self.qh

    def jac(self, q):
        pk = self.pk
        no = pk.n_obs
        J = torch.zeros(q.shape[0], no, 5, dtype=F64)
        ar = torch.arange(no)
        J[:, ar, pk.iu0] += q[:, pk.iu1]
        J[:, ar, pk.iu1] += q[:, pk.iu0]
        return J

    def sample(self, y10, r_n):
        pk = self.pk
        s = (torch.as_tensor(y10[:, :5], dtype=F64) - self.s_mu) / self.s_sd
        r = (torch.as_tensor(y10[:, 8:10], dtype=F64) - self.r_mu) / self.r_sd
        q_est = None
        if self.kf:
            with torch.no_grad():
                q_est = self.kf_step(pk.to_q(s, r))
                s = pk.from_q(q_est, r)
        if self.S:                                   # torque applied over the last interval
            self.U.append((torch.as_tensor(self.M, dtype=F64) - self.u_mu) / self.u_sd)
        self.S.append(s)
        self.R.append(r)
        self.S, self.R, self.U = self.S[-self.Hp:], self.R[-self.Hp:], self.U[-(self.Hp - 1):]
        t0 = time.time()
        with torch.no_grad():
            if self.use_obs and len(self.S) == self.Hp and (len(self.solve_time) % self.obs_every == 0):
                Sh, Rh = torch.stack(self.S, 1), torch.stack(self.R, 1)
                Uh = torch.cat([torch.stack(self.U, 1), self.U[-1].unsqueeze(1)], 1)
                w, th, _ = pk.observe(pk.to_q(Sh, Rh), Uh, Rh)
                self.w = w if w is not None else self.w
                self.th = th if th is not None else self.th
            u = self.solve(s, r, r_n, q_est)
        self.solve_time.append(time.time() - t0)
        M = (u * self.u_sd + self.u_mu).numpy()
        self.M = np.clip(M, -self.Mmax, self.Mmax)
        self.u_prev = (torch.as_tensor(self.M, dtype=F64) - self.u_mu) / self.u_sd

    def solve(self, s, r, r_n, q0=None):
        pk, N = self.pk, self.N
        B = s.shape[0]
        q0 = pk.to_q(s, r) if q0 is None else q0
        A, G, _, _ = pk.disc_q(r)
        no = pk.n_obs
        Gm, Gu, gc, Gw = G[:, :, :no], G[:, :, no:no + 3], G[:, :, no + 3], G[:, :, no + 4:no + 4 + pk.n_w]
        # linearise the quadratic dictionary about q0
        quad0 = q0[:, pk.iu0] * q0[:, pk.iu1]
        J = torch.zeros(B, no, 5, dtype=F64)
        ar = torch.arange(no)
        J[:, ar, pk.iu0] += q0[:, pk.iu1]
        J[:, ar, pk.iu1] += q0[:, pk.iu0]
        At = A + torch.bmm(Gm, J)
        c = gc + torch.bmm(Gm, (quad0 - torch.bmm(J, q0.unsqueeze(-1)).squeeze(-1)).unsqueeze(-1)).squeeze(-1)
        if Gw.shape[-1]:
            c = c + torch.bmm(Gw, self.w.unsqueeze(-1)).squeeze(-1)
        cw = gc + (torch.bmm(Gw, self.w.unsqueeze(-1)).squeeze(-1) if Gw.shape[-1] else 0.0)
        self.last = (A, Gm, Gu, cw)
        Gu_e = Gu * (1 + self.th).unsqueeze(1)
        c = c + torch.bmm(Gu, (self.th * pk.u_off).unsqueeze(-1)).squeeze(-1)
        # condensed prediction  Q = Sx q0 + Su U + Sc
        Sx = torch.zeros(B, 5 * N, 5, dtype=F64)
        Su = torch.zeros(B, 5 * N, 3 * N, dtype=F64)
        Sc = torch.zeros(B, 5 * N, dtype=F64)
        Ak = torch.eye(5, dtype=F64).expand(B, -1, -1)
        ck = torch.zeros(B, 5, dtype=F64)
        pw = [torch.eye(5, dtype=F64).expand(B, -1, -1)]
        for k in range(N):
            pw.append(torch.bmm(At, pw[-1]))
        for k in range(N):
            ck = torch.bmm(At, ck.unsqueeze(-1)).squeeze(-1) + c
            Sx[:, 5 * k:5 * k + 5] = pw[k + 1]
            Sc[:, 5 * k:5 * k + 5] = ck
            for j in range(k + 1):
                Su[:, 5 * k:5 * k + 5, 3 * j:3 * j + 3] = torch.bmm(pw[k - j], Gu_e)
        # reference in q coordinates (tensions, line speed; v1, v2 unweighted)
        # reference over the horizon, extrapolated with its known derivative (ramp feed-forward)
        Td = torch.as_tensor(r_n["Td"], dtype=F64)
        dTd = torch.as_tensor(r_n["dTd"], dtype=F64)
        w1d = torch.as_tensor(r_n["w1d"], dtype=F64)
        dw1d = torch.as_tensor(r_n["dw1d"], dtype=F64)
        kk = torch.arange(1, N + 1, dtype=F64).view(1, N) * pk.dt
        qref = torch.zeros(B, N, 5, dtype=F64)
        qref[:, :, 0] = (Td[:, :1] + kk * dTd[:, :1] - pk.q_mu[0]) / pk.q_sd[0]
        qref[:, :, 1] = (Td[:, 1:2] + kk * dTd[:, 1:2] - pk.q_mu[1]) / pk.q_sd[1]
        qref[:, :, 2] = (w1d.view(B, 1) + kk * dw1d.view(B, 1) - pk.q_mu[2]) / pk.q_sd[2]
        Wd = self.Wq.repeat(N)
        err0 = qref.reshape(B, 5 * N) - torch.bmm(Sx, q0.unsqueeze(-1)).squeeze(-1) - Sc
        D = torch.eye(3 * N, dtype=F64) - torch.diag(torch.ones(3 * (N - 1), dtype=F64), -3)
        H = torch.bmm(Su.transpose(1, 2) * Wd.view(1, 1, -1), Su) + self.lam_du * (D.T @ D) + self.lam_u * torch.eye(3 * N, dtype=F64)
        d0 = torch.zeros(B, 3 * N, dtype=F64)
        d0[:, :3] = self.u_prev
        g = torch.bmm(Su.transpose(1, 2) * Wd.view(1, 1, -1), err0.unsqueeze(-1)).squeeze(-1) + self.lam_du * (d0 @ D)
        U = torch.linalg.solve(H, g.unsqueeze(-1)).squeeze(-1)
        return U[:, :3]


def build_core(model, root="out/datasets"):
    tr = load_trajs(root, model)
    nm = fit_norm(tr, model)
    nm.model = model
    W = make_windows(tr, ["train"], 32, 64, 8, fams=["NOM", "REF"], max_n=4000)
    pk = PIKO(nm, model=model, innov=False)
    b = Batch(W, nm, 32, f64=True)
    pk.init_gedmd(b)
    with torch.no_grad():
        qq = pk.to_q(b.s64, b.r64)
        pk.dq_sd_mpc = (qq[:, 1:] - qq[:, :-1]).reshape(-1, 5).std(0)
    pk.eval()
    return pk


def metrics(out, t, t_min=0.5):
    t_min = min(t_min, 0.5 * t[-1])
    x, ref, u = out["x"][0], out["ref"][0], out["u"][0]
    m = t >= t_min
    e = x[:, [0, 1, 3]] - ref[:, [0, 1, 2]]
    return dict(rmse_T1=float(np.sqrt((e[m, 0] ** 2).mean())), rmse_T2=float(np.sqrt((e[m, 1] ** 2).mean())),
                rmse_w1=float(np.sqrt((e[m, 2] ** 2).mean())), max_T=float(np.abs(e[m, :2]).max()),
                iae_T=float(np.abs(e[:, :2]).sum() * (t[1] - t[0])),
                torque_tv=float(np.abs(np.diff(u, axis=0)).sum(0).mean()), torque_rms=float(np.sqrt((u ** 2).mean())))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--scen", default="0,3")
    ap.add_argument("--out", default="out/bench_final")
    ap.add_argument("--t-end", type=float, default=T_END)
    ap.add_argument("--N", type=int, default=20)
    ap.add_argument("--obs-every", type=int, default=4)
    a = ap.parse_args()
    torch.set_num_threads(1)
    model = a.model
    pk = build_core(model)
    t_sim = np.arange(0.0, a.t_end + 1e-12, DT_SIM)
    se = STORE_EVERY[model]
    specs = make_specs(model, "PAPER", MODEL_SEED[model] + FAMILY_SEED["PAPER"], np.arange(0.0, T_END + 1e-12, DT_SIM))
    od = os.path.join(a.out, model, "mpc")
    os.makedirs(od, exist_ok=True)
    summary = {}
    for si in [int(v) for v in a.scen.split(",")]:
        sp = dict(specs[si])
        n = t_sim.size
        for k in ("ref", "D"):
            sp[k] = sp[k][:n]
        tag = sp["tag"]
        res = {}
        for ctrl in ("paper", "piko_mpc"):
            t0 = time.time()
            if ctrl == "paper":
                o = simulate_batch(model, [sp], t_sim, se, expert_opts=sp.get("expert_opts"), noise_seed=11)
            else:
                from ..simulate import TAU_MEAS
                pol = PIKOMPC(pk, model, every=se, N=a.N, obs_every=a.obs_every, meas_std=sp.get("noise"),
                              tau_meas=TAU_MEAS[model], dt_sim=DT_SIM)
                o = simulate_batch(model, [sp], t_sim, se, policy=pol, expert_opts=sp.get("expert_opts"), noise_seed=11)
            tt = o["t"]
            res[ctrl] = metrics(o, tt)
            res[ctrl]["wall_s"] = time.time() - t0
            if ctrl == "piko_mpc":
                res[ctrl]["solve_ms_mean"] = 1e3 * float(np.mean(pol.solve_time))
            np.savez_compressed(os.path.join(od, f"S{si + 1}_{ctrl}.npz"), t=tt, x=o["x"][0], u=o["u"][0], ref=o["ref"][0],
                                D=o["D"][0], blown=o["blown"])
            print(f"[{model}] {tag} | {ctrl}: " + ", ".join(f"{k} {v:.4g}" for k, v in res[ctrl].items()), flush=True)
        summary[f"S{si + 1}"] = dict(tag=tag, **res)
    json.dump(summary, open(os.path.join(od, "mpc_summary.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
