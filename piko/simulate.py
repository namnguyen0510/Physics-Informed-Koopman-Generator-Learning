"""
simulate.py -- batched closed-loop simulation (RK4 plant, ZOH control at dt_sim).

Returns a dict of arrays sampled every `store_every` steps:
  t (N,), x (B,N,8) true state, xdot (B,N,8) exact vector field at the samples,
  y (B,N,8) noisy measurement, u (B,N,3) applied torque, u_cmd (B,N,3) expert
  command (before dither), ref (B,N,3) [T1d,T2d,w1d], dref (B,N,3), D (B,N,5)
  true lumped disturbance, fy (B,N,3) fault evolution, geo (B,N,4) [Ru,Rr,Ju,Jr],
  wd (B,N,3) virtual velocity command, est (B,N,k) observer estimates.
"""
from __future__ import annotations

import numpy as np

from .plants import BatchParams, xdot, geometry, PARAMS
from .controllers import ITSMC_M1, FixedSMC_M2
from .scenarios import fault_y, prbs, ref_derivative


# sensor low-pass time constant used when a trajectory has measurement noise.
# M2's observer differentiates the measurement, and a sensor lag inside its
# ~1000 rad/s loop destabilises it, so M2 uses the raw (noisy) samples.
TAU_MEAS = {"M1": 1e-3, "M2": 0.0}


def make_expert(model, pnom, dt, **opts):
    if model == "M1":
        return ITSMC_M1(pnom, dt, **opts)
    return FixedSMC_M2(pnom, dt, **opts)


def rk4(x, M, p, D, fy, fa, dt):
    k1 = xdot(x, M, p, D, fy, fa)
    k2 = xdot(x + 0.5 * dt * k1, M, p, D, fy, fa)
    k3 = xdot(x + 0.5 * dt * k2, M, p, D, fy, fa)
    k4 = xdot(x + dt * k3, M, p, D, fy, fa)
    return x + dt / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)


def simulate_batch(model, specs, t_sim, store_every, policy=None, expert_opts=None,
                   shadow_expert=True, noise_seed=0, tau_meas=None):
    """Simulate B trajectories in parallel.

    policy : optional callable(t, y, ref, internal) -> M  replacing the expert
             (used for closed-loop evaluation of the FNN controller). If given
             and shadow_expert=True, the expert still runs on the same
             measurements so its command u_cmd is recorded (DAgger labels).
    """
    B = len(specs)
    dt = float(t_sim[1] - t_sim[0])
    N = t_sim.size
    pnom = BatchParams([PARAMS[model]] * B)
    ptrue = BatchParams([s["plant"] for s in specs])
    switch = [s.get("plant_switch") for s in specs]
    p_after = None
    if any(sw is not None for sw in switch):
        p_after = BatchParams([sw[1] if sw is not None else s["plant"] for s, sw in zip(specs, switch)])
        t_switch = np.array([sw[0] if sw is not None else 1e9 for sw in switch])
    ref = np.stack([s["ref"] for s in specs])                 # (B,N,3)
    dref = np.stack([ref_derivative(s["ref"], dt) for s in specs])
    D = np.stack([s["D"] for s in specs])                     # (B,N,5)
    fa = np.stack([s["fault"]["a"] for s in specs])            # (B,3)
    FY = np.stack([fault_y(t_sim[:, None], s["fault"]) for s in specs])   # (B,N,3)
    noise_std = np.stack([s["noise"] if s["noise"] is not None else np.zeros(8) for s in specs])
    has_noise = noise_std.sum(axis=1) > 0
    rng = np.random.default_rng(noise_seed)
    dither = np.zeros((B, N, 3))
    for b, s in enumerate(specs):
        if s["dither"] is not None:
            r2 = np.random.default_rng(s["dither"]["seed"])
            for j in range(3):
                dither[b, :, j] = prbs(r2, N, dt, s["dither"]["amp"][j])

    x = np.stack([s["x0"] for s in specs]).astype(np.float64)
    opts = dict(expert_opts or {})
    expert = make_expert(model, pnom, dt, **opts)
    y_f = x.copy()
    wd0 = np.stack([x[:, 2], ref[:, 0, 2], x[:, 4]], axis=1)
    g0 = geometry(x, ptrue)
    y10 = np.concatenate([y_f, g0[0][:, None], g0[1][:, None]], axis=1)
    expert.reset(y10, wd0)
    if policy is not None and hasattr(policy, "reset"):
        policy.reset(y10, ref[:, 0])

    ns = (N - 1) // store_every + 1
    out = {k: np.zeros((B, ns, d)) for k, d in
           [("x", 8), ("xdot", 8), ("y", 8), ("u", 3), ("u_cmd", 3), ("ref", 3), ("dref", 3),
            ("D", 5), ("fy", 3), ("geo", 4), ("wd", 3), ("est", 5), ("u_avg", 3)]}
    u_acc = np.zeros((B, 3))
    n_acc = 0
    out["t"] = t_sim[::store_every][:ns]
    if tau_meas is None:
        tau_meas = TAU_MEAS[model]
    alpha = dt / (tau_meas + dt)
    si = 0
    blown = np.zeros(B, bool)
    # state "kicks" (KICK family): sudden web-tension / speed perturbations applied to the plant
    kicks = {}
    for b, s_ in enumerate(specs):
        for (tk, dxk) in s_.get("kicks", []):
            kicks.setdefault(int(round(tk / dt)), []).append((b, np.asarray(dxk, float)))
    for n in range(N):
        if n in kicks:
            for b, dxk in kicks[n]:
                x[b] += dxk
        t = t_sim[n]
        p_now = ptrue
        if p_after is not None:
            # parameter switch (M2 scenario 3): rebuild per-sample params
            if np.any(t >= t_switch):
                p_now = _mix_params(ptrue, p_after, t >= t_switch)
        # measurement: true + band-limited noise (1st-order sensor filter when noisy)
        y_raw = x + rng.standard_normal(x.shape) * noise_std
        y_f = np.where(has_noise[:, None], y_f + alpha * (y_raw - y_f), x)
        geo_n = geometry(x, p_now)
        # measured output: filtered noisy states + roll radii (diameter sensors / estimator)
        y10 = np.concatenate([y_f, geo_n[0][:, None], geo_n[1][:, None]], axis=1)
        r_n = dict(Td=ref[:, n, 0:2], dTd=dref[:, n, 0:2], w1d=ref[:, n, 2], dw1d=dref[:, n, 2])
        M_exp, info = expert.step(t, y10, r_n)
        if policy is not None:
            M_cmd = policy(t, y10, r_n, M_exp)
        else:
            M_cmd = M_exp
        M_app = np.clip(M_cmd + dither[:, n], -pnom.M_max[:, None], pnom.M_max[:, None])
        if policy is not None:
            # expert runs in shadow mode: its estimators must see the torque really applied
            expert.set_applied(M_app)
        if n % store_every == 0 and si > 0 and n_acc > 0:
            out["u_avg"][:, si - 1] = u_acc / n_acc      # mean torque applied over [t_{k}, t_{k+1})
            u_acc[:] = 0.0
            n_acc = 0
        u_acc += M_app
        n_acc += 1
        if n % store_every == 0 and si < ns:
            out["x"][:, si] = x
            out["xdot"][:, si] = xdot(x, M_app, p_now, D[:, n], FY[:, n], fa)
            out["y"][:, si] = y_f
            out["u"][:, si] = M_app
            out["u_cmd"][:, si] = M_exp
            out["ref"][:, si] = ref[:, n]
            out["dref"][:, si] = dref[:, n]
            out["D"][:, si] = D[:, n]
            out["fy"][:, si] = FY[:, n]
            out["geo"][:, si] = np.stack(geo_n, axis=1)
            out["wd"][:, si] = info["wd"]
            if "K_hat" in info:
                out["est"][:, si] = info["K_hat"]
            else:
                out["est"][:, si, 2:5] = info["xi_hat"]
            si += 1
        if n < N - 1:
            x = rk4(x, M_app, p_now, D[:, n], FY[:, n], fa, dt)
            bad = ~np.isfinite(x).all(axis=1) | (np.abs(x[:, 0:2]) > 1e4).any(axis=1)
            if bad.any():
                blown |= bad
                x[bad] = np.nan_to_num(x[bad], nan=0.0, posinf=0.0, neginf=0.0)
    if n_acc > 0 and si > 0:
        out["u_avg"][:, si - 1] = u_acc / n_acc
    out["blown"] = blown
    out["plant_params"] = ptrue.matrix()
    out["param_names"] = np.array(BatchParams.NAMES)
    out["fault_a"] = fa
    out["fault_tf"] = np.stack([s["fault"]["tf"] for s in specs])
    out["fault_rho"] = np.stack([s["fault"]["rho"] for s in specs])
    out["noise_std"] = noise_std
    out["dither_amp"] = np.stack([s["dither"]["amp"] if s["dither"] is not None else np.zeros(3) for s in specs])
    out["tags"] = np.array([s["tag"] for s in specs])
    out["expert"] = expert.name
    return out


def _mix_params(pa, pb, mask):
    class P:
        pass
    q = P()
    for k in BatchParams.NAMES + ["ES"]:
        setattr(q, k, np.where(mask, getattr(pb, k), getattr(pa, k)))
    q.B = pa.B
    return q
