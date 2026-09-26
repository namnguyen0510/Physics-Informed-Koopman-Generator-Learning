"""
scenarios.py -- reference trajectories, disturbances, faults, noise and the
scenario-family catalogue used to build the M1-* / M2-* datasets.

Each family returns a list of per-trajectory specs (plain dicts) consumed by
simulate.simulate_batch().  All randomness goes through a seeded Generator,
so every dataset is exactly reproducible from (model, family, seed).
"""
from __future__ import annotations

import numpy as np

from .plants import PARAMS, UNCERTAIN_KEYS

# ----------------------------------------------------------------------------
# reference profiles
# ----------------------------------------------------------------------------

def ref_paper_M1(t):
    """M1 Sec. 5.1: T1d=10, T2d=15, w1d trapezoid 0 -> 40/3 rad/s (1 s) -> hold -> 0 at 4 s."""
    T1 = np.full_like(t, 10.0)
    T2 = np.full_like(t, 15.0)
    w1 = np.where(t < 1, 40.0 / 3 * t, np.where(t < 3, 40.0 / 3, (160.0 - 40.0 * t) / 3))
    w1 = np.maximum(w1, 0.0)
    return np.stack([T1, T2, w1], axis=-1)


def ref_paper_M2(t):
    """M2 Sec. V.A (the 1.5<=t<1.7 branch of T2d is printed as 10+10(t-0.5); we use the
    continuous 10+10(t-1.5))."""
    T1 = np.select([t < 0.5, t < 0.7, t < 2.5, t < 2.7],
                   [10.0, 10 - 10 * (t - 0.5), 8.0, 8 + 10 * (t - 2.5)], 10.0)
    T2 = np.select([t < 1.5, t < 1.7, t < 3.5, t < 3.7],
                   [10.0, 10 + 10 * (t - 1.5), 12.0, 12 - 10 * (t - 3.5)], 10.0)
    w1 = np.select([t < 1, t < 1.1, t < 2, t < 2.1],
                   [7.0, 7 + 430 * (t - 1), 50.0, 50 - 200 * (t - 2)], 30.0)
    return np.stack([T1, T2, w1], axis=-1)


PAPER_REF = {"M1": ref_paper_M1, "M2": ref_paper_M2}

# ranges for random references: (lo, hi, max slope per second)
REF_RANGES = {
    "M1": dict(T1=(6.0, 14.0, 20.0), T2=(10.0, 20.0, 20.0), w1=(2.0, 20.0, 20.0), w1_start=(0.0, 0.0)),
    "M2": dict(T1=(6.0, 12.0, 15.0), T2=(8.0, 14.0, 15.0), w1=(5.0, 60.0, 250.0), w1_start=(5.0, 10.0)),
}
REF_RANGES_OOD = {
    "M1": dict(T1=(12.0, 18.0, 40.0), T2=(18.0, 26.0, 40.0), w1=(18.0, 28.0, 40.0), w1_start=(0.0, 0.0)),
    "M2": dict(T1=(3.5, 6.0, 30.0), T2=(13.0, 18.0, 30.0), w1=(60.0, 85.0, 450.0), w1_start=(8.0, 12.0)),
}


def random_pwl(rng, t, lo, hi, max_slope, v0=None, n_switch=(1, 4), t_margin=0.3):
    """Random piecewise-linear profile with bounded slope."""
    Tend = t[-1]
    v = rng.uniform(lo, hi) if v0 is None else v0
    knots_t, knots_v = [0.0], [v]
    k = rng.integers(n_switch[0], n_switch[1] + 1)
    ts = np.sort(rng.uniform(t_margin, Tend - t_margin, size=k))
    for ti in ts:
        if ti <= knots_t[-1] + 0.05:
            continue
        nv = rng.uniform(lo, hi)
        dur = max(abs(nv - knots_v[-1]) / max_slope, 0.02) * rng.uniform(1.0, 3.0)
        knots_t += [ti, min(ti + dur, Tend)]
        knots_v += [knots_v[-1], nv]
    knots_t.append(Tend + 1e-9)
    knots_v.append(knots_v[-1])
    return np.interp(t, knots_t, knots_v)


def random_ref(rng, t, model, ood=False):
    R = (REF_RANGES_OOD if ood else REF_RANGES)[model]
    T1 = random_pwl(rng, t, *R["T1"])
    T2 = random_pwl(rng, t, *R["T2"])
    w0 = rng.uniform(*R["w1_start"])
    lo, hi, sl = R["w1"]
    w1 = random_pwl(rng, t, lo, hi, sl, v0=w0, n_switch=(1, 3))
    if model == "M1":
        # M1-like start from standstill with a ramp, optional stop at the end
        ramp = np.clip(t / rng.uniform(0.5, 1.2), 0, 1)
        w1 = w1 * ramp
        if rng.random() < 0.5:
            t_stop = rng.uniform(3.0, 3.6)
            w1 = w1 * np.clip((t[-1] - t) / (t[-1] - t_stop), 0, 1)
    return np.stack([T1, T2, w1], axis=-1)


def ref_derivative(r, dt):
    return np.gradient(r, dt, axis=0)


# ----------------------------------------------------------------------------
# disturbances  D = [D_T1, D_T2, D_u, D_1, D_r]  (N/s, N/s, rad/s^2 x3)
# ----------------------------------------------------------------------------
DIST_SCALE = {"M1": np.array([0.0, 0.0, 3.0, 3.0, 3.0]),      # M1 has matched D_w only
              "M2": np.array([10.0, 10.0, 8.0, 8.0, 8.0])}


def dist_paper_M1(t, amp=1.0):
    """M1 scenario 4: 0.4 sin(4t+1) + 2 cos(0.2t+4) on every roller (matched)."""
    d = 0.4 * np.sin(4 * t + 1) + 2.0 * np.cos(0.2 * t + 4)
    D = np.zeros((t.size, 5))
    D[:, 2:5] = amp * d[:, None]
    return D


def rect_pulse(t, amp, period, phase=0.0):
    return amp * np.sign(np.sin(2 * np.pi * (t / period) + phase) + 1e-12)


def dist_paper_M2_s1(t):
    """M2 scenario 1: rectangular pulse +/-10 on K_T1 with 0.08 s period (Fig. 7)."""
    D = np.zeros((t.size, 5))
    D[:, 0] = rect_pulse(t, 10.0, 0.08)
    return D


def dist_random(rng, t, model, scale=1.0):
    """Mixture of multisine, rectangular pulses, steps and bias per channel."""
    S = DIST_SCALE[model] * scale
    D = np.zeros((t.size, 5))
    for c in range(5):
        if S[c] == 0:
            continue
        kind = rng.choice(["multisine", "pulse", "step", "paper"], p=[0.4, 0.25, 0.2, 0.15])
        if kind == "multisine":
            for _ in range(3):
                f = 10 ** rng.uniform(-1, 1)            # 0.1-10 Hz
                D[:, c] += S[c] * rng.uniform(0.1, 0.5) * np.sin(2 * np.pi * f * t + rng.uniform(0, 2 * np.pi))
        elif kind == "pulse":
            D[:, c] = rect_pulse(t, S[c] * rng.uniform(0.2, 1.0), rng.uniform(0.05, 1.0), rng.uniform(0, 2 * np.pi))
        elif kind == "step":
            ts = rng.uniform(0.3, 3.5)
            D[:, c] = S[c] * rng.uniform(-1, 1) * (t >= ts)
        else:
            d = 0.4 * np.sin(4 * t + 1) + 2.0 * np.cos(0.2 * t + 4)
            D[:, c] = S[c] / 2.4 * d
        D[:, c] += S[c] * 0.1 * rng.uniform(-1, 1)      # small bias
    return D


# ----------------------------------------------------------------------------
# faults (M2 eq. 4): y_i = 1 - exp(-rho_i (t - t_fi)), effective M_i -> (1 - a_i y_i) M_i
# ----------------------------------------------------------------------------

def fault_random(rng, n_act=(1, 2), a_rng=(0.1, 0.6)):
    a = np.zeros(3)
    tf = np.full(3, 1e9)
    rho = np.full(3, 1e5)
    k = rng.integers(n_act[0], n_act[1] + 1)
    idx = rng.choice(3, size=k, replace=False)
    for i in idx:
        a[i] = rng.uniform(*a_rng)
        tf[i] = rng.uniform(0.5, 3.3)
        rho[i] = 1e5 if rng.random() < 0.5 else 10 ** rng.uniform(-0.3, 1.0)   # abrupt / incipient
    return dict(a=a, tf=tf, rho=rho)


def fault_none():
    return dict(a=np.zeros(3), tf=np.full(3, 1e9), rho=np.full(3, 1e5))


def fault_y(t, fault):
    return np.where(t >= fault["tf"], 1.0 - np.exp(-fault["rho"] * np.maximum(t - fault["tf"], 0.0)), 0.0)


# ----------------------------------------------------------------------------
# measurement noise std: [T1, T2, wu, w1, wr, phiu, phi1, phir]
# (levels chosen so the paper controllers stay well-behaved; M1's web stiffness
#  E*S = 2.5e6 N makes it ~1000x more sensitive to speed-measurement noise than M2)
# ----------------------------------------------------------------------------
NOISE_STD = {"M1": np.array([0.01, 0.01, 0.002, 0.002, 0.002, 5e-5, 5e-5, 5e-5]),
             "M2": np.array([0.005, 0.005, 0.005, 0.005, 0.005, 1e-4, 1e-4, 1e-4])}

NOISE_SCALE = {"M1": (0.5, 1.5), "M2": (0.3, 0.9)}          # NOISE family multiplier range
NOISE_SCALE_COMBO = {"M1": (0.5, 1.0), "M2": (0.3, 0.7)}

# state-kick magnitudes (KICK family): uniform in +/- these values
KICK_T = {"M1": 2.0, "M2": 2.0}           # N
KICK_W = {"M1": 0.05, "M2": 1.0}          # rad/s

# torque dither (PRBS) amplitude range for system-identification excitation
DITHER_RNG = {"M1": (0.005, 0.03), "M2": (0.02, 0.15)}


def prbs(rng, n, dt, amp, hold_rng=(0.005, 0.05)):
    out = np.zeros(n)
    i = 0
    while i < n:
        h = max(1, int(rng.uniform(*hold_rng) / dt))
        out[i:i + h] = amp * rng.choice([-1.0, 1.0])
        i += h
    return out


# ----------------------------------------------------------------------------
# initial conditions
# ----------------------------------------------------------------------------

def x0_consistent(model, ref0, p, dT=(0.0, 0.0), dw1=0.0, w1_override=None):
    T1 = ref0[0] + dT[0]
    T2 = ref0[1] + dT[1]
    w1 = ref0[2] + dw1 if w1_override is None else w1_override
    wu = p.R1 * w1 / p.Ru0
    wr = p.R1 * w1 / p.Rr0
    return np.array([T1, T2, wu, w1, wr, 0.0, 0.0, 0.0])


# ----------------------------------------------------------------------------
# family catalogue
# ----------------------------------------------------------------------------
FAMILIES = {
    # suffix : (n_traj, short description)
    "PAPER": (None, "Exact replicas of the paper's simulation scenarios S1-S4 (validation showcase)"),
    "NOM":   (12, "Paper reference, nominal plant, randomised initial tension/speed offsets (convergence from ICs)"),
    "REF":   (24, "Random piecewise-linear tension/speed references, nominal plant"),
    "UNC":   (24, "Random references + parametric uncertainty +/-20% on the paper's uncertain set"),
    "DIST":  (24, "Random references + external disturbances (multisine / pulse / step / paper-type)"),
    "NOISE": (16, "Random references + band-limited measurement noise + mild disturbance"),
    "FAULT": (24, "Random references + actuator loss-of-effectiveness faults (abrupt/incipient) + 10% unc."),
    "EXC":   (24, "Random references + PRBS torque dither on top of the expert (system-ID excitation)"),
    "KICK":  (32, "Random references + impulsive disturbances (5-20 ms pulses, dT<=2 N) -> expert recovery (DART-style)"),
    "COMBO": (16, "Uncertainty + disturbance + noise + fault + dither together (stress, in-distribution)"),
    "OOD":   (16, "Held-out extrapolation: references outside training ranges, 30% unc., 1.5x dist., faults"),
}


def make_specs(model, family, seed, t_sim):
    """Return list of trajectory specs for one (model, family)."""
    rng = np.random.default_rng(seed)
    pn = PARAMS[model]
    keys = UNCERTAIN_KEYS[model]
    n = FAMILIES[family][0]
    specs = []

    def base(ref, plant=pn, D=None, fault=None, noise=None, dither=None, x0=None, tag="", **kw):
        if D is None:
            D = np.zeros((t_sim.size, 5))
        if x0 is None:
            x0 = x0_consistent(model, ref[0], plant)
        return dict(ref=ref, plant=plant, D=D, fault=fault or fault_none(), noise=noise,
                    dither=dither, x0=x0, tag=tag, **kw)

    if family == "PAPER":
        r = PAPER_REF[model](t_sim)
        if model == "M1":
            up = lambda pct: pn.__class__(**{**pn.as_dict(), **{k: getattr(pn, k) * (1 + pct) for k in keys}}, name="M1")
            x0 = np.array([10.0, 15.0, 0, 0, 0, 0, 0, 0])
            specs.append(base(r, x0=x0, tag="S1: nominal, ITSMC", expert_opts=dict(use_eso=False)))
            specs.append(base(r, plant=up(0.10), x0=x0, tag="S2: +10% unc., ITSMC", expert_opts=dict(use_eso=False)))
            specs.append(base(r, plant=up(0.10), x0=x0, tag="S3: +10% unc., ITSMC-ESO", expert_opts=dict(use_eso=True)))
            specs.append(base(r, plant=up(0.20), D=dist_paper_M1(t_sim), noise=NOISE_STD[model], x0=x0,
                              tag="S4: +20% unc., dist., noise, ITSMC-ESO", expert_opts=dict(use_eso=True)))
        else:
            up = lambda pct: pn.__class__(**{**pn.as_dict(), **{k: getattr(pn, k) * (1 + pct) for k in keys}}, name="M2")
            for (T10, T20, w10) in [(15, 5, 0), (8, 8, 0), (5, 15, 10), (12, 12, 20)]:
                x0 = np.array([T10, T20, 0, w10, 0, 0, 0, 0], float)
                x0[2] = pn.R1 * w10 / pn.Ru0
                x0[4] = pn.R1 * w10 / pn.Rr0
                specs.append(base(r, D=dist_paper_M2_s1(t_sim), x0=x0,
                                  tag=f"S1: pulse dist., IC T=({T10},{T20}), w1={w10}"))
            x0 = np.array([8.0, 8.0, 0, 0, 0, 0, 0, 0])
            specs.append(base(r, plant=up(0.20), x0=x0, tag="S2: +20% unc., FixedSMC (no observer)",
                              expert_opts=dict(use_observer=False)))
            D = np.zeros((t_sim.size, 5))
            D[:, 0] = rect_pulse(t_sim, 10.0, 0.08) * (t_sim < 4 / 3)
            fault = dict(a=np.array([0.25, 0.5, 0.0]), tf=np.array([8 / 3, 8 / 3, 1e9]), rho=np.array([1e5, 1e5, 1e5]))
            specs.append(base(r, D=D, fault=fault, x0=x0, plant_switch=(4 / 3, up(0.20)),
                              tag="S3: pulse dist. -> +20% unc. at 4/3 s -> faults at 8/3 s"))
            D4 = np.zeros((t_sim.size, 5))
            D4[:, 0:2] = 5.0 * np.sin(2 * np.pi * 1.0 * t_sim)[:, None]
            D4[:, 2:5] = (0.4 * np.sin(4 * t_sim + 1) + 2 * np.cos(0.2 * t_sim + 4))[:, None]
            specs.append(base(r, D=D4, noise=0.6 * NOISE_STD[model], x0=x0, tag="S4: sinusoidal dist. + noise"))
        return specs

    for i in range(n):
        if family == "NOM":
            r = PAPER_REF[model](t_sim)
            dT = rng.uniform(-3, 3, size=2)
            dw = rng.uniform(-3, 3) if model == "M2" else 0.0
            x0 = x0_consistent(model, r[0], pn, dT=dT, dw1=dw)
            x0[3] = max(x0[3], 0.0)
            specs.append(base(r, x0=x0, tag=f"NOM-{i}"))
            continue
        ood = family == "OOD"
        r = random_ref(rng, t_sim, model, ood=ood)
        plant, D, fault, noise, dither = pn, None, None, None, None
        if family in ("UNC", "COMBO", "EXC"):
            plant = pn.perturbed(0.20, rng, keys)
        if family == "FAULT":
            plant = pn.perturbed(0.10, rng, keys)
            fault = fault_random(rng)
        if family in ("DIST",):
            D = dist_random(rng, t_sim, model, 1.0)
        if family == "NOISE":
            noise = NOISE_STD[model] * rng.uniform(*NOISE_SCALE[model])
            D = dist_random(rng, t_sim, model, 0.3)
        if family == "EXC":
            amp = rng.uniform(*DITHER_RNG[model], size=3)
            dither = dict(amp=amp, seed=int(rng.integers(1 << 30)))
        if family == "KICK":
            # impulsive disturbances: short (5-20 ms) high-amplitude pulses on every channel whose
            # integral moves T by up to +/-KICK_T and w by up to +/-KICK_W -> expert recovery data
            plant = pn.perturbed(0.10, rng, keys)
            D = np.zeros((t_sim.size, 5))
            tk = rng.uniform(0.2, 0.5)
            while tk < t_sim[-1] - 0.15:
                tau = rng.uniform(0.005, 0.02)
                m = (t_sim >= tk) & (t_sim < tk + tau)
                D[m, 0:2] += rng.uniform(-1, 1, 2) * KICK_T[model] / tau
                D[m, 2:5] += rng.uniform(-1, 1, 3) * KICK_W[model] / tau
                tk += rng.uniform(0.25, 0.6)
        if family == "COMBO":
            D = dist_random(rng, t_sim, model, 1.0)
            noise = NOISE_STD[model] * rng.uniform(*NOISE_SCALE_COMBO[model])
            fault = fault_random(rng, n_act=(1, 1), a_rng=(0.1, 0.4))
            amp = 0.5 * rng.uniform(*DITHER_RNG[model], size=3)
            dither = dict(amp=amp, seed=int(rng.integers(1 << 30)))
        if family == "OOD":
            plant = pn.perturbed(0.30, rng, keys)
            D = dist_random(rng, t_sim, model, 1.5)
            if rng.random() < 0.5:
                fault = fault_random(rng, n_act=(1, 2), a_rng=(0.4, 0.7))
        x0 = x0_consistent(model, r[0], plant, dT=rng.uniform(-1, 1, size=2))
        specs.append(base(r, plant=plant, D=D, fault=fault, noise=noise, dither=dither, x0=x0, tag=f"{family}-{i}"))
    return specs
