"""
plants.py -- batched two-span roll-to-roll (R2R) plant models M1 and M2.

M1 : V.T. Dang et al., "Finite-time velocity sensorless integral sliding mode
     control for roll-to-roll systems under matched disturbances",
     ISA Transactions 164 (2025) 61-74.            (eqs. 1-3, Sec. 5.1)
M2 : H.T. Nguyen et al., "A Fixed-Time Convergence Control of Roll-to-Roll
     Systems With a Fault-Tolerant Mechanism", IEEE Access 13 (2025)
     138247-138262.                                  (eqs. 1-4, Sec. V.A)

Both papers share the same two-span structure (M2 adds an unmatched tension
disturbance D_T and an actuator-fault term Y_w C_w):

    dT/dt = F_T(T, w, R) + G_T(T, R) w_m + D_T
    dw/dt = F_w(T, w, R, J) + G_w(J) M + D_w + Y_w C_w
    dphi/dt = w

    f_T1 = -(R1/L1) T1 w1 + (E S R1/L1) w1        g_T1 = (Ru/L1)(Tud - E S)
    f_T2 =  (R1/L2) w1 (T1 - E S)                 g_T2 = (Rr/L2)(E S - T2)
    f_u  =  (Ru T1 - bfu wu + chi kap rho Ru^3 wu^2)/Ju     g_u = -1/Ju
    f_1  =  (R1 (T2 - T1) - bf1 w1)/J1                       g_1 =  1/J1
    f_r  = (-Rr T2 - bfr wr - chi kap rho Rr^3 wr^2)/Jr      g_r =  1/Jr

    Ru = Ru0 - phi_u chi/(2 pi)      Rr = Rr0 + phi_r chi/(2 pi)
    Ju = Ju0 + (pi rho kap/2)(Ru^4 - Ru0^4)   (same for r)

Notation map  M1 -> M2 : chi -> hbar (thickness), kappa -> zeta (width),
rho -> nu (density).  Note: M1 prints f_T2 with L1; with L1 = L2 this is
immaterial, we use L2 (physically correct).  M2's actuator fault model:
    y_i(t) = 1 - exp(-rho_i (t - t_fi)) for t >= t_fi, C_i = g_i (-a_i M_i)
i.e. an incipient/abrupt loss of effectiveness of magnitude a_i.

Everything here is vectorised over a leading batch dimension B so a whole
scenario family is integrated in one pass.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict, fields, replace
from typing import Dict, Tuple

import numpy as np

TWO_PI = 2.0 * np.pi

# state layout ---------------------------------------------------------------
STATE_NAMES = ["T1", "T2", "wu", "w1", "wr", "phiu", "phi1", "phir"]
INPUT_NAMES = ["Mu", "M1", "Mr"]
NX, NU = 8, 3
UNCERTAIN_KEYS_M1 = ("chi", "rho", "bfu", "bf1", "bfr", "Ju0", "Jr0")          # M1 Sec. 5.1
UNCERTAIN_KEYS_M2 = ("Ju0", "Jr0", "J1", "bfu", "bf1", "bfr", "rho", "E", "chi")  # M2 Sec. V.A


@dataclass(frozen=True)
class R2RParams:
    Ru0: float
    R1: float
    Rr0: float
    L1: float
    L2: float
    E: float
    S: float
    chi: float      # thickness            (M2: hbar)
    kappa: float    # width                (M2: zeta)
    rho: float      # density              (M2: nu)
    Ju0: float
    J1: float
    Jr0: float
    bfu: float
    bf1: float
    bfr: float
    Tud: float
    M_max: float    # actuator saturation |M_i| <= M_max (not in the papers; loose safety limit)
    name: str = "R2R"

    @property
    def ES(self) -> float:
        return self.E * self.S

    def as_dict(self) -> Dict[str, float]:
        d = asdict(self)
        d.pop("name")
        return d

    def perturbed(self, pct: float, rng: np.random.Generator, keys) -> "R2RParams":
        """Uniform multiplicative uncertainty in [-pct, +pct] on `keys`."""
        upd = {k: getattr(self, k) * (1.0 + pct * rng.uniform(-1.0, 1.0)) for k in keys}
        return replace(self, **upd)


# M1 -- ISA Trans. 164 (2025), Sec. 5.1
M1_PARAMS = R2RParams(
    Ru0=0.04, R1=0.015, Rr0=0.015, L1=0.5, L2=0.5, E=2.5e9, S=1.0e-3,
    chi=2.0e-3, kappa=0.5, rho=800.0, Ju0=7.0e-3, J1=7.0e-3, Jr0=7.0e-3,
    bfu=2.533e-5, bf1=2.533e-5, bfr=2.533e-5, Tud=10.0, M_max=5.0, name="M1")

# M2 -- IEEE Access 13 (2025), Sec. V.A
M2_PARAMS = R2RParams(
    Ru0=0.10, R1=0.02, Rr0=0.10, L1=1.2, L2=1.2, E=1.6e8, S=1.2e-5,
    chi=1.2e-4, kappa=0.1, rho=800.0, Ju0=4.56e-3, J1=8.67e-4, Jr0=4.52e-3,
    bfu=5.0e-3, bf1=6.5e-3, bfr=4.6e-3, Tud=4.0, M_max=20.0, name="M2")

PARAMS = {"M1": M1_PARAMS, "M2": M2_PARAMS}
UNCERTAIN_KEYS = {"M1": UNCERTAIN_KEYS_M1, "M2": UNCERTAIN_KEYS_M2}


class BatchParams:
    """Parameters stored as (B,) arrays so each trajectory can have its own plant."""

    NAMES = [f.name for f in fields(R2RParams) if f.name != "name"]

    def __init__(self, plist):
        self.B = len(plist)
        for k in self.NAMES:
            setattr(self, k, np.array([getattr(p, k) for p in plist], dtype=np.float64))
        self.ES = self.E * self.S

    def matrix(self) -> np.ndarray:
        return np.stack([getattr(self, k) for k in self.NAMES], axis=1)


def geometry(x: np.ndarray, p, radii=None) -> Tuple[np.ndarray, ...]:
    """Radii and inertias of the unwinder/rewinder (eqs. 2-3). x: (B, 8).

    radii=(Ru, Rr) overrides the angle-based radius (controllers use the
    measured roll radius -- diameter sensors are standard on winders -- so a
    thickness error does not corrupt the web-kinematic feed-forward)."""
    if radii is None:
        Ru = p.Ru0 - x[:, 5] * p.chi / TWO_PI
        Rr = p.Rr0 + x[:, 7] * p.chi / TWO_PI
    else:
        Ru, Rr = radii
    Ru = np.maximum(Ru, 0.2 * p.Ru0)                     # guard for very long runs
    c = 0.5 * np.pi * p.rho * p.kappa
    Ju = p.Ju0 + c * (Ru ** 4 - p.Ru0 ** 4)
    Jr = p.Jr0 + c * (Rr ** 4 - p.Rr0 ** 4)
    return Ru, Rr, Ju, Jr


def FG(x: np.ndarray, p, radii=None):
    """Return F (B,5) and diagonal G (B,5) of X_dot = F + G U, U=[wu, wr, Mu, M1, Mr]."""
    T1, T2, wu, w1, wr = (x[:, i] for i in range(5))
    Ru, Rr, Ju, Jr = geometry(x, p, radii)
    ES = p.ES
    fT1 = -(p.R1 / p.L1) * T1 * w1 + (ES * p.R1 / p.L1) * w1
    gT1 = (Ru / p.L1) * (p.Tud - ES)
    fT2 = (p.R1 / p.L2) * w1 * (T1 - ES)
    gT2 = (Rr / p.L2) * (ES - T2)
    ck = p.chi * p.kappa * p.rho
    fu = (Ru * T1 - p.bfu * wu + ck * Ru ** 3 * wu ** 2) / Ju
    gu = -1.0 / Ju
    f1 = (p.R1 * (T2 - T1) - p.bf1 * w1) / p.J1
    g1 = np.ones_like(T1) / p.J1
    fr = (-Rr * T2 - p.bfr * wr - ck * Rr ** 3 * wr ** 2) / Jr
    gr = 1.0 / Jr
    F = np.stack([fT1, fT2, fu, f1, fr], axis=1)
    G = np.stack([gT1, gT2, gu, g1, gr], axis=1)
    return F, G


def xdot(x: np.ndarray, M: np.ndarray, p, D: np.ndarray | None = None,
         fault_y: np.ndarray | None = None, fault_a: np.ndarray | None = None) -> np.ndarray:
    """Full plant vector field.

    x (B,8), M (B,3) applied torques, D (B,5) lumped disturbances
    [D_T1, D_T2, D_u, D_1, D_r] (N/s, N/s, rad/s^2 x3),
    fault_y (B,3) fault evolution y_i(t) in [0,1], fault_a (B,3) loss magnitudes a_i.
    Actuator fault (M2 eq. 1,4): w_dot += y_i * g_i * (-a_i M_i).
    """
    F, G = FG(x, p)
    dx = np.empty_like(x)
    dx[:, 0] = F[:, 0] + G[:, 0] * x[:, 2]
    dx[:, 1] = F[:, 1] + G[:, 1] * x[:, 4]
    Meff = M
    if fault_y is not None:
        Meff = M * (1.0 - fault_y * fault_a)
    dx[:, 2:5] = F[:, 2:5] + G[:, 2:5] * Meff
    if D is not None:
        dx[:, 0:5] += D
    dx[:, 5:8] = x[:, 2:5]
    return dx


def physics_residual(x, xd, M, p, D=None, fault_y=None, fault_a=None):
    """r = xd - f(x, M): the quantity a PINN drives to zero (collocation residual)."""
    return xd - xdot(x, M, p, D, fault_y, fault_a)
