"""
controllers.py -- batched expert controllers used to generate closed-loop data.

M1 expert : ITSMC-ESO (Dang et al., ISA Trans. 2025)
    tension loop  (eqs. 5, 7/25)  s_T = e_T + k1 int sig(e_T)^{q/p}
        w_v = -G_T^{-1}[F_T + k1 sig(e_T)^{q/p} + k2 sgn(s_T) + k3 s_T + k4 sig(s_T)^{q/p} - dT_d]
    velocity loop (eqs. 9, 11/26)  s_w = e_w + k5 int sig(e_w)^{q/p}
        M   = -G_w^{-1}[xi_hat + k5 sig(e_w)^{q/p} + k6 sgn(s_w) + k7 s_w + k8 sig(s_w)^{q/p} - dw_d]
    finite-time ESO (eq. 15) on the measured roll angles phi -> (w_hat, xi_hat).
    Option `use_eso=False` gives the sensor-based ITSMC of eqs. (7), (11).

M2 expert : FixedSMC + fixed-time SMO + nonlinear 1st-order filter (Nguyen et al., IEEE Access 2025)
    observer (eqs. 17-22) estimates K = D + Y C for all 5 channels,
    tension loop (eqs. 34-36), filter (eq. 40), velocity loop (eqs. 44-46).

Implementation notes (deviations, all deliberate and documented in the README):
  * sgn(.) is replaced by tanh(./eps) (both papers' Remark 5 / Remark 8).
  * M1: the virtual command w_v is passed through a first-order command filter
    (tau_f) to obtain dw_d -- the paper does not state how dw_d is computed.
  * M1: with the paper's ESO constants (lambda=(3,3,1), mu=0.1, delta=0.155)
    the linear-region observer poles are at about -52, -0.09+/-0.04j rad/s,
    i.e. xi_hat essentially never converges within a 4 s run.  We keep the fal
    structure and exponents but rescale the gains to a bandwidth w_o (default
    100 rad/s): lambda1/mu*a11 = 3 w_o, lambda2*a12 = 3 w_o^2, lambda3*mu*a13 = w_o^3.
  * M2: the observer needs e_dot = X_dot - varsigma_dot; X_dot is obtained by
    backward differencing the (low-pass filtered) measurement.
  * Actuator saturation |M_i| <= M_max is applied to the commanded torque.
"""
from __future__ import annotations

import numpy as np

from .plants import FG, geometry, BatchParams


def sig(x, a):
    return np.sign(x) * np.abs(x) ** a


def ssat(x, eps):
    """smooth sign"""
    return np.tanh(x / eps)


def fal(e, gamma, delta):
    ae = np.abs(e)
    out = np.where(ae > delta, np.sign(e) * ae ** gamma, e / delta ** (1.0 - gamma))
    return out


# ============================================================================
#  M1 : ITSMC / ITSMC-ESO
# ============================================================================
class ITSMC_M1:
    name = "ITSMC"

    def __init__(self, pnom: BatchParams, dt: float, use_eso: bool = False,
                 k1=50.0, k2=100.0, k3=30.0, k4=50.0, k5=50.0, k6=100.0, k7=30.0, k8=50.0,
                 q=3, p=5, eps_T=0.05, eps_w=0.02, tau_f=0.001,
                 eso_wo=100.0, gam=(0.7, 0.4, 0.1), delta=0.155):
        self.p = pnom
        self.dt = dt
        self.use_eso = use_eso
        if use_eso:
            self.name = "ITSMC-ESO"
        self.k = dict(k1=k1, k2=k2, k3=k3, k4=k4, k5=k5, k6=k6, k7=k7, k8=k8)
        self.r = q / p
        self.eps_T, self.eps_w, self.tau_f = eps_T, eps_w, tau_f
        self.gam, self.delta = gam, delta
        a1 = delta ** (gam[0] - 1)
        a2 = delta ** (gam[1] - 1)
        a3 = delta ** (gam[2] - 1)
        # linear-region equivalent gains: beta = (3wo, 3wo^2, wo^3)
        self.l1 = 3 * eso_wo / a1          # = lambda1/mu
        self.l2 = 3 * eso_wo ** 2 / a2     # = lambda2
        self.l3 = eso_wo ** 3 / a3         # = lambda3*mu
        self.B = pnom.B

    def reset(self, x0_meas, wd0):
        B = self.B
        self.IT = np.zeros((B, 2))       # int sig(e_T)^{q/p}
        self.Iw = np.zeros((B, 3))       # int sig(e_w)^{q/p}
        self.wf = wd0.copy()             # filtered virtual command [wu_d, w1_d, wr_d]
        self.phi_hat = x0_meas[:, 5:8].copy()
        self.w_hat = x0_meas[:, 2:5].copy()
        F, G = FG(x0_meas, self.p)
        self.xi_hat = F[:, 2:5].copy()
        self.M_prev = np.zeros((B, 3))
        self.initialised = False

    def step(self, t, y, ref):
        """y: measured state (B,8); ref: dict with Td (B,2), dTd (B,2), w1d, dw1d (B,)."""
        k, r, p, dt = self.k, self.r, self.p, self.dt
        if self.use_eso:
            w_est = self.w_hat
        else:
            w_est = y[:, 2:5]
        xe = y[:, 0:8].copy()
        xe[:, 2:5] = w_est
        rad = (y[:, 8], y[:, 9])                # measured roll radii
        F, G = FG(xe, p, rad)
        # ---------------- tension loop ----------------
        eT = y[:, 0:2] - ref["Td"]
        sT = eT + k["k1"] * self.IT
        corr = (k["k1"] * sig(eT, r) + k["k2"] * ssat(sT, self.eps_T)
                + k["k3"] * sT + k["k4"] * sig(sT, r) - ref["dTd"])
        # w_v = w_ff + w_fb ; w_ff = -F_T/G_T is the web-kinematic (no-slip) part,
        # w_fb = -corr/G_T the sliding-mode correction.
        wff = -F[:, 0:2] / G[:, 0:2]
        wfb = -corr / G[:, 0:2]
        wv2 = wff + wfb
        wd = np.stack([wv2[:, 0], ref["w1d"], wv2[:, 1]], axis=1)
        # derivative of w_v: analytic for w_ff (dominant; the web stiffness E S makes
        # any lag here catastrophic), filtered backward difference for w_fb.
        ES = p.ES
        T1, T2 = y[:, 0], y[:, 1]
        w1 = w_est[:, 1]
        Ru, Rr, Ju, Jr = geometry(xe, p, rad)
        dw1 = (self.xi_hat[:, 1] if self.use_eso else F[:, 3]) + G[:, 3] * self.M_prev[:, 1]
        dT1, dT2 = ref["dTd"][:, 0], ref["dTd"][:, 1]
        dRu = -p.chi * w_est[:, 0] / (2 * np.pi)
        dRr = p.chi * w_est[:, 2] / (2 * np.pi)
        a = p.R1 * w1 * (ES - T1)
        da = p.R1 * (dw1 * (ES - T1) - w1 * dT1)
        bu = Ru * (ES - p.Tud)
        dbu = dRu * (ES - p.Tud)
        br = Rr * (ES - T2)
        dbr = dRr * (ES - T2) - Rr * dT2
        dwff_u = (da * bu - a * dbu) / bu ** 2
        dwff_r = (da * br - a * dbr) / br ** 2
        if not self.initialised:
            self.wfb_prev = wfb.copy()
            self.dwfb = np.zeros_like(wfb)
            self.initialised = True
        raw = (wfb - self.wfb_prev) / dt
        self.dwfb += (dt / (self.tau_f + dt)) * (raw - self.dwfb)
        self.wfb_prev = wfb.copy()
        dwf = np.stack([dwff_u + self.dwfb[:, 0], ref["dw1d"], dwff_r + self.dwfb[:, 1]], axis=1)
        # ---------------- velocity loop ----------------
        ew = w_est - wd
        sw = ew + k["k5"] * self.Iw
        xi = self.xi_hat if self.use_eso else F[:, 2:5]
        vcorr = (k["k5"] * sig(ew, r) + k["k6"] * ssat(sw, self.eps_w)
                 + k["k7"] * sw + k["k8"] * sig(sw, r) - dwf)
        M = -(xi + vcorr) / G[:, 2:5]
        M = np.clip(M, -p.M_max[:, None], p.M_max[:, None])
        # ---------------- internal state updates (Euler) ----------------
        self.IT += dt * sig(eT, r)
        self.Iw += dt * sig(ew, r)
        if self.use_eso:
            e = self.phi_hat - y[:, 5:8]
            g1, g2, g3 = self.gam
            dphi = self.w_hat - self.l1 * fal(e, g1, self.delta)
            dw = self.xi_hat - self.l2 * fal(e, g2, self.delta) + G[:, 2:5] * M
            dxi = -self.l3 * fal(e, g3, self.delta)
            self.phi_hat += dt * dphi
            self.w_hat += dt * dw
            self.xi_hat += dt * dxi
        self.M_prev = M
        info = dict(wd=wd, sT=sT, sw=sw, xi_hat=self.xi_hat.copy(), w_hat=w_est.copy())
        return M, info

    def set_applied(self, M_app):
        """Shadow mode (DAgger): the torque actually applied to the plant came from another
        policy; internal estimators must use it, not the expert's own command."""
        self.M_prev = M_app.copy()


# ============================================================================
#  M2 : FixedSMC + fixed-time SMO + nonlinear first-order filter
# ============================================================================
class FixedSMC_M2:
    name = "FixedSMC-FxTSMO"

    def __init__(self, pnom: BatchParams, dt: float, use_observer: bool = True,
                 xi_T=(75.0, 75.0, 75.0), xi_w=(75.0, 75.0, 75.0), gamma=3 / 5, phi=5 / 3,
                 k0=1e3, a0=25.0, k1=25.0, b=(50.0, 50.0, 50.0), k2=50.0, k3=50.0,
                 alpha=3 / 5, beta=5 / 3, P1=5.0, P2=5.0, P3=10.0, mu2=20.0, Tf=0.01,
                 eps_obs=1.0, eps_f=0.5, edot_clip=100.0, tau_e=1e-3):
        self.p = pnom
        self.dt = dt
        self.use_obs = use_observer
        if not use_observer:
            self.name = "FixedSMC"
        self.xiT, self.xiw = xi_T, xi_w
        self.g, self.ph = gamma, phi
        self.k0, self.a0, self.k1, self.b, self.k2, self.k3 = k0, a0, k1, b, k2, k3
        self.al, self.be = alpha, beta
        self.P1, self.P2, self.P3, self.mu2, self.Tf = P1, P2, P3, mu2, Tf
        self.eps_obs, self.eps_f = eps_obs, eps_f
        self.edot_clip = edot_clip
        self.tau_e = tau_e
        self.B = pnom.B

    def reset(self, x0_meas, wd0):
        B = self.B
        self.K_hat = np.zeros((B, 5))
        self.eo = np.zeros((B, 5))
        self.edot_f = np.zeros((B, 5))
        self.X_prev = None
        self.f_prev = None
        self.M_prev = np.zeros((B, 3))
        self.wfd = None                        # filter state [u_ud, u_rd]
        self.s = np.zeros((B, 5))

    def _observer_update(self, X, F, G):
        """Fixed-time SMO (17)-(22), discretised with a trapezoidal model increment:
        e_dot_k = (X_k - X_{k-1})/dt - 0.5 (f_{k-1} + f_k) - K_hat, with
        f = F + G U evaluated with the torque actually applied over the step."""
        dt = self.dt
        if self.X_prev is not None:
            U_now = np.concatenate([X[:, [2]], X[:, [4]], self.M_prev], axis=1)
            f_now = F + G * U_now
            edot_raw = (X - self.X_prev) / dt - 0.5 * (self.f_prev + f_now) - self.K_hat
            self.edot_f += (dt / (self.tau_e + dt)) * (edot_raw - self.edot_f)
            edot = self.edot_f
            # the beta>1 power terms are clipped so the explicit 10 kHz update stays
            # stable for large transient errors (continuous-time law unchanged near 0)
            ec = np.clip(edot, -self.edot_clip, self.edot_clip)
            deo = self.b[0] * sig(edot, self.al) + self.b[1] * sig(ec, self.be) + self.b[2] * edot
            s = edot + self.a0 * self.eo
            sc = np.clip(s, -self.edot_clip, self.edot_clip)
            dK = (self.a0 * deo + self.k0 * ssat(s, self.eps_obs) + self.k1 * sig(s, self.al)
                  + self.k2 * sig(sc, self.be) + self.k3 * s)
            self.eo += dt * deo
            self.K_hat += dt * dK
            self.s = s

    def step(self, t, y, ref):
        p, dt = self.p, self.dt
        X = y[:, 0:5]
        F, G = FG(y[:, 0:8], p, (y[:, 8], y[:, 9]))
        if self.use_obs:
            self._observer_update(X, F, G)
        Kh = self.K_hat if self.use_obs else np.zeros_like(self.K_hat)
        # ---------------- tension loop (34)-(36) ----------------
        eT = X[:, 0:2] - ref["Td"]
        num = (F[:, 0:2] + Kh[:, 0:2] - ref["dTd"]
               + self.xiT[0] * sig(eT, self.g) + self.xiT[1] * sig(eT, self.ph) + self.xiT[2] * eT)
        wmd = -num / G[:, 0:2]
        # ---------------- nonlinear first-order filter (40) ----------------
        if self.wfd is None:
            self.wfd = wmd.copy()
        z = self.wfd - wmd
        dwfd = -(self.P1 * sig(z, self.g) + self.P2 * sig(z, self.ph) + self.P3 * z
                 + self.mu2 * ssat(z, self.eps_f)) / self.Tf
        wd = np.stack([self.wfd[:, 0], ref["w1d"], self.wfd[:, 1]], axis=1)
        dwd = np.stack([dwfd[:, 0], ref["dw1d"], dwfd[:, 1]], axis=1)
        # ---------------- velocity loop (44)-(46) ----------------
        ew = X[:, 2:5] - wd
        num = (F[:, 2:5] + Kh[:, 2:5] - dwd
               + self.xiw[0] * sig(ew, self.g) + self.xiw[1] * sig(ew, self.ph) + self.xiw[2] * ew)
        M = -num / G[:, 2:5]
        M = np.clip(M, -p.M_max[:, None], p.M_max[:, None])
        # bookkeeping for the next observer update
        U = np.concatenate([X[:, [2]], X[:, [4]], M], axis=1)
        self.f_prev = F + G * U
        self._FGX = (F, G, X)
        self.X_prev = X.copy()
        self.M_prev = M
        self.wfd = self.wfd + dt * dwfd
        s = self.s
        info = dict(wd=wd, K_hat=self.K_hat.copy(), s=s)
        return M, info

    def set_applied(self, M_app):
        """Shadow mode (DAgger): observer model increment uses the torque actually applied."""
        F, G, X = self._FGX
        self.M_prev = M_app.copy()
        self.f_prev = F + G * np.concatenate([X[:, [2]], X[:, [4]], M_app], axis=1)
