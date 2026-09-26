"""
bench/piko.py -- PIKO: Physics-Informed Koopman Operator network with an unknown-input observer.

Idea
----
Written in the *kinematic* coordinates
        q = [T1, T2, w1, v1, v2],   v1 = R1 w1 - Ru wu,  v2 = Rr wr - R1 w1   (web-speed mismatches)
the two-span web model is a polynomial vector field of degree two in q whose coefficients depend on
the (measured, slowly varying) roll radii rho = (Ru, Rr) only through a handful of known functions
(the radius and the inertia law J(R) = J0 + pi/2 rho_w kappa (R^4 - R0^4)).  Hence the state-inclusive
dictionary  psi(q, rho) = [q, q_i q_j, g_theta(q, rho)]  spans the vector field almost exactly and a
*radius-scheduled Koopman generator* acting on psi is an (almost) exact finite-dimensional model:

    d/dt q  = C(rho) [ Lp(rho) psi + Bp(rho) u~ + cp(rho) + Ep w ]           (5 rows)
    Lp(rho) = sum_k sigma_k(rho) Lp_k                                          (LPV generator)

The generator is identified on seven *physical* rows p = [dT1, dT2, dw1, dwu, dwr, -dRu/dt wu, dRr/dt wr]
that are free of cancellation, and composed into the q rows by the exact kinematic map C(rho)
(v1' = R1 w1' - Ru wu' - Ru' wu, ...).  This matters on M1, where |v1| ~ 1e-6 R1 w1: any model that
regresses v1' (or T1' in raw state coordinates) directly must resolve a 1e-4..1e-6 cancellation.

Pieces
------
* sigma(rho): physics-derived scheduling functions [1, Ru/Ju, 1/(Ru Ju), 1/Ju, 1/Ru^2, (same for Rr)]
  with a structural mask (which function may act on which physical row).  Ablations: poly / LTI.
* Discretisation: exact zero-order hold of the stiff linear block (augmented matrix exponential,
  float64, re-scheduled every step because C(rho) changes by ~1e-3/32 steps), dictionary held
  constant over a step and re-encoded (Koopman with re-encoding, period `relift_every`; 0 = pure
  linear Koopman propagation of the lifted state -- ablation).
* Actuator model u~ = u + theta * (u + u_offset) (loss-of-effectiveness faults).
* Unknown-input observer: closed-form Gauss-Newton moving-horizon estimate of the constant lumped
  disturbance w (5, one per physical row) and actuator effectiveness change theta (3) from the
  history window (forgetting factor, learnable row weights, Tikhonov regularisation).
* Learned closure observables g_theta(q, rho) (small MLP) enter the generator linearly.
* Initialisation: generator-EDMD (least squares on exact derivative labels, per physical row).
* Training: multi-step prediction loss + PINN residual of the physical rows (exact derivatives,
  with the estimated w, theta); the identified generator is fine-tuned with a 50x smaller lr.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from .models import Base, mlp, NormalEq, _chunks
from ..plants import PARAMS

F64 = torch.float64
P_NAMES = ["dT1", "dT2", "dw1", "dwu", "dwr", "-dRu*wu", "dRr*wr"]


def sched_poly(r, order=2):
    one = torch.ones_like(r[..., :1])
    if order == 0:
        return one
    if order == 1:
        return torch.cat([one, r], -1)
    r1, r2 = r[..., :1], r[..., 1:2]
    return torch.cat([one, r1, r2, r1 * r1, r2 * r2, r1 * r2], -1)


class PIKO(Base):
    name = "PIKO"
    uses_sdot = True
    needs_f64 = True

    def __init__(self, nm, model="M1", n_learn=0, width=64, use_obs=True, use_fault=True, exact=True,
                 sched="phys", gamma=0.95, lam_obs=1e-3, lam_fault=3e-2, relift_every=1, resched_every=8,
                 quad=True, obs_rows=(1.0, 1.0, 1.0, 1.0, 1.0), gn_iters=2, use_eta=False, lam_eta=1e-2,
                 struct=True, foh=False, uctx=True, use_df=False, df_width=32, accept=0.5,
                 innov=True, innov_hidden=128, innov_bidir=0):
        super().__init__(nm)
        R = lambda a: torch.as_tensor(a, dtype=F64)
        for k in ("s_mu", "s_sd", "r_mu", "r_sd"):
            self.register_buffer(k + "64", R(getattr(nm, k)))
        self.register_buffer("u_off", R(nm.u_mu / nm.u_sd))
        self.register_buffer("q_mu", torch.zeros(5, dtype=F64))
        self.register_buffer("q_sd", torch.ones(5, dtype=F64))
        self.register_buffer("p_sd", torch.ones(7, dtype=F64))            # physical-row rate scales
        self.register_buffer("qd_sd", torch.ones(5, dtype=F64))           # q-rate scales
        self.R1, self.dt = float(nm.R1), float(nm.dt)
        pp = PARAMS[model]
        self.geo = dict(Ju0=pp.Ju0, Jr0=pp.Jr0, Ru0=pp.Ru0, Rr0=pp.Rr0, c=0.5 * torch.pi * pp.rho * pp.kappa)
        self.use_obs, self.use_fault, self.exact = use_obs, use_obs and use_fault, exact
        self.sched = sched
        self.gamma, self.lam_obs, self.lam_fault, self.gn_iters = gamma, lam_obs, lam_fault, gn_iters
        self.use_eta, self.lam_eta = use_obs and use_eta, lam_eta
        self.foh = foh and exact
        self.accept = accept
        self.uctx = uctx
        self.relift_every, self.resched_every = relift_every, resched_every
        self.quad = quad
        iu = torch.triu_indices(5, 5)
        self.register_buffer("iu0", iu[0])
        self.register_buffer("iu1", iu[1])
        self.n_q = iu.shape[1] if quad else 0
        self.n_known = 5 + self.n_q
        self.n_l = n_learn
        self.nz = self.n_known + self.n_l                                   # dictionary size
        self.n_obs = self.nz - 5
        self.n_w = 5 if use_obs else 0
        # ---- scheduling functions and structural mask (n_s, 7)
        self.n_s = {"phys": 9, "poly": 6, "lti": 1}[sched]
        mask = torch.ones(self.n_s, 7, dtype=F64)
        if sched == "phys":
            mask.zero_()
            mask[0, 0:3] = 1.0                   # T1, T2, w1 rows: radius-independent
            mask[1:4, 3] = 1.0                   # dwu: Ru/Ju, 1/(Ru Ju), 1/Ju
            mask[5:8, 4] = 1.0                   # dwr: Rr/Jr, 1/(Rr Jr), 1/Jr
            mask[4, 5] = 1.0                     # -dRu/dt wu ~ wu^2 = (R1 w1 - v1)^2 / Ru^2
            mask[8, 6] = 1.0                     #  dRr/dt wr ~ wr^2 = (v2 + R1 w1)^2 / Rr^2
        self.register_buffer("smask", mask)
        self.register_buffer("sig_ref", torch.ones(self.n_s, dtype=F64))
        if sched == "phys":
            with torch.no_grad():
                self.sig_ref.copy_(self._sig_raw(R(nm.r_mu).view(1, 2))[0])
        # structural sparsity of the dictionary (which monomials / torques may appear in which
        # physical row under which scheduling function), derived from the web model; dense otherwise
        dm, bm, cm = self._struct_masks() if (struct and sched == "phys") else (
            mask[:, :, None].expand(-1, -1, self.nz).clone(), mask[:, :, None].expand(-1, -1, 3).clone(), mask.clone())
        dm[:, :, self.n_known:] = 0.0          # learned observables act on the q rows directly (Lg below)
        self.register_buffer("dmask", dm)
        self.register_buffer("bmask", bm)
        self.register_buffer("cmask", cm)
        # ---- learned closure observables
        if self.n_l:
            self.g = mlp(5 + 2 + (9 if uctx else 0), self.n_l, width, 2).double()
            with torch.no_grad():
                self.g[-1].weight.mul_(0.1)
                self.g[-1].bias.zero_()
        # ---- generator of the physical rows (per-step units)
        n_s, nz = self.n_s, self.nz
        self.Lp = nn.Parameter(torch.zeros(n_s, 7, nz, dtype=F64))
        self.Bp = nn.Parameter(torch.zeros(n_s, 7, 3, dtype=F64))
        self.cp = nn.Parameter(torch.zeros(n_s, 7, dtype=F64))
        self.Ep = nn.Parameter(torch.zeros(7, max(self.n_w, 1), dtype=F64))
        with torch.no_grad():
            self.Ep[:5, :self.n_w] = torch.eye(5, dtype=F64)[:, :self.n_w] * 0.01
        # identified physics is kept fixed during training (Adam's per-parameter steps would destroy the
        # 1e-5-relative balance of the winder rows); learning happens in q units:
        for prm in (self.Lp, self.Bp, self.cp, self.Ep):
            prm.requires_grad_(False)
        self.Lq = nn.Parameter(torch.zeros(5, 5, dtype=F64))                # residual linear generator (implicit)
        self.Bq = nn.Parameter(torch.zeros(5, 3, dtype=F64))
        self.cq = nn.Parameter(torch.zeros(5, dtype=F64))
        self.Lg = nn.Parameter(torch.zeros(5, max(self.n_l, 1), dtype=F64))  # closure observables -> q rows
        self.damp_raw = nn.Parameter(torch.full((5,), -12.0, dtype=F64))     # learnable extra damping (hedging)
        # exogenous-input forecaster: in closed loop the future torques carry the controller's reaction
        # to disturbances; a small non-causal 1-D CNN over the torque sequence (+ observer estimates)
        # forecasts a per-step lumped input d_k acting on the q rows (Koopman input channel)
        # closed-loop innovation network: in logged closed-loop data the actuator commands are a noisy
        # measurement of the state through the (unknown) control law.  A small GRU reads the torque
        # stream along the Koopman rollout and emits innovations Delta q (zero-initialised).  It is
        # switched off for counterfactual / MPC use, where torques are decisions, not measurements.
        self.innov = innov
        if innov:
            self.corr = InnovationCorrector(nm, innov_hidden, bidir=innov_bidir)
        self.use_df = use_df
        if use_df:
            nin = 6 + 5 + 3 + 5 + 5
            self.df = nn.Sequential(nn.Conv1d(nin, df_width, 5, padding=2), nn.SiLU(),
                                    nn.Conv1d(df_width, df_width, 5, padding=4, dilation=2), nn.SiLU(),
                                    nn.Conv1d(df_width, 5, 5, padding=2)).double()
            with torch.no_grad():
                self.df[-1].weight.zero_()
                self.df[-1].bias.zero_()
        # ---- observable rows (only for pure-linear Koopman propagation, relift_every != 1)
        if relift_every != 1:
            self.Lo = nn.Parameter(torch.zeros(n_s, self.n_obs, nz, dtype=F64))
            self.Bo = nn.Parameter(torch.zeros(n_s, self.n_obs, 3, dtype=F64))
            self.co = nn.Parameter(torch.zeros(n_s, self.n_obs, dtype=F64))
        self.obs_row = nn.Parameter(torch.log(torch.expm1(R(list(obs_rows)))))
        # observer regularisation (log) and disturbance internal model: the lumped input estimated
        # over the history is forecast as w_k = w * exp(-lambda k) (lambda learnable, init ~0 = constant)
        self.log_lam = nn.Parameter(torch.log(R([lam_obs, lam_fault, lam_eta])))
        self.w_decay_raw = nn.Parameter(torch.full((max(self.n_w, 1),), -9.0, dtype=F64))
        self.no_pinn = False
        self.pinn_fams = ("NOM", "REF")

    def _struct_masks(self):
        """(row, sigma) -> allowed linear terms, quadratic monomials, torques (standardised q:
        0 T1, 1 T2, 2 w1, 3 v1, 4 v2).  Learned observables are allowed wherever the pair is active."""
        spec = {  # (row, sigma): (linear, quadratic, torques)
            (0, 0): ([0, 2, 3], [(0, 2)], []),                               # dT1: ES v1, Tud(a - v1), T1 w1
            (1, 0): ([0, 1, 2, 4], [(0, 2), (1, 4), (1, 2)], []),            # dT2: ES v2, w1 T1, (v2 + a) T2
            (2, 0): ([0, 1, 2], [], [1]),                                    # dw1: T1, T2, w1, M1
            (3, 1): ([0, 2, 3], [(2, 2), (2, 3), (3, 3)], []),               # dwu, Ru/Ju: T1, ck (a - v1)^2
            (3, 2): ([2, 3], [], []),                                        # dwu, 1/(Ru Ju): bfu (a - v1)
            (3, 3): ([], [], [0]),                                           # dwu, 1/Ju: Mu
            (4, 5): ([1, 2, 4], [(2, 2), (2, 4), (4, 4)], []),               # dwr, Rr/Jr: T2, ck (v2 + a)^2
            (4, 6): ([2, 4], [], []),                                        # dwr, 1/(Rr Jr): bfr (v2 + a)
            (4, 7): ([], [], [2]),                                           # dwr, 1/Jr: Mr
            (5, 4): ([2, 3], [(2, 2), (2, 3), (3, 3)], []),                  # -dRu/dt wu = chi/2pi (a - v1)^2/Ru^2
            (6, 8): ([2, 4], [(2, 2), (2, 4), (4, 4)], []),                  # dRr/dt wr = chi/2pi (v2 + a)^2/Rr^2
        }
        qidx = {(int(i), int(j)): 5 + n for n, (i, j) in enumerate(zip(self.iu0.tolist(), self.iu1.tolist()))}
        dm = torch.zeros(self.n_s, 7, self.nz, dtype=F64)
        bm = torch.zeros(self.n_s, 7, 3, dtype=F64)
        cm = torch.zeros(self.n_s, 7, dtype=F64)
        for (row, k), (lin, quad, tq) in spec.items():
            cm[k, row] = 1.0
            for i in lin:
                dm[k, row, i] = 1.0
            if self.quad:
                for ij in quad:
                    dm[k, row, qidx[ij]] = 1.0
            for j in tq:
                bm[k, row, j] = 1.0
        return dm, bm, cm

    # ================================================================== coordinates
    def phys_r(self, rn):
        return rn * self.r_sd64 + self.r_mu64

    def to_q(self, sn, rn):
        sp = sn * self.s_sd64 + self.s_mu64
        rp = self.phys_r(rn)
        R1 = self.R1
        q = torch.stack([sp[..., 0], sp[..., 1], sp[..., 3], R1 * sp[..., 3] - rp[..., 0] * sp[..., 2],
                         rp[..., 1] * sp[..., 4] - R1 * sp[..., 3]], -1)
        return (q - self.q_mu) / self.q_sd

    def from_q(self, qn, rn):
        q = qn * self.q_sd + self.q_mu
        rp = self.phys_r(rn)
        R1 = self.R1
        w1 = q[..., 2]
        wu = (R1 * w1 - q[..., 3]) / rp[..., 0]
        wr = (q[..., 4] + R1 * w1) / rp[..., 1]
        sp = torch.stack([q[..., 0], q[..., 1], wu, w1, wr], -1)
        return (sp - self.s_mu64) / self.s_sd64

    def p_labels(self, sn, sdn, rn, rd):
        """exact physical-row rates (float64): [dT1, dT2, dw1 (q units), dwu, dwr (state units),
        -dRu/dt wu, dRr/dt wr (SI)]"""
        sp = sn * self.s_sd64 + self.s_mu64
        return torch.stack([sdn[..., 0] * self.s_sd64[0] / self.q_sd[0], sdn[..., 1] * self.s_sd64[1] / self.q_sd[1],
                            sdn[..., 3] * self.s_sd64[3] / self.q_sd[2], sdn[..., 2], sdn[..., 4],
                            -rd[..., 0] * sp[..., 2], rd[..., 1] * sp[..., 4]], -1)

    def compose(self, rn):
        """exact kinematic map C(rho) (B,5,7): physical-row rates -> standardised q rates"""
        rp = self.phys_r(rn)
        B = rn.shape[0]
        C = rn.new_zeros(B, 5, 7)
        qs, ss = self.q_sd, self.s_sd64
        C[:, 0, 0] = 1.0
        C[:, 1, 1] = 1.0
        C[:, 2, 2] = 1.0
        C[:, 3, 2] = self.R1 * qs[2] / qs[3]
        C[:, 3, 3] = -rp[:, 0] * ss[2] / qs[3]
        C[:, 3, 5] = 1.0 / qs[3]
        C[:, 4, 4] = rp[:, 1] * ss[4] / qs[4]
        C[:, 4, 2] = -self.R1 * qs[2] / qs[4]
        C[:, 4, 6] = 1.0 / qs[4]
        return C

    # ================================================================== dictionary
    def lift(self, q, rn, uc=None):
        """dictionary psi(q, rho[, torque context]).  The learned observables may also see the local
        torque context uc = [u_j, u_j - u_{j-1}, u_{j+1} - u_j] (input-dependent observables, as in
        Koopman-with-inputs), letting the closure learn the average intra-sample torque effect."""
        parts = [q]
        if self.quad:
            parts.append(q[..., self.iu0] * q[..., self.iu1])
        if self.n_l:
            gin = [q, rn]
            if self.uctx:
                gin.append(uc if uc is not None else q.new_zeros(q.shape[:-1] + (9,)))
            parts.append(self.g(torch.cat(gin, -1)))
        return torch.cat(parts, -1)

    @staticmethod
    def ctx(u, j):
        n = u.shape[1]
        a, b = max(j - 1, 0), min(j + 1, n - 1)
        return torch.cat([u[:, j], u[:, j] - u[:, a], u[:, b] - u[:, j]], -1)

    @staticmethod
    def ctx_seq(u):
        um = torch.cat([u[:, :1], u[:, :-1]], 1)
        up = torch.cat([u[:, 1:], u[:, -1:]], 1)
        return torch.cat([u, u - um, up - u], -1)

    def jac_quad(self, q, S):
        """d(q_i q_j)/dq applied to sensitivities S (B,5,n) -> (B,n_q,n)"""
        return q[:, self.iu0, None] * S[:, self.iu1] + q[:, self.iu1, None] * S[:, self.iu0]

    # ================================================================== scheduling / generator
    def _sig_raw(self, rp):
        g = self.geo
        Ru, Rr = rp[..., 0:1], rp[..., 1:2]
        Ju = g["Ju0"] + g["c"] * (Ru ** 4 - g["Ru0"] ** 4)
        Jr = g["Jr0"] + g["c"] * (Rr ** 4 - g["Rr0"] ** 4)
        return torch.cat([torch.ones_like(Ru), Ru / Ju, 1 / (Ru * Ju), 1 / Ju, 1 / Ru ** 2,
                          Rr / Jr, 1 / (Rr * Jr), 1 / Jr, 1 / Rr ** 2], -1)

    def sig(self, rn):
        if self.sched == "phys":
            return self._sig_raw(self.phys_r(rn)) / self.sig_ref
        if self.sched == "poly":
            return sched_poly(rn, 2)
        return torch.ones_like(rn[..., :1])

    def p_blocks(self, rn):
        """physical-row generator at radii rn (B,2): Lp (B,7,nz), Bp (B,7,3), cp (B,7)"""
        w = self.sig(rn)
        L = torch.einsum("bk,kij->bij", w, self.Lp * self.dmask)
        Bm = torch.einsum("bk,kij->bij", w, self.Bp * self.bmask)
        c = w @ (self.cp * self.cmask)
        return L, Bm, c

    def p_full(self, rn, eta=None):
        """[Lp | Bp | cp | Ep] (B,7,nz+3+1+n_w); eta (B,5) scales the physical rows T1,T2,w1,wu,wr
        (uncertain stiffness E*S and inertias: a whole row of the web model carries 1/J or E*S)"""
        L, Bm, c = self.p_blocks(rn)
        P = torch.cat([L, Bm, c.unsqueeze(-1)], -1)
        if eta is not None:
            P = torch.cat([P[:, :5] * (1 + eta).unsqueeze(-1), P[:, 5:]], 1)
        E = self.Ep.expand(rn.shape[0], -1, -1)[:, :, :self.n_w]
        return torch.cat([P, E], -1)

    def residual(self):
        """learned residual generator of the q rows in q units (B-independent): (5, nz+3+1+n_w)"""
        res = [self.Lq - torch.diag(nn.functional.softplus(self.damp_raw)), self.Lq.new_zeros(5, self.n_known - 5)]
        if self.n_l:
            res.append(self.Lg[:, :self.n_l])
        res += [self.Bq, self.cq.unsqueeze(-1), self.Lq.new_zeros(5, self.n_w)]
        return torch.cat(res, -1)

    def q_blocks(self, rn, eta=None, phys_only=False):
        """generator of the q rows: F = C [Lp | Bp | cp | Ep] (+ learned residual) -> (B,5,nz+3+1+n_w)"""
        F = torch.bmm(self.compose(rn), self.p_full(rn, eta))
        return F if phys_only else F + self.residual().unsqueeze(0)

    def disc_q(self, rn, eta=None):
        """Exponential-integrator discretisation of dq = Lqq q + F [dictionary-rest, u, 1, w] with the
        dictionary held over the step:  q+ = e^{Lqq} q + phi1(Lqq) F [...] + Psi(Lqq) Fu du,
        where du is the torque slope over the step (first-order hold of the interval-averaged torque;
        Psi = phi1/2 - phi2 is the exact response to a zero-mean linear ramp).
        Returns (Aq (B,5,5), G (B,5,nz-5+3+1+n_w), Gs (B,5,3) or None, phi1 (B,5,5))."""
        # the matrix exponential only sees the (fixed) identified physics -> no backward through expm;
        # the learned residual is integrated with phi1 (exponential Euler)
        with torch.no_grad():
            Fp = self.q_blocks(rn, None if eta is None else eta.detach(), phys_only=True)
        Rz = self.residual()
        Lqq, Fr = Fp[:, :, :5], Fp[:, :, 5:] + Rz[:, 5:].unsqueeze(0)
        if False:
            # keep the (first-order) gradient path of eta outside the exponential
            Fe = torch.bmm(self.compose(rn), self.p_full(rn, eta)) - torch.bmm(self.compose(rn), self.p_full(rn, eta.detach()))
            Fr = Fr + Fe[:, :, 5:]
            Rq = Rz[:, :5].unsqueeze(0) + Fe[:, :, :5]
        else:
            Rq = Rz[:, :5].unsqueeze(0)
        B = rn.shape[0]
        if not self.exact:
            Phi = torch.eye(5, dtype=F64).expand(B, -1, -1)
            return Phi + Lqq + Rq, Fr, None, Phi
        n = 15 if self.foh else 10
        M = rn.new_zeros(B, n, n)
        M[:, :5, :5] = Lqq
        M[:, :5, 5:10] = torch.eye(5, dtype=F64)
        if self.foh:
            M[:, 5:10, 10:15] = torch.eye(5, dtype=F64)
        with torch.no_grad():
            X = torch.linalg.matrix_exp(M)
        Phi = X[:, :5, 5:10]
        Gs = None
        if self.foh:
            no = self.n_obs
            Gs = torch.bmm(0.5 * Phi - X[:, :5, 10:15], Fr[:, :, no:no + 3])
        return X[:, :5, :5] + torch.bmm(Phi, Rq.expand(B, -1, -1)), torch.bmm(Phi, Fr), Gs, Phi

    def disc_full(self, rn):
        """pure-Koopman ablation: full lifted generator (q rows + observable rows), exact ZOH"""
        F = self.q_blocks(rn)                                                # (B,5,nz+4+nw)
        w = self.sig(rn)
        Lo = torch.einsum("bk,kij->bij", w, self.Lo)
        Bo = torch.einsum("bk,kij->bij", w, self.Bo)
        co = w @ self.co
        B, nz, nw = rn.shape[0], self.nz, self.n_w
        Fo = torch.cat([Lo, Bo, co.unsqueeze(-1), rn.new_zeros(B, self.n_obs, nw)], -1)
        Fall = torch.cat([F, Fo], 1)                                         # (B,nz,nz+4+nw)
        if not self.exact:
            A = torch.eye(nz, dtype=F64).expand(B, -1, -1) + Fall[:, :, :nz]
            return A, Fall[:, :, nz:]
        n = nz + 4 + nw
        M = rn.new_zeros(B, n, n)
        M[:, :nz, :] = Fall
        X = torch.linalg.matrix_exp(M)
        return X[:, :nz, :nz], X[:, :nz, nz:]

    # ================================================================== one step
    def forecast_inputs(self, u, Q, w, th, eta):
        """per-step lumped input d (B, L, 5) in q units per step from the torque sequence"""
        B, L, _ = u.shape
        feats = [u, u - torch.cat([u[:, :1], u[:, :-1]], 1)]
        for v, n in ((w, 5), (th, 3), (eta, 5)):
            feats.append((v if v is not None else u.new_zeros(B, n)).detach().unsqueeze(1).expand(-1, L, -1))
        feats.append(Q[:, -1].unsqueeze(1).expand(-1, L, -1))
        x = torch.cat(feats, -1).transpose(1, 2)
        return self.df(x).transpose(1, 2)

    def innov_in(self, q, u, j, rn, est):
        du = u[:, j] - u[:, max(j - 1, 0)]
        return torch.cat([q, u[:, j], du, rn, est], -1)

    def step(self, z, uk, w, rn_k, rn_next, k, mats, du=None, uc_next=None, dk=None, dq=None):
        """z: lifted state (B,nz); returns next lifted state"""
        ones = z.new_ones(z.shape[0], 1)
        extra = [uk, ones] + ([w] if w is not None else [])
        if self.relift_every == 1:
            Aq, G, Gs = mats[:3]
            q = torch.bmm(Aq, z[:, :5].unsqueeze(-1)).squeeze(-1) + \
                torch.bmm(G, torch.cat([z[:, 5:]] + extra, -1).unsqueeze(-1)).squeeze(-1)
            if Gs is not None and du is not None:
                q = q + torch.bmm(Gs, du.unsqueeze(-1)).squeeze(-1)
            if dk is not None:
                q = q + torch.bmm(mats[3], dk.unsqueeze(-1)).squeeze(-1)
            if dq is not None:
                q = q + dq
            q = torch.nan_to_num(q, nan=1e3).clamp(-1e4, 1e4)
            return self.lift(q, rn_next, uc_next)
        A, G = mats[:2]
        z = torch.bmm(A, z.unsqueeze(-1)).squeeze(-1) + torch.bmm(G, torch.cat(extra, -1).unsqueeze(-1)).squeeze(-1)
        if self.relift_every and (k + 1) % self.relift_every == 0:
            z = self.lift(z[:, :5], rn_next, uc_next)
        return torch.nan_to_num(z, nan=1e3).clamp(-1e4, 1e4)

    @staticmethod
    def slope(u, j):
        """torque change per step around sample j from the interval averages (central difference)"""
        n = u.shape[1]
        a, b = max(j - 1, 0), min(j + 1, n - 1)
        return (u[:, b] - u[:, a]) / max(b - a, 1)

    def mats_at(self, rn, k, eta=None):
        if self.relift_every == 1:
            return self.disc_q(rn, eta)
        return self.disc_full(rn)

    # ================================================================== observer
    def observe(self, Q, u, r):
        """Gauss-Newton moving-horizon estimate of the lumped additive input w (5), the actuator
        effectiveness change theta (3) and the physical-row scales eta (5) from the history window.
        Returns (w, theta, eta); unused parts are None."""
        if not self.use_obs:
            return None, None, None
        B, Hp, _ = Q.shape
        rw = nn.functional.softplus(self.obs_row) ** 2
        nw = self.n_w
        nth = 3 if self.use_fault else 0
        ne = 5 if (self.use_eta and self.relift_every == 1) else 0
        nx = nw + nth + ne
        no, nq = self.n_obs, self.n_q
        xi = Q.new_zeros(B, nx)
        ones = Q.new_ones(B, 1)
        ll = self.log_lam.exp()
        lam = torch.cat([ll[0].expand(nw), ll[1].expand(nth), ll[2].expand(ne)]).unsqueeze(0).expand(B, -1)
        best_xi, best_J, dg_ref = None, None, None
        # Gauss-Newton with safeguard: every iterate is re-simulated over the history and the one with
        # the lowest regularised residual (including xi = 0, the physics-only prediction) is returned
        for it in range(self.gn_iters + 1):
            last = it == self.gn_iters
            w = xi[:, :nw]
            th = xi[:, nw:nw + nth] if nth else None
            eta = xi[:, nw + nth:] if ne else None
            z = self.lift(Q[:, 0], r[:, 0], self.ctx(u, 0))
            S = Q.new_zeros(B, 5, nx)
            G_ = Q.new_zeros(B, nx, nx)
            h = Q.new_zeros(B, nx)
            J = Q.new_zeros(B)
            for k in range(Hp - 1):
                uk = u[:, k]
                ubar = uk + self.u_off
                if nth:
                    uk = uk + th * ubar
                du = self.slope(u, k)
                if nth:
                    du = du * (1 + th)
                if self.relift_every == 1:
                    Aq, Gm, Gs, Phi = self.disc_q(r[:, k], eta)
                    if not last:
                        dS = torch.bmm(Aq, S)
                        if self.quad:
                            dS = dS + torch.bmm(Gm[:, :, :nq], self.jac_quad(z[:, :5], S))
                        add = [Gm[:, :, no + 4:no + 4 + nw]]
                        if nth:
                            add.append(Gm[:, :, no:no + 3] * ubar.unsqueeze(1))
                        if ne:
                            P = self.p_full(r[:, k])[:, :5, :self.nz + 4]
                            f = torch.bmm(P, torch.cat([z, uk, ones], -1).unsqueeze(-1)).squeeze(-1)
                            C = self.compose(r[:, k])[:, :, :5]
                            add.append(torch.bmm(Phi, C * f.unsqueeze(1)))
                        S = dS + torch.cat(add, -1)
                    z = self.step(z, uk, w, r[:, k], r[:, k + 1], k, (Aq, Gm, Gs), du, self.ctx(u, k + 1))
                else:
                    A, Gm = self.disc_full(r[:, k])
                    if not last:
                        add = [Gm[:, :5, 4:4 + nw]]
                        if nth:
                            add.append(Gm[:, :5, :3] * ubar.unsqueeze(1))
                        S = torch.bmm(A[:, :5, :5], S) + torch.cat(add, -1)
                    z = self.step(z, uk, w, r[:, k], r[:, k + 1], k, (A, Gm), None, self.ctx(u, k + 1))
                e = Q[:, k + 1] - z[:, :5]
                wk = self.gamma ** (Hp - 2 - k)
                J = J + wk * (rw * e * e).sum(-1)
                if not last:
                    G_ = G_ + wk * torch.einsum("r,bri,brj->bij", rw, S, S)
                    h = h + wk * torch.einsum("r,bri,br->bi", rw, S, e)
            if dg_ref is None:
                dg_ref = G_.diagonal(dim1=1, dim2=2).detach() + 1e-30
            J = J + (lam * dg_ref * xi * xi).sum(-1)
            J = torch.nan_to_num(J, nan=1e30)
            if best_J is None:
                best_J, best_xi = J, xi
                J0 = J
            else:
                # accept an estimate only if it explains a substantial part of the history misfit
                # (evidence of unknown inputs); otherwise keep the nominal physics (xi = 0)
                better = ((J < best_J) & (J < self.accept * J0)).unsqueeze(-1)
                best_xi = torch.where(better, xi, best_xi)
                best_J = torch.minimum(J, best_J)
            if last:
                break
            A_ = G_ + torch.diag_embed(lam * dg_ref)
            xi = xi + torch.linalg.solve(A_, (h - lam * dg_ref * xi).unsqueeze(-1)).squeeze(-1)
            parts = [xi[:, :nw]]
            if nth:
                parts.append(xi[:, nw:nw + nth].clamp(-0.7, 0.7))
            if ne:
                parts.append(xi[:, nw + nth:].clamp(-0.4, 0.4))
            xi = torch.nan_to_num(torch.cat(parts, 1))
        xi = best_xi
        return xi[:, :nw], (xi[:, nw:nw + nth] if nth else None), (xi[:, nw + nth:] if ne else None)

    # ================================================================== rollout
    def rollout(self, s_hist, u, r, H, return_aux=False):
        Hp = s_hist.shape[1]
        Q = self.to_q(s_hist, r[:, :Hp])
        w, th, eta = self.observe(Q, u[:, :Hp], r[:, :Hp])
        z = self.lift(Q[:, -1], r[:, Hp - 1], self.ctx(u, Hp - 1))
        dseq = self.forecast_inputs(u, Q, w, th, eta) if self.use_df else None
        out = []
        re = 1 if self.relift_every == 1 else self.resched_every
        mats = None
        for k in range(H):
            j = Hp - 1 + k
            if mats is None or k % re == 0:
                mats = self.mats_at(r[:, j], k, eta)
            uk = u[:, j] if th is None else u[:, j] + th * (u[:, j] + self.u_off)
            du = self.slope(u, j) if th is None else self.slope(u, j) * (1 + th)
            wk = None if w is None else w * torch.exp(-nn.functional.softplus(self.w_decay_raw) * (k + 1))
            dk = dseq[:, j] if (dseq is not None and self.relift_every == 1) else None
            z = self.step(z, uk, wk, r[:, j], r[:, j + 1], k, mats, du, self.ctx(u, j + 1), dk)
            out.append(self.from_q(z[:, :5], r[:, j + 1]))
        out = torch.stack(out, 1)
        if return_aux:
            return out, dict(w=w, th=th, eta=eta)
        return out

    @torch.no_grad()
    def phys_mats(self, b, H):
        """Per-step Koopman/observer operators for the corrector loop (no gradient; the identified core
        is fixed):  q_{k+1} = Aq_k q_k + Gm_k quad(q_k) + Gu_k u~_k + c_k  (c_k includes the estimated
        lumped input).  Returns dict of float32 tensors (+ float64 initial state)."""
        assert self.relift_every == 1 and self.n_l == 0
        Hp = b.Hp
        s_hist, u, r = b.s64[:, :Hp], b.u64[:, :Hp + H], b.r64[:, :Hp + H]
        Q = self.to_q(s_hist, r[:, :Hp])
        w, th, eta = self.observe(Q, u[:, :Hp], r[:, :Hp])
        B = Q.shape[0]
        no = self.n_obs
        Aq, Gm, Gu, c, ue = [], [], [], [], []
        for k in range(H):
            j = Hp - 1 + k
            A_, G_, _, _ = self.disc_q(r[:, j], eta)
            uk = u[:, j] if th is None else u[:, j] + th * (u[:, j] + self.u_off)
            ck = G_[:, :, no + 3]
            if w is not None:
                wk = w * torch.exp(-nn.functional.softplus(self.w_decay_raw) * (k + 1))
                ck = ck + torch.bmm(G_[:, :, no + 4:no + 4 + self.n_w], wk.unsqueeze(-1)).squeeze(-1)
            Aq.append(A_)
            Gm.append(G_[:, :, :no])
            Gu.append(G_[:, :, no:no + 3])
            c.append(ck)
            ue.append(uk)
        est = torch.cat([v if v is not None else Q.new_zeros(B, n) for v, n in ((w, 5), (th, 3), (eta, 5))], -1)
        st = lambda L: torch.stack(L, 1).float()
        return dict(q0=Q[:, -1], Qh=Q, Aq=st(Aq), Gm=st(Gm), Gu=st(Gu), c=st(c), ue=st(ue), est=est.float())

    def koopman_from_mats(self, P, r64, Hp, H):
        """pure Koopman prediction from cached operators (used when the corrector is off)"""
        q = P["q0"]
        out = []
        for k in range(H):
            q = self._kstep(q, P, k)
            out.append(self.from_q(q, r64[:, Hp + k]))
        return torch.stack(out, 1)

    def _kstep(self, q, P, k):
        quad = q[:, self.iu0] * q[:, self.iu1]
        qn = torch.bmm(P["Aq"][:, k].double(), q.unsqueeze(-1)).squeeze(-1) \
            + torch.bmm(P["Gm"][:, k].double(), quad.unsqueeze(-1)).squeeze(-1) \
            + torch.bmm(P["Gu"][:, k].double(), P["ue"][:, k].double().unsqueeze(-1)).squeeze(-1) + P["c"][:, k].double()
        return torch.nan_to_num(qn, nan=1e3).clamp(-1e4, 1e4)

    def get_mats(self, b, H):
        P = getattr(b, "piko_P", None)
        if P is not None and P["Aq"].shape[1] >= H:
            return P
        return self.phys_mats(b, H)

    def forward_batch(self, b, H, return_aux=False):
        use_corr = self.innov and not getattr(self, "innov_off", False)
        if self.core_trainable() or (not use_corr and getattr(b, "piko_P", None) is None):
            Hp = b.Hp
            res = self.rollout(b.s64[:, :Hp], b.u64[:, :Hp + H], b.r64[:, :Hp + H], H, return_aux)
            out, aux = (res[0].float(), res[1]) if return_aux else (res.float(), None)
            return (out, aux) if return_aux else out
        P = self.get_mats(b, H)
        if use_corr:
            out = self.corr(self, b, P, H).float()
        else:
            out = self.koopman_from_mats(P, b.r64, b.Hp - 1 + 1, H).float()
        aux = dict(w=None, th=None, eta=None)
        return (out, aux) if return_aux else out

    def core_trainable(self):
        return getattr(self, "train_core", False)

    @torch.no_grad()
    def cache_physics(self, b, H, chunk=512):
        """precompute the (fixed) per-step Koopman/observer operators of a batch so that the
        corrector loop trains at recurrent-network speed"""
        parts = []
        for i in range(0, b.n, chunk):
            parts.append(self.phys_mats(b.sel(torch.arange(i, min(i + chunk, b.n))), H))
        b.piko_P = {k: torch.cat([p[k] for p in parts]) for k in parts[0]}
        b.extras = tuple(set(getattr(b, "extras", ())) | {"piko_P"})

    def forward(self, s_hist, u, r, H):
        return self.rollout(s_hist.double(), u.double(), r.double(), H).float()

    # ================================================================== physics loss
    def p_model(self, s, ui, r, w=None, th=None):
        """physical-row rates predicted by the generator (B,L,7), in label units"""
        Z = self.lift(self.to_q(s, r), r)
        if th is not None:
            ui = ui + th.unsqueeze(1) * (ui + self.u_off)
        Bsz, L = s.shape[:2]
        rf = r.reshape(-1, 2)
        Lp, Bp, cp = self.p_blocks(rf)
        f = torch.bmm(Lp, Z.reshape(-1, self.nz, 1)).squeeze(-1) + torch.bmm(Bp, ui.reshape(-1, 3, 1)).squeeze(-1) + cp
        f = f.view(Bsz, L, 7)
        if w is not None:
            f = f + (self.Ep[:, :self.n_w] @ w.unsqueeze(-1)).squeeze(-1).unsqueeze(1)
        return f / self.dt

    def aux_loss(self, b, H, aux, lam_phys=0.1, **kw):
        if self.no_pinn or lam_phys == 0 or not self.core_trainable():
            return 0.0
        import numpy as _np
        m = torch.as_tensor(_np.isin(b.fam, self.pinn_fams))
        if not m.any():
            return 0.0
        L = b.Hp + H
        s, r = b.s64[m, :L], b.r64[m, :L]
        Z = self.lift(self.to_q(s, r), r, self.ctx_seq(b.u64[m, :L]))
        Bn, Ln = s.shape[:2]
        rf = r.reshape(-1, 2)
        F = self.q_blocks(rf)
        X = torch.cat([Z.reshape(-1, self.nz), b.ui64[m, :L].reshape(-1, 3), Z.new_ones(Bn * Ln, 1)], -1)
        f = torch.bmm(F[:, :, :self.nz + 4], X.unsqueeze(-1)).squeeze(-1) / self.dt
        c = b.sel(torch.where(m)[0])
        y = self.p_to_qdot(c)[:, :L].reshape(-1, 5)
        return lam_phys * (((f - y) / self.qd_sd) ** 2).mean() * m.float().mean()

    # ================================================================== gEDMD initialisation
    @torch.no_grad()
    def fit_norm(self, b):
        self.q_mu.zero_()
        self.q_sd.fill_(1.0)
        q = self.to_q(b.s64.reshape(-1, 5), b.r64.reshape(-1, 2))
        self.q_mu.copy_(q.mean(0))
        self.q_sd.copy_(q.std(0) + 1e-15)
        y = self.p_labels(b.s64, b.sdot64, b.r64, b.rd64).reshape(-1, 7)
        self.p_sd.copy_(y.std(0) + 1e-15)
        if self.innov:
            qq = self.to_q(b.s64, b.r64)
            self.corr.dq_sd.copy_((qq[:, 1:] - qq[:, :-1]).reshape(-1, 5).std(0) + 1e-12)
        self.qd_sd.copy_(self.p_to_qdot(b).reshape(-1, 5).std(0) + 1e-15)

    @torch.no_grad()
    def init_gedmd(self, b, lam=1e-10, b_norm=None):
        """generator-EDMD on exact derivative labels, one least-squares problem per physical row with
        the regressors allowed by the structural mask: [psi_known (x) sigma, u_inst (x) sigma, sigma]"""
        self.fit_norm(b_norm if b_norm is not None else b)
        nk, n_s = self.n_known, self.n_s

        def design(c):
            r = c.r64.reshape(-1, 2)
            q = self.to_q(c.s64.reshape(-1, 5), r)
            zk = self.lift(q, r)[:, :nk]
            w = self.sig(r)
            X = torch.cat([(zk.unsqueeze(-1) * w.unsqueeze(1)).flatten(1),
                           (c.ui64.reshape(-1, 3).unsqueeze(-1) * w.unsqueeze(1)).flatten(1), w], 1)
            Y = self.p_labels(c.s64, c.sdot64, c.r64, c.rd64).reshape(-1, 7) * self.dt
            Yo = None
            if self.relift_every != 1 and self.n_q:
                qd = self.p_to_qdot(c).reshape(-1, 5)
                Yo = (qd[:, self.iu0] * q[:, self.iu1] + q[:, self.iu0] * qd[:, self.iu1]) * self.dt
            return X, Y, Yo

        X0, _, _ = design(b.sel(torch.arange(min(b.n, 512))))
        sc = X0.std(0) + 1e-12
        sc[nk * n_s + 3 * n_s] = X0[:, nk * n_s + 3 * n_s].abs().mean()      # constant column: rms instead of std
        ne, neo = NormalEq(), NormalEq()
        for c in _chunks(b, 256):
            X, Y, Yo = design(c)
            ne.add(X / sc, Y)
            if Yo is not None:
                neo.add(X / sc, Yo)
        iS, iU = nk * n_s, 3 * n_s
        Wt = torch.zeros(X0.shape[1], 7, dtype=F64)
        for i in range(7):
            # design column order: z-block (d, k) -> d*n_s + k ; u-block (j, k) ; const-block k
            allowed = torch.cat([self.dmask[:, i, :nk].T.reshape(-1), self.bmask[:, i, :].T.reshape(-1), self.cmask[:, i]])
            idx = torch.where(allowed > 0)[0]
            if len(idx):
                Wt[idx, i] = ne.solve(lam, idx=idx, col=i)
        Wt = Wt / sc[:, None]
        self.Lp.zero_()
        self.Bp.zero_()
        self.cp.zero_()
        self.Lp[:, :, :nk] = Wt[:iS].view(nk, n_s, 7).permute(1, 2, 0)
        self.Bp[:] = Wt[iS:iS + iU].view(3, n_s, 7).permute(1, 2, 0)
        self.cp[:] = Wt[iS + iU:].view(n_s, 7)
        if self.relift_every != 1:
            self.Lo.zero_()
            self.Bo.zero_()
            self.co.zero_()
            if self.n_q:
                Wo = neo.solve(lam) / sc[:, None]
                nq = self.n_q
                self.Lo[:, :nq, :nk] = Wo[:iS].view(nk, n_s, nq).permute(1, 2, 0)
                self.Bo[:, :nq] = Wo[iS:iS + iU].view(3, n_s, nq).permute(1, 2, 0)
                self.co[:, :nq] = Wo[iS + iU:].view(n_s, nq)
            if self.n_l:
                self.Lo[0, self.n_q:, nk:] = -0.01 * torch.eye(self.n_l, dtype=F64)

    def p_to_qdot(self, c):
        y = self.p_labels(c.s64, c.sdot64, c.r64, c.rd64)                   # (B,L,7)
        B, L = y.shape[:2]
        C = self.compose(c.r64.reshape(-1, 2))
        return torch.bmm(C, y.reshape(-1, 7, 1)).view(B, L, 5)

    def param_groups(self, lr):
        if not self.core_trainable():
            return [dict(params=[p for p in self.corr.parameters()], lr=lr)] if self.innov else [dict(params=[self.cq], lr=0.0)]
        named = dict(self.named_parameters())
        slow = [named[n] for n in ("Lq", "Bq", "cq", "Lo", "Bo", "co") if n in named and named[n].requires_grad]
        hyp = [named[n] for n in ("obs_row", "log_lam", "w_decay_raw", "damp_raw") if n in named]
        ids = {id(p) for p in slow + hyp}
        rest = [p for p in self.parameters() if id(p) not in ids and p.requires_grad]
        return [dict(params=slow, lr=lr * 0.05), dict(params=hyp, lr=lr * 10), dict(params=rest, lr=lr)]


class InnovationCorrector(nn.Module):
    """Closed-loop innovation network inside the Koopman recursion.

    In logged closed-loop data the actuator commands are a (noisy) measurement of the state through
    the unknown control law and they react to disturbances/faults the physics cannot see.  At every
    step the Koopman operator proposes the physics increment dq_k = K(q_k) - q_k; a GRU (warmed up on
    the observed history) reads the current state, dq_k, the torque stream and the observer
    estimates, and returns a trust gate g_k in (0,2) and an innovation (learned unknown input) d_k:
        q_{k+1} = q_k + g_k * dq_k + d_k
    Zero-initialised (g = 1, d = 0) it *is* the Koopman predictor; the state stays in the physics
    coordinates, so the physics keeps acting on the corrected state at every step.  Switched off
    (innov_off) for counterfactual prediction and MPC, where torques are decisions."""

    def __init__(self, nm, hidden=128, gate_T_logit=0.0, bidir=0):
        super().__init__()
        from .data import Feat
        self.F = Feat(nm)
        # optional backward GRU over the torque sequence: the innovation at step k may also read the
        # torques applied after k (in closed loop they reveal the state at k; "smoothing" instead of
        # "filtering").  Off by default.
        self.bidir = bidir
        if bidir:
            self.bgru = nn.GRU(6, bidir, batch_first=True)
        nin = Feat.N_IN + 5 + 3 + 13 + bidir
        self.cell = nn.GRUCell(nin, hidden)
        self.head = nn.Sequential(nn.Linear(hidden + nin, 128), nn.SiLU(), nn.Linear(128, 10))
        self.register_buffer("dq_sd", torch.ones(5, dtype=F64))
        with torch.no_grad():
            self.head[-1].weight.zero_()
            self.head[-1].bias.zero_()
            self.head[-1].bias[:2] = gate_T_logit

    def forward(self, pk, b, P, H):
        Hp = b.Hp
        s, u, r, r64 = b.s, b.u, b.r, b.r64
        est = torch.asinh(P["est"] * 10.0)
        Qh = P["Qh"]
        B = s.shape[0]
        L = Hp + H
        uu = u[:, :L]
        ctx = None
        if self.bidir:
            dU = uu - torch.cat([uu[:, :1], uu[:, :-1]], 1)
            ctx = self.bgru(torch.flip(torch.cat([uu, dU], -1), [1]))[0].flip(1)       # (B, L, bidir)
        ex = (lambda j: [ctx[:, j]]) if ctx is not None else (lambda j: [])
        h = s.new_zeros(B, self.cell.hidden_size)
        for k in range(Hp - 1):                       # warm-up on the observed history
            du = u[:, k] - u[:, max(k - 1, 0)]
            dqh = ((Qh[:, k + 1] - Qh[:, k]) / self.dq_sd).float()
            h = self.cell(torch.cat([self.F.step_input(s[:, k], u[:, k], r[:, k]), torch.asinh(dqh), du, est] + ex(k), -1), h)
        q = P["q0"]
        sh = s[:, Hp - 1]
        out = []
        for k in range(H):
            j = Hp - 1 + k
            dq = pk._kstep(q, P, k) - q
            du = u[:, j] - u[:, max(j - 1, 0)]
            x = torch.cat([self.F.step_input(sh, u[:, j], r[:, j]), torch.asinh((dq / self.dq_sd).float()), du, est] + ex(j), -1)
            h = self.cell(x, h)
            o = self.head(torch.cat([h, x], -1)).double()
            q = q + 2 * torch.sigmoid(o[:, :5]) * dq + o[:, 5:] * self.dq_sd
            q = torch.nan_to_num(q, nan=1e3).clamp(-1e4, 1e4)
            s64 = pk.from_q(q, r64[:, j + 1])
            sh = s64.float()
            out.append(s64)
        return torch.stack(out, 1)
