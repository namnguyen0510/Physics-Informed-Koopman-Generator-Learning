"""
bench/models.py -- baseline architectures for multi-step R2R state prediction.

Every model maps (s_hist[B,Hp,5], u[B,Hp+H,3], rho[B,Hp+H,2], H) -> s_pred[B,H,5] in
standardised units (see data.py).  Transition k -> k+1 uses u[k], rho[k].

Autoregressive (AR) models predict normalised one-step increments scaled by the
train-set increment std, s_{k+1} = s_k + ds_sd * head(.), and are rolled out on
their own predictions.  Sequence-to-sequence models (MLP-Direct, TCN, Transformer)
predict a fixed block of H_block steps; longer horizons are produced block-wise
using their own predictions as new history.

Classical (closed-form) baselines: DMDc, EDMDc (random-Fourier dictionary),
SINDYc (STLSQ on exact derivatives), ESN (reservoir + ridge readout).
"""
from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn

from .data import Feat, PHYS_NAMES


def mlp(n_in, n_out, width=128, depth=2, act=nn.SiLU):
    layers, d = [], n_in
    for _ in range(depth):
        layers += [nn.Linear(d, width), act()]
        d = width
    layers.append(nn.Linear(d, n_out))
    return nn.Sequential(*layers)


class Base(nn.Module):
    name = "base"
    kind = "ar"          # "ar" | "block" | "closed"
    uses_sdot = False    # trained with derivative labels?
    H_block = None

    def __init__(self, nm):
        super().__init__()
        self.nm = nm
        self.F = Feat(nm)
        self.register_buffer("ds_sd", nm.t("ds_sd"))

    # default AR rollout for Markov one-step models ------------------------------------
    def step(self, s, u, r, state):
        raise NotImplementedError

    def init_state(self, s_hist, u, r):
        return None

    def forward(self, s_hist, u, r, H):
        Hp = s_hist.shape[1]
        state = self.init_state(s_hist, u, r)
        s = s_hist[:, -1]
        out = []
        for k in range(H):
            j = Hp - 1 + k
            s, state = self.step(s, u[:, j], r[:, j], state)
            out.append(s)
        return torch.stack(out, 1)


# ============================================================================ Markov MLP
class MLPAR(Base):
    name = "MLP-AR"

    def __init__(self, nm, width=160, depth=3):
        super().__init__(nm)
        self.net = mlp(Feat.N_IN, 5, width, depth)

    def step(self, s, u, r, state):
        return s + self.ds_sd * self.net(self.F.step_input(s, u, r)), state


class NODE(Base):
    """continuous-time neural ODE ds/dt = f(s, u, rho), RK4 with n_sub substeps"""
    name = "NeuralODE"

    def __init__(self, nm, width=160, depth=3, n_sub=2):
        super().__init__(nm)
        self.net = mlp(Feat.N_IN, 5, width, depth)
        self.register_buffer("sdot_sd", nm.t("sdot_sd"))
        self.dt = nm.dt
        self.n_sub = n_sub

    def f(self, s, u, r):
        return self.sdot_sd * self.net(self.F.step_input(s, u, r))

    def step(self, s, u, r, state):
        h = self.dt / self.n_sub
        for _ in range(self.n_sub):
            k1 = self.f(s, u, r)
            k2 = self.f(s + 0.5 * h * k1, u, r)
            k3 = self.f(s + 0.5 * h * k2, u, r)
            k4 = self.f(s + h * k3, u, r)
            s = s + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        return s, state


# ============================================================================ recurrent
class RecurrentAR(Base):
    """encoder-decoder recurrent model: warm-up on history (teacher forcing), then AR."""

    def __init__(self, nm, cell="lstm", hidden=128):
        super().__init__(nm)
        R = {"lstm": nn.LSTM, "gru": nn.GRU, "rnn": nn.RNN}[cell]
        self.rnn = R(Feat.N_IN, hidden, batch_first=True)
        self.head = mlp(hidden + Feat.N_IN, 5, 128, 1)
        self.name = {"lstm": "LSTM", "gru": "GRU", "rnn": "RNN"}[cell]

    def forward(self, s_hist, u, r, H):
        Hp = s_hist.shape[1]
        xh = self.F.step_input(s_hist[:, :-1], u[:, :Hp - 1], r[:, :Hp - 1])
        _, st = self.rnn(xh)
        s = s_hist[:, -1]
        out = []
        for k in range(H):
            j = Hp - 1 + k
            x = self.F.step_input(s, u[:, j], r[:, j])
            o, st = self.rnn(x.unsqueeze(1), st)
            s = s + self.ds_sd * self.head(torch.cat([o[:, 0], x], -1))
            out.append(s)
        return torch.stack(out, 1)


class LRU(Base):
    """Linear Recurrent Unit (Orvieto et al. 2023): diagonal complex linear recurrence +
    nonlinear readout; a modern state-space-model (S4/S5-family) baseline."""
    name = "LRU-SSM"

    def __init__(self, nm, N=96, r_min=0.5, r_max=0.999):
        super().__init__(nm)
        u1, u2 = torch.rand(N), torch.rand(N)
        self.nu_log = nn.Parameter(torch.log(-0.5 * torch.log(u1 * (r_max ** 2 - r_min ** 2) + r_min ** 2)))
        self.theta_log = nn.Parameter(torch.log(u2 * math.pi * 2))
        self.B_re = nn.Parameter(torch.randn(N, Feat.N_IN) / math.sqrt(2 * Feat.N_IN))
        self.B_im = nn.Parameter(torch.randn(N, Feat.N_IN) / math.sqrt(2 * Feat.N_IN))
        self.head = mlp(2 * N + Feat.N_IN, 5, 128, 2)
        self.N = N

    def lam(self):
        mod = torch.exp(-torch.exp(self.nu_log))
        th = torch.exp(self.theta_log)
        gamma = torch.sqrt(1 - mod ** 2)
        return mod * torch.cos(th), mod * torch.sin(th), gamma

    def rec(self, hr, hi, x, lr, li, g):
        bu_r, bu_i = x @ self.B_re.T, x @ self.B_im.T
        return lr * hr - li * hi + g * bu_r, lr * hi + li * hr + g * bu_i

    def forward(self, s_hist, u, r, H):
        Hp = s_hist.shape[1]
        B = s_hist.shape[0]
        lr, li, g = self.lam()
        hr = s_hist.new_zeros(B, self.N)
        hi = s_hist.new_zeros(B, self.N)
        for j in range(Hp - 1):
            hr, hi = self.rec(hr, hi, self.F.step_input(s_hist[:, j], u[:, j], r[:, j]), lr, li, g)
        s = s_hist[:, -1]
        out = []
        for k in range(H):
            j = Hp - 1 + k
            x = self.F.step_input(s, u[:, j], r[:, j])
            hr, hi = self.rec(hr, hi, x, lr, li, g)
            s = s + self.ds_sd * self.head(torch.cat([hr, hi, x], -1))
            out.append(s)
        return torch.stack(out, 1)


# ============================================================================ block (seq2seq) models
class BlockModel(Base):
    kind = "block"

    def forward(self, s_hist, u, r, H):
        """block-autoregressive for H > H_block"""
        Hp = s_hist.shape[1]
        outs, sh = [], s_hist
        done = 0
        while done < H:
            j0 = done                      # offset of the current history window inside u/r
            ub, rb = u[:, j0:j0 + Hp + self.H_block], r[:, j0:j0 + Hp + self.H_block]
            if ub.shape[1] < Hp + self.H_block:        # pad the tail with the last input
                pad = Hp + self.H_block - ub.shape[1]
                ub = torch.cat([ub, ub[:, -1:].expand(-1, pad, -1)], 1)
                rb = torch.cat([rb, rb[:, -1:].expand(-1, pad, -1)], 1)
            pb = self.block(sh, ub, rb)
            outs.append(pb)
            done += self.H_block
            sh = torch.cat([sh, pb], 1)[:, -Hp:]
        return torch.cat(outs, 1)[:, :H]


class MLPDirect(BlockModel):
    name = "MLP-Direct"

    def __init__(self, nm, Hp, H_block, width=256):
        super().__init__(nm)
        self.H_block = H_block
        n_in = Hp * Feat.N_IN + H_block * 5
        self.net = mlp(n_in, H_block * 5, width, 2)

    def block(self, sh, ub, rb):
        Hp = sh.shape[1]
        xh = self.F.step_input(sh, ub[:, :Hp], rb[:, :Hp]).flatten(1)
        xf = torch.cat([ub[:, Hp - 1:Hp - 1 + self.H_block], rb[:, Hp - 1:Hp - 1 + self.H_block]], -1).flatten(1)
        d = self.net(torch.cat([xh, xf], 1)).view(-1, self.H_block, 5)
        return sh[:, -1:] + torch.cumsum(d * self.ds_sd, 1)


def _tokens(F, sh, ub, rb, H_block):
    """token sequence over history+block: [s*m, phys*m, m, u, rho] ; m=1 on observed states"""
    B, Hp, _ = sh.shape
    L = Hp + H_block
    s_full = torch.cat([sh, sh[:, -1:].expand(-1, H_block, -1)], 1)
    m = torch.cat([sh.new_ones(B, Hp, 1), sh.new_zeros(B, H_block, 1)], 1)
    ph = F.phys(s_full, rb[:, :L])
    return torch.cat([s_full * m, ph * m, m, ub[:, :L], rb[:, :L]], -1)


class CausalConv(nn.Module):
    def __init__(self, c, dil, k=3):
        super().__init__()
        self.pad = (k - 1) * dil
        self.conv = nn.Conv1d(c, c, k, dilation=dil)
        self.act = nn.SiLU()
        self.norm = nn.GroupNorm(1, c)

    def forward(self, x):
        y = self.conv(nn.functional.pad(x, (self.pad, 0)))
        return x + self.act(self.norm(y))


class TCN(BlockModel):
    name = "TCN"

    def __init__(self, nm, H_block, ch=64, dils=(1, 2, 4, 8, 16, 32)):
        super().__init__(nm)
        self.H_block = H_block
        n_tok = 5 + len(PHYS_NAMES) + 1 + 3 + 2
        self.inp = nn.Conv1d(n_tok, ch, 1)
        self.blocks = nn.Sequential(*[CausalConv(ch, d) for d in dils])
        self.out = nn.Conv1d(ch, 5, 1)

    def block(self, sh, ub, rb):
        Hp = sh.shape[1]
        x = _tokens(self.F, sh, ub, rb, self.H_block).transpose(1, 2)
        y = self.out(self.blocks(self.inp(x))).transpose(1, 2)        # (B, L, 5)
        d = y[:, Hp - 1:Hp - 1 + self.H_block]                          # position t predicts s_{t+1}
        return sh[:, -1:] + torch.cumsum(d * self.ds_sd, 1)


class TransformerM(BlockModel):
    name = "Transformer"

    def __init__(self, nm, Hp, H_block, d=64, heads=4, layers=3):
        super().__init__(nm)
        self.H_block = H_block
        n_tok = 5 + len(PHYS_NAMES) + 1 + 3 + 2
        self.inp = nn.Linear(n_tok, d)
        self.pos = nn.Parameter(torch.randn(1, Hp + H_block, d) * 0.02)
        el = nn.TransformerEncoderLayer(d, heads, 4 * d, dropout=0.0, batch_first=True, norm_first=True,
                                        activation="gelu")
        self.enc = nn.TransformerEncoder(el, layers)
        self.out = nn.Linear(d, 5)
        L = Hp + H_block
        self.register_buffer("mask", torch.triu(torch.full((L, L), float("-inf")), 1))

    def block(self, sh, ub, rb):
        Hp = sh.shape[1]
        x = self.inp(_tokens(self.F, sh, ub, rb, self.H_block)) + self.pos
        y = self.out(self.enc(x, mask=self.mask))
        d = y[:, Hp - 1:Hp - 1 + self.H_block]
        return sh[:, -1:] + torch.cumsum(d * self.ds_sd, 1)


# ============================================================================ Deep Koopman (Lusch et al. 2018, with control)
class DeepKoopman(Base):
    """autoencoder Koopman with control: z = enc(s, rho); z+ = K z + B u; s = dec(z).
    Linear latent rollout (no re-encoding); Markov (no history)."""
    name = "DeepKoopman"

    def __init__(self, nm, nz=32, width=128):
        super().__init__(nm)
        self.enc = mlp(5 + 2, nz, width, 2)
        self.dec = mlp(nz, 5, width, 2)
        self.K = nn.Parameter(torch.eye(nz) + 0.01 * torch.randn(nz, nz))
        self.B = nn.Parameter(0.01 * torch.randn(nz, 3 + 2 + 1))

    def encode(self, s, r):
        return self.enc(torch.cat([s, r], -1))

    def forward(self, s_hist, u, r, H, return_latent=False):
        Hp = s_hist.shape[1]
        z = self.encode(s_hist[:, -1], r[:, Hp - 1])
        out, zs = [], []
        for k in range(H):
            j = Hp - 1 + k
            uin = torch.cat([u[:, j], r[:, j], torch.ones_like(u[:, j, :1])], -1)
            z = z @ self.K.T + uin @ self.B.T
            zs.append(z)
            out.append(self.dec(z))
        out = torch.stack(out, 1)
        if return_latent:
            return out, torch.stack(zs, 1)
        return out

    def aux_loss(self, b, H, pred_latent):
        """reconstruction + latent consistency (Lusch et al.)"""
        Hp = b.Hp
        s_all, r_all = b.s[:, :Hp + H], b.r[:, :Hp + H]
        z_true = self.encode(s_all, r_all)
        rec = ((self.dec(z_true) - s_all) ** 2).mean()
        lat = ((pred_latent - z_true[:, Hp:Hp + H]) ** 2).mean()
        return rec + 0.1 * lat


# ============================================================================ closed-form baselines
class NormalEq:
    """accumulate X^T X and X^T Y in float64 chunk by chunk (memory-safe least squares)"""

    def __init__(self):
        self.G = None

    def add(self, X, Y):
        X, Y = X.double(), Y.double()
        if self.G is None:
            self.G, self.h, self.n = X.T @ X, X.T @ Y, X.shape[0]
        else:
            self.G += X.T @ X
            self.h += X.T @ Y
            self.n += X.shape[0]

    def solve(self, lam=1e-8, idx=None, col=None):
        G, h = self.G, self.h
        if idx is not None:
            G = G[idx][:, idx]
            h = h[idx] if col is None else h[idx, col]
        elif col is not None:
            h = h[:, col]
        reg = lam * torch.trace(G) / G.shape[0]
        return torch.linalg.solve(G + reg * torch.eye(G.shape[0], dtype=G.dtype), h)


def _chunks(b, n=512):
    for i in range(0, b.n, n):
        yield b.sel(torch.arange(i, min(i + n, b.n)))


class ClosedForm(Base):
    kind = "closed"

    def fit_windows(self, b):
        raise NotImplementedError


class DMDc(ClosedForm):
    """linear model s+ - s = A s + B u + C rho + c (least squares) -- classical DMD with control"""
    name = "DMDc"

    def fit_windows(self, b):
        ne = NormalEq()
        for c in _chunks(b):
            s, u, r = c.s[:, :-1].reshape(-1, 5), c.u[:, :-1].reshape(-1, 3), c.r[:, :-1].reshape(-1, 2)
            ds = (c.s[:, 1:] - c.s[:, :-1]).reshape(-1, 5)
            ne.add(torch.cat([s, u, r, torch.ones_like(s[:, :1])], 1), ds)
        self.W = ne.solve(1e-9).float()

    def step(self, s, u, r, state):
        X = torch.cat([s, u, r, torch.ones_like(s[:, :1])], 1)
        return s + X @ self.W, state


class EDMDc(ClosedForm):
    """EDMD with control: psi = [s, phys(s), RFF(s, rho)], psi+ = K psi + B u + c; linear lifted
    rollout, state read from the first 5 lifted coordinates (classical Koopman baseline)."""
    name = "EDMDc-RFF"

    def __init__(self, nm, D=200, sigma=1.0, seed=0):
        super().__init__(nm)
        g = torch.Generator().manual_seed(seed)
        self.register_buffer("Wf", torch.randn(7, D, generator=g) / sigma)
        self.register_buffer("bf", torch.rand(D, generator=g) * 2 * math.pi)
        self.D = D

    def lift(self, s, r):
        return torch.cat([s, self.F.phys(s, r), math.sqrt(2.0 / self.D) * torch.cos(torch.cat([s, r], -1) @ self.Wf + self.bf)], -1)

    def fit_windows(self, b):
        ne = NormalEq()
        for c in _chunks(b):
            s0, s1 = c.s[:, :-1].reshape(-1, 5), c.s[:, 1:].reshape(-1, 5)
            r0, r1 = c.r[:, :-1].reshape(-1, 2), c.r[:, 1:].reshape(-1, 2)
            u0 = c.u[:, :-1].reshape(-1, 3)
            P0, P1 = self.lift(s0, r0), self.lift(s1, r1)
            ne.add(torch.cat([P0, u0, r0, torch.ones_like(u0[:, :1])], 1), P1 - P0)
        self.W = ne.solve(1e-8).float()

    def forward(self, s_hist, u, r, H):
        Hp = s_hist.shape[1]
        z = self.lift(s_hist[:, -1], r[:, Hp - 1])
        out = []
        for k in range(H):
            j = Hp - 1 + k
            X = torch.cat([z, u[:, j], r[:, j], torch.ones_like(u[:, j, :1])], 1)
            z = z + X @ self.W
            z = torch.nan_to_num(z, nan=1e3).clamp(-1e3, 1e3)
            out.append(z[:, :5])
        return torch.stack(out, 1)


class SINDYc(ClosedForm):
    """sparse regression (STLSQ) of ds/dt on a quadratic library of [s, u, rho, phys];
    uses the exact derivative labels; integrated with RK4."""
    name = "SINDYc"
    uses_sdot = True

    def __init__(self, nm, thresh=0.02, iters=10, n_sub=2):
        super().__init__(nm)
        self.thresh, self.iters, self.n_sub = thresh, iters, n_sub
        self.register_buffer("sdot_sd", nm.t("sdot_sd"))
        self.dt = nm.dt

    def lib(self, s, u, r):
        x = torch.cat([s, u, r, self.F.phys(s, r)], -1)
        n = x.shape[-1]
        iu = torch.triu_indices(n, n)
        quad = (x.unsqueeze(-1) * x.unsqueeze(-2))[..., iu[0], iu[1]]
        return torch.cat([torch.ones_like(x[..., :1]), x, quad], -1)

    def fit_windows(self, b):
        # column scaling from a sample, then normal equations of the scaled library
        c0 = b.sel(torch.arange(min(b.n, 512)))
        Th0 = self.lib(c0.s.reshape(-1, 5), c0.ui.reshape(-1, 3), c0.r.reshape(-1, 2))
        sc = Th0.std(0).double() + 1e-9
        sc[0] = 1.0
        ne = NormalEq()
        for c in _chunks(b):
            Th = self.lib(c.s.reshape(-1, 5), c.ui.reshape(-1, 3), c.r.reshape(-1, 2)).double() / sc
            ne.add(Th, (c.sdot.reshape(-1, 5) / self.sdot_sd).double())
        Xi = ne.solve(1e-10)
        for _ in range(self.iters):
            small = Xi.abs() < self.thresh
            Xi[small] = 0
            for j in range(5):
                big = torch.where(~small[:, j])[0]
                if len(big):
                    Xi[big, j] = ne.solve(1e-10, idx=big, col=j)
        self.Xi = (Xi / sc[:, None]).float()
        self.sparsity = float((Xi.abs() > 0).float().mean())

    def f(self, s, u, r):
        return self.sdot_sd * (self.lib(s, u, r) @ self.Xi)

    def step(self, s, u, r, state):
        h = self.dt / self.n_sub
        for _ in range(self.n_sub):
            k1 = self.f(s, u, r)
            k2 = self.f(s + 0.5 * h * k1, u, r)
            k3 = self.f(s + 0.5 * h * k2, u, r)
            k4 = self.f(s + h * k3, u, r)
            s = s + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
            s = torch.nan_to_num(s, nan=1e3).clamp(-1e3, 1e3)
        return s, state


class ESN(ClosedForm):
    """echo-state network: fixed random leaky reservoir, ridge readout of the one-step increment."""
    name = "ESN"

    def __init__(self, nm, N=600, rho=0.9, leak=0.3, in_scale=0.4, seed=0):
        super().__init__(nm)
        g = np.random.default_rng(seed)
        W = g.standard_normal((N, N)) * (g.random((N, N)) < 0.05)
        W *= rho / np.max(np.abs(np.linalg.eigvals(W)))
        self.register_buffer("Wr", torch.as_tensor(W, dtype=torch.float32))
        self.register_buffer("Win", torch.as_tensor(g.uniform(-in_scale, in_scale, (N, Feat.N_IN + 1)), dtype=torch.float32))
        self.leak, self.N = leak, N

    def upd(self, h, x):
        xb = torch.cat([x, torch.ones_like(x[:, :1])], -1)
        return (1 - self.leak) * h + self.leak * torch.tanh(h @ self.Wr.T + xb @ self.Win.T)

    def fit_windows(self, b):
        ne = NormalEq()
        for c in _chunks(b):
            B, L, _ = c.s.shape
            h = c.s.new_zeros(B, self.N)
            for j in range(L - 1):
                x = self.F.step_input(c.s[:, j], c.u[:, j], c.r[:, j])
                h = self.upd(h, x)
                if j >= 16:                      # discard reservoir transient
                    ne.add(torch.cat([h, x, torch.ones_like(x[:, :1])], 1), (c.s[:, j + 1] - c.s[:, j]) / self.ds_sd)
        self.Wout = ne.solve(1e-6).float()

    def forward(self, s_hist, u, r, H):
        Hp = s_hist.shape[1]
        h = s_hist.new_zeros(s_hist.shape[0], self.N)
        for j in range(Hp - 1):
            h = self.upd(h, self.F.step_input(s_hist[:, j], u[:, j], r[:, j]))
        s = s_hist[:, -1]
        out = []
        for k in range(H):
            j = Hp - 1 + k
            x = self.F.step_input(s, u[:, j], r[:, j])
            h = self.upd(h, x)
            z = torch.cat([h, x, torch.ones_like(x[:, :1])], 1)
            s = s + self.ds_sd * (z @ self.Wout)
            s = torch.nan_to_num(s, nan=1e3).clamp(-1e3, 1e3)
            out.append(s)
        return torch.stack(out, 1)
