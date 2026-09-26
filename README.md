# PIKO — Physics-Informed Koopman Operator network with an unknown-input observer

PIKO is the new model in this benchmark. It predicts the two-span web
(T1, T2, ωu, ω1, ωr) over 256 samples from a 32-sample history, the future torques and the measured roll radii.
It has three layers. Each is a separate switch in the code (`r2r_nn/bench/piko.py`):

| layer | what it is | trained how | used for |
|---|---|---|---|
| **Koopman core** | radius-scheduled Koopman generator on a physics dictionary, exact exponential integrator | identified in closed form (generator EDMD on exact derivatives, 2 s) | every prediction; the MPC model |
| **Unknown-input observer** | Gauss–Newton moving-horizon estimate of lumped disturbances *w* and actuator effectiveness *θ* | closed form per window, no parameters to train | every prediction |
| **Closed-loop innovation network** | GRU inside the Koopman recursion that gates the physics increment and adds a learned unknown input | gradient descent on the multi-step loss (core frozen, operators cached) | closed-loop replay only; switched off for counterfactual inputs and MPC |

## 1. Coordinates: why the model works in (T1, T2, ω1, v1, v2)

The web-speed mismatches

    v1 = R1·ω1 − Ru·ωu,      v2 = Rr·ωr − R1·ω1

drive the tensions (dT1/dt = (ES/L1)·v1 + …). On M1, E·S = 2.5·10⁶ N, so |v1| is about 10⁻⁶ of R1·ω1.
Any model that works in the raw state has to resolve a 10⁻⁶ cancellation between two large velocities.
This held in every experiment with raw-state coordinates:

- least squares for the tension rows was ill-conditioned;
- EDMD and SINDy fits were unstable off the data manifold;
- float32 rounding alone produced 5 % errors in v1.

PIKO works in the kinematic coordinates

    q = [T1, T2, ω1, v1, v2]      (standardised; ωu, ωr are recovered exactly from q and the radii)

In q, the web model is a polynomial of degree two. Its coefficients depend on the radii only through a few known functions:

    dT1/dt = (ES/L1)·v1 + (Tud/L1)·(R1ω1 − v1) − (R1/L1)·T1·ω1                          (constant coefficients)
    dv1/dt = R1·dω1/dt − Ru·dωu/dt − (dRu/dt)·ωu,   ωu = (R1ω1 − v1)/Ru
    dωu/dt = (Ru/Ju)·T1 − (bfu/(Ru·Ju))·(R1ω1 − v1) + (χκρ·Ru/Ju)·(R1ω1 − v1)² − Mu/Ju
    −(dRu/dt)·ωu = (χ/2π)·(R1ω1 − v1)²/Ru²                                              (and mirror terms for the rewinder)

## 2. Koopman core

**Dictionary.** It is state-inclusive: ψ(q) = [q, q_i·q_j (i ≤ j)], so 5 + 15 observables. An optional learned closure g_θ(q, ρ) exists but is off by default.

**Physical rows and composition.** The generator is identified on seven cancellation-free physical rows:

    p = [dT1, dT2, dω1, dωu, dωr, −(dRu/dt)·ωu, (dRr/dt)·ωr]

These are mapped to q-rates by the exact kinematic map C(ρ). For example, v̇1 = R1·ω̇1 − Ru·ω̇u − Ṙu·ωu.
This is the step that makes M1 learnable: the dωu row can be fitted to 10⁻⁷ relative accuracy, and v1 inherits that accuracy.

**Scheduling (LPV).** The model is

    L(ρ) = Σ_k σ_k(ρ)·L_k,    σ(ρ) = [1, Ru/Ju, 1/(Ru·Ju), 1/Ju, 1/Ru², Rr/Jr, 1/(Rr·Jr), 1/Jr, 1/Rr²]

It uses the nominal inertia law J(R) = J0 + (π/2)·ρ_w·κ·(R⁴ − R0⁴).

**Structural sparsity.** For each physical row and scheduling function, only the monomials and torques that appear in the web model are allowed. For example, the dT1 row may use v1, ω1, T1·ω1 and a constant. This is the physics-informed prior. Without it (ablation `PIKO-core-dense`), the identified generator is equally accurate on the training manifold but diverges off it.

**Identification (generator EDMD).** Each physical row is a small ridge least-squares problem on exact derivative labels. It uses the nominal clean families NOM and REF only, fits in about 2 s, and reproduces the true vector field to about 10⁻⁵ relative error, including 1.4× outside the training range.

**Discretisation.** An exponential integrator, re-scheduled every step:

    q_{k+1} = e^{L_qq}·q_k + φ1(L_qq)·F·[q⊗q, ũ_k, 1, w]

- e^{L_qq} and φ1 come from one 10×10 augmented matrix exponential in float64.
- The stiff 170 Hz web mode of M1 (ω·Δt ≈ 1) is integrated exactly.
- The quadratic observables are held over the step and re-encoded from q afterwards (Koopman with re-encoding).
- The ablation `PIKO-core-linear` propagates the full lifted state linearly instead.

## 3. Unknown-input observer

Over the 32-sample history, the core is free-run from the first sample. The observer fits:

- a constant lumped input w ∈ R⁵, one per physical row: tension, winder and motor disturbances;
- an actuator effectiveness change θ ∈ R³, with ũ = u + θ·(u + ū).

The fit is weighted least squares with forgetting (γ = 0.95) and relative Tikhonov regularisation, solved with two Gauss–Newton steps using exact sensitivities.

It includes a safeguard: every iterate is re-simulated. An estimate is accepted only if it removes at least half of the history misfit; otherwise the nominal physics (w = θ = 0) is kept. Without this safeguard, aliasing-induced misfit (10 kHz torque chatter seen at 1 kHz) was interpreted as a disturbance. That broke nominal predictions under new torques.

## 4. Closed-loop innovation network

Logged torques come from a feedback controller. They are a noisy measurement of the state and they react within one sample to disturbances, faults and kicks that no physics model can see. Sequence models exploit this "controller leak".

PIKO exploits it inside the Koopman recursion instead of replacing the physics:

    dq_k = K(q_k) − q_k                          (Koopman increment)
    h_k  = GRU([state features, dq_k, Δu_k, observer estimates], h_{k−1})     (warmed up on the history)
    q_{k+1} = q_k + g_k ⊙ dq_k + d_k,   g_k = 2·sigmoid(·) ∈ (0, 2),  d_k = learned unknown input

- At initialisation (g = 1, d = 0) PIKO equals the Koopman predictor.
- The state stays in physical coordinates, so the physics acts on the corrected state at every step. This is what keeps 256-step forecasts stable.
- Because the core is fixed, the per-step operators of every training window are precomputed once (75 s). The network then trains at the speed of a plain GRU.

**Deployment modes.**

- *Replay* (logged closed-loop inputs): full PIKO.
- *Counterfactual / MPC* (torques are decisions): innovation network off, `innov_off=True`.

The benchmark reports both.
