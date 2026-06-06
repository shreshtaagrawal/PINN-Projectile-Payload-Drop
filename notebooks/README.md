# PINN-Projectile-Payload-Drop

A **Physics-Informed Neural Network (PINN)** framework for projectile / payload-drop
trajectory modelling and inverse parameter estimation recovering launch velocity,
release angle from data, building toward targeting under real-world
effects (drag, crosswind, vibration, turbulence).

This README is written as a learning document. I know ML in theory but am new to coding
from scratch, so every notebook is documented with four things: the **ideology** (why this
choice), the **physics** (the law being enforced), the **math** (how that law becomes a loss),
and the **code reasoning** (what the key lines do).

The project tells one story in two directions:

> **Forward** (physics → trajectory): Notebooks 1 → 2 → 3.
> **Inverse** (data → physics): Notebooks 4 → 5.
---

## 0. One-paragraph mental model

A normal network learns by copying labelled examples. A **PINN** learns by being told the
**differential equation** its answer must obey, and is penalised whenever it breaks that law, 
so it needs little or no labelled data. The same network, run "backwards," can also treat an
unknown physical constant (a launch angle, a drag coefficient) as a **trainable parameter** and
discover it from a few noisy measurements. Both halves of this project rest on one PyTorch
feature: `torch.autograd.grad`, which differentiates the network's *output* with respect to its
*input* exactly, letting us put a derivative (i.e. a law of motion) directly inside the loss.

---

## 1. Why a PINN at all? (ideology)

For an *ideal* projectile there's a clean formula, so a PINN looks like overkill and in
Notebooks 1–2 it deliberately is. The point is to **validate the machinery on a problem we can
check by hand** before turning off the safety rails. The payoff arrives in Notebook 3: once air
drag is added, the equations become **coupled and nonlinear with no closed-form solution**, and
in Notebooks 4–5 we use the same tool to **discover hidden physics from data**, the canonical
real-world use of PINNs. Build trust on the known, then extend to the unknown.

---

## 2. The toolkit every notebook reuses

Rather than repeat these five times, here are the shared ideas; each walkthrough only explains
what's *new*.

- **Automatic differentiation.** `torch.autograd.grad(out, x, create_graph=True)` gives exact
  derivatives of an output w.r.t. an input. `create_graph=True` is mandatory whenever we need a
  *second* derivative (acceleration), because it keeps the graph alive to differentiate again.
- **`Tanh` activation, never `ReLU`.** PINNs differentiate the network twice. `ReLU` is
  piecewise-linear, so its second derivative is zero everywhere → the physics loss would be
  meaningless. `Tanh` is infinitely smooth, so accelerations are well-defined.
- **No activation on the output layer.** Positions can be any real number; an activation would
  clamp the range.
- **Normalisation + scale-balanced losses.** Inputs/outputs are scaled to ~`[0,1]` (nets train
  best on `O(1)` numbers), and *every loss term is divided by its natural physical scale*
  (`u`, `g`, a curvature scale…). This makes each term order 1, so the loss **weights express
  priorities** instead of secretly being unit conversions.
- **Adam optimiser + a decaying learning rate** (StepLR or CosineAnnealing): big steps early to
  explore, tiny steps late to settle into the minimum.
- **Collocation points:** random input locations where the physics residual is enforced, so the
  law holds across the *whole* domain, resampled every epoch ("on-the-fly") so the net can't
  memorise.

---

## 3. Repository structure & status

```
PINN-Projectile-Payload-Drop/
├── README.md
└── notebooks/
    ├── 01_Ideal_Projectile_FIXED_u_theta_PINN.ipynb     ✅ single trajectory (forward)
    ├── 02_Ideal_Projectile_GENERAL_u_theta_PINN.ipynb   ✅ family over (u, θ)  (forward)
    ├── 03_Projectile_DRAG_RK4_PINN_Sensitivity.ipynb    ✅ drag + RK4 truth + sensitivity (forward)
    ├── 04_Inverse_Projectile_Vacuum_PINN.ipynb          ✅ recover (u, θ) from noisy data (inverse)
    └── 05_Inverse_Projectile_Drag_PINN.ipynb            ✅ discover drag coeff k from data (inverse)
```

All five notebooks are complete and runnable.

---

## 4. The physics ladder

| # | Notebook | Direction | Inputs → Outputs | Core physics | Ground truth |
|---|---|---|---|---|---|
| 1 | Ideal fixed | forward | `x → y` | $y''=-g/v_x^2$ (constant curvature) | analytical parabola |
| 2 | Ideal general | forward | `(x,u,θ) → y` | same, generalised | analytical parabola |
| 3 | Drag | forward | `(t,θ,k) → (x,y)` | coupled nonlinear ODE system | **RK4** (no formula) |
| 4 | Inverse vacuum | inverse | `x → y` + trainable `(C,m)` | constant curvature | known hidden `(u,θ)` |
| 5 | Inverse drag | inverse | `t → (x,y)` + trainable `k` | nonlinear drag ODE | RK4 at hidden `k` |

---

## 5. Notebook 1 — `01_Ideal_Projectile_FIXED` (single trajectory)

**Aim:** map one input `x` → one output `y`, reproducing a single parabola for fixed `u=30`,
`θ=30°`.

**Physics.** Eliminating time from ideal projectile motion gives
`y(x) = tan θ · x − (g / 2v_x²) x²`, whose key facts become four losses: start `y(0)=0` (IC),
launch slope `y'(0)=v_y/v_x` (slope), landing `y(x_max)=0` (BC), and the law
`y''(x) = −g/v_x² = const` enforced everywhere (physics).

**The physics loss (the heart):**
```python
y_col   = pinn(x_col)
dy_dx   = torch.autograd.grad(y_col, x_col, torch.ones_like(y_col), create_graph=True)[0]
d2y_dx2 = torch.autograd.grad(dy_dx, x_col, torch.ones_like(dy_dx), create_graph=True)[0]
loss_phys = torch.mean((d2y_dx2 - physics_constant)**2)   # physics_constant = -g/vx²
```
`create_graph=True` on the first grad is what lets the second grad exist.

**Sizing reasoning.** One smooth curve is an easy target → a small net (1→32×3→1, ~2,209 params)
is plenty; more capacity would just risk wiggles in a curve that should be a clean parabola.
Weights `λ_phys=500 ≫ λ_ic=1` because the second-derivative residual is numerically tiny and
would otherwise be ignored; `λ_slope=10` because the launch angle is what we care about most.

---

## 6. Notebook 2 — `02_Ideal_Projectile_GENERAL` (family of trajectories)

**Aim:** one network for *any* launch in a range — inputs become `(x, u, θ)`, so the net learns
a **surface** over launch space, not a single curve.

**What's new vs NB1:** inputs normalised to `[0,1]`; bigger net (3→64×5→1, ~16,961 params)
for the richer family; **on-the-fly** sampling of 20 `(u,θ)` pairs per epoch. The normalisation
subtlety to watch: the output is normalised (`y_phys = y_norm · y_ref`) but we differentiate
w.r.t. *raw* `x`, so the physical derivative is `dy_norm/dx · y_ref` — getting that conversion
wrong silently corrupts the physics loss.

**Honest result.** The `(u,θ)` sensitivity sweep gave **mean RMSE ≈ 21%**, best ≈ **0.95%**,
with only **~56% of the training range under 5%** — accurate in the core, weak near the edges
(low-angle cases where the range formula is sensitive). This is a real to-do, not a victory; the
likely fixes are adaptive loss balancing, longer training, and a hard-constraint output form that
bakes in `y(0)=0` and `y(x_max)=0` so those terms can't fight the physics term.

---

## 7. Notebook 3 — `03_Projectile_DRAG_RK4_PINN_Sensitivity` (the step up)

**Aim:** add real **quadratic air drag** and generalise over launch angle `θ` and drag
coefficient `k`. This is the first problem with **no closed-form answer**.

**Physics.** With drag opposing velocity and scaling as speed², the state `s=[x,y,v_x,v_y]` obeys
the **coupled nonlinear system**
```
ẋ = v_x,   v̇_x = −k·v·v_x
ẏ = v_y,   v̇_y = −g − k·v·v_y,   v = √(v_x²+v_y²)
```
The `v = √(v_x²+v_y²)` term couples horizontal and vertical motion → nonlinear → no formula.
Note `k=0` recovers the vacuum parabola, so this notebook *contains* NB1–2 as a special case.

**Two new tools, and why:**
1. **RK4 ground truth.** Without a formula we integrate numerically. RK4 samples the slope four
   times per step and takes a weighted average, giving 4th-order accuracy (error ~`Δt⁴`). It
   plays the exact role the parabola played before. Crucially, RK4 is **validated against the
   exact *linear*-drag solution** (a case that *does* have a formula) to ~`10⁻⁷` m error before
   it's trusted on the quadratic case — proper scientific practice.
2. **Time-domain PINN** `(t, θ, k) → (x, y)`. With drag, time is the natural variable, and
   velocities/accelerations fall out of autograd w.r.t. `t`.

**Loss (an ODE *system*, no data):** initial position `x(0)=y(0)=0`; initial velocity
`ẋ(0)=u cos θ`, `ẏ(0)=u sin θ`; a **landing anchor** `y(T_f)=0` (with `T_f` from a precomputed
RK4 **flight-time lookup table**, bilinearly interpolated so training stays fast); and the
**physics residuals** `R_x = ẍ + k v ẋ`, `R_y = ÿ + k v ẏ + g`, each normalised by `g`.

**A normalisation trick worth understanding.** Because `t_norm = t/T_ref` lives inside the
graph, differentiating an output w.r.t. the *physical* time leaf makes autograd automatically
carry the `1/T_ref` factor — so we only multiply by the position scale to get a physical
velocity. Physics stays exact while the net works in tidy numbers.

**Sensitivity.** A `(θ,k)` heatmap (extended *beyond* the training box) of RMSE-vs-RK4 shows
where the surrogate is trustworthy and how it degrades — worst in the high-drag, steep-angle
corner where the trajectory is most asymmetric, and beyond the training box (PINNs extrapolate
poorly because physics was only enforced inside it). Exact numbers are in the notebook output.

---

## 8. Notebook 4 — `04_Inverse_Projectile_Vacuum` (the flip: data → physics)

**Aim:** given ~15 **noisy** `(x,y)` points and *no* knowledge of the launch, recover `u` and
`θ`. (Tested against a hidden truth `u=28 m/s, θ=42°`, noise std 0.3 m, so recovery can be
checked.)

**What's new — trainable physics parameters.** Beyond fitting data, we make the unknown physics
**`nn.Parameter`s optimised jointly with the network**:
```python
C = torch.nn.Parameter(torch.tensor(0.0))   # curvature  d²y/dx²  (= -g/vx²)
m = torch.nn.Parameter(torch.tensor(0.0))   # launch slope dy/dx(0) (= vy/vx)
optimizer = torch.optim.Adam(list(net.parameters()) + [C, m], lr=3e-3)
```
Three losses: **data** (net passes near the points — the new term that injects observations),
**physics** (net curvature must equal `C` everywhere), **IC** (`y(0)=0`, `y'(0)=m`). The data
pins the curve, the physics ties its shape to `(C,m)`, so `(C,m)` get dragged to the values that
make physics agree with the data — i.e. the hidden truth.

**The decisive lesson — reparametrisation.** Making `v_x, v_y` the unknowns directly trains
badly, because curvature depends on `v_x` through `−g/v_x²` (a weak, badly-scaled gradient).
Discovering the **well-conditioned coefficients** `C` and `m` (which the data constrains
directly) and *then* converting back is what works:
```
v_x = √(−g/C),  v_y = m·v_x,  u = √(v_x²+v_y²),  θ = arctan(v_y/v_x)
```
This conditioning insight is the most important idea in the notebook.

---

## 9. Notebook 5 — `05_Inverse_Projectile_Drag` (discovering a hidden coefficient)

**Aim:** the canonical inverse PINN — from ~20 noisy points, **discover the hidden drag
coefficient `k`** of a *nonlinear* ODE with no closed-form solution. (Tested against hidden
`u=30, θ=45°, k=0.012`; data manufactured with the validated RK4 integrator.)

**What's new — a constrained unknown.** `k` must be ≥ 0 (negative drag is unphysical), so we
don't optimise it directly; we optimise an unconstrained number through `softplus`:
```python
k_raw = torch.nn.Parameter(torch.tensor(-2.0))    # starts far from the truth (fair test)
def k_value(): return F.softplus(k_raw) * 0.05    # always ≥ 0, scaled to a realistic band
```
A time-domain net `t → (x,y)` (1→48×4→2) fits the data while the **physics residuals carry the
trainable `k`** (`R_x = ẍ + k v ẋ`, `R_y = ÿ + k v ẏ + g`, normalised by `g`). The only `k` that
lets a data-fitting curve also satisfy the equations is the true one, so `k` converges to it.
Loss weights `5·data + 2·phys + 10·ic` lean on the terms carrying hard facts.

**The arc, completed.** NB1→3 went physics→trajectory; NB4→5 reversed it, data→physics. Same
three ingredients throughout: autograd derivatives, a normalised `Tanh` network, and a
scale-balanced weighted loss.

---

## 10. Honest results & known limitations

- **NB2** is accurate in its core but ~56%-under-5%-RMSE overall — needs adaptive loss balancing
  / hard constraints before it's a reliable building block.
- **NB3–5** demonstrate the method against known truths; the precise RMSE / recovered values are
  printed in each notebook's output (read them off the committed runs rather than quoting from
  memory).
- **Extrapolation** is poor outside training ranges — expected, since physics is only enforced
  inside the collocation box. Report results *inside* the trained region.
- **No uncertainty yet:** every estimate is a single point with no error bar — the first thing to
  add for credibility (see roadmap).

---

## 11. Roadmap to the research goal

The publishable contribution is **inverse estimation of launch parameters under combined,
realistic disturbances**, with a systematic robustness study. Remaining notebooks:

| # | Planned notebook | Adds | Why it matters |
|---|---|---|---|
| 6 | `06_Crosswind_3D` | lateral wind force / third dimension | real drops aren't planar |
| 7 | `07_Stochastic_Disturbances` | vibration + turbulence (random forcing) | the genuinely novel, hard regime |
| 8 | `08_Inverse_Combined_Targeting` | recover `(u, θ, k)` jointly under drag+wind+noise, with **uncertainty** | the actual research deliverable |

Cross-cutting tasks that turn notebooks into a paper: a **classical baseline** (RK4 + an
optimiser) to beat; **noise/sparsity robustness sweeps** (how recovery degrades with fewer/
noisier points); **uncertainty quantification** (repeat fits over many noisy datasets, or
ensembles, to put error bars on recovered parameters); and **writing as you go**, targeting an
arXiv preprint / SciML workshop first.

---

## 12. References

- Raissi, Perdikaris, Karniadakis (2019), *Physics-informed neural networks: A deep learning
  framework for solving forward and inverse problems involving nonlinear PDEs*, J. Comput. Phys.
  378:686–707. — the founding paper; read its inverse-problem sections (they underpin NB4–5).
- Karniadakis, Kevrekidis, Lu, Perdikaris, Wang, Yang (2021), *Physics-informed machine
  learning*, Nature Reviews Physics 3(6):422–440. — the field map.
- Cuomo et al. (2022), *Scientific ML Through PINNs: Where we are and What's Next*, J. Sci.
  Comput. 92(3):88. — most beginner-friendly survey.
- Wang, Teng, Perdikaris (2021), *Understanding and mitigating gradient flow pathologies in
  PINNs*, SIAM J. Sci. Comput. 43(5):A3055–A3081. — directly fixes the NB2 loss-balancing issue.
- Wang, Yu, Perdikaris (2022), *When and why PINNs fail to train: a neural tangent kernel
  perspective*, J. Comput. Phys. 449:110768.
- Lu, Meng, Mao, Karniadakis, *DeepXDE* — a mature PINN library worth reading for structure.

---

## 13. Glossary

- **PINN** — network trained to obey a differential equation via its loss, not to copy data.
- **Forward vs inverse** — physics→trajectory vs data→physics (discovering unknown parameters).
- **Residual** — "law value minus what the network produced"; zero = law obeyed.
- **Collocation points** — random inputs where the physics loss is enforced.
- **RK4** — 4th-order Runge–Kutta; the numerical "truth" once no formula exists.
- **`nn.Parameter`** — a tensor the optimiser trains; here, the unknown physics (`C`, `m`, `k`).
- **`softplus`** — smooth map to positive numbers; keeps a discovered coefficient ≥ 0.
- **Reparametrisation** — solving for well-conditioned unknowns, then converting back to physics.
- **Loss weights (λ / w)** — priorities among loss terms; the fiddliest PINN hyperparameters.
