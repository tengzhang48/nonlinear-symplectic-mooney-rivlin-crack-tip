# Theory notes and claim boundaries

These notes provide a compact derivation map for the public code and figures
associated with

> Teng Zhang, “Constrained asymptotic crack-tip fields of a Mooney–Rivlin
> sheet in plane stress: a symplectic analysis” (2026).

They are not a substitute for the manuscript. Their purpose is to make the
theory-to-code-to-figure chain inspectable without publishing internal
development records or superseded calculations.

## 1. Plane-stress reduction

Let `F` be the in-plane deformation gradient and `J=det(F)` its areal
Jacobian. Incompressibility and plane stress eliminate the thickness stretch
as `lambda_3=J^(-1)`. The reduced two-dimensional Mooney–Rivlin energy is

```text
W = c1 [F:F + J^(-2) - 3]
  + c2 [J^2 + (F:F) J^(-2) - 3],        c1>0, c2>0.
```

Direct differentiation gives the in-plane first Piola stress

```text
P = 2 c1 [F - J^(-2) F^(-T)]
  + 2 c2 [J^(-2) F + (J^2 - J^(-2) F:F) F^(-T)].
```

This is the stress used by the asymptotic flux calculation and by the finite-
element implementation. The three-dimensional Cauchy stress is `sigma=P F^T`
because the full three-dimensional determinant is one; the in-plane `J` is
not a volume ratio.

Public checks:

- `analysis/leading_field.py`
- `analysis/energy_release_rate.py`
- `fem/theory_field.py`
- `verification/verify_equations.py`

## 2. Radial Hamiltonian formulation

Set

```text
xi = log(r).
```

The crack tip is `xi -> -infinity`, and powers `r^Lambda` become exponentials
in radial time. The action between two circles can be written as

```text
A = integral L(q, partial_xi q) dxi,
```

where `q(theta)` is the deformed-position trace on a circle. Its conjugate
momentum is the force per unit angle,

```text
p = r P e_r.
```

When a stiff energy contribution becomes the kinematic constraint `C(F)=0`,
the constrained action is

```text
A[y,chi] = integral [W1(F) + chi C(F)] r dr dtheta.
```

The second variation gives a linearized Jacobi system. The multiplier
increment has no radial derivative and hence no independent momentum; its
reaction enters the displacement momenta through the incremental total
traction. Symmetry of the second variation gives the Lagrange identity

```text
d/dxi integral_0^pi (u . pi_bar - u_bar . pi) dtheta
  = [u . sigma_bar_theta - u_bar . sigma_theta]_0^pi.
```

The endpoint term determines whether the pairing is conserved. This project
establishes the conserved pairing on the closed normal-opening sector. The
paper-scope higher-order lane contains only the closed first-material rung and
the restricted scalar opening-resonance check described below. Other
higher-order scripts remain research provenance and are not promoted to a
completed all-grade reaction-carrying endpoint operator.

Public checks:

- `verification/verify_equations.py`, opening-sector identities
- `analysis/second_variation_reduction.py`
- `analysis/derive_constraint_row.py`

`analysis/symplectic_dae.py` is retained only as a historical spectral
scaffold. Its non-opening rows are not the completed canonical operator, and
the script is outside the standard verification lane.

## 3. Why the second invariant becomes a constraint

Near the tip, the largest principal stretch obeys `lambda_1^2 ~ 1/r`. At
fixed incompressible product

```text
lambda_2 lambda_3 = lambda_1^(-1),
```

the `I2` term increasingly penalizes `lambda_2^2+lambda_3^2`. Its minimum is
therefore

```text
lambda_2 = lambda_3 = lambda_1^(-1/2).
```

In the reduced variables this gives the leading pointwise constraint

```text
J^4 = |grad(y2)|^2.
```

The limit is singular: the constrained zone shrinks as `c2 -> 0`, although
the limiting kinematics inside that zone remain uniaxial for every fixed
`c2>0`.

A scale estimate for the constrained-zone edge and the later on-manifold
material crossover are

```text
r_*  = P^2 c2 / (4 c1),
r_I2 ~ P^2 (c1/c2)^2.
```

Thus the stated two-dimensional constrained asymptotics require the overlap
annulus

```text
t_s << r << min(r_*, r_I2),
```

where `t_s` is the sheet thickness. They are not a statement about the
sub-thickness three-dimensional core or about radii beyond either crossover.

Public checks:

- `analysis/leading_field.py`
- `analysis/derive_constraint_row.py`
- `verification/verify_equations.py`

## 4. Leading constrained map and its null family

Define

```text
f(theta) = sin(theta/2),
s = r sin^2(theta/2).
```

The formal constrained/null-family map is

```text
y2 = P r^(1/2) f(theta),
y1 = c0 + Cs s + P^(-1/2) r^(5/4) g(theta) + ... .
```

The angular residual profile satisfies

```text
(5/4) f' g - (1/2) f g' = 1/sqrt(2).
```

The constraint determines the component of `grad(y1)` perpendicular to
`grad(y2)` but leaves the exact null family

```text
y1 -> y1 + F(y2).
```

`Cs s` is its first analytic Mode-I member. It is a physical displacement,
not a gauge. The local theory does not select `Cs`.

The angular equation also admits the homogeneous member

```text
Ch sin(theta/2)^(5/2).
```

The analytic-axis representative sets `Ch=0`, but the local calculation does
not select it. Both the `Cs` and `Ch` angular factors vanish at `theta=0`.
The matching-circle campaign can therefore fit

```text
Y1(r,0) = c0 + A_ax r^q + D r^(7/4)
```

without either coefficient. The final core/angular refinement gives
`q=1.251529` and `A_ax/A_ax,pred=1.012420`, an amplitude error below 1.3%.
A free two-power audit of the final case returns
`1.24980 <= q <= 1.25152`, so the recovered value is not produced by fixing
the nuisance exponent. Its next power is consistent with the `r^(7/4)`
in-plane slot, which checks the radial ordering but not that correction's
angular profile or amplitude. The full-angle residual is consistent with the
ODE family after `Ch` is admitted. The campaign supports the asymptotic class
for the tested strip but does not select `Cs`, `Ch`, or the analytic-axis
member.
The six retained profiles come from same-cell exact-restriction consistency
solves driven by six new global matching-circle strip configurations. Four
form the plotted core/angular sequence and two test the matching radius. The
independently refined local submodel does not enter this result.

On the crack face, the physical `Cs r` term can dominate the residual, so no
universal raw crack-face-shape exponent is claimed.

Public checks and data:

- `analysis/leading_field.py`
- `analysis/check_g_selection.py`
- `verification/verify_equations.py`
- `fem/GLOBAL_LOCAL_RESULTS.md`
- `data/fem/global_local/global_local_campaign_summary_2026-07-23.json`
- `data/fem/global_local/global_local_profile_lam16_core2p5e6_nt120_rm1e2.npz`
- `figures/rendered/fig_r54_axis_convergence.*`

`analysis/profile_mode_audit.py` has a separate numerical role: it reproduces
the ESI strip table of fitted `b`, raw slopes, and nested target-free `q`
values. That short-window table is not the dedicated matching-circle
convergence test.

## 5. Parameter-free leading predictions

The principal-stretch magnitudes and compensated Jacobian are

```text
lambda_1 = (P/2) r^(-1/2),
lambda_2 = lambda_3 = sqrt(2/P) r^(1/4),
J r^(1/4) = sqrt(P/2).
```

Direct evaluation of the finite-strain energy flux on the superposed leading
map gives

```text
G = (pi/2) c1 P^2,
sigma_22 r = G/pi.
```

The limiting flux is independent of the local null coefficient `Cs`. For a
Rivlin–Thomas pure-shear strip of height `h` and grip stretch `lambda`,

```text
G = h (c1+c2) (lambda^2 + lambda^(-2) - 2),
```

so the remote loading predicts `P` without fitting it to a specimen-scale
simulation. The strip is the sole FEM comparison used by the paper.

Public checks and figures:

- `analysis/energy_release_rate.py`
- `analysis/check_g_selection.py`
- `analysis/check_row1_flux.py`
- paper Figures 3 and 5–7 in `figures/rendered/`

The preserved disk BVP is not a second validation geometry. Its full-arc
Dirichlet condition imposes crack-parallel compression and is not equivalent
to the strip loading; its high-load branch also loses global geometric
admissibility. It is quarantined from the evidence chain.

## 6. Closed normal-opening sector

Because the in-plane and opening increments carry different radial weights,
the pencil label `Lambda` represents

```text
delta y1 ~ r^Lambda a(theta),
delta y2 ~ r^(Lambda-3/4) b(theta).
```

Let `nu=Lambda-3/4`. The closed opening equation is

```text
-b'' = nu^2 b,
b(0)=0,
b'(pi)=0.
```

Its positive opening powers are

```text
nu = 1/2, 3/2, 5/2, ...,
b(theta) = sin(nu theta).
```

Restoring the full radial field `u2=exp(nu xi)b(theta)` and its normalized
momentum `tau2=partial_xi u2`, the conserved opening pairing is

```text
Omega_op(V,W) = integral_0^pi (u2_V tau2_W - u2_W tau2_V) dtheta.
```

Nonzero dual pairings require

```text
Lambda' = 3/2 - Lambda.
```

This is an exact statement within the closed opening sector. The public code
does not promote it
to a completed coupled spectrum or a finite-radius extractor for every
higher-order amplitude.

Public check:

- `verification/verify_equations.py`, opening-sector endpoint, orthogonality,
  and dual-pair gates

## 7. Bounded higher-order results and the physics map

The first finite constitutive correction on the chosen `F=0` base has pencil
label `Lambda=7/4`. It is not a fitted nuisance exponent. Its closed
calculation gives

```text
delta y2  = (c2/c1) r v(theta),
delta y1  = (c2/c1) P^(-3/2) r^(7/4) w(theta),
v(theta)  = -(2/3) sin(theta/2),
w(0)      = 4 sqrt(2)/21.
```

The total face traction closes at this rung and no new local amplitude is
introduced. This statement is conditional on the selected `F=0`
representative.

The `Lambda=11/4` stationary-background calculation is a separate formal
checkpoint. It is not a globally matched physical mode. The audited
restricted interaction of the `7/4` and `11/4` sources reaches the
`Lambda=13/4` opening kernel. Since opening displacement has radial power
`Lambda-3/4`, this label corresponds to `r^(5/2)`. A nonzero restricted
Fredholm projection forces an
`r^(5/2) log(r/r0) sin(5 theta/2)` companion in that formal opening
coefficient.

This is the stopping line. The complete same-grade source, reaction-carrying
coupled response, and specimen-selected net logarithmic amplitude have not
been assembled. The opening harmonic `B r^(3/2) sin(3 theta/2)` and contour
shears `Q_k s^k` remain parallel matching sectors whose amplitudes come from
the outer specimen. Figure 2 maps these established and open objects. It is
not a data plot or a complete higher-order expansion.

Public checks:

- `analysis/gate4_first_material_background.py`
- `analysis/verify_lambda13_restricted_log.py`
- `figures/rendered/fig_asymap.*`

Other formal `Q_k s^k` calculations remain outside the manuscript claim
ledger and standard verification runner. They do not complete the all-source
tower, mixed adjoints, physical endpoint matching, or specimen-scale
selection of `B` and `Q_k`.

Retained non-paper research scripts:

- `analysis/coupled_shear_completion.py`
- `analysis/qk_full_mr_ordering.py`
- `analysis/qk_full_mr_completion.py`
- `analysis/qk_later_companions.py`

## 8. Evidence labels used in this repository

| Label | Meaning |
|---|---|
| exact identity | symbolic or analytic statement checked without a fitted physical coefficient |
| numerical consistency | deterministic evaluation of a stated formula or stored field |
| stored-data comparison | reanalysis of the exact curated arrays used by a figure |
| fresh FEM solve | independent specimen boundary-value calculation in the pinned FEniCSx environment |
| conditional asymptotic statement | requires the named branch, endpoint class, or matching assumption |
| open | not needed by the reproduced paper claim and not promoted by the public code |

For the exact commands and manuscript-figure map, see
[`REPRODUCIBILITY.md`](REPRODUCIBILITY.md) and [`FIGURES.md`](FIGURES.md).
