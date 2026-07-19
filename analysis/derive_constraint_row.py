#!/usr/bin/env python3
"""Step-by-step derivation and validation of the linearized-constraint DAE row.

Row 4 of the Hamiltonian DAE pencil is the linearization of the leading
pointwise constraint about the leading field. This script derives it from
scratch with sympy and checks, numerically against the leading g-profile, that
the derived angular equation is proportional to the row-4 expression coded in
the public spectral scaffold in ``symplectic_dae.py``.

The constraint.  On the leading similarity field y1=r^a1 g, y2=r^a2 f one finds
  J = det F = r^(a1+a2-2) (a1 f' g - a2 f g') = r^(-1/4) Delta,   Delta=a1 f'g-a2 f g',
  |grad y2|^2 = F21^2+F22^2 = r^(2a2-2)(a2^2 f^2 + f'^2) = r^(-1) G,  G=a2^2 f^2+f'^2.
The leading row-1 balance is the pointwise constraint Delta^4 = G, i.e.
  (J r^{1/4})^4 = |grad y2|^2 r   <=>   J^4 = |grad y2|^2.
We linearize C := J^4 - (F21^2+F22^2) = 0 about the leading field with
  y1 = r^a1 g(theta) + eps r^mu a(theta),
  y2 = r^a2 f(theta) + eps r^nu b(theta),   nu = mu - 3/4.
"""
from __future__ import annotations

import numpy as np
import sympy as sp

from leading_field import solve_g_leading, g_family

# ---------------------------------------------------------------- symbolic
r, th, eps, mu = sp.symbols("r theta epsilon mu", positive=True)
a1, a2 = sp.Rational(5, 4), sp.Rational(1, 2)
nu = mu - sp.Rational(3, 4)
f = sp.sin(th / 2)
g = sp.Function("g")
a = sp.Function("a")
b = sp.Function("b")

y1 = r ** a1 * g(th) + eps * r ** mu * a(th)
y2 = r ** a2 * f + eps * r ** nu * b(th)


def dX1(Y):  # d/dX1 = cos th d_r - (sin th / r) d_th
    return sp.cos(th) * sp.diff(Y, r) - sp.sin(th) / r * sp.diff(Y, th)


def dX2(Y):  # d/dX2 = sin th d_r + (cos th / r) d_th
    return sp.sin(th) * sp.diff(Y, r) + sp.cos(th) / r * sp.diff(Y, th)


F11, F12 = dX1(y1), dX2(y1)
F21, F22 = dX1(y2), dX2(y2)
J = sp.expand(F11 * F22 - F12 * F21)
C = J ** 4 - (F21 ** 2 + F22 ** 2)

# Step 1: the leading constraint (eps^0) is r^-1 (Delta^4 - 1/4); it vanishes
# exactly when Delta = a1 f' g - a2 f g' = 2^(-1/2) (the g-ODE).
C0 = sp.simplify(C.subs(eps, 0) * r)
print("Step 1  leading C * r  =", sp.simplify(C0))

# Step 2: linearize.  dC/deps|_0, then strip the common power r^(mu-9/4).
dC = sp.diff(C, eps).subs(eps, 0)
ang = sp.simplify(dC * r ** (sp.Rational(9, 4) - mu))
ang = sp.simplify(ang)
has_r = ang.has(r)
print("Step 2  linearized angular constraint still depends on r? ", has_r)

# substitute concrete symbols for g, g', a, a', b, b' so we can evaluate
gs, gps, as_, aps, bs, bps = sp.symbols("gs gps as aps bs bps")
subs = {
    sp.Derivative(g(th), th): gps, g(th): gs,
    sp.Derivative(a(th), th): aps, a(th): as_,
    sp.Derivative(b(th), th): bps, b(th): bs,
}
ang_s = sp.simplify(ang.subs(subs))

# ---------------------------------------------------------------- row 4 (code)
# f' tau1 - a2 f a' + a1 g b' - g' tau2 - sqrt2 (a2 f tau2 + f' b'),  tau1=mu a, tau2=nu b
fp_e = sp.diff(f, th)
row4 = (fp_e * mu * as_ - a2 * f * aps + a1 * gs * bps - gps * nu * bs
        - sp.sqrt(2) * (a2 * f * nu * bs + fp_e * bps))

# Exact symbolic check on the positive leading branch:
#   a1 f' g - a2 f g' = 1/sqrt(2).
leading_constraint = sp.Eq(a1 * fp_e * gs - a2 * f * gps, 1 / sp.sqrt(2))
gps_positive_branch = sp.solve(leading_constraint, gps)[0]
exact_residual = sp.trigsimp(
    sp.simplify((ang_s - sp.sqrt(2) * row4).subs(gps, gps_positive_branch))
)
print("Step 3  exact symbolic residual after positive-branch constraint =",
      exact_residual)
if exact_residual != 0:
    raise SystemExit("exact constraint-row derivation did NOT match row 4")

# ---------------------------------------------------------------- numeric check
ang_fn = sp.lambdify((th, gs, gps, as_, aps, bs, bps, mu), ang_s, "numpy")
row4_fn = sp.lambdify((th, gs, gps, as_, aps, bs, bps, mu), row4, "numpy")

sol = solve_g_leading()
gfun, gpfun = g_family(sol, A0=0.0)
rng = np.random.default_rng(0)

ratios = []
for trial in range(6):
    mu_val = rng.uniform(1.0, 3.5)
    # random smooth test perturbations a(theta), b(theta) and their derivatives
    k1, k2, p1, p2 = rng.uniform(0.5, 3, 4)
    th_grid = np.linspace(0.3, np.pi - 0.3, 25)
    av = np.sin(k1 * th_grid + p1); apv = k1 * np.cos(k1 * th_grid + p1)
    bv = np.sin(k2 * th_grid + p2); bpv = k2 * np.cos(k2 * th_grid + p2)
    gv = gfun(th_grid); gpv = gpfun(th_grid)
    A_ = ang_fn(th_grid, gv, gpv, av, apv, bv, bpv, mu_val)
    R_ = row4_fn(th_grid, gv, gpv, av, apv, bv, bpv, mu_val)
    m = np.abs(R_) > 1e-6
    ratios.append(A_[m] / R_[m])

ratios = np.concatenate(ratios)
ratio_mean = float(np.mean(ratios))
ratio_spread = float(np.max(ratios) - np.min(ratios))
print(f"\nStep 4  derived(angular constraint) / row4(code):")
print(f"        mean ratio = {ratio_mean:.10f}   spread over (theta,a,b,mu) = {ratio_spread:.2e}")
ok = ratio_spread < 1e-9
print(f"\n[{'PASS' if ok else 'FAIL'}] derived linearized constraint is proportional to coded row 4 "
      f"(constant factor {ratio_mean:.6g})")
print("        => the derivation reproduces the row 4 used in the pencil.")
if not ok:
    raise SystemExit("constraint-row derivation did NOT match the coded row 4")
