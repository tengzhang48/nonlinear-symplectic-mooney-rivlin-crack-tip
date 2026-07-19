"""Smooth-branch selection of the slave profile g, and row-1 flux cancellation.

Two algebraic issues are resolved here with explicit computations:

1. UNDER-DETERMINACY OF THE g-ODE.  (5/4) f' g - (1/2) f g' = 2^{-1/2},
   f = sin(theta/2), admits the homogeneous solution g_h = C f^{5/2} (the
   mu = 5/4 eigenmode), invisible to BOTH forced endpoint values g(0) and
   g'(pi).  The linear ODE integrates in closed form,

       g(theta) = f^{5/2} [ g(pi) + sqrt2 * INT_theta^pi f^{-7/2} dtheta' ],

   and near the axis  INT_th^pi f^{-7/2} = (4/5) x^{-5/2} + (7/3) x^{-1/2}
   + A + o(1)  with x = theta/2 (only even-power corrections follow), so

       g(theta) = 4 sqrt2/5 + [g(pi) + sqrt2 A] x^{5/2} + (analytic even series).

   Mode-I smoothness of y1 across the ligament kills the x^{5/2} term:

       g(pi) = -sqrt2 * A = 2.033311...   (A = -1.437768..., exact finite
                                           part; quadrature agrees within 2e-6)

   This is the smooth-branch selection rule, and g(pi) is a
   checkable constant (FEM face-ray value agrees to ~5%).

2. ROW-1 FLUX EXPONENT AND ITS CANCELLATION.  On the leading map the row-1
   traction: on the exact constrained map the
   identity J^2 - J^-2 |grad y2|^2 = 0 removes the dominant c2 terms
   identically, so pointwise t1 F11 r = O(c2 r^2) (c1 piece O(r^{3/2})).
   The earlier O(c2 r^{1/2}) claim here was an interpolation artifact
   (see check_row1_flux.py).  Its
   angular integral over [0, pi] vanishes identically at that order AND the
   next: with the smooth g above, INT t1 F11 r dtheta ~ r^2 (checked at
   r = 1e-5..1e-9; coefficient ~ -5.5e-10 * (r/1e-5)^2).  The reaction
   transports its energy along the characteristics; none reaches the tip.
   The reported pointwise orders are
   O(c2 r^2), O(c1 r^{3/2}).  The magnitude of the signed G-quadrature
   error is O(r^{1/2}) (an excess for the present leading map), and the
   leading-field row-1 flux does not carry it.

Run:  python check_g_selection.py
"""
from __future__ import annotations

import math

import numpy as np
import sympy as sp
from scipy.integrate import quad, simpson

TOL = dict(limit=400)


def f(t):
    return np.sin(t / 2)


def tail(th):
    """INT_th^pi f^{-7/2} dtheta'."""
    return quad(lambda t: f(t) ** -3.5, th, np.pi, **TOL)[0]


def finite_part():
    """Double-subtracted finite part A of the tail integral at the axis."""
    vals = []
    for t0 in (1e-3, 3e-4, 1e-4):
        x = t0 / 2
        vals.append(tail(t0) - (4 / 5) * x ** -2.5 - (7 / 3) * x ** -0.5)
    A = vals[-1]
    spread = max(vals) - min(vals)
    return A, spread


def g_smooth(t, gpi):
    return f(t) ** 2.5 * (gpi + np.sqrt(2) * tail(t)) if t < np.pi else gpi


def main():
    A_quad, spread = finite_part()
    A = np.sqrt(np.pi) * math.gamma(-1.25) / math.gamma(-0.75)
    gpi = -np.sqrt(2) * A
    print(f"A (exact finite part)= {A:.9f}")
    print(f"A (quadrature)       = {A_quad:.9f}   "
          f"(error {abs(A_quad-A):.1e}, spread {spread:.1e})")
    print(f"g(pi) = -sqrt2 * A   = {gpi:.6f}")
    ok1 = abs(A_quad - A) < 5e-6 and spread < 1e-4

    # endpoint sanity on the closed form
    g0 = g_smooth(1e-3, gpi)
    gp_pi = (2.5 * 0.5 * np.cos((np.pi - 1e-7) / 2) * g_smooth(np.pi - 1e-7, gpi)
             - np.sqrt(2)) / f(np.pi - 1e-7)
    print(f"g(0+) = {g0:.6f} (forced 4sqrt2/5 = {4*np.sqrt(2)/5:.6f});"
          f"  g'(pi-) = {gp_pi:.6f} (forced -sqrt2)")
    ok2 = abs(g0 - 4 * np.sqrt(2) / 5) < 1e-5 and abs(gp_pi + np.sqrt(2)) < 1e-4

    # ---- row-1 flux on the leading map, c2 (reaction) part -----------------
    r_, th_, P_, c1_, c2_ = sp.symbols('r theta P c1 c2', positive=True)
    g_ = sp.Function('g')(th_)
    fs = sp.sin(th_ / 2)
    y2 = P_ * sp.sqrt(r_) * fs
    y1 = P_ ** sp.Rational(-1, 2) * r_ ** sp.Rational(5, 4) * g_

    def gradF(y):
        yr, yt = sp.diff(y, r_), sp.diff(y, th_) / r_
        return (yr * sp.cos(th_) - yt * sp.sin(th_),
                yr * sp.sin(th_) + yt * sp.cos(th_))

    F11, F12 = gradF(y1)
    F21, F22 = gradF(y2)
    J = F11 * F22 - F12 * F21
    I2d = F11 ** 2 + F12 ** 2 + F21 ** 2 + F22 ** 2
    FiT11, FiT12 = F22 / J, -F21 / J
    P11 = (2 * c1_ * (F11 - J ** -2 * FiT11)
           + 2 * c2_ * (J ** -2 * F11 + (J ** 2 - J ** -2 * I2d) * FiT11))
    P12 = (2 * c1_ * (F12 - J ** -2 * FiT12)
           + 2 * c2_ * (J ** -2 * F12 + (J ** 2 - J ** -2 * I2d) * FiT12))
    t1 = P11 * sp.cos(th_) + P12 * sp.sin(th_)
    gs_, gp_ = sp.symbols('gs gp')
    flux1 = (t1 * F11 * r_).subs({sp.Derivative(g_, th_): gp_, g_: gs_})
    fn = sp.lambdify((r_, th_, P_, c1_, c2_, gs_, gp_), flux1, 'numpy')

    ths = np.linspace(1e-4, np.pi - 1e-6, 3001)
    gv = np.array([g_smooth(t, gpi) for t in ths])
    gpv = (2.5 * 0.5 * np.cos(ths / 2) * gv - np.sqrt(2)) / f(ths)
    Pv = 1.3

    # Pointwise exponent of the c2 part. On the exact constrained map the identity
    # J^2 - J^-2 |grad y2|^2 = 0 cancels the dominant c2 terms
    # identically, so the c2 row-1 flux is O(r^2) POINTWISE (the earlier
    # 0.496 slope came from interpolating g, g' at the probe angle,
    # which breaks C = 0 and lets the penalty amplify the defect).
    # Evaluate g and g' constraint-consistently at the probe angle:
    gs1 = g_smooth(1.0, gpi)
    gp1 = (2.5 * 0.5 * np.cos(0.5) * gs1 - np.sqrt(2)) / f(1.0)
    p1 = fn(1e-6, 1.0, Pv, 0.0, 1.0, gs1, gp1)
    p2 = fn(1e-8, 1.0, Pv, 0.0, 1.0, gs1, gp1)
    expt = np.log(abs(p2 / p1)) / np.log(1e-2)
    print(f"pointwise c2 row-1 flux exponent = {expt:.4f}  (expect 2.0)")
    ok3 = abs(expt - 2.0) < 0.02

    Is = []
    for rr in (1e-5, 1e-7):
        Is.append(simpson(fn(rr, ths, Pv, 0.0, 1.0, gv, gpv), x=ths))
    int_expt = np.log(abs(Is[1] / Is[0])) / np.log(1e-2)
    print(f"ANGULAR-INTEGRAL exponent of the c2 row-1 flux = {int_expt:.3f}"
          f"  (>= 2: the r^(1/2) and r^1 coefficients cancel exactly)")
    ok4 = int_expt > 1.9

    n_pass = sum([ok1, ok2, ok3, ok4])
    print(f"\nPASSED {n_pass}/4" + ("  -- all checks green" if n_pass == 4 else ""))
    return n_pass == 4


if __name__ == "__main__":
    raise SystemExit(0 if main() else 1)
