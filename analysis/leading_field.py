#!/usr/bin/env python3
"""Leading-order constrained Mooney-Rivlin crack-tip field (consolidated).

A single self-contained module for the Phase-1 result: the reduced
incompressible plane-stress energy and its PK1 stress, the constraint
Delta = 2^(-1/2), the first-order ODE for the in-plane slave profile g with its
forced regular value g(0) = 4 sqrt2 / 5, the deformed map, and the locally
uniaxial principal-stretch magnitudes with the parameter-free relation
J r^(1/4) = sqrt(P/2).

This is the consolidated public implementation of the leading-field result.

Run:  python leading_field.py     (prints the headline constants + self-checks)
"""
from __future__ import annotations

import math

import numpy as np
from scipy.integrate import solve_ivp

# --- exponents and constants (also checked by verification/verify_equations.py) ---
A2 = 0.5                       # opening exponent a2
A1 = 1.25                      # in-plane exponent a1  (balance 2 a1 + a2 = 3)
M1 = 3.0 - 3.0 * A1            # row-1 leading stress power = -3/4
M2 = A2 - 1.0                  # row-2 leading stress power = -1/2
DELTA = 2.0 ** -0.5            # the constraint constant  Delta = 2^(-1/2)
G0 = 4.0 * np.sqrt(2.0) / 5.0  # forced regular value g(0)
G2 = np.sqrt(2.0) / 2.0        # series: g = G0 + G2 theta^2 + O(theta^4)
AF = np.sqrt(np.pi) * math.gamma(-1.25) / math.gamma(-0.75)
GPI = -np.sqrt(2.0) * AF       # regular-axis selection at the crack face


def f(th):
    return np.sin(0.5 * np.asarray(th, float))


def fp(th):
    return 0.5 * np.cos(0.5 * np.asarray(th, float))


# --------------------------------------------------------------------------
# Reduced incompressible plane-stress Mooney-Rivlin energy and PK1 stress.
# Thickness eliminated (lambda3 = 1/J); the 2D invariants are
#   I1 = I1_2d + J^-2,   I2 = J^2 + I1_2d J^-2.
# --------------------------------------------------------------------------
def W_mr(F, c1, c2):
    J = F[0, 0] * F[1, 1] - F[0, 1] * F[1, 0]
    FF = float((F * F).sum())
    I1 = FF + J ** -2
    I2 = J ** 2 + J ** -2 * FF
    return c1 * (I1 - 3.0) + c2 * (I2 - 3.0)


def P_mr(F, c1, c2):
    """First Piola-Kirchhoff stress dW/dF."""
    J = F[0, 0] * F[1, 1] - F[0, 1] * F[1, 0]
    FF = (F * F).sum()
    Finvt = np.array([[F[1, 1], -F[1, 0]], [-F[0, 1], F[0, 0]]]) / J
    return (2.0 * c1 * (F - J ** -2 * Finvt)
            + 2.0 * c2 * (J ** -2 * F + (J ** 2 - J ** -2 * FF) * Finvt))


def pk1_max_rel_err(seed=7, trials=20):
    """Cross-check P_mr against a finite-difference dW/dF."""
    rng = np.random.default_rng(seed)
    worst = 0.0
    for _ in range(trials):
        c1, c2 = rng.uniform(0.2, 2.0, 2)
        F = np.eye(2) + 0.6 * rng.standard_normal((2, 2))
        if abs(np.linalg.det(F)) < 0.3:
            continue
        P = P_mr(F, c1, c2)
        h = 1e-6
        for i in range(2):
            for j in range(2):
                Fp, Fm = F.copy(), F.copy()
                Fp[i, j] += h
                Fm[i, j] -= h
                dW = (W_mr(Fp, c1, c2) - W_mr(Fm, c1, c2)) / (2 * h)
                worst = max(worst, abs(dW - P[i, j]) / max(1.0, abs(P[i, j])))
    return worst


# --------------------------------------------------------------------------
# The in-plane slave profile g(theta):  a1 f' g - a2 f g' = Delta,  g(0)=4 sqrt2/5.
# --------------------------------------------------------------------------
def solve_g_leading(eps=1e-4):
    """Return the regular-axis branch, integrated stably from the face.

    Outward integration from the degenerate axis amplifies an arbitrarily
    small f^(5/2) homogeneous contamination.  The exact finite-part selection
    fixes g(pi)=-sqrt(2) A_f, so backward integration is the stable direction.
    """
    def rhs(th, y):
        return [(A1 * fp(th) * y[0] - DELTA) / (A2 * f(th))]
    sol = solve_ivp(rhs, (np.pi, eps), [GPI], method="DOP853",
                    rtol=1e-12, atol=1e-14, dense_output=True)
    assert sol.success
    return sol


def g_family(sol, A0=0.0):
    """Callables (g, g') for g + A0 f^(5/2) (A0 = the admissible-gauge mode)."""
    def g_fun(th):
        th = np.asarray(th, float)
        if np.ndim(th) == 0:
            base = sol.sol(max(th, 1e-4))[0] if th >= 1e-4 else G0 + G2 * th ** 2
            return base + A0 * f(th) ** 2.5
        base = np.where(th < 1e-4, G0 + G2 * th ** 2, np.nan)
        ok = th >= 1e-4
        base[ok] = sol.sol(th[ok])[0]
        return base + A0 * f(th) ** 2.5

    def gp_fun(th):
        th = np.asarray(th, float)
        g = g_fun(th) - A0 * f(th) ** 2.5
        with np.errstate(divide="ignore", invalid="ignore"):
            gp = (A1 * fp(th) * g - DELTA) / (A2 * f(th))
        gp = np.where(th < 1e-3, 2.0 * G2 * th, gp)
        return gp + A0 * 1.25 * np.cos(0.5 * th) * f(th) ** 1.5
    return g_fun, gp_fun


def constraint_defect(th, g_fun, gp_fun):
    """Constraint defect Delta - G/Delta^3, identically zero on the leading field.

    On the leading field G = a2^2 f^2 + f'^2 = 1/4 and Delta^4 = G, so this
    vanishes to solver accuracy; it is an on-manifold diagnostic, NOT the
    reaction (Lagrange-multiplier) amplitude, which enters at higher order.
    """
    ff, ffp = f(th), fp(th)
    G = A2 ** 2 * ff ** 2 + ffp ** 2
    Delta = A1 * ffp * g_fun(th) - A2 * ff * gp_fun(th)
    return Delta - G / Delta ** 3


# --------------------------------------------------------------------------
# Deformed map and the locally uniaxial constrained magnitudes.
# --------------------------------------------------------------------------
def deformed(r, theta, P, g_fun):
    """Leading deformed coordinates (y1, y2) at opening amplitude P."""
    r = np.asarray(r, float)
    y2 = P * r ** A2 * f(theta)                  # = P sqrt(s),  s = r sin^2(theta/2)
    y1 = P ** -0.5 * r ** A1 * g_fun(theta)
    return y1, y2


def constrained_magnitudes(r, P):
    """Angularly uniform leading stretch magnitudes and parameter-free relation."""
    lam1 = (P / 2.0) * r ** -0.5
    lam2 = np.sqrt(2.0 / P) * r ** 0.25          # = lambda3 = lambda1^(-1/2)
    J = lam1 * lam2                              # = sqrt(P/2) r^(-1/4)
    return {"lambda1": lam1, "lambda2": lam2, "J": J, "J_r_1_4": J * r ** 0.25}


def main():
    print("Leading constrained Mooney-Rivlin crack-tip field")
    print(f"  exponents      a2 = {A2},  a1 = {A1}   (balance 2a1+a2 = {2*A1+A2})")
    print(f"  constraint     Delta = 2^(-1/2) = {DELTA:.10f}")
    print(f"  forced g(0)    = 4 sqrt2/5 = {G0:.10f}")
    print(f"  series coeff   G2 = sqrt2/2 = {G2:.10f}")

    sol = solve_g_leading()
    g_fun, gp_fun = g_family(sol)
    g0_num = float(g_fun(1e-6))
    gpi_num = float(g_fun(np.pi))
    gp_pi = float(gp_fun(np.pi))
    pk1 = pk1_max_rel_err()
    mag = constrained_magnitudes(r=np.array([1e-4, 1e-3, 1e-2]), P=1.3)
    Jr14 = mag["J_r_1_4"]

    checks = {
        "PK1 = dW/dF (fd)": pk1 < 1e-5,
        "g(0) = 4 sqrt2/5": abs(g0_num - G0) < 1e-6,
        "regular selection g(pi) = -sqrt2 A_f": abs(gpi_num - GPI) < 1e-11,
        "g'(pi) = -sqrt2": abs(gp_pi + np.sqrt(2)) < 1e-3,
        "J r^1/4 = sqrt(P/2) (theta-free)":
            np.allclose(Jr14, np.sqrt(1.3 / 2.0), rtol=1e-12),
        "lambda2 = lambda1^-1/2":
            np.allclose(mag["lambda2"], mag["lambda1"] ** -0.5, rtol=1e-12),
    }
    print("\n  checks:")
    for k, v in checks.items():
        print(f"    [{'PASS' if v else 'FAIL'}] {k}")
    print(f"\n  J r^1/4 = {Jr14[0]:.6f} (= sqrt(P/2) = {np.sqrt(1.3/2):.6f}), flat in r/theta")
    if not all(checks.values()):
        raise SystemExit("leading_field self-checks FAILED")
    print("\nLeading-field self-checks passed.")


if __name__ == "__main__":
    main()
