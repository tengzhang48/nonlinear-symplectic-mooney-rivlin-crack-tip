#!/usr/bin/env python3
"""Energy release rate of the constrained Mooney-Rivlin crack tip.

Main result (verified here by two independent routes):

    G = J-integral = (pi/2) c1 P^2        [exact]

with P the opening intensity (y2 = P r^{1/2} sin(theta/2) + ...).  The leading
J-flux is carried by the opening row alone: the W-term of the Eshelby flux is
theta-independent at leading order (W r -> c1 P^2 / 4) and integrates to zero
against cos(theta); the in-plane row and every c2 contribution vanish as
r -> 0.  Hence the RELATION G(P) is shared with the neo-Hookean class (same
opening row); the I2 physics is in what P means kinematically (the constrained
uniaxial state), not in the leading flux.

Corollary:  sigma22 * r -> c1 P^2 / 2 = G / pi   (flat in theta).

Route 1 (symbolic): build F exactly on the two-term map
    y1 = Q1 r^{5/4} g(theta),  y2 = P r^{1/2} sin(theta/2)
with g, g' arbitrary symbols, form the exact reduced-plane-stress W and
PK1 = dW/dF, take the r -> 0+ limit of the circular J-integrand
    L(theta) = lim r [ W cos(theta) - t1 F11 - t2 F21 ],   t_i = P_iJ N_J,
check L is independent of (g, g', Q1, c2), and integrate exactly.

Route 2 (numeric, algorithmically independent): no asymptotic dropping at all.
Evaluate the FULL W and PK1 (all terms, c2 included) on the composite leading
field with the actual g(theta) profile from the Delta-ODE, quadrature the
J-integral on circles r = 1e-8 .. 1e-3, and check J(r) -> (pi/2) c1 P^2 with
the finite-r drift decaying as the omitted-correction estimate.

Specimen chain (no FEM required): for the clamped pure-shear strip of
reference height h (Rivlin-Thomas), J over the remote boundary is exact:
    G = h W_inf,   W_inf = (c1 + c2)(lambda^2 + lambda^-2 - 2)
(for the pure-shear state (1, lambda, 1/lambda) one has I1 = I2 exactly), so

    P(lambda) = sqrt[ 2 h (c1+c2)(lambda^2 + lambda^-2 - 2) / (pi c1) ].

Run:  python energy_release_rate.py
"""
from __future__ import annotations

import numpy as np
import sympy as sp

from leading_field import A1, A2, f as f_np, fp as fp_np, solve_g_leading, g_family, W_mr, P_mr

PI_HALF = np.pi / 2.0


# ==========================================================================
# Route 1: symbolic limit of the circular J-integrand, exact integration.
# ==========================================================================
def route1_symbolic(verbose=True):
    s, th = sp.symbols("s theta", positive=True)   # r = s**4 keeps powers integer
    P, Q1, c1, c2 = sp.symbols("P Q1 c1 c2", positive=True)
    gs, gps = sp.symbols("g gp", real=True)        # g(theta), g'(theta) pointwise
    r = s ** 4
    fs = sp.sin(th / 2)
    fps = sp.cos(th / 2) / 2

    a1, a2 = sp.Rational(5, 4), sp.Rational(1, 2)
    dy1_dr = a1 * Q1 * r ** (a1 - 1) * gs
    dy1_dth = Q1 * r ** a1 * gps
    dy2_dr = a2 * P * r ** (a2 - 1) * fs
    dy2_dth = P * r ** a2 * fps

    def grad(dr_, dth_):
        return (sp.cos(th) * dr_ - sp.sin(th) / r * dth_,
                sp.sin(th) * dr_ + sp.cos(th) / r * dth_)

    F11, F12 = grad(dy1_dr, dy1_dth)
    F21, F22 = grad(dy2_dr, dy2_dth)
    J = sp.expand(F11 * F22 - F12 * F21)
    FF = F11 ** 2 + F12 ** 2 + F21 ** 2 + F22 ** 2

    # reduced incompressible plane-stress MR energy and PK1 (as in the paper)
    Fm = sp.Matrix([[sp.Symbol("F11"), sp.Symbol("F12")],
                    [sp.Symbol("F21"), sp.Symbol("F22")]])
    Jm = Fm.det()
    FFm = sum(Fm[i, j] ** 2 for i in range(2) for j in range(2))
    Wm = c1 * (FFm + Jm ** -2 - 3) + c2 * (Jm ** 2 + FFm * Jm ** -2 - 3)
    dW = {(i, j): sp.diff(Wm, Fm[i, j]) for i in range(2) for j in range(2)}
    subsF = {sp.Symbol("F11"): F11, sp.Symbol("F12"): F12,
             sp.Symbol("F21"): F21, sp.Symbol("F22"): F22}

    W = Wm.subs(subsF)
    P11, P12 = dW[(0, 0)].subs(subsF), dW[(0, 1)].subs(subsF)
    P21, P22 = dW[(1, 0)].subs(subsF), dW[(1, 1)].subs(subsF)

    t1 = P11 * sp.cos(th) + P12 * sp.sin(th)      # N = e_r on the circle
    t2 = P21 * sp.cos(th) + P22 * sp.sin(th)
    integrand = r * (W * sp.cos(th) - t1 * F11 - t2 * F21)

    L = sp.limit(sp.together(integrand), s, 0, "+")
    L = sp.simplify(L)

    # expected limit: c1 P^2 [ cos(th)/4 - f((1/2)cos(th) f - sin(th) f') ]
    L_expected = c1 * P ** 2 * (sp.cos(th) / 4
                                - fs * (sp.cos(th) * fs / 2 - sp.sin(th) * fps))
    diff = sp.simplify(L - L_expected)

    ok_indep = not (L.has(gs) or L.has(gps) or L.has(Q1) or L.has(c2))
    ok_form = diff == 0
    Jint = sp.integrate(L, (th, -sp.pi, sp.pi))
    Jint = sp.simplify(Jint)
    ok_val = sp.simplify(Jint - sp.pi / 2 * c1 * P ** 2) == 0

    if verbose:
        print("Route 1 (symbolic):")
        print(f"  limiting integrand L(theta) = {sp.simplify(L)}")
        print(f"  L independent of g, g', Q1, c2 : {ok_indep}")
        print(f"  L equals expected closed form  : {ok_form}")
        print(f"  integral over (-pi, pi)        : {Jint}   [expect pi/2 c1 P^2]")
    return ok_indep and ok_form and ok_val


# ==========================================================================
# Route 2: full numeric J(r) on the composite leading field (nothing dropped).
# ==========================================================================
def _field_F(rr, th, Pamp, g_fun, gp_fun):
    """Exact deformation gradient of the composite leading map at (rr, theta).

    y1 even in theta, y2 odd; g is tabulated on [0, pi].
    """
    ath = np.abs(th)
    sgn = np.sign(th)
    Q1 = Pamp ** -0.5
    g = g_fun(ath)
    gp = gp_fun(ath) * sgn                      # g odd derivative continuation
    ff = f_np(th)                               # sin(th/2), odd, valid for th<0
    ffp = fp_np(th)                             # (1/2)cos(th/2), even

    dy1_dr = A1 * Q1 * rr ** (A1 - 1.0) * g
    dy1_dth = Q1 * rr ** A1 * gp
    dy2_dr = A2 * Pamp * rr ** (A2 - 1.0) * ff
    dy2_dth = Pamp * rr ** A2 * ffp

    c, s = np.cos(th), np.sin(th)
    F11 = c * dy1_dr - s / rr * dy1_dth
    F12 = s * dy1_dr + c / rr * dy1_dth
    F21 = c * dy2_dr - s / rr * dy2_dth
    F22 = s * dy2_dr + c / rr * dy2_dth
    return F11, F12, F21, F22


def j_integral_circle(rr, Pamp, g_fun, gp_fun, c1=1.0, c2=1.0, n_th=4001):
    th = np.linspace(-np.pi, np.pi, n_th)
    F11, F12, F21, F22 = _field_F(rr, th, Pamp, g_fun, gp_fun)
    vals = np.empty_like(th)
    for i in range(th.size):
        F = np.array([[F11[i], F12[i]], [F21[i], F22[i]]])
        W = W_mr(F, c1, c2)
        PK = P_mr(F, c1, c2)
        c, s = np.cos(th[i]), np.sin(th[i])
        t1 = PK[0, 0] * c + PK[0, 1] * s
        t2 = PK[1, 0] * c + PK[1, 1] * s
        vals[i] = rr * (W * c - t1 * F11[i] - t2 * F21[i])
    return np.trapezoid(vals, th)


def route2_numeric(Pamp=1.3, verbose=True):
    sol = solve_g_leading()
    g_fun, gp_fun = g_family(sol, A0=0.0)
    radii = np.array([1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3])
    target = PI_HALF * Pamp ** 2                  # c1 = 1
    rows, signed_rel = [], []
    for rr in radii:
        Jmr = j_integral_circle(rr, Pamp, g_fun, gp_fun, c1=1.0, c2=1.0)
        Jc2z = j_integral_circle(rr, Pamp, g_fun, gp_fun, c1=1.0, c2=0.0)
        rows.append((rr, Jmr / target, Jc2z / target))
        signed_rel.append(Jmr / target - 1.0)
    signed_rel = np.array(signed_rel)
    rel = np.abs(signed_rel)
    # The composite field omits the relative-r^{1/2} (gap-half) correction, so
    # the magnitude of the signed finite-r truncation error must decay like
    # r^{1/2}: check the slope, then
    # extrapolate J(r) = J0 + C r^{1/2} to r = 0 from the two smallest radii.
    slope = np.polyfit(np.log(radii[:4]), np.log(rel[:4]), 1)[0]
    r1, r2 = radii[0], radii[1]
    J1, J2 = rows[0][1], rows[1][1]
    C = (J2 - J1) / (np.sqrt(r2) - np.sqrt(r1))
    J0 = J1 - C * np.sqrt(r1)
    if verbose:
        print("\nRoute 2 (numeric, full W and PK1 on the composite field):")
        print("      r        J_MR/(pi/2 c1 P^2)   J_{c2=0}/(pi/2 c1 P^2)")
        for rr, a, b in rows:
            print(f"  {rr:8.1e}   {a:16.10f}   {b:18.10f}")
        sign = "excess" if rows[0][1] > 1 else "deficit"
        print(f"  finite-r signed-error slope = {slope:.4f} "
              f"({sign}; gap-half prediction: 1/2)")
        print(f"  r^(1/2)-extrapolated  J/(pi/2 c1 P^2) = {J0:.10f}")
    ok_slope = abs(slope - 0.5) < 0.02
    ok_extrap = abs(J0 - 1.0) < 1e-6
    ok_excess = bool(np.all(signed_rel > 0.0))
    # class-universality of the leading flux: c2 does not change the limit
    ok_c2 = abs(rows[0][2] - rows[0][1]) < 2e-4
    return ok_slope and ok_extrap and ok_excess and ok_c2, rows


# ==========================================================================
# Corollary and the pure-shear strip chain.
# ==========================================================================
def corollary_sigma22(Pamp=1.3, rr=1e-7, verbose=True):
    """sigma22 * r -> c1 P^2 / 2 = G/pi, flat in theta (sigma = PK1 F^T, J3d=1)."""
    sol = solve_g_leading()
    g_fun, gp_fun = g_family(sol, A0=0.0)
    th = np.linspace(-np.pi + 1e-6, np.pi - 1e-6, 721)
    F11, F12, F21, F22 = _field_F(rr, th, Pamp, g_fun, gp_fun)
    s22 = np.empty_like(th)
    for i in range(th.size):
        F = np.array([[F11[i], F12[i]], [F21[i], F22[i]]])
        PK = P_mr(F, 1.0, 1.0)
        s22[i] = PK[1, 0] * F[1, 0] + PK[1, 1] * F[1, 1]
    val = s22 * rr
    pred = Pamp ** 2 / 2.0
    spread = float((val.max() - val.min()) / pred)
    err = float(abs(val.mean() / pred - 1.0))
    if verbose:
        print(f"\nCorollary: sigma22*r vs c1 P^2/2 = G/pi at r={rr:.0e}:"
              f"  mean/pred-1 = {err:.2e},  theta-spread = {spread:.2e}")
    return err < 5e-3 and spread < 5e-3


def strip_chain(verbose=True):
    lam, c1, c2 = sp.symbols("lambda c1 c2", positive=True)
    l1, l2, l3 = sp.Integer(1), lam, 1 / lam            # pure shear state
    I1 = l1 ** 2 + l2 ** 2 + l3 ** 2
    I2 = l1 ** 2 * l2 ** 2 + l2 ** 2 * l3 ** 2 + l3 ** 2 * l1 ** 2
    ok = sp.simplify(I1 - I2) == 0
    Winf = sp.simplify(c1 * (I1 - 3) + c2 * (I2 - 3))
    if verbose:
        print("\nPure-shear strip (height h, no FEM):  I1 == I2 :", ok)
        print(f"  W_inf = {sp.factor(Winf)}")
        print("  G = h W_inf  =>  P(lambda) = sqrt[2 h (c1+c2)(lam^2+lam^-2-2)/(pi c1)]")
        for lv in (1.5, 2.0, 3.0):
            Pv = np.sqrt(2 * 1.0 * 2.0 * (lv ** 2 + lv ** -2 - 2) / (np.pi * 1.0))
            print(f"    h=1, c1=c2=1, lambda={lv}:  P = {Pv:.5f}")
    return ok


def main():
    print("Energy release rate of the constrained MR crack tip: G = (pi/2) c1 P^2")
    ok1 = route1_symbolic()
    ok2, _ = route2_numeric()
    ok3 = corollary_sigma22()
    ok4 = strip_chain()
    checks = {
        "route 1: symbolic limit + exact integral = (pi/2) c1 P^2": ok1,
        "route 2: positive finite-r excess -> (pi/2) c1 P^2, c2-independent": ok2,
        "corollary: sigma22 * r = c1 P^2 / 2 = G/pi, flat in theta": ok3,
        "strip: I1 = I2 in pure shear (G = h (c1+c2)(lam^2+lam^-2-2))": ok4,
    }
    print("\n  checks:")
    for k, v in checks.items():
        print(f"    [{'PASS' if v else 'FAIL'}] {k}")
    if not all(checks.values()):
        raise SystemExit("energy_release_rate self-checks FAILED")
    print("\nEnergy-release-rate self-checks passed (G = (pi/2) c1 P^2).")


if __name__ == "__main__":
    main()
