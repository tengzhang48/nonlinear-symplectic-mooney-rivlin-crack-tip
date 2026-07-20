#!/usr/bin/env python3
"""Energy release rate of the constrained Mooney-Rivlin crack tip.

Main result (cross-checked here by two complementary computational routes):

    G = J-integral = (pi/2) c1 P^2        [exact]

with P the opening intensity (y2 = P r^{1/2} sin(theta/2) + ...).  The leading
J-flux is carried by the opening row alone: the W-term of the Eshelby flux is
theta-independent at leading order (W r -> c1 P^2 / 4) and integrates to zero
against cos(theta); the in-plane row and every c2 contribution vanish as
r -> 0.  Hence the RELATION G(P) is shared with the neo-Hookean class (same
opening row); the I2 physics is in what P means kinematically (the constrained
uniaxial state), not in the leading flux.

Corollary:  sigma22 * r -> c1 P^2 / 2 = G / pi   (flat in theta).

Route 1 (symbolic): build F exactly on the superposed truncated map
    y1 = C_s r sin(theta/2)^2 + Q1 r^{5/4} g(theta),
    y2 = P r^{1/2} sin(theta/2)
with g, g' arbitrary symbols, form the exact reduced-plane-stress W and
PK1 = dW/dF, and extract the exact r -> 0+ coefficient of the circular
J-integrand
    L(theta) = lim r [ W cos(theta) - t1 F11 - t2 F21 ],   t_i = P_iJ N_J,
check L is independent of (g, g', Q1, C_s, c2), and integrate exactly.

Route 2 (numeric, separate implementation): no asymptotic dropping at all.
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
# Route 1: symbolic Laurent coefficient of the circular J-integrand.
# ==========================================================================
def _laurent_coefficients(expr, variable):
    """Return exact coefficients of a finite Laurent expression.

    On the route-1 map, the determinant is a monomial in ``variable``.  The
    full energy-flux integrand is therefore a finite Laurent polynomial even
    when a general-purpose simplifier leaves a common power in a denominator.
    Extracting powers term by term is both explicit and fast.
    """
    coefficients = {}
    for raw_term in sp.Add.make_args(sp.expand(expr)):
        term = sp.factor_terms(raw_term, variable)
        power = term.as_powers_dict().get(variable, sp.Integer(0))
        if power.is_Integer is not True:
            raise AssertionError(f"noninteger Laurent power {power}: {term}")
        coefficient = term / variable ** power
        if coefficient.has(variable):
            raise AssertionError(
                f"failed to extract the {variable}-power from term {term}"
            )
        coefficients[power] = coefficients.get(power, sp.Integer(0)) + coefficient
    return coefficients


def route1_symbolic(verbose=True):
    eps, th = sp.symbols("epsilon theta", positive=True)  # r = eps**4
    P, Q1, c1, c2 = sp.symbols("P Q1 c1 c2", positive=True)
    Cs = sp.symbols("C_s", real=True)
    gs, gps = sp.symbols("g gp", real=True)        # g(theta), g'(theta) pointwise
    fs = sp.sin(th / 2)
    hs = sp.cos(th / 2)
    fps = hs / 2
    ct, st = sp.cos(th), sp.sin(th)

    # Exact polar derivatives after r=eps**4.  The regular C_s row is
    # parallel to the opening row and must cancel from the determinant.
    dy1_dr = Cs * fs ** 2 + sp.Rational(5, 4) * Q1 * eps * gs
    invr_dy1_dth = Cs * st / 2 + Q1 * eps * gps
    dy2_dr = P * fs / (2 * eps ** 2)
    invr_dy2_dth = P * fps / eps ** 2

    def cartesian_gradient(dr_, invr_dth_):
        return (ct * dr_ - st * invr_dth_,
                st * dr_ + ct * invr_dth_)

    F_direct = sp.Matrix([
        cartesian_gradient(dy1_dr, invr_dy1_dth),
        cartesian_gradient(dy2_dr, invr_dy2_dth),
    ])
    F = sp.Matrix([
        [-Cs * fs ** 2
         + eps * Q1 * (sp.Rational(5, 4) * gs * ct - gps * st),
         Cs * fs * hs
         + eps * Q1 * (sp.Rational(5, 4) * gs * st + gps * ct)],
        [-P * fs / (2 * eps ** 2), P * hs / (2 * eps ** 2)],
    ])
    ok_gradient = all(
        sp.trigsimp(sp.factor_terms(sp.expand(F_direct[i, j] - F[i, j]))) == 0
        for i in range(2) for j in range(2)
    )

    J_direct = sp.expand(F.det())
    j_minus_one = P * Q1 * (
        sp.Rational(5, 8) * gs * hs - sp.Rational(1, 2) * gps * fs
    )
    ok_null_J = (
        not J_direct.has(Cs)
        and sp.trigsimp(eps * J_direct - j_minus_one) == 0
    )

    # Constitutive identity on a generic matrix, kept separate from the
    # crack-tip substitution.
    z11, z12, z21, z22 = sp.symbols("z11 z12 z21 z22")
    F_generic = sp.Matrix([[z11, z12], [z21, z22]])
    J_generic = F_generic.det()
    FF_generic = sum(
        F_generic[i, j] ** 2 for i in range(2) for j in range(2)
    )
    W_generic = (
        c1 * (FF_generic + J_generic ** -2 - 3)
        + c2 * (J_generic ** 2 + FF_generic * J_generic ** -2 - 3)
    )
    PK1_from_W = sp.Matrix(
        2, 2, lambda i, j: sp.diff(W_generic, F_generic[i, j])
    )
    PK1_closed = (
        2 * c1 * (F_generic - J_generic ** -2 * F_generic.inv().T)
        + 2 * c2 * (
            J_generic ** -2 * F_generic
            + (J_generic ** 2 - J_generic ** -2 * FF_generic)
            * F_generic.inv().T
        )
    )
    ok_pk1 = all(
        sp.factor(PK1_from_W[i, j] - PK1_closed[i, j]) == 0
        for i in range(2) for j in range(2)
    )

    # J=j_minus_one/eps turns every inverse determinant into a monomial.
    # Form the full W and PK1 and extract the exact eps**0 coefficient.
    J = j_minus_one / eps
    Jinv = eps / j_minus_one
    Jm2 = Jinv ** 2
    J2 = J ** 2
    FinvT = sp.Matrix([[F[1, 1], -F[1, 0]],
                       [-F[0, 1], F[0, 0]]]) * Jinv
    FF = sum(F[i, j] ** 2 for i in range(2) for j in range(2))
    W = c1 * (FF + Jm2 - 3) + c2 * (J2 + FF * Jm2 - 3)
    PK1 = (
        2 * c1 * (F - Jm2 * FinvT)
        + 2 * c2 * (Jm2 * F + (J2 - Jm2 * FF) * FinvT)
    )
    traction = PK1 * sp.Matrix([ct, st])
    integrand = eps ** 4 * (W * ct - traction.dot(F[:, 0]))
    laurent = _laurent_coefficients(integrand, eps)
    negative = {
        power: sp.trigsimp(sp.factor(coefficient))
        for power, coefficient in laurent.items()
        if power < 0
    }
    ok_limit = all(coefficient == 0 for coefficient in negative.values())
    L = sp.trigsimp(sp.factor(laurent.get(sp.Integer(0), sp.Integer(0))))

    # expected limit: c1 P^2 [ cos(th)/4 - f((1/2)cos(th) f - sin(th) f') ]
    L_expected = c1 * P ** 2 * (sp.cos(th) / 4
                                - fs * (sp.cos(th) * fs / 2 - sp.sin(th) * fps))
    diff = sp.trigsimp(L - L_expected)

    dependencies = (gs, gps, Q1, Cs, c2)
    ok_indep = (
        not any(L.has(symbol) for symbol in dependencies)
        and all(sp.diff(L, symbol) == 0 for symbol in dependencies)
    )
    ok_form = diff == 0
    ok_constant = sp.trigsimp(L - c1 * P ** 2 / 4) == 0
    Jint = sp.integrate(L, (th, -sp.pi, sp.pi))
    Jint = sp.simplify(Jint)
    ok_val = sp.simplify(Jint - sp.pi / 2 * c1 * P ** 2) == 0

    if verbose:
        print("Route 1 (symbolic):")
        print(f"  exact map gradient and C_s-null determinant : "
              f"{ok_gradient and ok_null_J}")
        print(f"  generic PK1 equals dW/dF                    : {ok_pk1}")
        print(f"  Laurent powers in full r-weighted flux      : "
              f"{sorted(int(power) for power in laurent)}")
        print(f"  no negative-power residue                   : {ok_limit}")
        print(f"  limiting integrand L(theta) = {L}")
        print(f"  L independent of g, g', Q1, C_s, c2 : {ok_indep}")
        print(f"  L equals expected closed form  : {ok_form}")
        print(f"  L simplifies to c1 P^2 / 4     : {ok_constant}")
        print(f"  integral over (-pi, pi)        : {Jint}   [expect pi/2 c1 P^2]")
    return all((ok_gradient, ok_null_J, ok_pk1, ok_limit, ok_indep,
                ok_form, ok_constant, ok_val))


# ==========================================================================
# Route 2: full numeric J(r) on the composite leading field (nothing dropped).
# ==========================================================================
def _field_F(rr, th, Pamp, g_fun, gp_fun, C_s=0.0):
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

    dy1_dr = C_s * ff ** 2 + A1 * Q1 * rr ** (A1 - 1.0) * g
    dy1_dth = 2.0 * C_s * rr * ff * ffp + Q1 * rr ** A1 * gp
    dy2_dr = A2 * Pamp * rr ** (A2 - 1.0) * ff
    dy2_dth = Pamp * rr ** A2 * ffp

    c, s = np.cos(th), np.sin(th)
    F11 = c * dy1_dr - s / rr * dy1_dth
    F12 = s * dy1_dr + c / rr * dy1_dth
    F21 = c * dy2_dr - s / rr * dy2_dth
    F22 = s * dy2_dr + c / rr * dy2_dth
    return F11, F12, F21, F22


def j_integral_circle(rr, Pamp, g_fun, gp_fun, c1=1.0, c2=1.0,
                      C_s=0.0, n_th=4001):
    th = np.linspace(-np.pi, np.pi, n_th)
    F11, F12, F21, F22 = _field_F(rr, th, Pamp, g_fun, gp_fun, C_s=C_s)
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
        "route 1: C_s-inclusive symbolic limit + exact integral = (pi/2) c1 P^2": ok1,
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
