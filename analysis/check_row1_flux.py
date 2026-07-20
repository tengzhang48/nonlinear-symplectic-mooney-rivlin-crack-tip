#!/usr/bin/env python3
"""Row-1 flux and signed finite-radius J-error on the selected C_s=0 map.

On the manifold, J^2 - J^-2 |grad y2|^2 = 0 identically, so the c2
row-1 flux is pointwise O(r^2) and the c1 row-1 flux O(r^{3/2}); the
r^{1/2} finite-radius excess of the circle J-integral is real and is
carried by the OPENING-ROW c2 traction flux (coefficient pi c2 P; the
angle-independent c2 energy term integrates to zero against cos theta):
the finite-radius error of this truncated C_s=0 map, not a path
dependence of the exact solution.  Constraint-consistent evaluation throughout:
g'(theta) from the exact g-ODE at the same point as g.
"""
import numpy as np

from leading_field import solve_g_leading, g_family

sol = solve_g_leading()
gf, _ = g_family(sol, A0=0.0)
P, c1, c2 = 1.3, 1.0, 1.0
G_exact = np.pi / 2 * c1 * P**2


def fields(r, th):
    f, fp = np.sin(th / 2), np.cos(th / 2) / 2
    g = np.array([float(gf(t)) for t in np.atleast_1d(th)])
    gp = ((5 / 4) * fp * g - 2**-0.5) / ((1 / 2) * f)
    Q1 = P**-0.5
    F11 = (5 / 4) * Q1 * r**0.25 * g
    F12 = Q1 * r**0.25 * gp
    F21 = (1 / 2) * P * r**-0.5 * f
    F22 = P * r**-0.5 * fp
    return F11, F12, F21, F22


def row1_flux(r, th, use_c1, use_c2):
    F11, F12, F21, F22 = fields(r, np.array([th]))
    J = F11 * F22 - F12 * F21
    FF = F11**2 + F12**2 + F21**2 + F22**2
    Ft11, Ft12 = F22 / J, -F21 / J
    P11 = (2 * use_c1 * (F11 - J**-2 * Ft11)
           + 2 * use_c2 * (J**-2 * F11 + (J**2 - J**-2 * FF) * Ft11))
    P12 = (2 * use_c1 * (F12 - J**-2 * Ft12)
           + 2 * use_c2 * (J**-2 * F12 + (J**2 - J**-2 * FF) * Ft12))
    # traction on the circle (reference normal e_r) is the polar radial
    # component P_{1r} itself; F_{11} = dy1/dX1 = cos F_1r - sin F_1th
    Fx11 = np.cos(th) * F11 - np.sin(th) * F12
    return float((P11 * Fx11 * r)[0])


def Jcirc(r, n=4000):
    th = np.linspace(1e-6, np.pi - 1e-6, n)
    F11, F12, F21, F22 = fields(r, th)
    J = F11 * F22 - F12 * F21
    FF = F11**2 + F12**2 + F21**2 + F22**2
    W = c1 * (FF + J**-2 - 3) + c2 * (J**2 + FF * J**-2 - 3)
    Ft = [[F22 / J, -F21 / J], [-F12 / J, F11 / J]]
    Fs = [[F11, F12], [F21, F22]]
    integ = W * np.cos(th) * r
    for i in range(2):
        Pr = (2 * c1 * (Fs[i][0] - J**-2 * Ft[i][0])
              + 2 * c2 * (J**-2 * Fs[i][0]
                          + (J**2 - J**-2 * FF) * Ft[i][0]))
        Pt = (2 * c1 * (Fs[i][1] - J**-2 * Ft[i][1])
              + 2 * c2 * (J**-2 * Fs[i][1]
                          + (J**2 - J**-2 * FF) * Ft[i][1]))
        Fx = np.cos(th) * Fs[i][0] - np.sin(th) * Fs[i][1]
        integ = integ - Pr * Fx * r
    return 2 * np.trapezoid(integ, th)


def slope(vals, rs):
    return np.log(abs(vals[0] / vals[1])) / np.log(rs[0] / rs[1])


ok = True
rs = (1e-6, 1e-8)
s_c2 = slope([row1_flux(r, 1.0, 0.0, 1.0) for r in rs], rs)
s_c1 = slope([row1_flux(r, 1.0, 1.0, 0.0) for r in rs], rs)
print(f"pointwise row-1 flux exponents: c2 part {s_c2:.4f} (expect 2), "
      f"c1 part {s_c1:.4f} (expect 1.5)")
ok &= abs(s_c2 - 2.0) < 0.02 and abs(s_c1 - 1.5) < 0.02

rd = (1e-3, 1e-5)
signed_errors = np.array([Jcirc(r) - G_exact for r in rd])
s_err = slope(signed_errors, rd)
is_excess = bool(np.all(signed_errors > 0.0))
print(f"signed J-error exponent = {s_err:.4f} (expect 0.5; Jcirc-G > 0)")
ok &= abs(s_err - 0.5) < 0.02 and is_excess

# component decomposition of the excess: opening-row c2 flux carries it
def signed_error_components(r, n=4000):
    th = np.linspace(1e-6, np.pi - 1e-6, n)
    F11, F12, F21, F22 = fields(r, th)
    J = F11 * F22 - F12 * F21
    FF = F11**2 + F12**2 + F21**2 + F22**2
    P2r_c2 = 2 * c2 * (J**-2 * F21 + (J**2 - J**-2 * FF) * (-F12 / J))
    F21c = np.cos(th) * F21 - np.sin(th) * F22
    Iopen = 2 * np.trapezoid(-P2r_c2 * F21c * r, th)
    Wc2 = c2 * (J**2 + FF * J**-2 - 3)
    IW = 2 * np.trapezoid(Wc2 * np.cos(th) * r, th)
    return Iopen, IW

Io, Iw = signed_error_components(1e-6)
co = Io / 1e-3
print(f"opening-row c2 flux coefficient = {co:.6f} "
      f"(predict pi c2 P = {np.pi * c2 * P:.6f}); "
      f"c2 energy-term contribution = {Iw / 1e-3:.2e}")
ok &= abs(co - np.pi * c2 * P) < 1e-3 and abs(Iw / 1e-3) < 1e-4

print("[PASS]" if ok else "[FAIL]", "row-1 flux and signed J-error "
      "scalings + component decomposition")
raise SystemExit(0 if ok else 1)
