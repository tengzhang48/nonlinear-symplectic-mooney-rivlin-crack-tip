#!/usr/bin/env python3
"""Completion-sequence steps 2-4 for the constrained-limit reduction.

Step 2 (scaled action): exact decomposition of the full MR energy
  W_c2 = c2 [2 J^-2 |grad y2|^2 + J^-2 |grad y1|^2 - 3] + c2 J^-2 C,
so the formal constrained-limit leading action used in the interior
reduction retains the c1 part + multiplier term, and the on-manifold c2
energy 2 c2 J^2 is the next order and supplies the level-one forcing.
The r^(1/2) signed J-quadrature error is instead carried by the opening-row
c2 traction flux; its angle-independent energy projection is zero
(Gates F, G and check_row1_flux.py).

Step 3 (normalization and scaling): in the fixed action convention
  C = J^4 - |grad y2|^2,
the base action multiplier is
  chi0 = c1 P^-3 r^(3/2) psi0(theta),
and its homogeneous angular profile is f^(3/2).  Its stress product
chi0*C_F has radial exponent 1/4.  The monomial factor
r^(1/4)f^(3/2)=r^(-1/2)s^(3/4) matches the old transport bookkeeping,
but the scalar action multiplier chi0 is not the old reaction-stress
variable (Gate H).

Step 4 (base reaction with boundary conditions): the c1 base face
traction vanishes identically at theta = pi (via g'(pi) = -sqrt2), so
traction-free faces require psi0(pi) = 0; axis regularity fixes
psi0(0) = -8. Gate J verifies the exact regular solution, its axis series,
the homogeneous branch, and the face compatibility defect psi_reg(pi)=10;
an f^(3/2) admixture of coefficient -10 is required already at the base
level.  Both numbers refer to the dimensionless psi0 in this stated
constraint normalization.
"""
from __future__ import annotations

import sympy as sp

# ---------------------------------------------------------------- Gate F
F = {k: sp.Symbol(f"F{k}") for k in ("11", "12", "21", "22")}
J = F["11"] * F["22"] - F["12"] * F["21"]
FF = sum(v**2 for v in F.values())
g1 = F["11"]**2 + F["12"]**2          # |grad y1|^2
g2 = F["21"]**2 + F["22"]**2          # |grad y2|^2
C = J**4 - g2
Wc2_full = J**2 + FF * J**-2 - 3
Wc2_claim = 2 * J**-2 * g2 + J**-2 * g1 - 3 + J**-2 * C
okF = sp.cancel(sp.together(Wc2_full - Wc2_claim)) == 0
print(f"[{'PASS' if okF else 'FAIL'}] F: exact decomposition of the c2 "
      "energy (multiplier term + on-manifold remainder)")

# ---------------------------------------------------------------- Gate G
# Derive the on-manifold r-powers from a1=5/4, a2=1/2 rather than
# certifying a literal table.  Here eJ is the exponent of J, eg_i that of
# |grad y_i|^2.  C=0 supplies 4eJ=eg2.
a1, a2 = sp.Rational(5, 4), sp.Rational(1, 2)
eJ = a1 + a2 - 2
eg1, eg2 = 2 * a1 - 2, 2 * a2 - 2
orders = {
    "c1 opening  c1|grad y2|^2": eg2,
    "c2 on-manifold  2 c2 J^2": 2 * eJ,
    "c1 in-plane  c1|grad y1|^2": eg1,
    "c1 volumetric  c1 J^-2": -2 * eJ,
    "c2 J^-2 |grad y1|^2": -2 * eJ + eg1,
}
expected_orders = (-1, -sp.Rational(1, 2), sp.Rational(1, 2),
                   sp.Rational(1, 2), 1)
okG = (sp.simplify(4 * eJ - eg2) == 0
       and tuple(orders.values()) == expected_orders)
print(f"[{'PASS' if okG else 'FAIL'}] G: derived on-manifold energy ordering "
      "(r-powers): " + "; ".join(f"{name} ~ r^{power}"
                                    for name, power in orders.items()))
print("       => formal leading constrained action retains c1 + multiplier;")
print("          the 2 c2 J^2 term is the r^(1/2) relative correction "
      "(level-one forcing; the signed J-quadrature error is carried by")
print("          the opening-row c2 traction flux, pi c2 P r^(1/2), not "
      "by the angle-independent energy term).")

# ---------------------------------------------------------------- Gate H
r = sp.Symbol("r", positive=True)
P, c1 = sp.symbols("P c1", positive=True)
fpos = sp.Symbol("f", positive=True)     # sin(theta/2) > 0 on (0, pi)
s_ = r * fpos**2
# Restored balance: C_F1 ~ P^(5/2) r^(-5/4), so matching the c1 base
# row-1 stress c1 P^(-1/2) r^(1/4) fixes chi0 ~ c1 P^-3 r^(3/2).
chi_P_power = sp.solve(sp.Eq(sp.Symbol("x") + sp.Rational(5, 2),
                             -sp.Rational(1, 2)), sp.Symbol("x"))[0]
chi_r_power = sp.solve(sp.Eq(sp.Symbol("z") - sp.Rational(5, 4),
                             sp.Rational(1, 4)), sp.Symbol("z"))[0]
map_identity = sp.simplify(sp.powsimp(
    r**sp.Rational(1, 4) * fpos**sp.Rational(3, 2)
    - r**sp.Rational(-1, 2) * s_**sp.Rational(3, 4), force=True)) == 0

th = sp.Symbol("theta", real=True)
fh = sp.sin(th / 2)
fph = sp.diff(fh, th)
psi_h = fh**sp.Rational(3, 2)
cA_h, cB_h = -sp.sqrt(2) / 2 * fh, 3 * sp.sqrt(2) / 4 * fph
homogeneous_residual = sp.simplify(cA_h * sp.diff(psi_h, th)
                                   + cB_h * psi_h)
okH = (chi_P_power == -3 and chi_r_power == sp.Rational(3, 2)
       and map_identity and homogeneous_residual == 0)
print(f"[{'PASS' if okH else 'FAIL'}] H: chi0 = c1 P^-3 r^(3/2) psi0; "
      "psi_h=f^(3/2) solves the homogeneous transport; its stress-factor "
      "monomial r^(1/4)f^(3/2)=r^(-1/2)s^(3/4)")
print("       This is a scaling bridge only, not an equality between the "
      "action multiplier and the old scaffold reaction variable.")

# ---------------------------------------------------------------- Gate J
# cG = elastic residual coefficient (reaction part carries psi0):
# ODE: cA psi0' + cB psi0 + cG = 0.  EXACT regular particular solution
# (verified symbolically, zero residual):
#   psi_reg = 4 + 6/f^2 - (15 sqrt2/2) g f'/f^2,   psi_reg(pi) = 10.
import sympy as _sp
_th = _sp.Symbol("theta", positive=True)
_f = _sp.sin(_th/2); _fp = _sp.diff(_f, _th)
_g = _sp.Function("g")
_gp = (_sp.Rational(5,4)*_fp*_g(_th) - 1/_sp.sqrt(2))/(_sp.Rational(1,2)*_f)
_psi = 4 + 6/_f**2 - _sp.Rational(15,2)*_sp.sqrt(2)*_g(_th)*_fp/_f**2
_psip = _sp.diff(_psi, _th).replace(_sp.Derivative(_g(_th), _th), _gp)
_D3 = 2*_sp.sqrt(2)
_vr = _sp.Rational(5,2)*_g(_th) - 2*_D3*_fp
_vth = 2*_gp + _D3*_f
_cA, _cB = -_sp.sqrt(2)/2*_f, 3*_sp.sqrt(2)/4*_fp
_cG = _sp.Rational(5,4)*_vr + _sp.diff(_vth, _th).replace(
    _sp.Derivative(_g(_th), _th), _gp)
_res = _sp.simplify(_sp.expand((_cA*_psip + _cB*_psi + _cG).replace(
    _sp.Derivative(_g(_th), _th), _gp)))
okJ1 = _res == 0
print(f"[{'PASS' if okJ1 else 'FAIL'}] J1: exact regular particular "
      "solution psi_reg verified (zero symbolic residual)")
face_val = _sp.simplify(_psi.subs(_th, _sp.pi))
okJ2 = _sp.simplify(face_val - 10) == 0
print(f"[{'PASS' if okJ2 else 'FAIL'}] J2: psi_reg(pi) = 10 exactly; "
      "traction-free face requires psi0(pi) = 0, so H = -10 exactly "
      "in the stated dimensionless normalization")
# The base elastic angular traction coefficient is 2g'+Delta^-3 f.
# With g'(pi)=-sqrt(2), Delta^-3=2sqrt(2), it vanishes.  The action-
# multiplier traction is proportional to -f*psi0, hence psi0(pi)=0.
base_elastic_face = _sp.simplify(
    2 * (-_sp.sqrt(2)) + 2 * _sp.sqrt(2) * _f.subs(_th, _sp.pi))
hom_res = _sp.simplify((_cA * _sp.diff(_f**_sp.Rational(3, 2), _th)
                        + _cB * _f**_sp.Rational(3, 2)))
okJ3 = base_elastic_face == 0 and hom_res == 0
print(f"[{'PASS' if okJ3 else 'FAIL'}] J3: base elastic face traction "
      "cancels and f^(3/2) is the exact homogeneous reaction profile")

# Derive g4 and g6 from the base constraint ODE, then verify the advertised
# axis series rather than merely printing it.  Because psi_reg contains
# f^-2, its theta^4 coefficient depends on g through theta^6; stopping g at
# theta^4 gives a spurious coefficient.
G4, G6 = _sp.symbols("G4 G6")
g_series = (4 * _sp.sqrt(2) / 5 + _sp.sqrt(2) / 2 * _th**2
            + G4 * _th**4 + G6 * _th**6)
g_ode_series = _sp.series(
    _sp.Rational(5, 4) * _fp * g_series
    - _sp.Rational(1, 2) * _f * _sp.diff(g_series, _th)
    - 1 / _sp.sqrt(2), _th, 0, 7).removeO().expand()
G4_exact = _sp.solve(_sp.Eq(g_ode_series.coeff(_th, 4), 0), G4)[0]
G6_exact = _sp.solve(
    _sp.Eq(g_ode_series.subs(G4, G4_exact).coeff(_th, 6), 0), G6
)[0]
g_series_exact = g_series.subs({G4: G4_exact, G6: G6_exact})
psi_axis = _sp.series(_psi.replace(_g(_th), g_series_exact),
                      _th, 0, 5).removeO().expand()
okJ4 = (G4_exact == -7 * _sp.sqrt(2) / 96
        and G6_exact == 307 * _sp.sqrt(2) / 80640
        and psi_axis.coeff(_th, 0) == -8
        and psi_axis.coeff(_th, 2) == 3
        and psi_axis.coeff(_th, 4) == -_sp.Rational(1, 7))
print(f"[{'PASS' if okJ4 else 'FAIL'}] J4: exact axis series "
      f"psi_reg = {psi_axis} (g4={G4_exact}, g6={G6_exact})")
print("[RESULT] base compatibility defect psi_reg(pi)=10: the unique "
      "traction-free outer profile is psi0=psi_reg-10 f^(3/2), with "
      "a theta^(3/2) (C^1) axis term.  The coefficient -10 is tied to "
      "C=J^4-|grad y2|^2 and the dimensionless psi0 normalization.")
ok_all = okF and okG and okH and okJ1 and okJ2 and okJ3 and okJ4
print(f"\nGATES: {'ALL PASS' if ok_all else 'FAILURES PRESENT'}")
raise SystemExit(0 if ok_all else 1)
