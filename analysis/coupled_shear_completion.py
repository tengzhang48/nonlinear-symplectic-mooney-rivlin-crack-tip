#!/usr/bin/env python3
"""Leading Q_k row-one response and first constrained-action opening.

The calculation is linearized about the selected C_s=0 base representative;
it does not determine the regular C_s s amplitude.

The opening order is internal to W1 + chi*C.  Q_k-dependent half-power terms
from the restored on-manifold c2 remainder are outside this script.  Their
direct row-one contribution enters earlier, at common label k+1/2; see
``qk_full_mr_ordering.py``.

This script works in the leading constrained action

    A[y, chi] = integral (F:F + J**(-2) - 3 + chi*C) dA,
    C = J**4 - |grad y2|**2,

in the P = c1 = 1 gauge.  It does not use the decoupled five-row
scaffold.  Instead it substitutes the characteristic shear

    delta y1 = r**k a(theta),  a = sin(theta/2)**(2*k),
    delta y2 = 0,

directly into the constraint and the leading row-1 Euler--Lagrange
equation.  The latter determines the action-multiplier increment

    delta chi = r**(k + 1/4) p_k(theta).

Within this action, the bare shear also generates a weaker row-2 residual,
3/2 radial powers after its leading label.  The script derives that source,
its nonzero face traction, and the unique nonresonant opening BVP that cancels
it.  The constraint/in-plane companions at that later label are completed in
``qk_later_companions.py``.

All reported identities are symbolic.  Endpoint gates are evaluated for the
integer family k >= 2 (some symbolic endpoint limits depend on integer
assumptions that SymPy does not propagate reliably).
"""
from __future__ import annotations

import sympy as sp


theta = sp.symbols("theta", real=True)
k = sp.symbols("k", integer=True, positive=True)
C_h = sp.symbols("C_h", real=True)

f = sp.sin(theta / 2)
fp = sp.diff(f, theta)
a = f ** (2 * k)

GATES: list[tuple[str, bool]] = []


def gate(name: str, condition: bool, detail: str = "") -> None:
    passed = bool(condition)
    GATES.append((name, passed))
    suffix = f"  ({detail})" if detail else ""
    print(f"[{'PASS' if passed else 'FAIL'}] {name}{suffix}")


# -------------------------------------------------------------------------
# K0. Derive, rather than assume, the row-1 derivative of the constraint.
F11, F12, F21, F22 = sp.symbols("F11 F12 F21 F22", real=True)
J = F11 * F22 - F12 * F21
C_constraint = J**4 - F21**2 - F22**2
CF1 = (sp.diff(C_constraint, F11), sp.diff(C_constraint, F12))
gate("K0: exact row-1 derivative C_F1=4 J^3(F22,-F21)",
     sp.simplify(CF1[0] - 4 * J**3 * F22) == 0
     and sp.simplify(CF1[1] + 4 * J**3 * F21) == 0)

rho = sp.symbols("rho", positive=True)
J0 = sp.sqrt(2) / 2 * rho ** (-sp.Rational(1, 4))
F210 = sp.Rational(1, 2) * rho ** (-sp.Rational(1, 2)) * f
F220 = rho ** (-sp.Rational(1, 2)) * fp
CF1_base = (4 * J0**3 * F220, -4 * J0**3 * F210)
CF1_claim = (sp.sqrt(2) * rho ** (-sp.Rational(5, 4)) * fp,
             -rho ** (-sp.Rational(5, 4)) * f / sp.sqrt(2))
gate("K0: base C_F1 polar coefficients and r^(-5/4) scaling",
     all(sp.simplify(value - claim) == 0
         for value, claim in zip(CF1_base, CF1_claim)))


# -------------------------------------------------------------------------
# K1. The shear is tangent to the constraint manifold.
# For delta y2 = 0, the coefficient of delta J is
# k*a*f' - (1/2)*f*a'.  Therefore both delta J and delta C vanish.
delta_J = sp.trigsimp(k * a * fp - sp.Rational(1, 2) * f * sp.diff(a, theta))
gate("K1: characteristic shear has delta J = delta C = 0",
     sp.simplify(delta_J) == 0)


# -------------------------------------------------------------------------
# K2. Direct row-1 equilibrium.
# Since delta J = 0 and delta y2 = 0, the elastic row-1 stress increment
# is 2*grad(delta y1).  On the base orbit, the row-1 derivative of C is
#
#   C_F1r = sqrt(2) r^(-5/4) f',
#   C_F1theta = -r^(-5/4) f/sqrt(2).
#
# Thus the coefficient of r^(k-2) in div(delta P_1) is
#
#   2*(k^2*a + a'') + [(2k-1)f'p - f p']/sqrt(2).
laplace_coefficient = k**2 * a + sp.diff(a, theta, 2)
laplace_claim = sp.Rational(1, 2) * k * (2 * k - 1) * f ** (2 * k - 2)
gate("K2: exact Laplacian coefficient for s^k",
     sp.trigsimp(sp.simplify(
         (laplace_coefficient - laplace_claim) / f ** (2 * k - 2)
     )) == 0)


# The forced multiplier transport and its general solution.
p_particular = (-2 * sp.sqrt(2) * k * (2 * k - 1)
                * sp.cos(theta / 2) * f ** (2 * k - 2))
p_homogeneous = f ** (2 * k - 1)
p_general = p_particular + C_h * p_homogeneous


def transport_residual(p: sp.Expr, forced: bool = True) -> sp.Expr:
    """Residual of f p'-(2k-1)f'p = sqrt(2)k(2k-1)f^(2k-2)."""
    rhs = (sp.sqrt(2) * k * (2 * k - 1) * f ** (2 * k - 2)
           if forced else 0)
    return sp.trigsimp(
        f * sp.diff(p, theta) - (2 * k - 1) * fp * p - rhs
    )


gate("K3: p_k solves the forced multiplier transport exactly",
     sp.simplify(transport_residual(p_particular)) == 0)
gate("K3: f^(2k-1) is the homogeneous multiplier branch",
     sp.simplify(transport_residual(p_homogeneous, forced=False)) == 0)


# Substitute into the divergence itself as an independent sign/factor gate.
reaction_divergence = ((2 * k - 1) * fp * p_particular
                       - f * sp.diff(p_particular, theta)) / sp.sqrt(2)
full_divergence = 2 * laplace_coefficient + reaction_divergence
gate("K4: elastic plus reaction row-1 divergence vanishes",
     sp.trigsimp(sp.simplify(full_divergence / f ** (2 * k - 2))) == 0)


# -------------------------------------------------------------------------
# K5. Face traction and axis parity select the particular solution.
# The coefficient of the angular row-1 traction is
#
#   T_1theta = 2*a' - f*p/sqrt(2).
#
# Its general-solution face value is -C_h/sqrt(2), so a traction-free face
# fixes C_h = 0.  For that choice the whole angular profile simplifies.
traction_general = sp.trigsimp(
    2 * sp.diff(a, theta) - f * p_general / sp.sqrt(2)
)
traction_particular = sp.trigsimp(traction_general.subs(C_h, 0))
traction_claim = 4 * k**2 * sp.cos(theta / 2) * f ** (2 * k - 1)
traction_reduced = sp.powsimp(sp.cancel(sp.together(
    (traction_particular - traction_claim)
    / (sp.cos(theta / 2) * f ** (2 * k - 1)))), force=True)
traction_numerator = sp.factor(sp.fraction(traction_reduced)[0])
gate("K5: coupled angular traction has the closed form 4 k^2 cos f^(2k-1)",
     sp.simplify(sp.trigsimp(traction_numerator)) == 0)
gate("K5: traction-free face uniquely sets the homogeneous coefficient to zero",
     sp.simplify(traction_general.subs(theta, sp.pi) + C_h / sp.sqrt(2)) == 0)


# Canonical radial momentum.  It is not the scaffold value tau1=k*a:
# reaction content changes its angular profile.
pi1 = sp.trigsimp(2 * k * a + sp.sqrt(2) * fp * p_particular)
pi1_claim = (2 * k * f ** (2 * k - 2)
             * (f**2 - (2 * k - 1) * sp.cos(theta / 2)**2))
gate("K6: reaction-carrying radial momentum profile",
     sp.trigsimp(sp.simplify(
         (pi1 - pi1_claim) / f ** (2 * k - 2)
     )) == 0)


# Explicit endpoint and smoothness checks for representative integer members.
for kval in (2, 3, 4):
    pk = sp.simplify(p_particular.subs(k, kval))
    tk = sp.simplify(traction_claim.subs(k, kval))
    even_gate = sp.simplify(pk.subs(theta, -theta) - pk) == 0
    endpoint_gate = (sp.simplify(pk.subs(theta, sp.pi)) == 0
                     and sp.simplify(tk.subs(theta, 0)) == 0
                     and sp.simplify(tk.subs(theta, sp.pi)) == 0)
    gate(f"K7: k={kval} multiplier is even and analytic at the axis", even_gate)
    gate(f"K7: k={kval} satisfies axis/face angular-traction endpoints",
         endpoint_gate)


p2 = sp.trigsimp(p_particular.subs(k, 2))
p2_series = sp.series(p2, theta, 0, 5)
gate("K8: Q2 multiplier begins -3 sqrt(2) theta^2 at the axis",
     sp.simplify(sp.expand(p2_series.removeO()).coeff(theta, 2)
                 + 3 * sp.sqrt(2)) == 0,
     f"p2={p2}; series={p2_series}")


# -------------------------------------------------------------------------
# K9. The Q_k shear is triangular, not a standalone full-vector solution.
# Although b=0 solves the more singular common-label opening row, the row-2
# stress receives a weaker contribution at r^(k-1/4) from W_{J^-2}, chi0*C_FF,
# and delta-chi*C_F.  Its divergence is r^(k-5/4) R_2k.  Work on the upper
# half-domain (x>0), where f^(3/2) has its real branch.
x = sp.symbols("x", positive=True)
fx = sp.sin(x / 2)
fpx = sp.diff(fx, x)
cx = sp.cos(x / 2)
gx = sp.Function("g")(x)
gpx = ((sp.Rational(5, 4) * fpx * gx - 1 / sp.sqrt(2))
       / (sp.Rational(1, 2) * fx))
psi_reg_x = (4 + 6 / fx**2
             - sp.Rational(15, 2) * sp.sqrt(2) * gx * fpx / fx**2)
psi0_x = psi_reg_x - 10 * fx**sp.Rational(3, 2)
ax = fx ** (2 * k)
apx = sp.diff(ax, x)
px = -2 * sp.sqrt(2) * k * (2 * k - 1) * cx * fx ** (2 * k - 2)

S2r = (sp.sqrt(2) * (4 - psi0_x) * apx
       + px * (-sp.sqrt(2) * gpx - fx))
S2theta = (sp.sqrt(2) * (psi0_x - 4) * k * ax
           + px * (sp.Rational(5, 4) * sp.sqrt(2) * gx - 2 * fpx))

# Recover these two coefficients directly by differentiating the constrained
# action stress.  In polar reference components,
#
#   P_2 = 2 F_2 - 2 J^(-3) cof(F)_2
#         + chi [4 J^3 cof(F)_2 - 2 F_2].
#
# This is a stronger gate than merely reusing the displayed S2 formulas.
eps = sp.symbols("eps", real=True)
P_amp, c1_amp, q_amp = sp.symbols("P_amp c1_amp q_amp", positive=True)
F11x = (P_amp**(-sp.Rational(1, 2))
        * sp.Rational(5, 4) * rho**sp.Rational(1, 4) * gx
        + eps * q_amp * P_amp**(2 - 2 * k)
        * rho**(k - 1) * k * ax)
F12x = (P_amp**(-sp.Rational(1, 2))
        * rho**sp.Rational(1, 4) * gpx
        + eps * q_amp * P_amp**(2 - 2 * k)
        * rho**(k - 1) * apx)
F21x = (P_amp * sp.Rational(1, 2)
        * rho**(-sp.Rational(1, 2)) * fx)
F22x = P_amp * rho**(-sp.Rational(1, 2)) * fpx
Jx = F11x * F22x - F12x * F21x
chix = (c1_amp * P_amp**-3 * rho**sp.Rational(3, 2) * psi0_x
        + eps * q_amp * c1_amp * P_amp**(-2 * k - sp.Rational(1, 2))
        * rho**(k + sp.Rational(1, 4)) * px)
cof2x = (-F12x, F11x)
P2x = tuple(2 * c1_amp * F2 - 2 * c1_amp * Jx**-3 * cof2
             + chix * (4 * Jx**3 * cof2 - 2 * F2)
             for F2, cof2 in zip((F21x, F22x), cof2x))
S2_direct = tuple(sp.trigsimp(sp.simplify(
    sp.diff(component, eps).subs(eps, 0)
    / (q_amp * c1_amp * P_amp**(sp.Rational(1, 2) - 2 * k)
       * rho**(k - sp.Rational(1, 4))))) for component in P2x)

# The same arbitrary-P,c1 differentiation recovers the leading row-one stress.
cof1x = (F22x, -F21x)
P1x = tuple(2 * c1_amp * F1 - 2 * c1_amp * Jx**-3 * cof1
             + chix * 4 * Jx**3 * cof1
             for F1, cof1 in zip((F11x, F12x), cof1x))
S1_direct = tuple(sp.trigsimp(sp.simplify(
    sp.diff(component, eps).subs(eps, 0)
    / (q_amp * c1_amp * P_amp**(2 - 2 * k) * rho**(k - 1))))
    for component in P1x)
S1_claim = (2 * k * ax + sp.sqrt(2) * fpx * px,
            2 * apx - fx * px / sp.sqrt(2))


def trig_numerator_zero(expression: sp.Expr) -> bool:
    """Test a rational trigonometric identity after clearing denominators."""
    numerator = sp.fraction(sp.cancel(sp.together(expression)))[0]
    # On 0 < theta < pi, f>0, so combining its symbolic integer-shifted
    # powers is branch-safe and avoids a SymPy false negative.
    numerator = sp.powsimp(numerator, force=True)
    return sp.simplify(sp.trigsimp(numerator)) == 0


direct_action_stress_ok = (
    sp.simplify(sp.trigsimp(sp.simplify(
        Jx.subs(eps, 0) * rho**sp.Rational(1, 4)
        / P_amp**sp.Rational(1, 2)
        - 1 / sp.sqrt(2)))) == 0
    and trig_numerator_zero(S1_direct[0] - S1_claim[0])
    and trig_numerator_zero(S1_direct[1] - S1_claim[1])
    and trig_numerator_zero(S2_direct[0] - S2r)
    and trig_numerator_zero(S2_direct[1] - S2theta)
)

R2k = ((k + sp.Rational(3, 4)) * S2r + sp.diff(S2theta, x))
R2k = R2k.replace(sp.Derivative(gx, x), gpx)
R2k_claim = (sp.sqrt(2) * k / 8 * fx ** (2 * k - 3)
             * (5 * sp.sqrt(2) * (4 * k + 1) * gx
                - 4 * (8 * k**2 - 10 * k + 9) * cx
                - 12 * (2 * k - 1) * cx**3))
R2k_reduced = sp.powsimp(sp.cancel(sp.together(
    (R2k - R2k_claim) / fx ** (2 * k - 4))), force=True)
R2k_numerator = sp.factor(sp.fraction(R2k_reduced)[0])
R2k_check = sp.simplify(sp.trigsimp(R2k_numerator))
gate("K9: variable-P,c1 action stress gives row one and exact row-2 source R_2k",
     direct_action_stress_ok and R2k_check == 0)

# At the face psi0=p=0 and a=1, so the bare Q_k row-2 angular traction is
# -4 sqrt(2) k.  It is canceled by an opening correction with b'(pi)=2sqrt(2)k.
S2theta_face = sp.simplify(S2theta.subs(x, sp.pi))
gate("K9: bare Q_k has later row-2 face traction -4 sqrt(2) k",
     sp.simplify(S2theta_face + 4 * sp.sqrt(2) * k) == 0)


# K10. A slaved opening delta y2 ~ r^(k+3/4) b_k cancels R_2k through
#   2[b_k''+(k+3/4)^2 b_k] + R_2k = 0,
#   b_k(0)=0, b_k'(pi)=2sqrt(2)k.
# The mixed homogeneous problem is nonresonant for every integer k because
# cos[(k+3/4)pi]=(-1)^(k+1)/sqrt(2), so this opening BVP is unique.
alpha_slaved = k + sp.Rational(3, 4)
gate("K10: slaved opening exponent matches the row-2 residual power",
     sp.simplify(alpha_slaved - 2 - (k - sp.Rational(5, 4))) == 0)
gate("K10: restored P power of the slaved opening is P^(1/2-2k)",
     sp.simplify((2 - 2 * k) - sp.Rational(3, 2)
                 - (sp.Rational(1, 2) - 2 * k)) == 0)
gate("K10: slaved opening BVP is nonresonant for integer k",
     sp.simplify(sp.cos(alpha_slaved * sp.pi)**2 - sp.Rational(1, 2)) == 0
     and sp.simplify(sp.cos(alpha_slaved * sp.pi)
                     - (-1)**(k + 1) / sp.sqrt(2)) == 0,
     "cos[(k+3/4)pi]=(-1)^(k+1)/sqrt(2)")


print("\nTriangular characteristic-shear response (P=c1=1 action gauge):")
print("  a_k(theta) =", a)
print("  p_k(theta) =", p_particular)
print("  p_2(theta) =", p2)
print("  pi_1,k(theta) =", pi1_claim)
print("  T_1theta,k(theta) =", traction_claim)
print("  div(delta P_2) first appears as r^(k-5/4) R_2k, with")
print("  R_2k(theta) =", R2k_claim)
print("  bare later row-2 face traction =", S2theta_face)
print("  slaved opening: delta y2 ~ r^(k+3/4) b_k,")
print("    2[b_k''+(k+3/4)^2 b_k] + R_2k = 0,")
print("    b_k(0)=0, b_k'(pi)=2 sqrt(2) k  (unique; nonresonant)")
print("\nRestored scaling for a unit mode in Eq. (weighted):")
print("  delta y1 = P^(2-2k) r^k a_k")
print("  delta chi = c1 P^(-2k-1/2) r^(k+1/4) p_k")
print("  delta y2_slaved = P^(1/2-2k) r^(k+3/4) b_k")

failed = [name for name, passed in GATES if not passed]
print("\n" + "=" * 68)
print(f"PASSED {len(GATES) - len(failed)}   FAILED {len(failed)}")
if failed:
    print("FAILURES:", failed)
    raise SystemExit(1)
print("All coupled-shear gates passed.")
