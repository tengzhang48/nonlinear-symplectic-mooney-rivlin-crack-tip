#!/usr/bin/env python3
"""Complete the constrained-action Q_k companion at Lambda = k + 3/2.

The stated companion is about the selected C_s=0 base representative. This
script does not determine the regular null-family amplitude or a full coupled
spectrum.

This script continues ``coupled_shear_completion.py``.  It works in the
formal leading action

    A[y, chi] = integral (F:F + J**(-2) - 3 + chi*C) dA,
    C = J**4 - |grad y2|**2,

in the P = c1 = 1 gauge.  For the characteristic shear Q_k, k >= 2, the
weaker row-two residual forces an opening at

    Lambda = k + 3/2,       nu = Lambda - 3/4 = k + 3/4.

The calculation below derives the two companions that were previously open,
within the admitted outer endpoint class: an analytic-even in-plane field A
and an even Puiseux-log action-multiplier field H. The forcing is
triangular: the lower Q_k field directly forces only the row-two equation at
this label.  The constraint and row-one equations have no additional direct
Q_k source.

The axis classification is part of the result. A is selected by analytic
Mode-I parity. For H, the base f^(3/2) multiplier determines a generally
nonzero |theta|^(7/2) coefficient, and the regular-singular transport
can produce a later theta^(2k+2) log|theta| term. The exact affine
logarithmic residues are checked for Q2 and Q3 by rebuilding the local series,
not by fitting samples. Exceptional cancellation at an isolated globally
selected B'(0) is not ruled out for every higher k.

All family gates are symbolic.  The local Laurent gates for k=2,3 are exact
representative endpoint checks; they do not replace the general identities.
"""
from __future__ import annotations

import sympy as sp


SQRT2 = sp.sqrt(2)
k = sp.symbols("k", integer=True, positive=True)
Lambda = k + sp.Rational(3, 2)
nu = k + sp.Rational(3, 4)

# Exact half-angle quotient ring on 0 < theta < pi.
f = sp.symbols("f", positive=True)
c = sp.symbols("c", nonnegative=True)
g = sp.symbols("g", real=True)
psi = sp.symbols("psi", real=True)
psi_prime = sp.symbols("psi_prime", real=True)
g_prime = sp.Rational(5, 4) * c * g / f - SQRT2 / f

# Generic companion jets.
A, A1, A2 = sp.symbols("A A1 A2", real=True)
B, B1, B2, B3 = sp.symbols("B B1 B2 B3", real=True)
H, H1 = sp.symbols("H H1", real=True)

GATES: list[tuple[str, bool]] = []


def gate(name: str, condition: bool, detail: str = "") -> None:
    """Record and print one executable assertion."""
    passed = bool(condition)
    GATES.append((name, passed))
    suffix = f"  ({detail})" if detail else ""
    print(f"[{'PASS' if passed else 'FAIL'}] {name}{suffix}")


def dtheta(expr: sp.Expr) -> sp.Expr:
    """Differentiate in the half-angle/base-field quotient ring."""
    return (
        sp.diff(expr, f) * c / 2
        - sp.diff(expr, c) * f / 2
        + sp.diff(expr, g) * g_prime
        + sp.diff(expr, psi) * psi_prime
        + sp.diff(expr, A) * A1
        + sp.diff(expr, A1) * A2
        + sp.diff(expr, B) * B1
        + sp.diff(expr, B1) * B2
        + sp.diff(expr, B2) * B3
        + sp.diff(expr, H) * H1
    )


def circle_numerator(expr: sp.Expr) -> sp.Expr:
    """Reduce a rational half-angle identity modulo c**2 = 1-f**2."""
    reduced = sp.powsimp(sp.cancel(sp.together(expr)), force=True)
    numerator = sp.powsimp(sp.expand(sp.fraction(reduced)[0]), force=True)
    polynomial = sp.Poly(numerator, c)
    circle = sp.Poly(c**2 - (1 - f**2), c)
    return sp.factor(polynomial.rem(circle).as_expr())


def circle_zero(expr: sp.Expr) -> bool:
    return circle_numerator(expr) == 0


# Base and companion vectors.
u = sp.Matrix([sp.Rational(5, 4) * g, g_prime])
v = sp.Matrix([f / 2, c / 2])
K = sp.Matrix([c / 2, -f / 2])
L = sp.Matrix([-g_prime, sp.Rational(5, 4) * g])
d = sp.Matrix([Lambda * A, A1])
e = sp.Matrix([nu * B, B1])
E = sp.Matrix([B1, -nu * B])
Q = sp.Matrix([-A1, Lambda * A])
D_raw = sp.det(sp.Matrix.hstack(d, v)) + sp.det(sp.Matrix.hstack(u, e))
M = v.dot(e)


# -------------------------------------------------------------------------
# L0. Derive the later-label stress and constraint directly from the action.
r, eps = sp.symbols("r eps", positive=True)
F1 = r**sp.Rational(1, 4) * u + eps * r ** (Lambda - 1) * d
F2 = r**-sp.Rational(1, 2) * v + eps * r ** (nu - 1) * e
J = sp.det(sp.Matrix.hstack(F1, F2))
cof1 = sp.Matrix([F2[1], -F2[0]])
cof2 = sp.Matrix([-F1[1], F1[0]])
chi = r**sp.Rational(3, 2) * psi + eps * r ** (
    Lambda + sp.Rational(1, 4)
) * H

P1 = 2 * F1 - 2 * J**-3 * cof1 + 4 * chi * J**3 * cof1
P2 = 2 * F2 - 2 * J**-3 * cof2 + chi * (4 * J**3 * cof2 - 2 * F2)
C_constraint = J**4 - F2.dot(F2)

dP1 = sp.diff(P1, eps).subs(eps, 0)
dP2 = sp.diff(P2, eps).subs(eps, 0)
dC = sp.diff(C_constraint, eps).subs(eps, 0)

S_claim = (
    2 * d
    + (24 + 6 * psi) * D_raw * K
    + SQRT2 * (psi - 4) * E
    + SQRT2 * H * K
)
V2_claim = (
    (24 + 6 * psi) * D_raw * L
    + SQRT2 * (psi - 4) * Q
    - 2 * psi * e
    + H * (SQRT2 * L - 2 * v)
)

direct_row1_ok = all(
    circle_zero(dP1[j] / r ** (Lambda - 1) - S_claim[j])
    for j in range(2)
)
direct_row2_ok = all(
    circle_zero(
        (
            dP2[j]
            - 2 * r ** (Lambda - sp.Rational(7, 4)) * e[j]
        )
        / r ** (Lambda - sp.Rational(1, 4))
        - V2_claim[j]
    )
    for j in range(2)
)
direct_constraint_ok = circle_zero(
    dC / r ** (Lambda - sp.Rational(9, 4))
    - (SQRT2 * D_raw - 2 * M)
)

gate(
    "L0: direct action differentiation gives the complete later row-one stress",
    direct_row1_ok,
)
gate(
    "L0: direct action differentiation separates 2e from the next row-two rung",
    direct_row2_ok,
)
gate(
    "L0: direct constraint differentiation gives sqrt(2)D-2M",
    direct_constraint_ok,
)


# -------------------------------------------------------------------------
# L1. The only lower-Q forcing at this label is the delayed row-two source.
aQ = f ** (2 * k)
aQ_prime = k * c * f ** (2 * k - 1)
pQ = -2 * SQRT2 * k * (2 * k - 1) * c * f ** (2 * k - 2)
psi0 = (
    4
    + 6 / f**2
    - sp.Rational(15, 4) * SQRT2 * g * c / f**2
    - 10 * f**sp.Rational(3, 2)
)

T2 = (
    SQRT2 * (psi0 - 4) * sp.Matrix([-aQ_prime, k * aQ])
    + pQ * (SQRT2 * L - 2 * v)
)
R2k = (
    SQRT2
    * k
    / 8
    * f ** (2 * k - 3)
    * (
        5 * SQRT2 * (4 * k + 1) * g
        - 4 * (8 * k**2 - 10 * k + 9) * c
        - 12 * (2 * k - 1) * c**3
    )
)

gate(
    "L1: bare Q_k delayed row-two divergence is exactly R_2k",
    circle_zero(nu * T2[0] + dtheta(T2[1]) - R2k),
)

# A separate exact differentiation of the bare-Q perturbation shows all of
# its linear stress powers.  In particular, row one occurs only at r^(k-1)
# and row two only at r^(k-1/4); there is no hidden r^(k+1/2) row-one term.
dQ = sp.Matrix([k * aQ, aQ_prime])
F1_Q = r**sp.Rational(1, 4) * u + eps * r ** (k - 1) * dQ
F2_Q = r**-sp.Rational(1, 2) * v
J_Q = sp.det(sp.Matrix.hstack(F1_Q, F2_Q))
cof1_Q = sp.Matrix([F2_Q[1], -F2_Q[0]])
cof2_Q = sp.Matrix([-F1_Q[1], F1_Q[0]])
chi_Q = r**sp.Rational(3, 2) * psi0 + eps * r ** (
    k + sp.Rational(1, 4)
) * pQ
P1_Q = 2 * F1_Q - 2 * J_Q**-3 * cof1_Q + 4 * chi_Q * J_Q**3 * cof1_Q
P2_Q = (
    2 * F2_Q
    - 2 * J_Q**-3 * cof2_Q
    + chi_Q * (4 * J_Q**3 * cof2_Q - 2 * F2_Q)
)
C_Q = J_Q**4 - F2_Q.dot(F2_Q)
T1 = 2 * dQ + SQRT2 * pQ * K
bare_q_action_ok = (
    sp.simplify(sp.diff(C_Q, eps).subs(eps, 0)) == 0
    and all(
        circle_zero(
            sp.diff(P1_Q[j], eps).subs(eps, 0) / r ** (k - 1) - T1[j]
        )
        for j in range(2)
    )
    and all(
        circle_zero(
            sp.diff(P2_Q[j], eps).subs(eps, 0)
            / r ** (k - sp.Rational(1, 4))
            - T2[j]
        )
        for j in range(2)
    )
)
gate(
    "L1: bare Q_k contributes no direct constraint or row-one source here",
    bare_q_action_ok,
    "forcing vector (F_C,F_1,F_2)=(0,0,R_2k)",
)

T2_face = sp.simplify(
    T2[1].subs({f: 1, c: 0, g_prime: -SQRT2})
)
gate(
    "L1: bare delayed row-two face load is -4 sqrt(2) k",
    sp.simplify(T2_face + 4 * SQRT2 * k) == 0,
)


# -------------------------------------------------------------------------
# L2. Forced opening BVP, uniqueness, Green representation, and endpoints.
opening_divergence = nu * (2 * nu * B) + dtheta(2 * B1)
gate(
    "L2: opening stress contributes 2(B''+nu^2 B)",
    sp.simplify(opening_divergence - 2 * (B2 + nu**2 * B)) == 0,
)
B1_bc = sp.symbols("B1_bc", real=True)
opening_face_total = T2_face + 2 * B1_bc
opening_face_solution = sp.solve(sp.Eq(opening_face_total, 0), B1_bc)
gate(
    "L2: row-two face traction fixes B'(pi)=2 sqrt(2) k",
    opening_face_solution == [2 * SQRT2 * k],
)

cos_nu_pi = (-1) ** (k + 1) / SQRT2
gate(
    "L2: the forced opening BVP is nonresonant for every integer k",
    sp.simplify(sp.cos(sp.pi * nu) - cos_nu_pi) == 0,
    "cos(nu*pi)=(-1)^(k+1)/sqrt(2)",
)

z, tau = sp.symbols("z tau", real=True)
R_test = sp.Function("R_test")
b1_test = sp.symbols("b1_test", real=True)
green_convolution = sp.Integral(
    sp.sin(nu * (z - tau)) * R_test(tau), (tau, 0, z)
)
B_green = b1_test * sp.sin(nu * z) / nu - green_convolution / (2 * nu)
I_c = sp.Integral(
    sp.cos(nu * (sp.pi - tau)) * R_test(tau), (tau, 0, sp.pi)
)
b1_green = (2 * SQRT2 * k + I_c / 2) / cos_nu_pi
gate(
    "L2: Green convolution solves the ODE and both endpoint conditions",
    sp.simplify(B_green.subs(z, 0)) == 0
    and sp.simplify(
        sp.diff(B_green, z, 2) + nu**2 * B_green + R_test(z) / 2
    )
    == 0
    and sp.simplify(
        sp.diff(B_green, z).subs(z, sp.pi).subs(b1_test, b1_green)
        - 2 * SQRT2 * k
    )
    == 0,
    "including the convolution sign and moving-limit derivative",
)

G_face = sp.symbols("G_face", real=True)
R_face = sp.simplify(R2k.subs({f: 1, c: 0, g: G_face}))
gate(
    "L2: exact source face value",
    sp.simplify(R_face - sp.Rational(5, 4) * k * (4 * k + 1) * G_face)
    == 0,
)

# The source derivative at the axis is exceptional only for k=2.
theta_axis = sp.symbols("theta_axis", positive=True)
f_axis = sp.sin(theta_axis / 2)
c_axis = sp.cos(theta_axis / 2)
g_axis_low = 4 * SQRT2 / 5 + SQRT2 * theta_axis**2 / 2
R_axis_expr = R2k.subs({f: f_axis, c: c_axis, g: g_axis_low})
Rprime_axis = {
    kval: sp.simplify(
        sp.limit(sp.diff(R_axis_expr.subs(k, kval), theta_axis), theta_axis, 0)
    )
    for kval in (2, 3, 4)
}
gate(
    "L2: R_2k'(0)=-6sqrt(2) for k=2 and zero for k>=3 representatives",
    Rprime_axis[2] == -6 * SQRT2
    and Rprime_axis[3] == 0
    and Rprime_axis[4] == 0,
)


# -------------------------------------------------------------------------
# L3. Algebraic constraint, parity selection, and exact endpoint anchors.
D_constrained = SQRT2 * M
E_B = D_constrained + nu * g_prime * B - sp.Rational(5, 4) * g * B1
constraint_transport = Lambda * c * A / 2 - f * A1 / 2 - E_B
gate(
    "L3: the undivided constraint is equivalent to D=sqrt(2)M",
    circle_zero(D_raw - D_constrained - constraint_transport),
)

A_h = f ** (2 * Lambda)
gate(
    "L3: f^(2Lambda)=f^(2k+3) is the homogeneous A branch",
    circle_zero(Lambda * c * A_h / 2 - f * dtheta(A_h) / 2)
    and sp.simplify(2 * Lambda - (2 * k + 3)) == 0,
    "signed odd; its even continuation is nonanalytic, so Mode-I excludes it",
)

# Rebuild the first two axis recurrences without dividing by f.
t = sp.symbols("t", positive=True)
b_axis_slope, b_axis_third = sp.symbols("b_axis_slope b_axis_third", real=True)
a_axis_value, a_axis_second = sp.symbols("a_axis_value a_axis_second", real=True)
f_t = sp.sin(t / 2)
fp_t = sp.diff(f_t, t)
g_t_low = 4 * SQRT2 / 5 + SQRT2 * t**2 / 2
B_t_low = b_axis_slope * t + b_axis_third * t**3 / 6
A_t_low = a_axis_value + a_axis_second * t**2 / 2
D_t_raw = (
    Lambda * fp_t * A_t_low
    - f_t * sp.diff(A_t_low, t) / 2
    - nu * sp.diff(g_t_low, t) * B_t_low
    + sp.Rational(5, 4) * g_t_low * sp.diff(B_t_low, t)
)
M_t = nu * f_t * B_t_low / 2 + fp_t * sp.diff(B_t_low, t)
axis_constraint_series = sp.series(
    SQRT2 * D_t_raw - 2 * M_t, t, 0, 3
).removeO().expand()
a0_claim = -SQRT2 * b_axis_slope / Lambda
a2_claim = (
    SQRT2
    * ((5 * nu - 3) * b_axis_slope - b_axis_third)
    / (Lambda - 1)
)
gate(
    "L3: algebraic axis constraint fixes A(0) and A''(0)",
    sp.simplify(axis_constraint_series.subs(
        {a_axis_value: a0_claim, a_axis_second: a2_claim}
    ))
    == 0,
    "A(0)=-sqrt(2)B'(0)/Lambda; A'(0)=0",
)

B_face, A_face, A1_face_symbol, B1_face_symbol = sp.symbols(
    "B_face A_face A1_face_symbol B1_face_symbol", real=True
)
constraint_face_residual = sp.simplify(
    (D_raw - D_constrained).subs(
        {
            f: 1,
            c: 0,
            g: G_face,
            A: A_face,
            A1: A1_face_symbol,
            B: B_face,
            B1: B1_face_symbol,
        }
    )
)
A1_face_solution = sp.solve(sp.Eq(constraint_face_residual, 0), A1_face_symbol)
A1_face_general = SQRT2 * nu * B_face + sp.Rational(5, 2) * G_face * B1_face_symbol
B1_face = 2 * SQRT2 * k
A1_face = A1_face_general.subs(B1_face_symbol, B1_face)
gate(
    "L3: the face constraint fixes A'(pi)",
    len(A1_face_solution) == 1
    and sp.simplify(A1_face_solution[0] - A1_face_general) == 0
    and sp.simplify(A1_face - SQRT2 * (nu * B_face + 5 * k * G_face)) == 0,
)


# -------------------------------------------------------------------------
# L4. Row-one transport, face reaction, and next-rung ordering.
U = (
    2 * d
    + (24 + 6 * psi) * D_constrained * K
    + SQRT2 * (psi - 4) * E
)
S = U + SQRT2 * H * K
row1_equilibrium = Lambda * S[0] + dtheta(S[1])
transport_claim = (
    f * H1
    - (2 * Lambda - 1) * c * H / 2
    - SQRT2 * (Lambda * U[0] + dtheta(U[1]))
)
gate(
    "L4: row-one equilibrium is exactly the multiplier transport",
    circle_zero(row1_equilibrium + transport_claim / SQRT2),
)

H_h = f ** (2 * Lambda - 1)
gate(
    "L4: f^(2Lambda-1)=f^(2k+2) is the homogeneous H branch",
    circle_zero(
        f * dtheta(H_h) - (2 * Lambda - 1) * c * H_h / 2
    ),
)

Utheta_face = sp.simplify(
    U[1].subs(
        {
            f: 1,
            c: 0,
            g: G_face,
            psi: 0,
            A: A_face,
            A1: A1_face,
            B: B_face,
            B1: B1_face,
        }
    )
)
H_face_from_traction = sp.simplify(SQRT2 * Utheta_face)
gate(
    "L4: total row-one face traction fixes H(pi)=20 k g(pi)",
    sp.simplify(H_face_from_traction - 20 * k * G_face) == 0,
    "the B(pi) dependence cancels exactly",
)

V2_face = sp.simplify(
    V2_claim[1].subs(
        {
            f: 1,
            c: 0,
            g: G_face,
            psi: 0,
            A: A_face,
            A1: A1_face,
            B: B_face,
            B1: B1_face,
            H: 20 * k * G_face,
        }
    )
)
V2_face_claim = (
    15 * SQRT2 * nu * G_face * B_face
    - 4 * SQRT2 * Lambda * A_face
    + sp.Rational(25, 2) * G_face**2 * B1_face
)
gate(
    "L4: generated V2 face load belongs to the subsequent Lambda+3/2 rung",
    sp.simplify(V2_face - V2_face_claim) == 0,
    "do not impose V2_theta(pi)=0 on the present block",
)


# -------------------------------------------------------------------------
# L5. Axis multiplier value and leading Puiseux coefficient.
b1, b3, Rprime0 = sp.symbols("b1 b3 Rprime0", real=True)
H0_claim = (
    2
    * ((16 * k**2 + 24 * k + 15) * b1 + 4 * b3)
    / ((2 * k + 1) * (k + 1))
)
gamma_claim = (
    -sp.Rational(5, 4)
    * SQRT2
    * (Rprime0 + (nu - 1) * (2 * nu - 1) * b1)
)

# Rebuild both coefficients from the local stress, retaining the exact
# f^(3/2) base channel.  Here b3 means B'''(0), not the theta^3 coefficient.
A0_local = -SQRT2 * b1 / Lambda
A2_local = SQRT2 * ((5 * nu - 3) * b1 - b3) / (Lambda - 1)
B_local = b1 * t + b3 * t**3 / 6
A_local = A0_local + A2_local * t**2 / 2
D_local = SQRT2 * (
    nu * f_t * B_local / 2 + fp_t * sp.diff(B_local, t)
)
K_local = sp.Matrix([fp_t, -f_t / 2])
E_local = sp.Matrix([sp.diff(B_local, t), -nu * B_local])
psi_reg_local = -8 + 3 * t**2 - t**4 / 7
psi_ns_local = -10 * f_t**sp.Rational(3, 2)
U_local = (
    2 * sp.Matrix([Lambda * A_local, sp.diff(A_local, t)])
    + (24 + 6 * (psi_reg_local + psi_ns_local)) * D_local * K_local
    + SQRT2 * (psi_reg_local + psi_ns_local - 4) * E_local
)
source_local = SQRT2 * (
    Lambda * U_local[0] + sp.diff(U_local[1], t)
)
H0_derived = sp.simplify(
    -2 * sp.limit(source_local, t, 0) / (2 * Lambda - 1)
)
gate(
    "L5: degenerate axis transport fixes H(0) exactly",
    sp.simplify(H0_derived - H0_claim) == 0,
    "H'(0)=0 and the homogeneous branch vanishes at the axis",
)

U_ns_local = psi_ns_local * (6 * D_local * K_local + SQRT2 * E_local)
source_ns = sp.expand(
    SQRT2 * (Lambda * U_ns_local[0] + sp.diff(U_ns_local[1], t))
)
source_7half = sp.simplify(
    sp.limit(source_ns / t**sp.Rational(7, 2), t, 0)
)
gamma_derived = sp.simplify(
    source_7half
    / (sp.Rational(7, 4) - (2 * Lambda - 1) / 2)
)
gate(
    "L5: exact |theta|^(7/2) coefficient from the base multiplier",
    sp.simplify(
        gamma_derived.subs(b3, -nu**2 * b1 - Rprime0 / 2)
        - gamma_claim
    )
    == 0,
    "generically nonzero; an exceptional selected B'(0) could cancel it",
)


# -------------------------------------------------------------------------
# L6. Exact representative Laurent gates for the later logarithmic resonance.
def regular_g_series(variable: sp.Symbol, order: int) -> sp.Expr:
    """Regular even g series obtained from the leading constraint ODE."""
    ff = sp.sin(variable / 2)
    ffp = sp.diff(ff, variable)
    coefficients = sp.symbols(f"gg0:{order + 3}")
    trial = sum(coefficients[j] * variable**j for j in range(order + 3))
    residual = sp.series(
        sp.Rational(5, 4) * ffp * trial
        - sp.Rational(1, 2) * ff * sp.diff(trial, variable)
        - 1 / SQRT2,
        variable,
        0,
        order + 2,
    ).removeO().expand()
    solved: dict[sp.Symbol, sp.Expr] = {}
    for power in range(order + 2):
        equation = sp.expand(residual.subs(solved)).coeff(variable, power)
        unknowns = [symbol for symbol in coefficients if symbol in equation.free_symbols]
        if unknowns:
            solved[unknowns[0]] = sp.solve(equation, unknowns[0])[0]
    return sp.series(trial.subs(solved), variable, 0, order + 2).removeO()


g_series_long = regular_g_series(t, 14)


def local_log_data(k_value: int) -> dict[str, sp.Expr]:
    """Rebuild the selected local A/B series and the H-transport residue."""
    kk = sp.Integer(k_value)
    lam = kk + sp.Rational(3, 2)
    freq = kk + sp.Rational(3, 4)
    m = int(2 * lam - 1)
    ff = sp.sin(t / 2)
    cc = sp.cos(t / 2)
    ffp = cc / 2
    vv = sp.symbols(f"v{k_value}", real=True)
    gg = g_series_long

    source_R = (
        SQRT2
        * kk
        / 8
        * ff ** (2 * kk - 3)
        * (
            5 * SQRT2 * (4 * kk + 1) * gg
            - 4 * (8 * kk**2 - 10 * kk + 9) * cc
            - 12 * (2 * kk - 1) * cc**3
        )
    )
    source_R = sp.series(source_R, t, 0, m + 5).removeO()

    odd_powers = list(range(1, m + 6, 2))
    b_coefficients = {
        power: vv if power == 1 else sp.symbols(f"bb{k_value}_{power}")
        for power in odd_powers
    }
    b_trial = sum(b_coefficients[power] * t**power for power in odd_powers)
    b_residual = sp.series(
        2 * (sp.diff(b_trial, t, 2) + freq**2 * b_trial) + source_R,
        t,
        0,
        m + 4,
    ).removeO().expand()
    b_solved: dict[sp.Symbol, sp.Expr] = {}
    for power in range(m + 4):
        equation = sp.expand(b_residual.subs(b_solved)).coeff(t, power)
        unknowns = [
            symbol
            for degree, symbol in b_coefficients.items()
            if degree != 1 and symbol in equation.free_symbols
        ]
        if unknowns:
            b_solved[unknowns[0]] = sp.solve(equation, unknowns[0])[0]
    b_series = sp.expand(b_trial.subs(b_solved))

    even_powers = list(range(0, m + 5, 2))
    a_coefficients = {
        power: sp.symbols(f"aa{k_value}_{power}") for power in even_powers
    }
    a_trial = sum(a_coefficients[power] * t**power for power in even_powers)
    bp = sp.diff(b_series, t)
    gp = sp.diff(gg, t)
    determinant = SQRT2 * (freq * ff * b_series / 2 + ffp * bp)
    forcing_A = determinant + freq * gp * b_series - sp.Rational(5, 4) * gg * bp
    a_residual = sp.series(
        lam * ffp * a_trial - ff * sp.diff(a_trial, t) / 2 - forcing_A,
        t,
        0,
        m + 5,
    ).removeO().expand()
    a_solved: dict[sp.Symbol, sp.Expr] = {}
    for power in even_powers:
        equation = sp.expand(a_residual.subs(a_solved)).coeff(t, power)
        unknowns = [
            symbol for symbol in a_coefficients.values() if symbol in equation.free_symbols
        ]
        if unknowns:
            a_solved[unknowns[0]] = sp.solve(equation, unknowns[0])[0]
    a_series = sp.expand(a_trial.subs(a_solved))

    psi_regular = sp.series(
        4
        + 6 / ff**2
        - sp.Rational(15, 2) * SQRT2 * gg * ffp / ff**2,
        t,
        0,
        m + 5,
    ).removeO()
    e_series = sp.Matrix([bp, -freq * b_series])
    k_vector = sp.Matrix([ffp, -ff / 2])
    u_series = (
        2 * sp.Matrix([lam * a_series, sp.diff(a_series, t)])
        + (24 + 6 * psi_regular) * determinant * k_vector
        + SQRT2 * (psi_regular - 4) * e_series
    )
    source_H = SQRT2 * (lam * u_series[0] + sp.diff(u_series[1], t))
    laurent = sp.series(source_H / ff ** (m + 1), t, 0, 1).removeO().expand()
    residue = sp.factor(laurent.coeff(t, -1))
    return {
        "v": vv,
        "b": b_series,
        "a": a_series,
        "residue": residue,
        "log_theta": sp.factor(residue / 2**m),
    }


log2 = local_log_data(2)
log3 = local_log_data(3)
v2 = log2["v"]
v3 = log3["v"]
residue2_claim = -3 * (8144955 * v2 - 9409136 * SQRT2) / 56320
residue3_claim = 9 * (49573615 * v3 + 19988608 * SQRT2) / 229376
gate(
    "L6: Q2 multiplier transport has the exact logarithmic-residue formula",
    sp.simplify(log2["residue"] - residue2_claim) == 0
    and sp.diff(residue2_claim, v2) != 0,
    "generic term L2*theta^6 log|theta|, L2=residue/64",
)
gate(
    "L6: Q3 multiplier transport has the exact logarithmic-residue formula",
    sp.simplify(log3["residue"] - residue3_claim) == 0
    and sp.diff(residue3_claim, v3) != 0,
    "generic term L3*theta^8 log|theta|, L3=residue/256",
)

gate(
    "L6: representative B/A axis recurrences agree with closed anchors",
    sp.simplify(log2["b"].coeff(t, 3) - (SQRT2 / 2 - 121 * v2 / 96))
    == 0
    and sp.simplify(
        log2["a"].coeff(t, 0) + 2 * SQRT2 * v2 / 7
    )
    == 0
    and sp.simplify(
        log2["a"].coeff(t, 2) - (293 * SQRT2 * v2 / 80 - sp.Rational(6, 5))
    )
    == 0
    and sp.simplify(log3["b"].coeff(t, 3) + 75 * v3 / 32) == 0
    and sp.simplify(
        log3["a"].coeff(t, 0) + 2 * SQRT2 * v3 / 9
    )
    == 0
    and sp.simplify(
        log3["a"].coeff(t, 2) - 477 * SQRT2 * v3 / 112
    )
    == 0,
)


# -------------------------------------------------------------------------
# L7. Restored dimensions and radial ordering.
x_Q = 2 - 2 * k
# The delayed bare-Q row-two stress contains J^-3 delta(cof_2 F), hence its
# P exponent is x_Q-3/2.  Elastic cancellation fixes the opening to that
# exponent.  Constraint balance then matches delta F1_A * F2^0 with
# F1^0 * delta F2_B, and the multiplier stress matches H*C_F1^0 with
# C_F1^0 ~ P^(5/2).
x_B = x_Q - sp.Rational(3, 2)
x_A = x_B - sp.Rational(3, 2)
x_H = x_A - sp.Rational(5, 2)
x_V2 = x_A - sp.Rational(3, 2)

# Re-run the complete action differentiation with arbitrary P and c1. This
# turns the exponent balance above into a component-level scaling check.
P_scale, c1_scale = sp.symbols("P_scale c1_scale", positive=True)
F1_scaled = (
    P_scale**-sp.Rational(1, 2) * r**sp.Rational(1, 4) * u
    + eps * P_scale**x_A * r ** (Lambda - 1) * d
)
F2_scaled = (
    P_scale * r**-sp.Rational(1, 2) * v
    + eps * P_scale**x_B * r ** (nu - 1) * e
)
J_scaled = sp.det(sp.Matrix.hstack(F1_scaled, F2_scaled))
cof1_scaled = sp.Matrix([F2_scaled[1], -F2_scaled[0]])
cof2_scaled = sp.Matrix([-F1_scaled[1], F1_scaled[0]])
chi_scaled = (
    c1_scale * P_scale**-3 * r**sp.Rational(3, 2) * psi
    + eps
    * c1_scale
    * P_scale**x_H
    * r ** (Lambda + sp.Rational(1, 4))
    * H
)
P1_scaled = (
    2 * c1_scale * F1_scaled
    - 2 * c1_scale * J_scaled**-3 * cof1_scaled
    + 4 * chi_scaled * J_scaled**3 * cof1_scaled
)
P2_scaled = (
    2 * c1_scale * F2_scaled
    - 2 * c1_scale * J_scaled**-3 * cof2_scaled
    + chi_scaled * (4 * J_scaled**3 * cof2_scaled - 2 * F2_scaled)
)
C_scaled = J_scaled**4 - F2_scaled.dot(F2_scaled)
scaled_constraint_ok = circle_zero(
    sp.diff(C_scaled, eps).subs(eps, 0)
    / (P_scale ** (sp.Rational(3, 2) - 2 * k)
       * r ** (Lambda - sp.Rational(9, 4)))
    - (SQRT2 * D_raw - 2 * M)
)
scaled_row1_ok = all(
    circle_zero(
        sp.diff(P1_scaled[j], eps).subs(eps, 0)
        / (c1_scale * P_scale**x_A * r ** (Lambda - 1))
        - S_claim[j]
    )
    for j in range(2)
)
scaled_row2_ok = all(
    circle_zero(
        (
            sp.diff(P2_scaled[j], eps).subs(eps, 0)
            - 2
            * c1_scale
            * P_scale**x_B
            * r ** (Lambda - sp.Rational(7, 4))
            * e[j]
        )
        / (c1_scale * P_scale**x_V2 * r ** (Lambda - sp.Rational(1, 4)))
        - V2_claim[j]
    )
    for j in range(2)
)
gate(
    "L7: restored companion amplitudes have the required P powers",
    sp.simplify(x_B - (sp.Rational(1, 2) - 2 * k)) == 0
    and sp.simplify(x_A - (-2 * k - 1)) == 0
    and sp.simplify(x_H - (-2 * k - sp.Rational(7, 2))) == 0
    and sp.simplify(x_V2 - (-2 * k - sp.Rational(5, 2))) == 0,
    "derived from lower row two, constraint balance, and C_F1~P^(5/2)",
)
gate(
    "L7: arbitrary-P,c1 action differentiation confirms all restored scales",
    scaled_constraint_ok and scaled_row1_ok and scaled_row2_ok,
    "y2:P^(1/2-2k), y1:P^(-2k-1), chi:c1 P^(-2k-7/2)",
)
gate(
    "L7: next row-two stress is 3/2 radial powers after the shared row-two stress",
    sp.simplify(
        (k + sp.Rational(5, 4)) - (k - sp.Rational(1, 4))
        - sp.Rational(3, 2)
    )
    == 0,
)


print("\nRung-closed Lambda=k+3/2 companion in the admitted endpoint class:")
print("  2(B''+nu^2 B)+R_2k=0, B(0)=0, B'(pi)=2sqrt(2)k")
print("  Lambda f'A-(f/2)A'=sqrt(2)[nu f B/2+f'B']+nu g'B-5gB'/4")
print("  f H'-(2Lambda-1)f'H=sqrt(2)[Lambda U_r+U_theta']")
print("  A(0)=-sqrt(2)B'(0)/Lambda; H(pi)=20k g(pi)")
print("  A_h=f^(2k+3) is excluded by Mode-I parity; H_h=f^(2k+2) is face-fixed")
print("  H is even Puiseux-log and can contain |theta|^(7/2) and")
print("  theta^(2k+2) log|theta| (selected Q2/Q3 are nonzero numerically).")
print("\nRestored fields:")
print("  delta y2_sl = q_k P^(1/2-2k) r^(k+3/4) B")
print("  delta y1_sl = q_k P^(-2k-1) r^(k+3/2) A")
print("  delta chi_sl = q_k c1 P^(-2k-7/2) r^(k+7/4) H")
print("  V2 is generated at row-two stress power r^(k+5/4);")
print("  its divergence/face load belongs to the next Lambda=k+3 rung.")

failed = [name for name, passed in GATES if not passed]
print("\n" + "=" * 76)
print(f"PASSED {len(GATES) - len(failed)}   FAILED {len(failed)}")
if failed:
    print("FAILURES:", failed)
    raise SystemExit(1)
print("All later-companion gates passed.")
