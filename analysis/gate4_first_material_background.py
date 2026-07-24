#!/usr/bin/env python3
"""Close the first material-background rung of the constrained MR hierarchy.

The exact scope and stopping line are fixed in
``GATE4_FIRST_MATERIAL_BACKGROUND_CONTRACT_2026-07-21.md``.  This script
derives the ``O(rho)``, ``Lambda=7/4`` material rung from

    A = W1 + rho*W2_exact + chi*C,

where ``rho=c2/c1`` and ``chi`` is the emergent constrained-outer reaction.
It does not close the distinct ``Lambda=11/4`` stationary-background rung.

All current-grade field, constraint, stress, equilibrium, reaction, endpoint,
gauge, and restored-scale assertions are exact.  Fixed-step arbitrary-
precision RK4 and Cash--Karp integrations only provide an independent global
continuation check of the selected analytic branch.
"""
from __future__ import annotations

import sys
from typing import Callable

import mpmath as mp
import sympy as sp


# ---------------------------------------------------------------------------
# Exact helpers and reporting.
# ---------------------------------------------------------------------------
GATES: list[tuple[str, bool]] = []


def gate(name: str, condition: bool, detail: str = "") -> None:
    passed = bool(condition)
    GATES.append((name, passed))
    suffix = f"  ({detail})" if detail else ""
    print(f"[{'PASS' if passed else 'FAIL'}] {name}{suffix}")


def circle_numerator(expr: sp.Expr, cvar: sp.Symbol, fvar: sp.Symbol) -> sp.Expr:
    """Reduce a rational numerator modulo c^2+f^2=1."""
    numerator = sp.fraction(sp.cancel(sp.together(expr)))[0]
    numerator = sp.powsimp(sp.expand(numerator), force=True)
    polynomial = sp.Poly(numerator, cvar)
    circle = sp.Poly(cvar**2 - (1 - fvar**2), cvar)
    return sp.factor(polynomial.rem(circle).as_expr())


def circle_zero(expr: sp.Expr) -> bool:
    return circle_numerator(expr, c, f) == 0


def matrix_circle_zero(matrix: sp.Matrix) -> bool:
    return all(circle_zero(component) for component in matrix)


SQRT2 = sp.sqrt(2)
f, c = sp.symbols("f c", positive=True)
g, w = sp.symbols("g w", real=True)
wp = sp.symbols("wp", real=True)
v, vp = sp.symbols("v vp", real=True)
psi0, p = sp.symbols("psi0 p", real=True)
z, rho = sp.symbols("z rho", positive=True)

# The leading determinant identity has already eliminated g'.
gp = sp.Rational(5, 4) * c * g / f - SQRT2 / f
u = sp.Matrix([sp.Rational(5, 4) * g, gp])
h0 = sp.Matrix([f / 2, c / 2])
d = sp.Matrix([sp.Rational(7, 4) * w, wp])
e = sp.Matrix([v, vp])
K = sp.Matrix([c / 2, -f / 2])
L = sp.Matrix([-gp, sp.Rational(5, 4) * g])
E = sp.Matrix([vp, -v])
D = sp.det(sp.Matrix.hstack(d, h0)) + sp.det(sp.Matrix.hstack(u, e))


# ---------------------------------------------------------------------------
# A. Exact action, multiplier gauge, and direct coefficient extraction.
# ---------------------------------------------------------------------------
X11, X12, X21, X22 = sp.symbols("X11 X12 X21 X22", real=True)
Xs1 = sp.Matrix([X11, X12])
Xs2 = sp.Matrix([X21, X22])
Js = X11 * X22 - X12 * X21
W1s = Xs1.dot(Xs1) + Xs2.dot(Xs2) + Js**-2 - 3
W2_exact = Js**2 + (Xs1.dot(Xs1) + Xs2.dot(Xs2)) * Js**-2 - 3
Cs = Js**4 - Xs2.dot(Xs2)
W2_tangent = 2 * Js**2 + Xs1.dot(Xs1) * Js**-2 - 3

gate(
    "A1: exact and tangent W2 extensions differ by -J^-2 C",
    sp.cancel(W2_exact - W2_tangent + Js**-2 * Cs) == 0,
)

chi_exact, chi_tangent = sp.symbols("chi_exact chi_tangent", real=True)
P_exact_generic = sp.Matrix(
    [
        sp.diff(W1s + rho * W2_exact + chi_exact * Cs, variable)
        for variable in (X11, X12, X21, X22)
    ]
)
P_tangent_generic = sp.Matrix(
    [
        sp.diff(W1s + rho * W2_tangent + chi_tangent * Cs, variable)
        for variable in (X11, X12, X21, X22)
    ]
).subs(chi_tangent, chi_exact - rho * Js**-2)
gauge_remainder = sp.Matrix(
    [-rho * Cs * sp.diff(Js**-2, variable) for variable in (X11, X12, X21, X22)]
)
gate(
    "A2: exact/tangent multiplier shift leaves total stress equal on C=0",
    all(
        sp.cancel(P_exact_generic[index] - P_tangent_generic[index]
                  - gauge_remainder[index]) == 0
        for index in range(4)
    ),
    "off-manifold difference is -rho*C*d(J^-2)/dF",
)

F1 = z * u + rho * z**3 * d
F2 = z**-2 * h0 + rho * e
chi = z**6 * psi0 + rho * z**8 * p
field_subs = {X11: F1[0], X12: F1[1], X21: F2[0], X22: F2[1]}
action = W1s + rho * W2_exact + chi * Cs
P1_direct = sp.Matrix(
    [sp.diff(action, variable).subs(field_subs, simultaneous=True)
     for variable in (X11, X12)]
)
P2_direct = sp.Matrix(
    [sp.diff(action, variable).subs(field_subs, simultaneous=True)
     for variable in (X21, X22)]
)
C_direct = Cs.subs(field_subs, simultaneous=True)


def rho_coefficient(expr: sp.Expr) -> sp.Expr:
    return sp.diff(expr, rho).subs(rho, 0)


M = 4 * u - 4 * SQRT2 * u.dot(u) * K
N = -4 * SQRT2 * u.dot(u) * L
S = (
    2 * d
    + (24 + 6 * psi0) * D * K
    + SQRT2 * (psi0 - 4) * E
    + SQRT2 * p * K
)
V = (
    (24 + 6 * psi0) * D * L
    + SQRT2 * (psi0 - 4) * sp.Matrix([-wp, sp.Rational(7, 4) * w])
    - 2 * psi0 * e
    + p * (SQRT2 * L - 2 * h0)
)
constraint_claim = z**-2 * (SQRT2 * D - 2 * h0.dot(e))
row1_claim = z**3 * (S + M)
row2_claim = 2 * e + 4 * h0 + z**6 * (V + N)

gate(
    "A3: direct action gives the sourced first constraint coefficient",
    circle_zero(rho_coefficient(C_direct) - constraint_claim),
)
gate(
    "A4: direct action gives the complete rho*r^(3/4) row-one stress",
    matrix_circle_zero(P1_direct.applyfunc(rho_coefficient) - row1_claim),
)
gate(
    "A5: direct action gives current row-two stress plus outgoing handoff",
    matrix_circle_zero(P2_direct.applyfunc(rho_coefficient) - row2_claim),
)
gate(
    "A6: tangent-gauge shift begins at -2*rho*r^(1/2) in unit gauge",
    circle_zero(-rho / (sp.det(sp.Matrix.vstack((z * u).T, (z**-2 * h0).T)))**2
                + 2 * rho * z**2),
)


# ---------------------------------------------------------------------------
# B. Opening BVP and first-order constraint/slaving equation.
# ---------------------------------------------------------------------------
v_selected = -sp.Rational(2, 3) * f
vp_selected = -c / 3
e_selected = e.subs({v: v_selected, vp: vp_selected})
Q = (2 * e + 4 * h0).subs({v: v_selected, vp: vp_selected})


def dtheta_fc(expr: sp.Expr) -> sp.Expr:
    return sp.diff(expr, f) * c / 2 - sp.diff(expr, c) * f / 2


gate(
    "B1: selected v solves the exact current row-two equilibrium",
    circle_zero(Q[0] + dtheta_fc(Q[1])),
    "equivalent to v''+v=-f/2",
)
gate(
    "B2: opening symmetry and total current-grade face traction hold",
    sp.simplify(v_selected.subs(f, 0)) == 0
    and sp.simplify(Q[1].subs({f: 1, c: 0})) == 0,
)
gate(
    "B3: the opening BVP is unique in the stated Mode-I endpoint class",
    sp.cos(sp.pi) != 0,
    "the remaining homogeneous A*sin(theta) has face derivative -A",
)

D1 = -SQRT2 * (1 + f**2) / 6
constraint_selected = (SQRT2 * D - 2 * h0.dot(e)).subs(
    {v: v_selected, vp: vp_selected}
)
wp_selected = sp.factor(sp.solve(sp.Eq(constraint_selected, 0), wp)[0])
wp_expected = (
    sp.Rational(7, 4) * c * w / f
    + sp.Rational(5, 6) * c * g / f
    - SQRT2 * (3 - f**2) / (3 * f)
)
gate(
    "B4: sourced constraint gives D1=-sqrt(2)(1+f^2)/6",
    circle_zero(
        D.subs({v: v_selected, vp: vp_selected, wp: wp_selected}) - D1
    ),
)
gate(
    "B5: direct constraint gives the historical w transport exactly",
    circle_zero(wp_selected - wp_expected),
)

w0 = sp.Rational(4, 21) * SQRT2
g0 = sp.Rational(4, 5) * SQRT2
w0_symbol = sp.symbols("w0_symbol", real=True)
# Use the undivided numerator of the transport.  Substituting f=0 into the
# already divided equation would manufacture 0/0 and hide the axis condition.
axis_constraint = sp.simplify(
    sp.cancel(f * wp_expected).subs({f: 0, c: 1, g: g0, w: w0_symbol})
)
axis_w_solution = sp.solve(sp.Eq(axis_constraint, 0), w0_symbol)[0]
face_wp = sp.simplify(wp_expected.subs({f: 1, c: 0}))
gate(
    "B6: undivided axis constraint forces w(0)=4sqrt(2)/21",
    sp.simplify(axis_w_solution - w0) == 0,
)
gate(
    "B7: undivided face constraint forces w'(pi)=-2sqrt(2)/3",
    sp.simplify(face_wp + 2 * SQRT2 / 3) == 0,
)

w_h = f ** sp.Rational(7, 2)
w_h_prime = dtheta_fc(w_h)
gate(
    "B8: f^(7/2) is the sole homogeneous w direction",
    circle_zero(
        (D.subs({
            v: v_selected,
            vp: vp_selected,
            w: w + w_h,
            wp: wp_selected + w_h_prime,
        }) - D1)
    )
    and sp.denom(sp.Rational(7, 2)) == 2,
    "excluded by analytic-even displacement regularity",
)


def central_coefficient(n: int) -> sp.Expr:
    return sp.binomial(2 * n, n) / sp.Integer(4) ** n


def g_series_coefficient(n: int) -> sp.Expr:
    return -4 * SQRT2 * central_coefficient(n) / (4 * n - 5)


def w_series_coefficient(n: int) -> sp.Expr:
    cn = central_coefficient(n)
    return -sp.Rational(4, 3) * SQRT2 * cn * (
        sp.Rational(10, 4 * n - 5)
        + sp.Rational(4 * n - 3, 2 * n - 1)
    ) / (4 * n - 7)


n_sym = sp.symbols("n", integer=True, nonnegative=True)
Cn = sp.symbols("C_n", nonzero=True)
gn = -4 * SQRT2 * Cn / (4 * n_sym - 5)
wn = -sp.Rational(4, 3) * SQRT2 * Cn * (
    10 / (4 * n_sym - 5) + (4 * n_sym - 3) / (2 * n_sym - 1)
) / (4 * n_sym - 7)
Cprev = 2 * n_sym * Cn / (2 * n_sym - 1)
w_recurrence_residual = (
    (2 * n_sym - sp.Rational(7, 2)) * wn
    - sp.Rational(5, 3) * gn
    + sp.Rational(2, 3) * SQRT2 * (3 * Cn - Cprev)
)
gate(
    "B9: exact all-n analytic-axis recurrences generate g and w",
    sp.simplify(w_recurrence_residual) == 0
    and sp.simplify(g_series_coefficient(0) - g0) == 0
    and sp.simplify(w_series_coefficient(0) - w0) == 0,
)

# At f=1, the w recurrence is summed by applying Gauss' theorem to the
# following exact partial fractions.  The last sum vanishes by analytic
# continuation of Gauss' formula; the first two give the Gamma ratios used
# by the independent numerical endpoint check below.
w_sum_factor = (
    10 / (4 * n_sym - 5) + (4 * n_sym - 3) / (2 * n_sym - 1)
) / (4 * n_sym - 7)
w_sum_partial = (
    -5 / (4 * n_sym - 5)
    + sp.Rational(33, 5) / (4 * n_sym - 7)
    + sp.Rational(1, 5) / (2 * n_sym - 1)
)
gate(
    "B10: analytic w-face series has the exact Gauss-summable decomposition",
    sp.simplify(w_sum_factor - w_sum_partial) == 0,
)


# ---------------------------------------------------------------------------
# C. Complete row-one reaction, total traction, and outgoing row-two handoff.
# ---------------------------------------------------------------------------
psi0_selected = (
    4 + 6 / f**2 - sp.Rational(15, 4) * SQRT2 * c * g / f**2
    - 10 * f ** sp.Rational(3, 2)
)


def dtheta_selected(expr: sp.Expr) -> sp.Expr:
    """Differentiate on the selected g,w,v background quotient."""
    return (
        sp.diff(expr, f) * c / 2
        - sp.diff(expr, c) * f / 2
        + sp.diff(expr, g) * gp
        + sp.diff(expr, w) * wp_expected
    )


d_selected = d.subs(wp, wp_expected)
U = (
    2 * d_selected
    + (24 + 6 * psi0_selected) * D1 * K
    + SQRT2 * (psi0_selected - 4) * sp.Matrix([vp_selected, -v_selected])
    + M
)
p_particular = (
    sp.Rational(128, 3) / f**2
    + sp.Rational(104, 15)
    + 8 * f**2
    - 10 * f ** sp.Rational(7, 2)
    - sp.Rational(355, 12) * SQRT2 * c * g / f**2
    - sp.Rational(15, 4) * SQRT2 * c * g
    - sp.Rational(35, 4) * SQRT2 * c * w / f**2
    + sp.Rational(25, 4) * g**2 / f**2
)
p_selected = p_particular - sp.Rational(238, 5) * f ** sp.Rational(5, 2)
reaction_source = SQRT2 * (
    sp.Rational(7, 4) * U[0] + dtheta_selected(U[1])
)
reaction_transport = (
    f * dtheta_selected(p_selected)
    - sp.Rational(5, 2) * (c / 2) * p_selected
)
T = U + SQRT2 * p_selected * K

gate(
    "C1: closed p solves the exact row-one reaction transport",
    circle_zero(reaction_transport - reaction_source),
)
gate(
    "C2: complete current-grade row-one stress is divergence-free exactly",
    circle_zero(sp.Rational(7, 4) * T[0] + dtheta_selected(T[1])),
)

p_h = f ** sp.Rational(5, 2)
gate(
    "C3: f^(5/2) is the homogeneous reaction direction",
    circle_zero(
        f * dtheta_fc(p_h) - sp.Rational(5, 2) * (c / 2) * p_h
    ),
)

U_theta_face = sp.simplify(U[1].subs({f: 1, c: 0}))
p_part_face = sp.simplify(p_particular.subs({f: 1, c: 0}))
p_face = sp.simplify(p_selected.subs({f: 1, c: 0}))
gate(
    "C4: total face traction selects p(pi)=25*g(pi)^2/4",
    sp.simplify(U_theta_face - 25 * SQRT2 * g**2 / 8) == 0
    and sp.simplify(p_part_face - 25 * g**2 / 4 - sp.Rational(238, 5)) == 0
    and sp.simplify(p_face - 25 * g**2 / 4) == 0
    and sp.simplify(T[1].subs({f: 1, c: 0})) == 0,
)

# Generate sufficient exact analytic-axis data to take the cancellations in
# p/f^2 honestly.  No archived coefficient is imported.
g_axis_series = sum(g_series_coefficient(index) * f ** (2 * index)
                    for index in range(4))
w_axis_series = sum(w_series_coefficient(index) * f ** (2 * index)
                    for index in range(4))
p_axis_series = sp.series(
    p_selected.subs({g: g_axis_series, w: w_axis_series, c: sp.sqrt(1 - f**2)}),
    f,
    0,
    4,
).removeO().expand()
gate(
    "C5: selected reaction has the exact regular-axis limit p(0)=-256/15",
    sp.simplify(p_axis_series.coeff(f, 0) + sp.Rational(256, 15)) == 0
    and sp.simplify(p_axis_series.coeff(f, 2) - sp.Rational(472, 3)) == 0
    and sp.simplify(
        p_axis_series.coeff(f ** sp.Rational(5, 2)) + sp.Rational(238, 5)
    ) == 0,
    "p=-256/15+(472/3)f^2-(238/5)f^(5/2)-10f^(7/2)+...",
)

H = (
    V.subs({
        v: v_selected,
        vp: vp_selected,
        wp: wp_expected,
        psi0: psi0_selected,
        p: p_selected,
    })
    + N
)
H_r_face = sp.simplify(H[0].subs({f: 1, c: 0}))
H_theta_face = sp.simplify(H[1].subs({f: 1, c: 0}))
gate(
    "C6: outgoing row-two face handoff has the exact nonzero form",
    sp.simplify(H_r_face + (75 * g**2 + 448) / 12) == 0
    and sp.simplify(H_theta_face + SQRT2 * (20 * g + 7 * w)) == 0,
    "it is handed to Lambda=13/4, not imposed at the current grade",
)


# ---------------------------------------------------------------------------
# D. Restored P,c1,c2 powers from a second direct-action extraction.
# ---------------------------------------------------------------------------
Pamp, c1amp = sp.symbols("P_amp c1_amp", positive=True)
F1_scaled = Pamp**-sp.Rational(1, 2) * z * u \
    + rho * Pamp**-sp.Rational(3, 2) * z**3 * d
F2_scaled = Pamp * z**-2 * h0 + rho * e
chi_scaled = (
    c1amp * Pamp**-3 * z**6 * psi0
    + rho * c1amp * Pamp**-4 * z**8 * p
)
scaled_subs = {
    X11: F1_scaled[0], X12: F1_scaled[1],
    X21: F2_scaled[0], X22: F2_scaled[1],
}
scaled_action = c1amp * W1s + c1amp * rho * W2_exact + chi_scaled * Cs
P1_scaled = sp.Matrix([
    sp.diff(scaled_action, variable).subs(scaled_subs, simultaneous=True)
    for variable in (X11, X12)
])
P2_scaled = sp.Matrix([
    sp.diff(scaled_action, variable).subs(scaled_subs, simultaneous=True)
    for variable in (X21, X22)
])
C_scaled = Cs.subs(scaled_subs, simultaneous=True)

gate(
    "D1: restored first constraint coefficient is rho*P*r^(-1/2)",
    circle_zero(rho_coefficient(C_scaled) - Pamp * constraint_claim),
)
gate(
    "D2: restored row-one material stress is c1*rho*P^(-3/2)r^(3/4)T",
    matrix_circle_zero(
        P1_scaled.applyfunc(rho_coefficient)
        - c1amp * Pamp**-sp.Rational(3, 2) * row1_claim
    ),
)
gate(
    "D3: restored row-two current/handoff scales are c1*rho and c1*rho*P^-3",
    matrix_circle_zero(
        P2_scaled.applyfunc(rho_coefficient)
        - c1amp * (2 * e + 4 * h0 + Pamp**-3 * z**6 * (V + N))
    ),
)
gate(
    "D4: restored exact/tangent gauge shift is -2*c2*P^-1*r^(1/2)",
    sp.simplify(
        -c1amp * rho
        / sp.det(sp.Matrix.vstack(
            (Pamp**-sp.Rational(1, 2) * z * u).T,
            (Pamp * z**-2 * h0).T,
        ))**2
        + 2 * c1amp * rho * Pamp**-1 * z**2
    ) == 0,
)


# ---------------------------------------------------------------------------
# E. Re-generate the accepted Q_k x level-one-background source.
# ---------------------------------------------------------------------------
k = sp.symbols("k", integer=True, positive=True)
a_k = f ** (2 * k)
dQ = sp.Matrix([k * a_k, k * c * f ** (2 * k - 1)])
pQ = -2 * SQRT2 * k * (2 * k - 1) * c * f ** (2 * k - 2)
Hk = sp.det(sp.Matrix.hstack(dQ, e_selected))
h1 = sp.Matrix([vp_selected, -v_selected])
B_k = ((24 + 6 * psi0_selected) * Hk + 6 * pQ * D1) * K \
    + SQRT2 * pQ * h1

gate(
    "E1: background determinant cross is H_k=(2k/3)f^(2k)f'",
    circle_zero((Hk - sp.Rational(2, 3) * k * a_k * c / 2) / a_k),
)
background_face_divergence = sp.simplify(
    ((k + sp.Rational(1, 2)) * B_k[0] + dtheta_selected(B_k[1]))
    .subs({f: 1, c: 0})
)
gate(
    "E2: regenerated background cross has the accepted face divergence",
    sp.simplify(
        background_face_divergence - sp.Rational(4, 3) * k * (5 * k - 1)
    ) == 0
    and sp.simplify(B_k[1].subs({f: 1, c: 0})) == 0,
)


def direct_q_background_gate(k_value: int) -> bool:
    """Extract q*rho directly, retaining arbitrary W_r and psi1."""
    kval = sp.Integer(k_value)
    qlocal, rholocal = sp.symbols("q_local rho_local", real=True)
    w_r, psi1_local, psi0_local = sp.symbols(
        "w_r psi1_local psi0_local", real=True
    )
    aval = f ** (2 * kval)
    dval = sp.Matrix([kval * aval, kval * c * f ** (2 * kval - 1)])
    pQval = -2 * SQRT2 * kval * (2 * kval - 1) \
        * c * f ** (2 * kval - 2)
    det_u_e = sp.det(sp.Matrix.hstack(u, e_selected))
    w_theta = sp.solve(
        sp.Eq(
            sp.det(sp.Matrix.hstack(sp.Matrix([w_r, sp.Symbol("w_theta")]), h0))
            + det_u_e,
            D1,
        ),
        sp.Symbol("w_theta"),
    )[0]
    Wlocal = sp.Matrix([w_r, w_theta])

    F1local = z * u + rholocal * z**3 * Wlocal \
        + qlocal * z ** (4 * k_value - 4) * dval
    F2local = z**-2 * h0 + rholocal * e_selected
    chilocal = (
        z**6 * psi0_local
        + rholocal * z**8 * psi1_local
        + qlocal * z ** (4 * k_value + 1) * pQval
    )
    substitutions = {
        X11: F1local[0], X12: F1local[1],
        X21: F2local[0], X22: F2local[1],
    }
    local_action = W1s + rholocal * W2_exact + chilocal * Cs
    local_p1 = sp.Matrix([
        sp.diff(local_action, variable).subs(substitutions, simultaneous=True)
        for variable in (X11, X12)
    ])

    def selected_cross(component: sp.Expr) -> sp.Expr:
        cross = sp.diff(sp.diff(component, qlocal), rholocal).subs(
            {qlocal: 0, rholocal: 0}
        )
        return sp.simplify(
            sp.limit(cross / z ** (4 * k_value - 2), z, 0, dir="+")
        )

    direct_cross = local_p1.applyfunc(selected_cross)
    Hval = sp.det(sp.Matrix.hstack(dval, e_selected))
    Bval = ((24 + 6 * psi0_local) * Hval + 6 * pQval * D1) * K \
        + SQRT2 * pQval * h1
    direct_c2 = 4 * dval - 8 * SQRT2 * u.dot(dval) * K
    return all(
        circle_zero(component / f ** (2 * k_value - 2))
        for component in direct_cross - direct_c2 - Bval
    )


for representative_k in (2, 3, 4):
    gate(
        f"E3.{representative_k}: direct full-action Q_{representative_k} x background extraction",
        direct_q_background_gate(representative_k),
        "arbitrary local W_r and psi1 retained",
    )


# ---------------------------------------------------------------------------
# F. Independent arbitrary-precision global continuation.
# ---------------------------------------------------------------------------
# These tolerances and meshes are fixed before the final numerical run.
MP_DPS = 70
mp.mp.dps = MP_DPS
# Starting too near the regular singular point unnecessarily amplifies the
# fixed-step error in the independent p transport.  The 36-term exact series
# is still far below the acceptance and discretization errors at these
# switches; no acceptance tolerance is changed by this conditioning choice.
THETA_START = mp.mpf("0.25")
THETA_START_ALT = mp.mpf("0.20")
SERIES_TERMS = 36
RK4_MESHES = (9600, 19200)
CK5_MESHES = (1600, 3200)
RAW_ENDPOINT_TOL = mp.mpf("2e-9")
EXTRAPOLATED_TOL = mp.mpf("2e-11")
METHOD_AGREEMENT_TOL = mp.mpf("2e-11")
SWITCH_TOL = mp.mpf("2e-9")


def mp_central(n: int) -> mp.mpf:
    return mp.binomial(2 * n, n) / mp.mpf(4) ** n


def mp_g_coefficient(n: int) -> mp.mpf:
    return -4 * mp.sqrt(2) * mp_central(n) / (4 * n - 5)


def mp_w_coefficient(n: int) -> mp.mpf:
    return (
        -mp.mpf(4) * mp.sqrt(2) / 3 * mp_central(n)
        * (mp.mpf(10) / (4 * n - 5)
           + mp.mpf(4 * n - 3) / (2 * n - 1))
        / (4 * n - 7)
    )


def analytic_axis_gw(
    theta: mp.mpf,
    terms: int = SERIES_TERMS,
) -> tuple[mp.mpf, mp.mpf]:
    fval = mp.sin(theta / 2)
    x = fval**2
    gval = mp.mpf("0")
    wval = mp.mpf("0")
    power = mp.mpf("1")
    for index in range(terms):
        gval += mp_g_coefficient(index) * power
        wval += mp_w_coefficient(index) * power
        power *= x
    return gval, wval


def reaction_particular_coefficients(terms: int) -> tuple[mp.mpf, ...]:
    """Generate the no-``f^(5/2)`` reaction series from its transport.

    Put ``t=sqrt(f)`` and ``q=t^4=f^2``.  The inhomogeneous transport is

        t p_t - 5 p = 4 R/c.

    Its analytic part contains only ``q^n`` powers; the direct material term
    additionally gives ``-10 t^7`` in ``p``.  The resonant coefficient at
    ``t^5`` is set to zero here and is selected later by numerical face
    shooting.  No closed reaction formula or face-selected coefficient enters
    this initializer.
    """
    # Coefficient n needs g,w through n+1 because R contains q^(-1)g,w.
    g_coeff = [mp_g_coefficient(index) for index in range(terms + 1)]
    w_coeff = [mp_w_coefficient(index) for index in range(terms + 1)]
    inv_c_coeff = [mp_central(index) for index in range(terms + 1)]

    def square_g(index: int) -> mp.mpf:
        return sum(
            g_coeff[left] * g_coeff[index - left]
            for left in range(index + 1)
        )

    def bracket(index: int) -> mp.mpf:
        value = mp.mpf("0")
        if index >= 1:
            value += 90 * g_coeff[index - 1]
        if index >= 0:
            value -= 360 * g_coeff[index]
        if index >= -1:
            value += 470 * g_coeff[index + 1]
            value += 210 * w_coeff[index + 1]
        return value

    # The apparent q^(-1) coefficient must cancel before division by the
    # indicial factor.  Keeping this fail-hard check here guards the singular
    # axis initializer independently of the closed p expression.
    r_minus_one = (
        -mp.mpf(232) / 3
        + mp.mpf(25) * square_g(0) / 4
        + mp.sqrt(2) * bracket(-1) / 12
    )
    if abs(r_minus_one) > mp.mpf("1e-60"):
        raise ArithmeticError("reaction source retained a q^(-1) pole")

    coefficients: list[mp.mpf] = []
    for index in range(terms):
        convolution = sum(
            inv_c_coeff[left] * bracket(index - left)
            for left in range(index + 2)
        )
        source_coefficient = (
            (-28 if index == 0 else 0)
            + (-8 if index == 1 else 0)
            + mp.mpf(25) * square_g(index + 1) / 4
            + mp.sqrt(2) * convolution / 12
        )
        coefficients.append(source_coefficient / (4 * index - 5))
    return tuple(coefficients)


P_PARTICULAR_COEFFICIENTS = reaction_particular_coefficients(SERIES_TERMS)


def analytic_axis_p_particular(theta: mp.mpf) -> mp.mpf:
    """Evaluate the transport-generated particular reaction at a switch."""
    fval = mp.sin(theta / 2)
    qval = fval**2
    value = mp.mpf("0")
    power = mp.mpf("1")
    for coefficient in P_PARTICULAR_COEFFICIENTS:
        value += coefficient * power
        power *= qval
    return value - 10 * fval ** mp.mpf("3.5")


def background_rhs(theta: mp.mpf, state: tuple[mp.mpf, ...]) -> tuple[mp.mpf, ...]:
    gval, wval, pval, hval = state
    fval = mp.sin(theta / 2)
    cval = mp.cos(theta / 2)
    sqrt2 = mp.sqrt(2)
    gpval = 5 * cval * gval / (4 * fval) - sqrt2 / fval
    wpval = (
        7 * cval * wval / (4 * fval)
        + 5 * cval * gval / (6 * fval)
        - sqrt2 * (3 - fval**2) / (3 * fval)
    )
    source = (
        -240 * cval * fval ** mp.mpf("5.5")
        - 96 * cval * fval**4
        - 336 * cval * fval**2
        + 75 * cval * gval**2
        - 928 * cval
        + 90 * sqrt2 * fval**4 * gval
        - 360 * sqrt2 * fval**2 * gval
        + 470 * sqrt2 * gval
        + 210 * sqrt2 * wval
    ) / (48 * fval**2)
    ppval = (mp.mpf(5) * cval * pval / 4 + source) / fval
    hpval = mp.mpf(5) * cval * hval / (4 * fval)
    return gpval, wpval, ppval, hpval


def add_scaled(state, increments):
    return tuple(
        value + sum(coefficient * derivative[index]
                    for coefficient, derivative in increments)
        for index, value in enumerate(state)
    )


def rk4_step(rhs, theta: mp.mpf, state, step: mp.mpf):
    k1v = rhs(theta, state)
    k2v = rhs(theta + step / 2, add_scaled(state, ((step / 2, k1v),)))
    k3v = rhs(theta + step / 2, add_scaled(state, ((step / 2, k2v),)))
    k4v = rhs(theta + step, add_scaled(state, ((step, k3v),)))
    return add_scaled(
        state,
        ((step / 6, k1v), (step / 3, k2v),
         (step / 3, k3v), (step / 6, k4v)),
    )


def ck5_step(rhs, theta: mp.mpf, state, step: mp.mpf):
    """One fixed Cash--Karp fifth-order step."""
    k1v = rhs(theta, state)
    k2v = rhs(theta + step / 5, add_scaled(state, ((step / 5, k1v),)))
    k3v = rhs(
        theta + 3 * step / 10,
        add_scaled(state, ((3 * step / 40, k1v), (9 * step / 40, k2v))),
    )
    k4v = rhs(
        theta + 3 * step / 5,
        add_scaled(state, ((3 * step / 10, k1v), (-9 * step / 10, k2v),
                           (6 * step / 5, k3v))),
    )
    k5v = rhs(
        theta + step,
        add_scaled(state, ((-11 * step / 54, k1v), (5 * step / 2, k2v),
                           (-70 * step / 27, k3v), (35 * step / 27, k4v))),
    )
    k6v = rhs(
        theta + 7 * step / 8,
        add_scaled(
            state,
            ((mp.mpf(1631) * step / 55296, k1v),
             (mp.mpf(175) * step / 512, k2v),
             (mp.mpf(575) * step / 13824, k3v),
             (mp.mpf(44275) * step / 110592, k4v),
             (mp.mpf(253) * step / 4096, k5v)),
        ),
    )
    return add_scaled(
        state,
        ((mp.mpf(37) * step / 378, k1v),
         (mp.mpf(250) * step / 621, k3v),
         (mp.mpf(125) * step / 594, k4v),
         (mp.mpf(512) * step / 1771, k6v)),
    )


STEPPERS: dict[str, tuple[Callable, int]] = {
    "RK4": (rk4_step, 4),
    "CK5": (ck5_step, 5),
}


def continue_background(method: str, steps: int, theta_start: mp.mpf = THETA_START):
    gstart, wstart = analytic_axis_gw(theta_start)
    fstart = mp.sin(theta_start / 2)
    state = (
        gstart,
        wstart,
        analytic_axis_p_particular(theta_start),
        fstart ** mp.mpf("2.5"),
    )
    step = (mp.pi - theta_start) / steps
    stepper = STEPPERS[method][0]
    theta = theta_start
    for _ in range(steps):
        state = stepper(background_rhs, theta, state, step)
        theta += step
    gface, wface, ppart_face, hface = state

    # Evaluate the unsolved U_theta stress at the face, then select the
    # homogeneous reaction coefficient from T_theta=U_theta-p/sqrt(2)=0.
    sqrt2 = mp.sqrt(2)
    gpface = -sqrt2
    wpface = -2 * sqrt2 / 3
    uface = (5 * gface / 4, gpface)
    ktheta = -mp.mpf(1) / 2
    d1face = -sqrt2 / 3
    psi0face = mp.mpf("0")
    etheta = mp.mpf(2) / 3
    mtheta = (
        4 * gpface
        - 4 * sqrt2 * (uface[0] ** 2 + uface[1] ** 2) * ktheta
    )
    utheta = (
        2 * wpface
        + (24 + 6 * psi0face) * d1face * ktheta
        + sqrt2 * (psi0face - 4) * etheta
        + mtheta
    )
    reaction_amplitude = (sqrt2 * utheta - ppart_face) / hface
    pface = ppart_face + reaction_amplitude * hface

    # Use the unreduced p-dependent face stress for the outgoing handoff.
    outgoing_r = (-75 * gface**2 + 6 * pface - 224) / 6
    outgoing_theta = (
        sqrt2
        * (-125 * gface**3 + 20 * gface * pface - 320 * gface - 112 * wface)
        / 16
    )
    return (
        gface,
        wface,
        pface,
        reaction_amplitude,
        hface,
        outgoing_r,
        outgoing_theta,
    )


def richardson(coarse, fine, order: int):
    factor = mp.mpf(2) ** order - 1
    return tuple(fine[index] + (fine[index] - coarse[index]) / factor
                 for index in range(len(fine)))


g_face_exact = (
    4 * mp.sqrt(2) / 5
    * mp.gamma(-mp.mpf(1) / 4) * mp.sqrt(mp.pi)
    / mp.gamma(-mp.mpf(3) / 4)
)
sum_45 = mp.gamma(-mp.mpf(5) / 4) * mp.sqrt(mp.pi) \
    / (4 * mp.gamma(-mp.mpf(3) / 4))
sum_47 = mp.gamma(-mp.mpf(7) / 4) * mp.sqrt(mp.pi) \
    / (4 * mp.gamma(-mp.mpf(5) / 4))
w_face_exact = -4 * mp.sqrt(2) / 3 * (-5 * sum_45 + mp.mpf(33) / 5 * sum_47)
p_face_exact = mp.mpf(25) * g_face_exact**2 / 4
reaction_amplitude_exact = -mp.mpf(238) / 5
reaction_h_face_exact = mp.mpf(1)
outgoing_r_face = -(75 * g_face_exact**2 + 448) / 12
outgoing_theta_face = -mp.sqrt(2) * (20 * g_face_exact + 7 * w_face_exact)
exact_endpoint = (
    g_face_exact,
    w_face_exact,
    p_face_exact,
    reaction_amplitude_exact,
    reaction_h_face_exact,
    outgoing_r_face,
    outgoing_theta_face,
)

numeric_runs: dict[tuple[str, int], tuple[mp.mpf, ...]] = {}
for method, meshes in (("RK4", RK4_MESHES), ("CK5", CK5_MESHES)):
    for mesh in meshes:
        numeric_runs[(method, mesh)] = continue_background(method, mesh)

rk_extrapolated = richardson(
    numeric_runs[("RK4", RK4_MESHES[0])],
    numeric_runs[("RK4", RK4_MESHES[1])],
    4,
)
ck_extrapolated = richardson(
    numeric_runs[("CK5", CK5_MESHES[0])],
    numeric_runs[("CK5", CK5_MESHES[1])],
    5,
)
raw_errors = [
    abs(numeric_runs[(method, meshes[1])][index] - exact_endpoint[index])
    for method, meshes in (("RK4", RK4_MESHES), ("CK5", CK5_MESHES))
    for index in range(len(exact_endpoint))
]
extrapolated_errors = [
    abs(estimate[index] - exact_endpoint[index])
    for estimate in (rk_extrapolated, ck_extrapolated)
    for index in range(len(exact_endpoint))
]
method_spread = max(
    abs(rk_extrapolated[index] - ck_extrapolated[index])
    for index in range(len(exact_endpoint))
)
alternate_switch = continue_background("CK5", CK5_MESHES[1], THETA_START_ALT)
switch_spread = max(
    abs(alternate_switch[index] - numeric_runs[("CK5", CK5_MESHES[1])][index])
    for index in range(len(exact_endpoint))
)

print("\nNumerical continuation diagnostics:")
endpoint_names = ("g", "w", "p", "C_p", "h", "H_r", "H_theta")
for key, value in numeric_runs.items():
    errors = tuple(
        abs(value[index] - exact_endpoint[index])
        for index in range(len(exact_endpoint))
    )
    print(
        f"  {key[0]} N={key[1]} errors({','.join(endpoint_names)}) = "
        + ", ".join(mp.nstr(error, 8) for error in errors)
    )
print(
    "  RK4 extrap errors = "
    + ", ".join(
        mp.nstr(abs(rk_extrapolated[index] - exact_endpoint[index]), 8)
        for index in range(len(exact_endpoint))
    )
)
print(
    "  CK5 extrap errors = "
    + ", ".join(
        mp.nstr(abs(ck_extrapolated[index] - exact_endpoint[index]), 8)
        for index in range(len(exact_endpoint))
    )
)
print(
    "  alternate-switch errors = "
    + ", ".join(
        mp.nstr(abs(alternate_switch[index] - exact_endpoint[index]), 8)
        for index in range(len(exact_endpoint))
    )
)

gate(
    "F1: finest continuations meet the fixed full-endpoint tolerance",
    max(raw_errors) < RAW_ENDPOINT_TOL,
    f"max error={mp.nstr(max(raw_errors), 7)}",
)
gate(
    "F2: both Richardson full endpoints meet the fixed tolerance",
    max(extrapolated_errors) < EXTRAPOLATED_TOL,
    f"max error={mp.nstr(max(extrapolated_errors), 7)}",
)
gate(
    "F3: RK4 and CK5 extrapolations of the same shooting problem agree",
    method_spread < METHOD_AGREEMENT_TOL,
    f"spread={mp.nstr(method_spread, 7)}",
)
gate(
    "F4: analytic-series switch variation is below the fixed tolerance",
    switch_spread < SWITCH_TOL,
    f"spread={mp.nstr(switch_spread, 7)}",
)

print("\nSelected analytic material-background rung (unit gauge):")
print("  v(theta)       = -(2/3) sin(theta/2)")
print("  w(0)           = 4sqrt(2)/21")
print("  w'(pi)         = -2sqrt(2)/3")
print("  p(0)           = -256/15")
print("  p(pi)          = 25 g(pi)^2/4")
print(f"  g(pi)          = {mp.nstr(g_face_exact, 24)}")
print(f"  w(pi)          = {mp.nstr(w_face_exact, 24)}  (derived branch value)")
print(f"  p(pi)          = {mp.nstr(p_face_exact, 24)}")
print("  outgoing H_r   =", mp.nstr(outgoing_r_face, 24))
print("  outgoing H_th  =", mp.nstr(outgoing_theta_face, 24))
print("\nRestored fields:")
print("  delta y2  = (c2/c1) r v")
print("  delta y1  = (c2/c1) P^(-3/2) r^(7/4) w")
print("  delta chi = c2 P^(-4) r^2 p  (exact-W2 convention)")
print("  gauge shift to tangent W2 = -2 c2 P^(-1) r^(1/2)+...")
print("\nStopping line: Lambda=7/4 material rung closed; outgoing row-two")
print("handoff recorded.  Lambda=11/4 stationary background and total Gate 4 open.")

failed = [name for name, passed in GATES if not passed]
print("\n" + "=" * 76)
print(f"PASSED {len(GATES) - len(failed)}   FAILED {len(failed)}")
if failed:
    print("FAILURES:")
    for name in failed:
        print(" -", name)
    sys.exit(1)
print("All first-material-background gates passed.")
