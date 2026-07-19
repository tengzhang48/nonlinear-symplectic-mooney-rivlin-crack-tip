#!/usr/bin/env python3
"""Complete first-material-order full-MR response at Lambda = k + 1/2.

This script closes the block that ``qk_full_mr_ordering.py`` only located.
It works to first order in both a characteristic shear amplitude ``q_k`` and

    rho = c2/c1,

and keeps every term of order ``q_k*rho``.  It does *not* claim a solution at
fixed finite rho to all material orders.

The calculation uses the exact reduced plane-stress energy as an off-manifold
extension of the constrained action,

    W1 = F:F + J**(-2) - 3,
    W2 = J**2 + (F:F)*J**(-2) - 3,
    C  = J**4 - |F2|**2,
    A/c1 = W1 + rho*W2 + chi*C.

Changing to the on-manifold-equivalent remainder
``2*J**2 + J**(-2)*|F1|**2 - 3`` shifts ``chi`` by
``-rho*J**(-2)``.  The total stress, which is what is checked below, is
unchanged.

On the upper half-domain set

    f = sin(theta/2),  c = cos(theta/2),  a_k = f**(2*k),  k >= 2.

The forced level-one opening background is ``v=-2*f/3``.  Its interaction
with Q_k forces a smooth in-plane response

    e_k = -(4*k/3)*f**(2*k)

at ``Lambda=k+1/2``.  The same-label opening response is identically zero.
The remaining mixed action multiplier has the closed form

    p_k1 = k*f**(2*k-2) * {
        10*g + (2*sqrt(2)/3)*c
        * [8*k**2 - 22*k - 3 - 3*(2*k-1)*f**2]
    }.

All family identities below are exact symbolic identities.  A separate
direct differentiation of the full action checks both stress rows and all
component signs for k=2,3,4; those three gates are explicitly labelled as
representative-family checks rather than a proof by sampling.
"""
from __future__ import annotations

import sympy as sp


# Half-angle variables on 0 < theta < pi.  Positivity lets SymPy combine the
# integer-shifted powers of f without introducing branch ambiguities.
f = sp.symbols("f", positive=True)
c = sp.symbols("c", nonnegative=True)
g = sp.symbols("g", real=True)
psi = sp.symbols("psi", real=True)
psi_prime = sp.symbols("psi_prime", real=True)
k = sp.symbols("k", integer=True, positive=True)

SQRT2 = sp.sqrt(2)
Lambda = k + sp.Rational(1, 2)
nu = k - sp.Rational(1, 4)

GATES: list[tuple[str, bool]] = []


def gate(name: str, condition: bool, detail: str = "") -> None:
    """Record and print one executable assertion."""
    passed = bool(condition)
    GATES.append((name, passed))
    suffix = f"  ({detail})" if detail else ""
    print(f"[{'PASS' if passed else 'FAIL'}] {name}{suffix}")


def dtheta(expr: sp.Expr) -> sp.Expr:
    """Differentiate in the exact half-angle/base-field quotient ring."""
    g_prime = sp.Rational(5, 4) * c * g / f - SQRT2 / f
    return sp.diff(expr, f) * c / 2 - sp.diff(expr, c) * f / 2 + sp.diff(
        expr, g
    ) * g_prime + sp.diff(expr, psi) * psi_prime


def circle_numerator(expr: sp.Expr) -> sp.Expr:
    """Return the numerator reduced modulo c**2=1-f**2.

    Callers first remove the known positive common power of f.  The remaining
    expressions are rational polynomials in c, so a polynomial remainder is
    an exact identity test on the half-angle circle.
    """
    reduced = sp.powsimp(sp.cancel(sp.together(expr)), force=True)
    numerator = sp.fraction(reduced)[0]
    numerator = sp.powsimp(sp.expand(numerator), force=True)
    polynomial = sp.Poly(numerator, c)
    circle = sp.Poly(c**2 - (1 - f**2), c)
    return sp.factor(polynomial.rem(circle).as_expr())


def circle_zero(expr: sp.Expr) -> bool:
    return circle_numerator(expr) == 0


# -------------------------------------------------------------------------
# M0. Off-manifold convention and multiplier gauge.
F11, F12, F21, F22 = sp.symbols("F11 F12 F21 F22", real=True)
F1_symbol = sp.Matrix([F11, F12])
F2_symbol = sp.Matrix([F21, F22])
J_symbol = F11 * F22 - F12 * F21
C_symbol = J_symbol**4 - F2_symbol.dot(F2_symbol)
W2_exact = (
    J_symbol**2
    + (F1_symbol.dot(F1_symbol) + F2_symbol.dot(F2_symbol)) / J_symbol**2
    - 3
)
W2_tangent = 2 * J_symbol**2 + F1_symbol.dot(F1_symbol) / J_symbol**2 - 3
gate(
    "M0: exact and tangent W2 extensions differ only by -J^-2 C",
    sp.simplify(W2_exact - W2_tangent + C_symbol / J_symbol**2) == 0,
)
cof1_symbol = sp.Matrix([F22, -F21])
FinvT1_symbol = cof1_symbol / J_symbol
stiff_stress_from_W2 = 2 * (
    J_symbol**2 - F2_symbol.dot(F2_symbol) / J_symbol**2
) * FinvT1_symbol
stiff_stress_from_C = 2 * C_symbol * cof1_symbol / J_symbol**3
gate(
    "M0: exact c2 row-one stiff stress is 2 J^-3 C cof_1(F)",
    all(
        sp.simplify(component) == 0
        for component in stiff_stress_from_W2 - stiff_stress_from_C
    ),
)


# -------------------------------------------------------------------------
# M1. Base identities and the level-one determinant coefficient.
fp = c / 2
g_prime = sp.Rational(5, 4) * c * g / f - SQRT2 / f
u = sp.Matrix([sp.Rational(5, 4) * g, g_prime])
h0 = sp.Matrix([f / 2, fp])
K = sp.Matrix([fp, -f / 2])
v = -sp.Rational(2, 3) * f
v_prime = -c / 3
h = sp.Matrix([v, v_prime])
h1 = sp.Matrix([v_prime, -v])

base_J = sp.det(sp.Matrix.hstack(u, h0))
gate(
    "M1: leading determinant is 1/sqrt(2)",
    circle_zero(base_J - 1 / SQRT2),
)

# The level-one constraint supplies this coefficient without requiring an
# explicit formula for the in-plane background w(theta).
D1_from_forcing = SQRT2 * (f * v / 2 + fp * v_prime)
D1 = -SQRT2 * (1 + f**2) / 6
gate(
    "M1: forced level-one determinant coefficient D1",
    circle_zero(D1_from_forcing - D1),
)


# -------------------------------------------------------------------------
# M2. Q_k x level-one constraint and the selected smooth response.
a = f ** (2 * k)
a_prime = k * c * f ** (2 * k - 1)
dQ = sp.Matrix([k * a, a_prime])
pQ = -2 * SQRT2 * k * (2 * k - 1) * c * f ** (2 * k - 2)

J_Qv = sp.det(sp.Matrix.hstack(dQ, h))
J_Qv_claim = sp.Rational(2, 3) * k * a * fp
gate(
    "M2: nonlinear constraint source J_Qv=(2k/3) a f'",
    circle_zero((J_Qv - J_Qv_claim) / f ** (2 * k)),
)

e = -sp.Rational(4, 3) * k * a
e_prime = -sp.Rational(4, 3) * k**2 * c * f ** (2 * k - 1)
dE = sp.Matrix([Lambda * e, e_prime])
constraint_response = Lambda * fp * e - f * e_prime / 2
gate(
    "M2: e_k=-(4k/3)f^(2k) cancels J_Qv exactly",
    circle_zero((constraint_response + J_Qv) / f ** (2 * k)),
)
# At mixed order, [C]_(q*rho)=4*J0^3*(J_Qv+D_e).  Substitution into
# 2*J0^-3*C*cof_1 gives the physical Taylor coefficient 8, with no missing
# factorial.  Restored c2 times rho makes this an O(q*c2^2/c1) tier.
C_mixed = 4 * base_J**3 * (J_Qv + constraint_response)
stiff_profile_from_C = 2 * base_J**-3 * C_mixed * K
stiff_profile_claim = 8 * (J_Qv + constraint_response) * K
gate(
    "M2: the radially earlier q_k*c2^2/c1 stiff tier cancels pointwise",
    all(
        circle_zero(component / f ** (2 * k))
        for component in stiff_profile_from_C - stiff_profile_claim
    )
    and all(
        circle_zero(component / f ** (2 * k))
        for component in stiff_profile_claim
    ),
    "actual Taylor profile 8(J_Qv+D_e)K; not the multiplier p_k1",
)

e_homogeneous = f ** (2 * k + 1)
gate(
    "M2: f^(2k+1) is the homogeneous constraint branch",
    circle_zero(
        (
            Lambda * fp * e_homogeneous
            - f * dtheta(e_homogeneous) / 2
        )
        / f ** (2 * k + 1)
    ),
    "excluded by analytic-axis Mode-I parity",
)


# -------------------------------------------------------------------------
# M3. The same-label opening block is homogeneous and nonresonant.
# Mode-I symmetry gives b(0)=0.  Then b=A sin(nu theta), and b'(pi)=0
# requires A*nu*cos(nu*pi)=0.  The cosine never vanishes for integer k.
gate(
    "M3: opening mixed endpoint problem is nonresonant",
    sp.simplify(sp.cos(sp.pi * nu) - (-1) ** k / SQRT2) == 0,
    "b''+nu^2 b=0, b(0)=0, b'(pi)=0, hence b=0",
)


# -------------------------------------------------------------------------
# M4. Complete row-one stress before and after the mixed multiplier.
# U contains, in order: the smooth response, the leading-Q multiplier crossed
# with the level-one geometry, and the direct c2 tangent.
direct_c2 = 4 * dQ - 8 * SQRT2 * u.dot(dQ) * K
U = 2 * dE + pQ * (6 * D1 * K + SQRT2 * h1) + direct_c2

U_r = -sp.Rational(2, 3) * k * f ** (2 * k - 2) * (
    (6 * k - 3) * f**4
    + (8 * k + 6) * f**2
    - (10 * k + 7)
) - 5 * SQRT2 * k * g * c * f ** (2 * k - 2)
U_theta = sp.Rational(1, 3) * k * f ** (2 * k - 1) * (
    c * (2 - 36 * k + (6 - 12 * k) * f**2) + 15 * SQRT2 * g
)

gate(
    "M4: compact U_r reduces to the closed component formula",
    circle_zero((U[0] - U_r) / f ** (2 * k - 2)),
)
gate(
    "M4: compact U_theta reduces to the closed component formula",
    circle_zero((U[1] - U_theta) / f ** (2 * k - 2)),
)


# M5. Independent mechanism decomposition before the response is applied.
# This records the direct c2 and level-one-background pieces separately, so
# co-location is demonstrated by components rather than inferred by powers.
background_cross = (
    ((24 + 6 * psi) * J_Qv + 6 * pQ * D1) * K + SQRT2 * pQ * h1
)
response_cross = 2 * dE + (24 + 6 * psi) * constraint_response * K
gate(
    "M5: response cancels the complete base-reaction source channel",
    all(
        circle_zero(component / f ** (2 * k - 2))
        for component in direct_c2 + background_cross + response_cross - U
    ),
    "the nonsmooth psi0 dependence drops from the completed block",
)


def polar_divergence(stress: sp.Matrix, radial_stress_power: sp.Expr) -> sp.Expr:
    return radial_stress_power * stress[0] + dtheta(stress[1])


direct_face_divergence = sp.simplify(
    polar_divergence(direct_c2, Lambda).subs({f: 1, c: 0})
)
background_face_divergence = sp.simplify(
    polar_divergence(background_cross, Lambda).subs({f: 1, c: 0, psi: 0})
)
gate(
    "M5: direct c2 source has face divergence 2k(2k-3)",
    sp.simplify(direct_face_divergence - 2 * k * (2 * k - 3)) == 0,
)
gate(
    "M5: level-one cross source has face divergence 4k(5k-1)/3",
    sp.simplify(
        background_face_divergence - sp.Rational(4, 3) * k * (5 * k - 1)
    )
    == 0,
)
gate(
    "M5: direct and background terms are both nonzero at the same label",
    sp.simplify(
        direct_face_divergence
        + background_face_divergence
        - sp.Rational(2, 3) * k * (16 * k - 11)
    )
    == 0,
    "component audit; the direct source alone is incomplete",
)
gate(
    "M5: only the direct term supplies the pre-response face traction",
    sp.simplify(direct_c2[1].subs({f: 1, c: 0}) - 5 * SQRT2 * k * g)
    == 0
    and sp.simplify(background_cross[1].subs({f: 1, c: 0, psi: 0})) == 0,
)


# M6. Closed mixed multiplier and its transport/face selection.
p_k1 = k * f ** (2 * k - 2) * (
    10 * g
    + sp.Rational(2, 3)
    * SQRT2
    * c
    * (8 * k**2 - 22 * k - 3 - 3 * (2 * k - 1) * f**2)
)

transport_residual = (
    f * dtheta(p_k1)
    - 2 * k * fp * p_k1
    - SQRT2 * (Lambda * U_r + dtheta(U_theta))
)
gate(
    "M6: p_k1 solves the mixed multiplier transport exactly",
    circle_zero(transport_residual / f ** (2 * k - 2)),
)

p_face = sp.simplify(p_k1.subs({f: 1, c: 0}))
traction_face = sp.simplify(
    (U_theta - SQRT2 * f * p_k1 / 2).subs({f: 1, c: 0})
)
gate(
    "M6: total face traction fixes p_k1(pi)=10k g(pi)",
    sp.simplify(p_face - 10 * k * g) == 0 and traction_face == 0,
)

# Adding C_h*f^(2k) is the sole homogeneous multiplier branch.  Its face
# traction is -C_h/sqrt(2), so the displayed closed form already has C_h=0.
C_h = sp.symbols("C_h", real=True)
p_homogeneous = f ** (2 * k)
homogeneous_residual = f * dtheta(p_homogeneous) - 2 * k * fp * p_homogeneous
homogeneous_face_traction = sp.simplify(
    (-SQRT2 * f * C_h * p_homogeneous / 2).subs({f: 1, c: 0})
)
gate(
    "M6: face traction uniquely removes the f^(2k) homogeneous addition",
    circle_zero(homogeneous_residual / f ** (2 * k))
    and sp.simplify(homogeneous_face_traction + C_h / SQRT2) == 0,
)


# M7. The total stress collapses to a g-independent polynomial pair.
S = U + SQRT2 * p_k1 * K
S_r = (
    sp.Rational(8, 3)
    * k
    * (k - 1)
    * f ** (2 * k - 2)
    * ((2 * k - 1) * c**2 - f**2)
)
S_theta = (
    -sp.Rational(8, 3)
    * k
    * (k - 1)
    * (2 * k + 1)
    * c
    * f ** (2 * k - 1)
)
gate(
    "M7: total radial stress has the closed polynomial form",
    circle_zero((S[0] - S_r) / f ** (2 * k - 2)),
)
gate(
    "M7: total angular stress has the closed polynomial form",
    circle_zero((S[1] - S_theta) / f ** (2 * k - 2)),
)
gate(
    "M7: completed row-one stress is divergence-free exactly",
    circle_zero(polar_divergence(sp.Matrix([S_r, S_theta]), Lambda)
                / f ** (2 * k - 2)),
)
gate(
    "M7: completed row-one angular traction vanishes at axis and face",
    sp.simplify(S_theta.subs({f: 1, c: 0})) == 0
    and all(
        sp.simplify(S_theta.subs(k, kval).subs({f: 0, c: 1})) == 0
        for kval in (2, 3, 4)
    ),
)

p_k1_normalized = k * (
    10 * g
    + sp.Rational(2, 3)
    * SQRT2
    * c
    * (8 * k**2 - 22 * k - 3 - 3 * (2 * k - 1) * f**2)
)
axis_coefficient = sp.simplify(
    p_k1_normalized.subs({f: 0, c: 1, g: 4 * SQRT2 / 5})
)
axis_coefficient_claim = (
    -sp.Rational(2, 3) * SQRT2 * k * (2 * k - 1) * (9 - 4 * k)
)
gate(
    "M7: mixed multiplier has the regular analytic-axis coefficient",
    sp.simplify(sp.expand(axis_coefficient - axis_coefficient_claim)) == 0,
    "p_k1~[-2sqrt(2)k(2k-1)(9-4k)/3] f^(2k-2)",
)

p_face_derivative = sp.simplify(dtheta(p_k1).subs({f: 1, c: 0}))
p_face_derivative_claim = (
    SQRT2 * k * (-8 * k**2 + 28 * k - 30) / 3
)
gate(
    "M7: mixed multiplier has the exact crack-face derivative",
    sp.factor(p_face_derivative - p_face_derivative_claim) == 0,
)


# -------------------------------------------------------------------------
# M8. Direct full-action component audit for representative family members.
def direct_action_gate(k_value: int) -> bool:
    """Differentiate W1+rho*W2+chi*C and extract the q*rho stresses.

    ``z=r**(1/4)`` makes every radial power integral.  The coefficient at
    row-one stress power z**(4*k-2) and the more singular row-two power
    z**(4*k-5) is selected by a one-sided z->0 limit.  The arbitrary symbol
    ``w_r`` survives the local level-one constraint parametrization, so its
    disappearance also checks independence from the unsolved representation
    of w(theta).
    """
    kval = sp.Integer(k_value)
    Lval = kval + sp.Rational(1, 2)
    z, rho_local, q_local = sp.symbols("z rho_local q_local", positive=True)
    w_r = sp.symbols("w_r", real=True)
    psi0_local, psi1_local, p_local = sp.symbols(
        "psi0_local psi1_local p_local", real=True
    )

    # Parameterize the otherwise unnecessary level-one in-plane gradient W
    # by det(W,h0)+det(u,h)=D1, retaining arbitrary radial component w_r.
    det_u_h = sp.det(sp.Matrix.hstack(u, h))
    w_theta = sp.simplify((w_r * c + 2 * det_u_h - 2 * D1) / f)
    W = sp.Matrix([w_r, w_theta])

    aval = f ** (2 * kval)
    dval = sp.Matrix([kval * aval, kval * c * f ** (2 * kval - 1)])
    eval_ = -sp.Rational(4, 3) * kval * aval
    Eval = sp.Matrix(
        [
            Lval * eval_,
            -sp.Rational(4, 3) * kval**2 * c * f ** (2 * kval - 1),
        ]
    )
    pQval = -2 * SQRT2 * kval * (2 * kval - 1) * c * f ** (2 * kval - 2)

    F1 = (
        z * u
        + rho_local * z**3 * W
        + q_local * z ** (4 * kval - 4) * dval
        + rho_local * q_local * z ** (4 * kval - 2) * Eval
    )
    F2 = z**-2 * h0 + rho_local * h
    chi = (
        z**6 * psi0_local
        + rho_local * z**8 * psi1_local
        + q_local * z ** (4 * kval + 1) * pQval
        + rho_local * q_local * z ** (4 * kval + 3) * p_local
    )

    X11, X12, X21, X22 = sp.symbols("X11 X12 X21 X22", real=True)
    Fs1 = sp.Matrix([X11, X12])
    Fs2 = sp.Matrix([X21, X22])
    Js = X11 * X22 - X12 * X21
    W1s = Fs1.dot(Fs1) + Fs2.dot(Fs2) + Js**-2 - 3
    W2s = Js**2 + (Fs1.dot(Fs1) + Fs2.dot(Fs2)) * Js**-2 - 3
    Cs = Js**4 - Fs2.dot(Fs2)
    action = W1s + rho_local * W2s + chi * Cs
    substitutions = {X11: F1[0], X12: F1[1], X21: F2[0], X22: F2[1]}
    P1 = sp.Matrix(
        [sp.diff(action, variable).subs(substitutions) for variable in (X11, X12)]
    )
    P2 = sp.Matrix(
        [sp.diff(action, variable).subs(substitutions) for variable in (X21, X22)]
    )

    def selected_cross(component: sp.Expr, z_power: int) -> sp.Expr:
        cross = sp.diff(sp.diff(component, q_local), rho_local).subs(
            {q_local: 0, rho_local: 0}
        )
        return sp.simplify(sp.limit(cross / z**z_power, z, 0, dir="+"))

    P1_cross = sp.Matrix(
        [selected_cross(component, 4 * k_value - 2) for component in P1]
    )
    P2_cross = sp.Matrix(
        [selected_cross(component, 4 * k_value - 5) for component in P2]
    )

    Kval = K
    Uval = (
        2 * Eval
        + pQval * (6 * D1 * Kval + SQRT2 * h1)
        + 4 * dval
        - 8 * SQRT2 * u.dot(dval) * Kval
    )
    P1_claim = Uval + SQRT2 * p_local * Kval

    return all(circle_zero(component) for component in P2_cross) and all(
        circle_zero(component) for component in P1_cross - P1_claim
    )


for representative_k in (2, 3, 4):
    gate(
        f"M8: direct full-action component audit at k={representative_k}",
        direct_action_gate(representative_k),
        "row two vanishes; row one equals U+sqrt(2)pK",
    )


# -------------------------------------------------------------------------
# M9. Restored radial and P scalings.
gate(
    "M9: constraint fixes the restored P power of e_k",
    sp.simplify((1 - 2 * k) + 1 - (2 - 2 * k)) == 0,
    "delta y1=q_k rho P^(1-2k) r^(k+1/2)e_k",
)
gate(
    "M9: a possible same-label opening has P^(5/2-2k)",
    sp.simplify((sp.Rational(5, 2) - 2 * k) - sp.Rational(1, 2)
                - (2 - 2 * k)) == 0,
    "its selected profile is b_k=0",
)
gate(
    "M9: multiplier scaling gives the common row-one stress power",
    sp.simplify((-2 * k - sp.Rational(3, 2)) + sp.Rational(5, 2)
                - (1 - 2 * k)) == 0
    and sp.simplify((k + sp.Rational(3, 4)) - sp.Rational(5, 4)
                    - (k - sp.Rational(1, 2))) == 0,
    "delta chi=q_k c2 P^(-2k-3/2) r^(k+3/4)p_k1",
)


print("\nComplete first-material-order Q_k block (P=c1=1 profiles):")
print("  Lambda = k + 1/2,  nu = k - 1/4")
print("  e_k     =", e)
print("  b_k     = 0")
print("  p_k1    =", p_k1)
print("  S_r     =", S_r)
print("  S_theta =", S_theta)
print("\nRestored amplitudes:")
print("  delta y1 = q_k (c2/c1) P^(1-2k) r^(k+1/2) e_k")
print("  delta y2 = 0 at this label")
print("  delta chi = q_k c2 P^(-2k-3/2) r^(k+3/4) p_k1")
print("\nScope: exact through O(q_k*c2/c1); fixed-rho higher material orders remain open.")

failed = [name for name, passed in GATES if not passed]
print("\n" + "=" * 72)
print(f"PASSED {len(GATES) - len(failed)}   FAILED {len(failed)}")
if failed:
    print("FAILURES:", failed)
    raise SystemExit(1)
print("All first-material-order full-MR completion gates passed.")
