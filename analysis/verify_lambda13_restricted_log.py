#!/usr/bin/env python3
"""Verify the manuscript-scoped restricted Lambda=13/4 logarithmic channel.

This program verifies two statements only:

1. At the opening-block pencil label ``Lambda=13/4``, the opening exponent is
   ``nu=Lambda-3/4=5/2`` and the scalar angular operator is exactly resonant.
2. On the selected ``C_s=0`` analytic-axis representative, the normalized
   restricted source coupling the first material correction to the stationary
   background has a converged nonzero derivative-free Fredholm projection.
   That formal scalar opening coefficient therefore requires a contribution
   proportional to

       r^(5/2) log(r/r0) sin(5 theta/2).

The reported ``kappa=-F_df/(2*pi)`` is the conditional normalization of this
restricted formal coefficient.  It is not a specimen-selected amplitude.

This program does not assemble the complete same-grade source, construct the
sourced coupled Jordan response, or select the net specimen-level logarithmic
amplitude.  The numerical result is a converged arbitrary-precision
calculation, not an interval proof.

No stored coefficient table or result generates the calculation. Exact axis
series, lower profiles, source components, and both integration routes are
regenerated in this file. A fixed final defect is used only as a regression
check after the numerical value has been computed.
"""
from __future__ import annotations

import sys
import time

import mpmath as mp
import sympy as sp


START = time.perf_counter()
R = sp.Rational
GATES: list[tuple[str, bool]] = []


def gate(name: str, condition: bool, detail: str = "") -> None:
    """Record and print a deterministic verification gate."""
    passed = bool(condition)
    GATES.append((name, passed))
    suffix = f"  ({detail})" if detail else ""
    print(f"[{'PASS' if passed else 'FAIL'}] {name}{suffix}", flush=True)


# ---------------------------------------------------------------------------
# A. Exact resonance, Fredholm convention, and radial-log normalization.
# ---------------------------------------------------------------------------
lambda_label = R(13, 4)
nu_exact = lambda_label - R(3, 4)
theta_s, r_s, r0_s = sp.symbols(
    "theta_s r_s r0_s", real=True, positive=True
)
opening_mode = sp.sin(nu_exact * theta_s)

gate(
    "A1: pencil label Lambda=13/4 has opening exponent nu=5/2",
    nu_exact == R(5, 2),
    "Lambda is the weighted pencil label, not the opening power",
)
gate(
    "A2: sin(5 theta/2) is the exact mixed-endpoint kernel",
    sp.simplify(sp.diff(opening_mode, theta_s, 2)
                + nu_exact**2 * opening_mode) == 0
    and opening_mode.subs(theta_s, 0) == 0
    and sp.simplify(sp.diff(opening_mode, theta_s)
                    .subs(theta_s, sp.pi)) == 0,
    "b(0)=0 and b'(pi)=0",
)

H_r = sp.Function("H_r")(theta_s)
H_theta = sp.Function("H_theta")(theta_s)
s_exact = sp.sin(nu_exact * theta_s)
c_exact = sp.cos(nu_exact * theta_s)
effective_source = nu_exact * H_r + sp.diff(H_theta, theta_s)
fredholm_identity = sp.simplify(
    s_exact * effective_source
    - nu_exact * (s_exact * H_r - c_exact * H_theta)
    - sp.diff(s_exact * H_theta, theta_s)
)
gate(
    "A3: derivative-free Fredholm identity is exact",
    fredholm_identity == 0
    and sp.sin(nu_exact * sp.pi) == 1
    and sp.cos(nu_exact * sp.pi) == 0,
    "the raw projected defect is nu*F_df after the face term is separated",
)


def opening_euler(field: sp.Expr) -> sp.Expr:
    """Scalar radial-Euler operator, including the manuscript factor two."""
    return 2 * (
        r_s * sp.diff(r_s * sp.diff(field, r_s), r_s)
        + sp.diff(field, theta_s, 2)
    )


radial_kernel = r_s**nu_exact * opening_mode
radial_log = (
    r_s**nu_exact * sp.log(r_s / r0_s) * opening_mode
)
kernel_residual = sp.simplify(opening_euler(radial_kernel))
log_residual = sp.simplify(opening_euler(radial_log) / radial_kernel)
kernel_norm = sp.integrate(
    opening_mode**2, (theta_s, 0, sp.pi)
)
F_symbol, kappa_symbol = sp.symbols("F_df kappa", real=True)
raw_projection = nu_exact * F_symbol
log_projection = log_residual * kernel_norm * kappa_symbol
kappa_formula = sp.solve(
    sp.Eq(raw_projection + log_projection, 0), kappa_symbol
)[0]
gate(
    "A4: derivative-free convention gives kappa=-F_df/(2*pi)",
    kernel_residual == 0
    and log_residual == 4 * nu_exact
    and log_residual == 10
    and kernel_norm == sp.pi / 2
    and sp.simplify(kappa_formula + F_symbol / (2 * sp.pi)) == 0,
    "the factor nu belongs to both the raw defect and the log projection",
)


# ---------------------------------------------------------------------------
# B. Regenerate analytic-axis series for the two closed lower profiles.
# ---------------------------------------------------------------------------
mp.mp.dps = 80
SQRT2 = mp.sqrt(2)
PI = mp.pi
# At the switch q=sqrt(sin(0.25/2))<0.354, terms beyond q^69 are below
# 3e-32.  This is far beneath the 1e-8 continuation tolerances while avoiding
# expensive high-order symbolic factorization on newer Python/SymPy builds.
COEFF_ORDER = 70


def build_stationary_coefficients() -> tuple[
    dict[int, sp.Expr], dict[int, sp.Expr], sp.Expr
]:
    """Generate exact q-series for the stationary opening and shear.

    Here q=sqrt(sin(theta/2)).  The homogeneous q^11 stationary-shear
    direction is set to zero; the restricted opening source below is
    independent of that normal-form direction.
    """
    sqrt2 = sp.sqrt(2)
    f, c, g = sp.symbols("f c g", positive=True)
    gp = R(5, 4) * c * g / f - sqrt2 / f
    leading = sp.Matrix([-gp, R(5, 4) * g])
    opening = sp.Matrix([f / 2, c / 2])
    psi = (
        4 + 6 / f**2
        - R(15, 4) * sqrt2 * c * g / f**2
        - 10 * f ** R(3, 2)
    )
    base_source = (
        -4 * sqrt2 * leading
        + psi * (sqrt2 * leading - 2 * opening)
    )

    def dtheta_base(expr: sp.Expr) -> sp.Expr:
        return (
            sp.diff(expr, f) * c / 2
            - sp.diff(expr, c) * f / 2
            + sp.diff(expr, g) * gp
        )

    stationary_forcing = (
        2 * base_source[0] + dtheta_base(base_source[1])
    )
    q = sp.symbols("q", positive=True)
    cq = sp.series(
        sp.sqrt(1 - q**4), q, 0, COEFF_ORDER
    ).removeO()
    gq = sp.series(
        sum(
            -4 * sqrt2 * sp.binomial(2 * n, n)
            / sp.Integer(4) ** n
            / (4 * n - 5)
            * q ** (4 * n)
            for n in range(COEFF_ORDER // 4 + 3)
        ),
        q,
        0,
        COEFF_ORDER,
    ).removeO()
    forcing_q = sp.series(
        stationary_forcing.subs({f: q**2, c: cq, g: gq}),
        q,
        0,
        COEFF_ORDER,
    ).removeO().expand()
    forcing_coeff = [
        forcing_q.coeff(q, n) for n in range(COEFF_ORDER)
    ]

    beta = R(-24, 65)
    b_coeff: dict[int, sp.Expr] = {
        0: sp.S(0),
        1: sp.S(0),
        2: 2 * beta,
        3: sp.S(0),
        4: sp.S(0),
    }
    for n in range(COEFF_ORDER - 4):
        if n + 4 in b_coeff:
            continue
        b_coeff[n + 4] = sp.expand(
            -8 * forcing_coeff[n]
            - (64 - n**2) * b_coeff.get(n, 0)
        ) / ((n + 4) * (n + 2))

    a_value, a_q, b_value, b_q = sp.symbols(
        "a_value a_q b_value b_q"
    )
    tangent = sp.Matrix([c / 2, -f / 2])
    stationary_d = sp.Matrix([R(11, 4) * a_value, c * a_q / (4 * q)])
    stationary_e = sp.Matrix([2 * b_value, c * b_q / (4 * q)])
    determinant_increment = (
        tangent.dot(stationary_d) + leading.dot(stationary_e)
    )
    constraint = (
        sqrt2 * determinant_increment
        - 2 * opening.dot(stationary_e)
    )
    a_q_solution = sp.solve(
        sp.Eq(
            constraint.subs({f: q**2, c: cq, g: gq}),
            0,
        ),
        a_q,
    )[0]
    b_series = sum(
        b_coeff[n] * q**n for n in sorted(b_coeff)
    )
    radial_a_source = sp.series(
        q * a_q_solution.subs({
            a_value: 0,
            b_value: b_series,
            b_q: sp.diff(b_series, q),
        }),
        q,
        0,
        COEFF_ORDER,
    ).removeO().expand()
    a_coeff: dict[int, sp.Expr] = {}
    for n in range(COEFF_ORDER):
        rhs = radial_a_source.coeff(q, n)
        if n == 11:
            continue
        a_coeff[n] = rhs / (n - 11)
    log_coeff = radial_a_source.coeff(q, 11)
    if sp.simplify(log_coeff + R(1905, 154) * sqrt2) != 0:
        raise ArithmeticError("stationary q^11 log coefficient mismatch")
    return b_coeff, a_coeff, log_coeff


print("building exact analytic-axis coefficients...", flush=True)
B_SYM, A_SYM, A_LOG_SYM = build_stationary_coefficients()


def to_mp(value: sp.Expr) -> mp.mpf:
    return mp.mpf(str(sp.N(value, mp.mp.dps)))


B_COEFF = {n: to_mp(value) for n, value in B_SYM.items()}
A_COEFF = {n: to_mp(value) for n, value in A_SYM.items()}
A_LOG = to_mp(A_LOG_SYM)
del B_SYM, A_SYM, A_LOG_SYM

SERIES_TERMS = 80


def central_coefficient(n: int) -> mp.mpf:
    return mp.binomial(2 * n, n) / mp.mpf(4) ** n


G_COEFF = [
    -4 * SQRT2 * central_coefficient(n) / (4 * n - 5)
    for n in range(SERIES_TERMS)
]
W_COEFF = [
    -mp.mpf(4) * SQRT2 / 3
    * central_coefficient(n)
    * (
        mp.mpf(10) / (4 * n - 5)
        + mp.mpf(4 * n - 3) / (2 * n - 1)
    )
    / (4 * n - 7)
    for n in range(SERIES_TERMS)
]


def evaluate_q4_series(coefficients: list[mp.mpf], q_value: mp.mpf) -> mp.mpf:
    total = mp.mpf("0")
    power = mp.mpf("1")
    q4 = q_value**4
    for coefficient in coefficients:
        total += coefficient * power
        power *= q4
    return total


def b_axis(q_value: mp.mpf) -> mp.mpf:
    return sum(
        B_COEFF[n] * q_value**n for n in sorted(B_COEFF)
    )


def db_dq_axis(q_value: mp.mpf) -> mp.mpf:
    return sum(
        n * B_COEFF[n] * q_value ** (n - 1)
        for n in sorted(B_COEFF)
        if n >= 1
    )


def a_axis(q_value: mp.mpf) -> mp.mpf:
    regular = sum(
        A_COEFF[n] * q_value**n for n in sorted(A_COEFF)
    )
    return regular + A_LOG * q_value**11 * mp.log(q_value)


# ---------------------------------------------------------------------------
# C. Lower-profile continuation and the restricted source.
# ---------------------------------------------------------------------------
def lower_rhs(
    theta: mp.mpf, state: tuple[mp.mpf, ...]
) -> tuple[mp.mpf, ...]:
    """Angular ODEs for g,w and the stationary b,a checkpoint."""
    g_value, w_value, b_value, b_prime, a_value = state
    f_value = mp.sin(theta / 2)
    c_value = mp.cos(theta / 2)
    g_prime = (
        5 * c_value * g_value / (4 * f_value)
        - SQRT2 / f_value
    )
    w_prime = (
        7 * c_value * w_value / (4 * f_value)
        + 5 * c_value * g_value / (6 * f_value)
        - SQRT2 * (3 - f_value**2) / (3 * f_value)
    )
    stationary_forcing = (
        mp.mpf(3)
        / (16 * f_value**3)
        * (
            25 * g_value**2
            + 15 * SQRT2 * c_value * g_value * (f_value**2 - 3)
            + 40
            - 40 * f_value**2
            - 32 * f_value**4
            - 40 * f_value ** mp.mpf("3.5")
            + 40 * f_value ** mp.mpf("5.5")
        )
    )
    b_second = -4 * b_value - stationary_forcing / 2
    a_prime = (
        2
        / (SQRT2 * f_value)
        * (
            SQRT2
            * (
                11 * c_value * a_value / 8
                - 2 * g_prime * b_value
                + 5 * g_value * b_prime / 4
            )
            - 2 * f_value * b_value
            - c_value * b_prime
        )
    )
    return g_prime, w_prime, b_prime, b_second, a_prime


def restricted_source(
    theta: mp.mpf, state: tuple[mp.mpf, ...]
) -> tuple[tuple[mp.mpf, mp.mpf], tuple[mp.mpf, mp.mpf]]:
    """Return inherited and material-by-stationary source components.

    Their sum is the audited restricted source.  It is not the complete
    source at the Lambda=13/4 pencil label.
    """
    g_value, w_value, b_value, b_prime, a_value = state
    f_value = mp.sin(theta / 2)
    c_value = mp.cos(theta / 2)
    g_prime, w_prime, _, _, a_prime = lower_rhs(theta, state)

    psi = (
        4
        + 6 / f_value**2
        - mp.mpf(15)
        / 4
        * SQRT2
        * c_value
        * g_value
        / f_value**2
        - 10 * f_value ** mp.mpf("1.5")
    )
    material_opening = (-mp.mpf(2) * f_value / 3, -c_value / 3)
    material_reaction = (
        mp.mpf(128) / (3 * f_value**2)
        + mp.mpf(104) / 15
        + 8 * f_value**2
        - 10 * f_value ** mp.mpf("3.5")
        - mp.mpf(355)
        / 12
        * SQRT2
        * c_value
        * g_value
        / f_value**2
        - mp.mpf(15) / 4 * SQRT2 * c_value * g_value
        - mp.mpf(35)
        / 4
        * SQRT2
        * c_value
        * w_value
        / f_value**2
        + mp.mpf(25) / 4 * g_value**2 / f_value**2
        - mp.mpf(238) / 5 * f_value ** mp.mpf("2.5")
    )
    tangent = (c_value / 2, -f_value / 2)
    leading = (-g_prime, 5 * g_value / 4)
    opening = (f_value / 2, c_value / 2)
    material_d = (7 * w_value / 4, w_prime)
    determinant_material = (
        tangent[0] * material_d[0]
        + tangent[1] * material_d[1]
        + leading[0] * material_opening[0]
        + leading[1] * material_opening[1]
    )
    inherited_v = (
        (24 + 6 * psi) * determinant_material * leading[0]
        + SQRT2 * (psi - 4) * (-w_prime)
        - 2 * psi * material_opening[0]
        + material_reaction
        * (SQRT2 * leading[0] - 2 * opening[0]),
        (24 + 6 * psi) * determinant_material * leading[1]
        + SQRT2 * (psi - 4) * (7 * w_value / 4)
        - 2 * psi * material_opening[1]
        + material_reaction
        * (SQRT2 * leading[1] - 2 * opening[1]),
    )
    leading_norm_squared = (5 * g_value / 4) ** 2 + g_prime**2
    inherited_n = (
        -4 * SQRT2 * leading_norm_squared * leading[0],
        -4 * SQRT2 * leading_norm_squared * leading[1],
    )
    inherited = (
        inherited_v[0] + inherited_n[0],
        inherited_v[1] + inherited_n[1],
    )

    stationary_d = (11 * a_value / 4, a_prime)
    stationary_e = (2 * b_value, b_prime)
    determinant_stationary = (
        tangent[0] * stationary_d[0]
        + tangent[1] * stationary_d[1]
        + leading[0] * stationary_e[0]
        + leading[1] * stationary_e[1]
    )
    rotated_stationary_d = (-a_prime, 11 * a_value / 4)
    opening_dot_stationary = (
        opening[0] * stationary_e[0]
        + opening[1] * stationary_e[1]
    )
    delta = SQRT2 / 2
    cross = tuple(
        2
        * (
            determinant_stationary * leading[index]
            + delta * rotated_stationary_d[index]
        )
        + 2
        * (
            stationary_e[index] * delta**-2
            - 2
            * opening[index]
            * delta**-3
            * determinant_stationary
        )
        - 2
        * (
            2
            * opening_dot_stationary
            * delta**-3
            * leading[index]
            + mp.mpf(1)
            / 4
            * (
                -3
                * delta**-4
                * determinant_stationary
                * leading[index]
                + delta**-3 * rotated_stationary_d[index]
            )
        )
        for index in range(2)
    )
    return inherited, cross


def defect_density(
    theta: mp.mpf, state: tuple[mp.mpf, ...]
) -> tuple[mp.mpf, mp.mpf]:
    inherited, cross = restricted_source(theta, state)
    sine = mp.sin(mp.mpf(5) * theta / 2)
    cosine = mp.cos(mp.mpf(5) * theta / 2)
    return (
        sine * inherited[0] - cosine * inherited[1],
        sine * cross[0] - cosine * cross[1],
    )


# ---------------------------------------------------------------------------
# D. Two deterministic continuation and quadrature routes.
# ---------------------------------------------------------------------------
def add_state(
    state: tuple[mp.mpf, ...],
    increments: tuple[tuple[mp.mpf, tuple[mp.mpf, ...]], ...],
) -> tuple[mp.mpf, ...]:
    return tuple(
        value
        + sum(coefficient * derivative[index]
              for coefficient, derivative in increments)
        for index, value in enumerate(state)
    )


def rk4_step(
    rhs_function, theta: mp.mpf, state: tuple[mp.mpf, ...], step: mp.mpf
) -> tuple[mp.mpf, ...]:
    k1 = rhs_function(theta, state)
    k2 = rhs_function(
        theta + step / 2, add_state(state, ((step / 2, k1),))
    )
    k3 = rhs_function(
        theta + step / 2, add_state(state, ((step / 2, k2),))
    )
    k4 = rhs_function(
        theta + step, add_state(state, ((step, k3),))
    )
    return add_state(
        state,
        (
            (step / 6, k1),
            (step / 3, k2),
            (step / 3, k3),
            (step / 6, k4),
        ),
    )


def cash_karp_step(
    rhs_function, theta: mp.mpf, state: tuple[mp.mpf, ...], step: mp.mpf
) -> tuple[mp.mpf, ...]:
    k1 = rhs_function(theta, state)
    k2 = rhs_function(
        theta + step / 5, add_state(state, ((step / 5, k1),))
    )
    k3 = rhs_function(
        theta + 3 * step / 10,
        add_state(
            state, ((3 * step / 40, k1), (9 * step / 40, k2))
        ),
    )
    k4 = rhs_function(
        theta + 3 * step / 5,
        add_state(
            state,
            (
                (3 * step / 10, k1),
                (-9 * step / 10, k2),
                (6 * step / 5, k3),
            ),
        ),
    )
    k5 = rhs_function(
        theta + step,
        add_state(
            state,
            (
                (-11 * step / 54, k1),
                (5 * step / 2, k2),
                (-70 * step / 27, k3),
                (35 * step / 27, k4),
            ),
        ),
    )
    k6 = rhs_function(
        theta + 7 * step / 8,
        add_state(
            state,
            (
                (1631 * step / 55296, k1),
                (175 * step / 512, k2),
                (575 * step / 13824, k3),
                (44275 * step / 110592, k4),
                (253 * step / 4096, k5),
            ),
        ),
    )
    return add_state(
        state,
        (
            (37 * step / 378, k1),
            (250 * step / 621, k3),
            (125 * step / 594, k4),
            (512 * step / 1771, k6),
        ),
    )


STEPPERS = {
    "RK4": (rk4_step, 4),
    "CK5": (cash_karp_step, 5),
}


def initial_state(theta0: mp.mpf) -> tuple[mp.mpf, ...]:
    q0 = mp.sqrt(mp.sin(theta0 / 2))
    c0 = mp.cos(theta0 / 2)
    return (
        evaluate_q4_series(G_COEFF, q0),
        evaluate_q4_series(W_COEFF, q0),
        b_axis(q0),
        db_dq_axis(q0) * c0 / (4 * q0),
        a_axis(q0),
        mp.mpf("0"),
        mp.mpf("0"),
    )


def continue_profiles(
    theta0: mp.mpf, steps: int, method: str
) -> tuple[mp.mpf, ...]:
    state = initial_state(theta0)
    stepper = STEPPERS[method][0]
    step = (PI - theta0) / steps

    def augmented_rhs(
        theta: mp.mpf, augmented: tuple[mp.mpf, ...]
    ) -> tuple[mp.mpf, ...]:
        profile_rhs = lower_rhs(theta, augmented[:5])
        inherited_density, cross_density = defect_density(
            theta, augmented[:5]
        )
        return profile_rhs + (inherited_density, cross_density)

    theta = theta0
    for _ in range(steps):
        state = stepper(augmented_rhs, theta, state, step)
        theta += step
    return state


def axis_integral(
    theta0: mp.mpf, intervals: int = 1500
) -> tuple[mp.mpf, mp.mpf]:
    """Midpoint integration on the endpoint segment using exact q-series."""
    inherited_total = mp.mpf("0")
    cross_total = mp.mpf("0")
    for index in range(intervals):
        theta = theta0 * (index + mp.mpf("0.5")) / intervals
        q_value = mp.sqrt(mp.sin(theta / 2))
        c_value = mp.cos(theta / 2)
        state = (
            evaluate_q4_series(G_COEFF, q_value),
            evaluate_q4_series(W_COEFF, q_value),
            b_axis(q_value),
            db_dq_axis(q_value) * c_value / (4 * q_value),
            a_axis(q_value),
        )
        inherited_density, cross_density = defect_density(theta, state)
        inherited_total += inherited_density
        cross_total += cross_density
    step = theta0 / intervals
    return inherited_total * step, cross_total * step


def richardson(
    coarse: tuple[mp.mpf, ...],
    fine: tuple[mp.mpf, ...],
    order: int,
) -> tuple[mp.mpf, ...]:
    factor = mp.mpf(2) ** order - 1
    return tuple(
        fine[index] + (fine[index] - coarse[index]) / factor
        for index in range(len(fine))
    )


RUNS: dict[tuple[str, int], tuple[mp.mpf, ...]] = {}
for method, meshes in (("RK4", (4800, 9600)), ("CK5", (1200, 2400))):
    for mesh in meshes:
        RUNS[(method, mesh)] = continue_profiles(
            mp.mpf("0.25"), mesh, method
        )
        print(f"{method} N={mesh} done", flush=True)

G_FACE = (
    12
    * mp.sqrt(2 * PI)
    / 5
    * mp.gamma(mp.mpf(3) / 4)
    / mp.gamma(mp.mpf(1) / 4)
)
SUM_45 = (
    mp.gamma(-mp.mpf(5) / 4)
    * mp.sqrt(PI)
    / (4 * mp.gamma(-mp.mpf(3) / 4))
)
SUM_47 = (
    mp.gamma(-mp.mpf(7) / 4)
    * mp.sqrt(PI)
    / (4 * mp.gamma(-mp.mpf(5) / 4))
)
W_FACE = (
    -4
    * SQRT2
    / 3
    * (-5 * SUM_45 + mp.mpf(33) / 5 * SUM_47)
)
B_FACE = mp.mpf(122) / 65 - 25 * G_FACE**2 / 26
B_PRIME_FACE = 5 * SQRT2 * G_FACE / 2

face_errors: list[mp.mpf] = []
for final_state in RUNS.values():
    face_errors.extend([
        abs(final_state[0] - G_FACE),
        abs(final_state[1] - W_FACE),
        abs(final_state[2] - B_FACE),
        abs(final_state[3] - B_PRIME_FACE),
    ])
gate(
    "N1: both continuations recover the exact lower-profile face anchors",
    max(face_errors) < mp.mpf("2e-8"),
    f"maximum error={mp.nstr(max(face_errors), 7)}",
)

axis_inherited, axis_cross = axis_integral(mp.mpf("0.25"))
ESTIMATES: dict[str, tuple[mp.mpf, ...]] = {}
for method, meshes in (("RK4", (4800, 9600)), ("CK5", (1200, 2400))):
    ESTIMATES[method] = richardson(
        RUNS[(method, meshes[0])],
        RUNS[(method, meshes[1])],
        STEPPERS[method][1],
    )

F_RK = (
    ESTIMATES["RK4"][5] + axis_inherited,
    ESTIMATES["RK4"][6] + axis_cross,
)
F_CK = (
    ESTIMATES["CK5"][5] + axis_inherited,
    ESTIMATES["CK5"][6] + axis_cross,
)
method_spread = max(
    abs(F_RK[0] - F_CK[0]),
    abs(F_RK[1] - F_CK[1]),
)
gate(
    "N2: RK4 and Cash-Karp defects agree after Richardson extrapolation",
    method_spread < mp.mpf("5e-9"),
    f"spread={mp.nstr(method_spread, 7)}",
)

alt_axis_inherited, alt_axis_cross = axis_integral(mp.mpf("0.20"))
alt_state = continue_profiles(mp.mpf("0.20"), 2400, "CK5")
F_ALT = (
    alt_state[5] + alt_axis_inherited,
    alt_state[6] + alt_axis_cross,
)
switch_spread = max(
    abs(F_ALT[0] - F_CK[0]),
    abs(F_ALT[1] - F_CK[1]),
)
gate(
    "N3: moving the axis-series switch leaves the defect stable",
    switch_spread < mp.mpf("5e-8"),
    f"spread={mp.nstr(switch_spread, 7)}",
)

A_FACE_NUMERICAL = ESTIMATES["CK5"][4]
face_state = (
    G_FACE,
    W_FACE,
    B_FACE,
    B_PRIME_FACE,
    A_FACE_NUMERICAL,
)
inherited_face, cross_face = restricted_source(PI, face_state)
cross_theta_face = (
    48
    * mp.sqrt(PI)
    * mp.gamma(mp.mpf(3) / 4)
    / mp.gamma(mp.mpf(1) / 4)
)
gate(
    "N4: restricted source reproduces its exact face anchors",
    abs(
        inherited_face[0] + (75 * G_FACE**2 + 448) / 12
    ) < mp.mpf("1e-30")
    and abs(
        inherited_face[1] + SQRT2 * (20 * G_FACE + 7 * W_FACE)
    ) < mp.mpf("1e-30")
    and abs(cross_face[0]) < mp.mpf("1e-30")
    and abs(cross_face[1] - cross_theta_face) < mp.mpf("1e-20"),
)

F_INHERITED = F_CK[0]
F_CROSS = F_CK[1]
F_DF = F_INHERITED + F_CROSS
F_REFERENCE = mp.mpf("-9.23334287572121841917201725973")
stability_envelope = max(method_spread, switch_spread)
gate(
    "N5: restricted derivative-free projection is converged and nonzero",
    abs(F_DF - F_REFERENCE) < mp.mpf("5e-8")
    and abs(F_DF) > 1
    and abs(F_DF) > mp.mpf("1e6") * stability_envelope,
    (
        f"F_df={mp.nstr(F_DF, 25)}; "
        f"stability envelope={mp.nstr(stability_envelope, 7)}"
    ),
)

KAPPA = -F_DF / (2 * PI)
KAPPA_REFERENCE = -F_REFERENCE / (2 * PI)
gate(
    "N6: corrected restricted log normalization is reproduced",
    abs(KAPPA - KAPPA_REFERENCE) < mp.mpf("1e-8"),
    f"kappa=-F_df/(2*pi)={mp.nstr(KAPPA, 25)}",
)


print("\nRestricted Fredholm data (diagnostic decomposition):")
print("  F(V+N)       =", mp.nstr(F_INHERITED, 30))
print("  F(cross)     =", mp.nstr(F_CROSS, 30))
print("  F_df         =", mp.nstr(F_DF, 30))
print("  nu*F_df      =", mp.nstr(mp.mpf(5) / 2 * F_DF, 30))
print("  kappa_unit   =", mp.nstr(KAPPA, 30))
print("  convention   = kappa_unit = -F_df/(2*pi)")
print("\nScope:")
print("  VERIFIED: exact scalar resonance plus a converged nonzero projection")
print("            for the normalized restricted rho x stationary source.")
print("  CONSEQUENCE: that formal scalar opening coefficient requires an")
print("               r^(5/2) log(r/r0) sin(5 theta/2) contribution.")
print("  OPEN: complete same-grade source, sourced coupled Jordan response,")
print("        and outer matching of the net specimen-level log amplitude.")
print("  EVIDENCE: converged arbitrary-precision numerics, not an interval proof.")

passed = sum(ok for _, ok in GATES)
total = len(GATES)
print(f"\nRestricted Lambda=13/4 log verifier: {passed}/{total} passed")
print(f"Runtime: {time.perf_counter() - START:.3f} s")
sys.exit(0 if passed == total else 1)
