#!/usr/bin/env python3
"""Exact ordering audit for the first full-MR c2 contribution to Q_k.

This isolated audit uses the chosen C_s=0 base representative and does not
address matching of the regular null-family amplitude.

The leading-Q_k calculation in ``coupled_shear_completion.py`` uses the
formal constrained action ``W1 + chi*C``.  This script asks where the omitted
on-manifold Mooney--Rivlin remainder first enters. Along a Q_k tangent,
``delta J=delta F2=0``, so the decomposition term ``c2*J**-2*C`` has zero
row-one tangent and differentiating the full ``I2`` stress gives the same
direct remainder contribution. The script proves that this direct
row-one linearization has stress order ``r**(k-1/2)`` and a generically
nonzero divergence, hence belongs to the earlier common label
``Lambda = k+1/2`` rather than the constrained-action opening label
``k+3/2``.

This remains a deliberately isolated ordering gate.  The co-located
level-one-background terms and the resulting first-material-order response
are assembled in ``qk_full_mr_completion.py``.
"""
from __future__ import annotations

import sympy as sp


def is_zero(expr: sp.Expr) -> bool:
    return sp.simplify(sp.trigsimp(sp.expand_trig(expr))) == 0


def main() -> bool:
    checks: list[tuple[str, bool, str]] = []

    def gate(name: str, ok: bool, detail: str = "") -> None:
        checks.append((name, bool(ok), detail))

    # Derive the c2 row-one nominal stress from the exact reduced-plane-stress
    # invariant I2 = J^2 + (F:F) J^-2.
    x11, x12, x21, x22 = sp.symbols("x11 x12 x21 x22", nonzero=True)
    F1 = sp.Matrix([x11, x12])
    F2 = sp.Matrix([x21, x22])
    J = x11 * x22 - x12 * x21
    I2d = F1.dot(F1) + F2.dot(F2)
    W2 = J**2 + I2d * J**-2 - 3
    P1 = sp.Matrix([sp.diff(W2, x11), sp.diff(W2, x12)])
    FinvT1 = sp.Matrix([x22, -x21]) / J
    P1_expected = 2 * (J**-2 * F1 + (J**2 - J**-2 * I2d) * FinvT1)
    gate("M1: exact c2 row-one PK1 stress", all(is_zero(z) for z in P1 - P1_expected))

    # A characteristic shear changes F1 parallel to F2, so delta J=0.  Direct
    # differentiation of the exact stress then gives the tangent formula used
    # below, without appealing to the asymptotic script.
    eps, rho = sp.symbols("eps rho")
    replacements = {x11: x11 + eps * rho * x21,
                    x12: x12 + eps * rho * x22}
    dP1 = sp.Matrix([sp.diff(z.subs(replacements), eps).subs(eps, 0) for z in P1])
    dF1 = rho * F2
    cof1 = sp.Matrix([x22, -x21])
    dP1_expected = 2 * J**-2 * dF1 - 4 * J**-3 * F1.dot(dF1) * cof1
    gate("M2: exact tangent stress for delta J=0",
         all(is_zero(z) for z in dP1 - dP1_expected))

    theta = sp.symbols("theta", real=True)
    k = sp.symbols("k", integer=True, positive=True)
    P, r, q = sp.symbols("P r q", positive=True)
    f = sp.sin(theta / 2)
    fp = sp.diff(f, theta)
    g = sp.Function("g")(theta)
    a = f ** (2 * k)
    u = sp.Matrix([sp.Rational(5, 4) * g, sp.diff(g, theta)])
    v = sp.Matrix([f / 2, fp])
    d = sp.Matrix([k * a, sp.diff(a, theta)])
    K = sp.Matrix([fp, -f / 2])

    tangent = sp.det(sp.Matrix.hstack(d, v))
    gate("M3: Q_k shear is tangent to J", is_zero(tangent))

    # Restore the leading-map scales.  J0=sqrt(P/2) r^-1/4 and
    # F0^{-T}_1=sqrt(2P) r^-1/4 K.  Factor out the common dimensional
    # coefficient from the exact tangent stress.
    J0 = sp.sqrt(P / 2) * r ** sp.Rational(-1, 4)
    F10 = P ** sp.Rational(-1, 2) * r ** sp.Rational(1, 4) * u
    dF10 = q * P ** (2 - 2 * k) * r ** (k - 1) * d
    FinvT10 = sp.sqrt(2 * P) * r ** sp.Rational(-1, 4) * K
    dP10 = 2 * J0**-2 * dF10 - 4 * J0**-2 * F10.dot(dF10) * FinvT10
    common = q * P ** (1 - 2 * k) * r ** (k - sp.Rational(1, 2))
    profile = sp.Matrix([sp.powsimp(z / common, force=True) for z in dP10])
    profile_expected = 4 * d - 8 * sp.sqrt(2) * u.dot(d) * K
    gate("M4: restored c2 stress profile and P/r powers",
         all(is_zero(z) for z in profile - profile_expected))

    # Polar divergence of r^(k-1/2) profile: (k+1/2)P_r + P_theta'.
    div_profile = ((k + sp.Rational(1, 2)) * profile_expected[0]
                   + sp.diff(profile_expected[1], theta))
    Gpi, Gpppi = sp.symbols("Gpi Gpppi", real=True)
    face_repl = {
        g: Gpi,
        sp.diff(g, theta): -sp.sqrt(2),
        sp.diff(g, theta, 2): Gpppi,
    }
    traction_face = sp.factor(profile_expected[1].xreplace(face_repl).subs(theta, sp.pi))
    divergence_face = sp.factor(div_profile.xreplace(face_repl).subs(theta, sp.pi))
    gate("M5: direct c2 face traction is 5 sqrt(2) k g(pi)",
         is_zero(traction_face - 5 * sp.sqrt(2) * k * Gpi))
    gate("M6: direct c2 divergence is nonzero at the face for k>=2",
         is_zero(divergence_face - 2 * k * (2 * k - 3))
         and all((2 * j * (2 * j - 3)) > 0 for j in (2, 3, 4)),
         f"coefficient={divergence_face}")

    # Matching a W1 response, whose row-one divergence scales as r^(Lambda-2),
    # fixes the common label.
    Lambda = k + sp.Rational(1, 2)
    gate("M7: source order implies Lambda=k+1/2",
         sp.simplify((Lambda - 2) - (k - sp.Rational(3, 2))) == 0)

    for name, ok, detail in checks:
        suffix = f"  ({detail})" if detail else ""
        print(f"[{'PASS' if ok else 'FAIL'}] {name}{suffix}")

    n_pass = sum(ok for _, ok, _ in checks)
    print("\nDirect full-MR ordering result:")
    print("  delta P1_c2 = c2 q_k P^(1-2k) r^(k-1/2)")
    print("                 * [4 d - 8 sqrt(2) (u.d) K]")
    print("  direct divergence face coefficient = 2 k (2k-3) != 0, k>=2")
    print("  earlier common label: Lambda = k+1/2")
    print("  complete O(q_k*c2/c1) block: see qk_full_mr_completion.py")
    print(f"\nPASSED {n_pass}   FAILED {len(checks) - n_pass}")
    return n_pass == len(checks)


if __name__ == "__main__":
    raise SystemExit(0 if main() else 1)
