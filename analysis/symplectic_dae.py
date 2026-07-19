#!/usr/bin/env python3
"""Historical five-row crack-tip spectral scaffold (consolidated).

This module assembles the earlier decoupled state
V_sc=(a,b,tau1,tau2,Phi_sc) and pencil A V_sc=mu E V_sc.  The exact
constrained-action reduction confirms the opening/constraint blocks but shows
that the reaction is forced and the canonical in-plane momentum is not tau1;
therefore this file is a spectral scaffold, not the completed Hamiltonian DAE.
Its locations are classified by combining the pencil with analytic mode
shapes and the regularity class at the symmetry axis:

  * the integer shear shapes mu = 2, 3, ... (their scaffold vectors have no
    reaction block; coupled_shear_completion.py supplies the nonzero physical
    action-multiplier reaction and corrected canonical momentum);
  * the obstruction stated spectrally: the smooth (even-power) reaction basis
    MISSES mu = 9/4, the half-power basis CONTAINS it.  (The opening block has
    a smooth harmonic b = sin(3 theta/2) at the same mu = 9/4 -- the
    obstruction statement is about the reaction family only.)

This file preserves the representative numerical scaffold needed for the
published hierarchy figure. It must not be read as a completed coupled
spectrum; the consolidated completion scripts carry the stronger results.

Because E is singular, an unfiltered QZ solve also returns grid-dependent
near-duplicate roots. They are printed for diagnosis but are not treated as
additional scaffold modes without the analytic, block-content, regularity and
refinement checks above.

Run:  python symplectic_dae.py     (scaffold locations + diagnostics)
"""
from __future__ import annotations

import numpy as np
from scipy.linalg import eig

from leading_field import A1, A2, f, fp, solve_g_leading, g_family

SQRT2 = np.sqrt(2.0)


# --------------------------------------------------------------------------
# Chebyshev-Lobatto collocation on theta in [0, pi].
# --------------------------------------------------------------------------
def cheb_lobatto(N):
    if N < 8:
        raise ValueError("N must be at least 8")
    j = np.arange(N + 1)
    x = np.cos(np.pi * j / N)
    c = np.ones(N + 1)
    c[0] = c[-1] = 2.0
    c = c * ((-1.0) ** j)
    X = np.tile(x, (N + 1, 1))
    dX = X - X.T
    D = (np.outer(c, 1.0 / c)) / (dX + np.eye(N + 1))
    D = D - np.diag(np.sum(D, axis=1))
    theta_desc = 0.5 * np.pi * (1.0 - x)
    Dtheta_desc = (2.0 / np.pi) * D
    order = np.argsort(theta_desc)
    theta = theta_desc[order]
    Dtheta = Dtheta_desc[np.ix_(order, order)]
    return theta, Dtheta, Dtheta @ Dtheta


def assemble_pencil(N=36):
    """Assemble the historical five-row scaffold A V_sc = mu E V_sc.

    Rows:
      tau1 = mu a
      tau2 + 3/4 b = mu b                                       (nu = mu - 3/4)
      -b'' + 3/4 tau2 = mu tau2                                 (row-2 harmonic)
      f' tau1 - a2 f a' + a1 g b' - g' tau2 - sqrt2(a2 f tau2 + f' b') = 0   (constraint)
      f Phi' + 3 f' Phi = 2 mu f' Phi                           (reaction transport)
    BCs: b(0)=0, b'(pi)=0, Phi(0)=0 (Phi(pi) free as the face-load channel).
    """
    theta, D, D2 = cheb_lobatto(N)
    n = theta.size
    dim = 5 * n
    A = np.zeros((dim, dim))
    E = np.zeros((dim, dim))
    sl_a, sl_b, sl_t1, sl_t2, sl_phi = (slice(k * n, (k + 1) * n) for k in range(5))

    sol = solve_g_leading()
    gfun, gpfun = g_family(sol, A0=0.0)
    gg, gp = np.asarray(gfun(theta)), np.asarray(gpfun(theta))
    ff, ffp = f(theta), fp(theta)

    r = 0
    for i in range(n):                                   # tau1 = mu a
        A[r + i, sl_t1.start + i] = 1.0
        E[r + i, sl_a.start + i] = 1.0
    r += n
    for i in range(n):                                   # tau2 + 3/4 b = mu b
        A[r + i, sl_t2.start + i] = 1.0
        A[r + i, sl_b.start + i] = 0.75
        E[r + i, sl_b.start + i] = 1.0
    r += n
    for i in range(n):                                   # -b'' + 3/4 tau2 = mu tau2
        rr = r + i
        if i == 0:
            A[rr, sl_b.start + 0] = 1.0
        elif i == n - 1:
            A[rr, sl_b] = D[-1, :]
        else:
            A[rr, sl_b] = -D2[i, :]
            A[rr, sl_t2.start + i] = 0.75
            E[rr, sl_t2.start + i] = 1.0
    r += n
    for i in range(n):                                   # linearized constraint
        rr = r + i
        A[rr, sl_t1.start + i] += ffp[i]
        A[rr, sl_a] += -A2 * ff[i] * D[i, :]
        A[rr, sl_b] += (A1 * gg[i] - SQRT2 * ffp[i]) * D[i, :]
        A[rr, sl_t2.start + i] += -(gp[i] + SQRT2 * A2 * ff[i])
    r += n
    for i in range(n):                                   # reaction transport
        rr = r + i
        if i == 0:
            A[rr, sl_phi.start + 0] = 1.0
        else:
            A[rr, sl_phi] += ff[i] * D[i, :]
            A[rr, sl_phi.start + i] += 3.0 * ffp[i]
            E[rr, sl_phi.start + i] += 2.0 * ffp[i]
    return theta, A, E


def finite_real_eigs(A, E, abs_max=8.0, imag_tol=1e-7):
    vals = eig(A, E, right=False)
    out = [float(z.real) for z in vals
           if np.isfinite(z) and abs(z) < abs_max and abs(z.imag) < imag_tol]
    return np.array(sorted(out))


def cluster(vals, tol=1e-4):
    if vals.size == 0:
        return []
    clusters, cur = [], [float(vals[0])]
    for v in vals[1:]:
        if abs(float(v) - np.mean(cur)) < tol:
            cur.append(float(v))
        else:
            clusters.append(float(np.mean(cur)))
            cur = [float(v)]
    clusters.append(float(np.mean(cur)))
    return clusters


CLASSIFIED_MODE_LOCATIONS = np.array([1.0, 1.25, 1.5, 2.0, 2.25, 2.5, 3.0])


def classified_mode_roots(vals, targets=CLASSIFIED_MODE_LOCATIONS, tol=5e-3):
    """Return the raw collocation root nearest each analytically classified mode.

    This is a location cross-check, not a stand-alone classifier for the
    singular pencil. Mode identity is established from the analytic families
    and eigenvector blocks described in the manuscript.
    """
    vals = np.asarray(vals, dtype=float)
    roots = []
    for target in np.asarray(targets, dtype=float):
        if vals.size == 0:
            continue
        root = float(vals[np.argmin(np.abs(vals - target))])
        if abs(root - target) <= tol:
            roots.append(root)
    return roots


def shear_mode_residual(theta, A, E, k):
    """Relative pencil residual of the exact G-quiet scaffold shear a=f^(2k)."""
    n = theta.size
    a = f(theta) ** (2 * k)
    vec = np.zeros(5 * n)
    vec[:n] = a
    vec[2*n:3*n] = k * a
    residual = A @ vec - k * (E @ vec)
    scale = np.linalg.norm(A @ vec) + abs(k) * np.linalg.norm(E @ vec)
    return float(np.linalg.norm(residual) / scale)


def reaction_basis_spectrum(powers, ntheta=1000):
    """Reaction transport restricted to a Phi basis {f^m}: returns its mu spectrum.

    Smooth (even-power) basis misses mu=9/4; half-power basis contains it.
    """
    theta = np.linspace(2e-5, np.pi - 2e-5, ntheta)
    ff, ffp = f(theta), fp(theta)
    Phi = np.column_stack([ff ** m for m in powers])
    DPhi = np.column_stack([np.zeros_like(theta) if m == 0
                            else m * ff ** (m - 1.0) * ffp for m in powers])
    A = ff[:, None] * DPhi + 3.0 * ffp[:, None] * Phi
    E = 2.0 * ffp[:, None] * Phi
    L, *_ = np.linalg.lstsq(E, A, rcond=None)
    vals = sorted(float(z.real) for z in eig(L, right=False) if abs(z.imag) < 1e-9)
    return vals


def main():
    theta, A, E = assemble_pencil(N=36)
    raw_roots = cluster(finite_real_eigs(A, E))
    classified_roots = classified_mode_roots(raw_roots)
    print("Historical five-row spectral scaffold")
    print("A V_sc = mu E V_sc,  V_sc=(a,b,tau1,tau2,Phi_sc)")
    print("(use coupled_shear_completion.py for the corrected Q_k reaction block)")
    print(f"  raw finite real QZ roots (|mu|<8):  {[round(v,4) for v in raw_roots]}")
    print("  classified mode locations (0.9<mu<3.1):  "
          f"{[round(v,4) for v in classified_roots]}")

    def has(s, t, tol=5e-4):
        return any(abs(v - t) < tol for v in s)

    smooth = reaction_basis_spectrum([2, 4, 6, 8, 10])
    half = reaction_basis_spectrum([1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5])
    print(f"  smooth (even) reaction basis spectrum:     {[round(v,3) for v in smooth]}")
    print(f"  half-power reaction basis spectrum:        {[round(v,3) for v in half]}")

    q2_residual = shear_mode_residual(theta, A, E, 2)
    q3_residual = shear_mode_residual(theta, A, E, 3)
    checks = {
        "classified collocation roots reproduce all seven mode locations":
            len(classified_roots) == len(CLASSIFIED_MODE_LOCATIONS)
            and max(abs(np.asarray(classified_roots) - CLASSIFIED_MODE_LOCATIONS)) < 5e-3,
        "Q2 shear shape a=f^4 satisfies the scaffold": q2_residual < 1e-10,
        "scaffold contains mu = 9/4 (opening harmonic + reaction branch)":
            has(classified_roots, 2.25),
        "Q3 shear shape a=f^6 satisfies the scaffold": q3_residual < 1e-10,
        "smooth reaction basis MISSES mu = 9/4":
            min(abs(np.array(smooth) - 2.25)) > 0.2,
        "half-power reaction basis CONTAINS mu = 9/4":
            min(abs(np.array(half) - 2.25)) < 1e-9,
    }
    print("\n  checks:")
    for k, v in checks.items():
        print(f"    [{'PASS' if v else 'FAIL'}] {k}")
    if not all(checks.values()):
        raise SystemExit("five-row scaffold self-checks FAILED")
    print("\nFive-row scaffold checks passed "
          "(classified locations reproduced; raw near-duplicates retained only as "
          "diagnostics; smooth scaffold reaction basis misses 9/4).")


if __name__ == "__main__":
    main()
