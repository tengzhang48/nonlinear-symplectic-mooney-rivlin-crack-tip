"""Archival extraction for the auxiliary focused-disk calculation.

Given a solved state (from mr_fem_solve.solve), sample the deformed
coordinates (Y1, Y2) and the in-plane Jacobian J = det F along rays. The
returned legacy diagnostics preserve provenance but are not paper gates:

  T1  finite-window powers: Y2(theta=pi) and J; the old near-axis Y1 residual
      estimator is retained in serialized provenance but is not interpreted
      as evidence for an asymptotic residual exponent.
  T2  constant-Delta signature:  J*r^{1/4} vs theta -> plateau (MR) / varies (NH)
  T3  parameter-free amplitude:  J*r^{1/4} = sqrt(P/2),  with P from Y2(pi).

Conventions: reference polar (r, theta); deformed coordinates
Y1 = X + u_x, Y2 = Y + u_y.  P is the fitted coefficient of Y2 = P r^{1/2} on
the crack face (theta -> pi), following the angular convention in this code.
"""
from __future__ import annotations

import numpy as np
import ufl
from dolfinx import fem, geometry


# ---------------------------------------------------------------- evaluation
def _make_evaluator(msh):
    bb = geometry.bb_tree(msh, msh.topology.dim)

    def evaluate(fun, pts2d, value_size):
        pts = np.column_stack([pts2d[:, 0], pts2d[:, 1], np.zeros(len(pts2d))])
        cand = geometry.compute_collisions_points(bb, pts)
        coll = geometry.compute_colliding_cells(msh, cand, pts)
        cells = np.full(len(pts), -1, dtype=np.int32)
        for i in range(len(pts)):
            links = coll.links(i)
            if len(links) > 0:
                cells[i] = links[0]
        vals = np.full((len(pts), value_size), np.nan)
        valid = np.where(cells >= 0)[0]
        if valid.size:
            v = fun.eval(pts[valid], cells[valid])
            vals[valid] = v.reshape(valid.size, value_size)
        return vals

    return evaluate


def _interp_J(msh, u, degree=2):
    Q = fem.functionspace(msh, ("DG", degree))
    d = msh.geometry.dim
    F = ufl.Identity(d) + ufl.grad(u)
    Jfun = fem.Function(Q)
    Jfun.interpolate(fem.Expression(ufl.det(F), Q.element.interpolation_points))
    return Jfun


def _interp_F(msh, u, degree=2):
    """Interpolate the 4 components of F = I + grad u into a DG vector space."""
    V4 = fem.functionspace(msh, ("DG", degree, (4,)))
    d = msh.geometry.dim
    F = ufl.Identity(d) + ufl.grad(u)
    Fvec = ufl.as_vector([F[0, 0], F[0, 1], F[1, 0], F[1, 1]])
    Ffun = fem.Function(V4)
    Ffun.interpolate(fem.Expression(Fvec, V4.element.interpolation_points))
    return Ffun


def _principal_stretches(Fcomp):
    """Principal stretches (lambda1 >= lambda2) of 2x2 F arrays, shape (n,4)."""
    F = Fcomp.reshape(-1, 2, 2)
    C = np.einsum("nki,nkj->nij", F, F)  # C = F^T F
    a, b, d = C[:, 0, 0], C[:, 0, 1], C[:, 1, 1]
    tr = a + d
    disc = np.sqrt(np.maximum((a - d) ** 2 + 4 * b ** 2, 0.0))
    eig_hi = 0.5 * (tr + disc)
    eig_lo = 0.5 * (tr - disc)
    lam1 = np.sqrt(np.maximum(eig_hi, 0.0))
    lam2 = np.sqrt(np.maximum(eig_lo, 0.0))
    return lam1, lam2


# ---------------------------------------------------------------- sampling
def sample_rays(res, thetas_deg, r_lo, r_hi, n_r=40):
    """Return list of per-ray dicts with arrays r, Y1, Y2, J (deformed coords)."""
    msh, u = res["msh"], res["u"]
    evaluate = _make_evaluator(msh)
    Jfun = _interp_J(msh, u)
    Ffun = _interp_F(msh, u)

    r = np.logspace(np.log10(r_lo), np.log10(r_hi), n_r)
    rays = []
    for th_deg in thetas_deg:
        th = np.deg2rad(th_deg)
        X = r * np.cos(th)
        Y = r * np.sin(th)
        pts = np.column_stack([X, Y])
        uvals = evaluate(u, pts, value_size=2)
        Jvals = evaluate(Jfun, pts, value_size=1)[:, 0]
        Fvals = evaluate(Ffun, pts, value_size=4)
        lam1, lam2 = _principal_stretches(Fvals)
        Y1 = X + uvals[:, 0]
        Y2 = Y + uvals[:, 1]
        rays.append({
            "theta_deg": th_deg, "r": r, "X": X, "Y": Y,
            "ux": uvals[:, 0], "uy": uvals[:, 1],
            "Y1": Y1, "Y2": Y2, "J": Jvals,
            "lam1": lam1, "lam2": lam2,
        })
    return rays


# ---------------------------------------------------------------- fitting
def fit_powerlaw(r, q, window):
    """log|q| = log A + p log r on the window; return (p, A, n, rms_log)."""
    lo, hi = window
    m = (r >= lo) & (r <= hi) & np.isfinite(q) & (np.abs(q) > 0)
    if m.sum() < 3:
        return np.nan, np.nan, int(m.sum()), np.nan
    lr = np.log(r[m])
    lq = np.log(np.abs(q[m]))
    A = np.column_stack([np.ones_like(lr), lr])
    (logA, p), *_ = np.linalg.lstsq(A, lq, rcond=None)
    rms = float(np.sqrt(np.mean((lq - (logA + p * lr)) ** 2)))
    return float(p), float(np.exp(logA)), int(m.sum()), rms


def fit_face_opening(r, q, window):
    """Fit q = P r^{1/2} + C r on the window (lstsq); return (P, C, n, rms).

    Theory-consistent face-opening estimator: the r^{1/2} coefficient is P and
    the r term is the gap-half correction's face trace.  A free-exponent
    power-law fit biases P by several percent when the fitted exponent is
    0.48-0.49 instead of exactly 1/2 over the finite fitting window.
    """
    lo, hi = window
    m = (r >= lo) & (r <= hi) & np.isfinite(q)
    if m.sum() < 3:
        return np.nan, np.nan, int(m.sum()), np.nan
    rr, qq = r[m], q[m]
    A = np.column_stack([np.sqrt(rr), rr])
    (P, C), *_ = np.linalg.lstsq(A, qq, rcond=None)
    rms = float(np.sqrt(np.mean((qq - A @ np.array([P, C])) ** 2)))
    return float(P), float(C), int(m.sum()), rms


def fit_offset_amplitude(r, q, window, exp):
    """Fit q = c0 + amp * r^exp (linear lstsq) on the window.

    Returns (c0, amp, n, rms). This legacy two-column estimator is retained for
    stored scalar compatibility; it is not the C_s-aware face estimator.
    """
    lo, hi = window
    m = (r >= lo) & (r <= hi) & np.isfinite(q)
    if m.sum() < 3:
        return np.nan, np.nan, int(m.sum()), np.nan
    rr, qq = r[m], q[m]
    A = np.column_stack([np.ones_like(rr), rr ** exp])
    (c0, amp), *_ = np.linalg.lstsq(A, qq, rcond=None)
    rms = float(np.sqrt(np.mean((qq - (c0 + amp * rr ** exp)) ** 2)))
    return float(c0), float(amp), int(m.sum()), rms


def fit_deriv_exponent(r, q, window):
    """Local exponent of the r-dependent part of q via d q / d log r.

    If q = c0 + amp r^p then dq/dlogr = p*amp*r^p, so a log-log fit of the
    derivative magnitude vs r has slope p (the constant c0 drops out).
    """
    lo, hi = window
    m = (r >= lo) & (r <= hi) & np.isfinite(q)
    if m.sum() < 4:
        return np.nan, int(m.sum())
    rr, qq = r[m], q[m]
    order = np.argsort(rr)
    rr, qq = rr[order], qq[order]
    lr = np.log(rr)
    dq = np.gradient(qq, lr)
    good = np.abs(dq) > 0
    if good.sum() < 3:
        return np.nan, int(good.sum())
    # midpoint radii
    A = np.column_stack([np.ones(good.sum()), lr[good]])
    (b0, p), *_ = np.linalg.lstsq(A, np.log(np.abs(dq[good])), rcond=None)
    return float(p), int(good.sum())


def get_ray(rays, theta_deg):
    return min(rays, key=lambda d: abs(d["theta_deg"] - theta_deg))


# ---------------------------------------------------------------- tests
def run_tests(res, window, thetas_deg=(2, 45, 90, 135, 178), n_r=40,
              r_sample=None):
    """Run T1/T2/T3 and return a structured dict."""
    if r_sample is None:
        # sample a bit wider than the fit window so the window is interior
        r_sample = (window[0] * 0.5, window[1] * 2.0)
    rays = sample_rays(res, thetas_deg, r_sample[0], r_sample[1], n_r=n_r)

    # --- P from crack face (theta -> pi, use the closest, ~178 deg) ---
    face = get_ray(rays, 178)
    # free-exponent diagnostic fit (biases P when the fitted exponent != 1/2)
    p_open, P_free, n_open, rms_open = fit_powerlaw(face["r"], face["Y2"], window)
    # primary estimator: P = coefficient of r^{1/2} in Y2 = P r^{1/2} + C r
    P, C_face, _, rms_face = fit_face_opening(face["r"], face["Y2"], window)
    # legacy offset fit (c0 + P r^{1/2}), kept for cross-comparison
    c0_open, P_amp, _, _ = fit_offset_amplitude(face["r"], face["Y2"], window, 0.5)

    # --- T1 exponents ---
    near = get_ray(rays, 2)
    # Legacy near-axis diagnostic retained in stored scalar provenance. It is
    # not used to establish the residual exponent; the strip-only ESI audit
    # fits c0+b*r+a*r^q across all rays and reports q as unresolved.
    c0_in, Q1, n_in, rms_in = fit_offset_amplitude(near["r"], near["Y1"], window, 1.25)
    p_in, n_in_d = fit_deriv_exponent(near["r"], near["Y1"], window)
    # J exponent: average the slope over all rays
    Jexps = []
    for ray in rays:
        pj, Aj, nj, _ = fit_powerlaw(ray["r"], ray["J"], window)
        Jexps.append(pj)
    Jexp_mean = float(np.nanmean(Jexps))
    Jexp_std = float(np.nanstd(Jexps))

    # --- T2 plateau: J*r^{1/4} vs theta at several radii inside window ---
    lo, hi = window
    r_probe = np.logspace(np.log10(lo), np.log10(hi), 5)
    plateau = {}  # theta_deg -> mean Jr14 over probe radii
    Jr14_table = {}
    for ray in rays:
        # interpolate J at probe radii (log-log linear in r)
        m = (ray["r"] >= r_sample[0]) & (ray["r"] <= r_sample[1]) & np.isfinite(ray["J"]) & (ray["J"] > 0)
        if m.sum() < 3:
            continue
        logJ = np.interp(np.log(r_probe), np.log(ray["r"][m]), np.log(ray["J"][m]))
        Jprobe = np.exp(logJ)
        Jr14 = Jprobe * r_probe ** 0.25
        Jr14_table[ray["theta_deg"]] = Jr14
        plateau[ray["theta_deg"]] = float(np.mean(Jr14))

    pv = np.array(list(plateau.values()))
    plateau_mean = float(np.mean(pv))
    plateau_rel_spread = float((pv.max() - pv.min()) / np.abs(plateau_mean)) if pv.size else np.nan

    # --- T3 parameter-free relation (primary P; free-fit P kept as diagnostic) ---
    sqrt_P_over_2 = float(np.sqrt(P / 2.0)) if np.isfinite(P) and P > 0 else np.nan
    t3_rel_err = float(abs(plateau_mean - sqrt_P_over_2) / sqrt_P_over_2) if np.isfinite(sqrt_P_over_2) else np.nan
    sqrtPf = float(np.sqrt(P_free / 2.0)) if np.isfinite(P_free) and P_free > 0 else np.nan
    t3_rel_err_free = float(abs(plateau_mean - sqrtPf) / sqrtPf) if np.isfinite(sqrtPf) else np.nan

    return {
        "case": {"c1": res["c1"], "c2": res["c2"], "lam": res["lam_target"]},
        "window": window,
        "P": P, "C_face": C_face, "rms_face": rms_face,
        "P_freefit": P_free, "p_open": p_open, "n_open": n_open, "rms_open": rms_open,
        "T3_rel_err_freefit": t3_rel_err_free,
        "P_amp_fixed_half": P_amp, "c0_open": c0_open,
        "Q1": Q1, "c0_inplane": c0_in, "p_inplane": p_in,
        "n_inplane": n_in, "rms_inplane": rms_in,
        "J_exp_mean": Jexp_mean, "J_exp_std": Jexp_std, "J_exp_per_ray": dict(zip([r["theta_deg"] for r in rays], Jexps)),
        "plateau_per_theta": plateau,
        "plateau_mean": plateau_mean,
        "plateau_rel_spread": plateau_rel_spread,
        "sqrt_P_over_2": sqrt_P_over_2,
        "T3_rel_err": t3_rel_err,
        "Jr14_table": {k: v.tolist() for k, v in Jr14_table.items()},
        "r_probe": r_probe.tolist(),
        "rays": rays,
    }


def summarize(tag, t):
    print(f"\n===== {tag}: c1={t['case']['c1']}, c2={t['case']['c2']}, lam={t['case']['lam']} =====")
    print(f"  window r in [{t['window'][0]:.2e}, {t['window'][1]:.2e}]")
    print(f"  T1 opening  exponent (->0.50): {t['p_open']:.4f}   P={t['P']:.5f} "
          f"(two-term r^1/2 + r; free-fit P={t['P_freefit']:.5f}, offset-fit P={t['P_amp_fixed_half']:.5f})")
    print(f"  T1 in-plane exponent (->1.25): {t['p_inplane']:.4f}   Q1={t['Q1']:.5f}  c0={t['c0_inplane']:.5f} (rms={t['rms_inplane']:.2e})")
    print(f"  T1 J        exponent (->-0.25): {t['J_exp_mean']:.4f} +/- {t['J_exp_std']:.4f}")
    print(f"  T2 J*r^1/4 plateau mean: {t['plateau_mean']:.5f}   rel spread over theta: {t['plateau_rel_spread']*100:.2f}%")
    print(f"     per theta: " + ", ".join(f"{k:g}d:{v:.4f}" for k, v in sorted(t['plateau_per_theta'].items())))
    print(f"  T3 sqrt(P/2) = {t['sqrt_P_over_2']:.5f}   vs plateau {t['plateau_mean']:.5f}   "
          f"rel err {t['T3_rel_err']*100:.2f}% (with free-fit P: {t['T3_rel_err_freefit']*100:.2f}%)")
