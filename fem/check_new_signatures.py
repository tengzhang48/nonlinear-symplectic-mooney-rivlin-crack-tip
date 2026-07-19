"""Check the two new near-tip signatures against the stored FEM cases.

Reads the curated ``data/fem/disk/fem_case_*.json`` files (no re-solve
needed) and tests:

  S1  Cauchy stress amplitude:  sigma1 * r -> c1 P^2 / 2  (= G / pi),
      flat in theta.  sigma1 is the largest principal Cauchy stress computed
      from the stored principal stretches via the incompressible plane-stress
      relation
          sigma1 = 2 (lam1^2 - lam3^2)(c1 + c2 lam2^2),  lam3 = 1/(lam1 lam2);
      near the tip the principal direction is asymptotically the opening
      direction, so sigma22 ~ sigma1 at leading order.
      IMPORTANT: this relation is CLASS-UNIVERSAL, not an I2 discriminator —
      the leading sigma22 sees only the opening row (2 c1 |grad y2|^2), whose
      angular profile sin(theta/2) is shared by the neo-Hookean control, so
      BOTH materials must be flat and on c1 P^2 / 2 (P from the two-term face
      fit Y2 = P r^{1/2} + C r).  The FEM confirms exactly that (both ~4-9%
      flat, values within ~4%), which is also why Han-Li's apparent Cauchy
      index is insensitive to c2/c1.  The I2-specific signatures remain the
      kinematic ones (J r^{1/4} plateau, Q1 anomaly, 5/4 exponent, 2/5 shape).

  S2  Tip-opening shape:  y2 ~ (y1 - c0)^{2/5} on the face for MR
      (a2/a1 = (1/2)/(5/4)), vs exponent ~ 1/2 for the neo-Hookean control
      ((1/2)/1).  Extracted two ways: (i) ratio of the fitted radial
      exponents p_open / p_inplane (robust: the deformed tip offset c0 drops
      out); (ii) direct log-log fit of Y2 against |Y1 - c0| on the face ray
      (what one measures from an image).  Route (ii) is c0-SENSITIVE in FEM
      (|Y1 - c0| is a difference of nearly equal numbers), so it is reported
      and gated only on the MR-vs-control CONTRAST; experimentally the tip
      position is visible in the image, so the ambiguity is milder there.

Run:  python check_new_signatures.py   (numpy only)
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

OUT = Path(__file__).resolve().parent.parent / "data" / "fem" / "disk"
WINDOW = (1e-4, 1e-3)


def fit_face_P(r, y2, window):
    lo, hi = window
    m = (r >= lo) & (r <= hi) & np.isfinite(y2)
    A = np.column_stack([np.sqrt(r[m]), r[m]])
    (P, C), *_ = np.linalg.lstsq(A, y2[m], rcond=None)
    return float(P)


def fit_offset(r, q, window, exp):
    lo, hi = window
    m = (r >= lo) & (r <= hi) & np.isfinite(q)
    A = np.column_stack([np.ones(m.sum()), r[m] ** exp])
    (c0, amp), *_ = np.linalg.lstsq(A, q[m], rcond=None)
    return float(c0), float(amp)


def loglog_slope(x, y):
    m = np.isfinite(x) & np.isfinite(y) & (x > 0) & (y > 0)
    if m.sum() < 3:
        return np.nan
    A = np.column_stack([np.ones(m.sum()), np.log(x[m])])
    (_, p), *_ = np.linalg.lstsq(A, np.log(y[m]), rcond=None)
    return float(p)


def derivative_exponent(r, q, window):
    """Fit the exponent of q-c0 through d q / d log(r), removing c0."""
    lo, hi = window
    m = (r >= lo) & (r <= hi) & np.isfinite(q)
    if m.sum() < 4:
        return np.nan
    rr, qq = r[m], q[m]
    order = np.argsort(rr)
    rr, qq = rr[order], qq[order]
    lr = np.log(rr)
    dq = np.gradient(qq, lr)
    good = np.isfinite(dq) & (np.abs(dq) > 0)
    if good.sum() < 3:
        return np.nan
    A = np.column_stack([np.ones(good.sum()), lr[good]])
    (_, p), *_ = np.linalg.lstsq(A, np.log(np.abs(dq[good])), rcond=None)
    return float(p)


def sigma1(lam1, lam2, c1, c2):
    lam3 = 1.0 / (lam1 * lam2)
    return 2.0 * (lam1 ** 2 - lam3 ** 2) * (c1 + c2 * lam2 ** 2)


def analyze(path):
    d = json.loads(path.read_text())
    c1, c2 = d["c1"], d["c2"]
    rays = {ray["theta_deg"]: {k: np.asarray(v) for k, v in ray.items()
                               if isinstance(v, list)} for ray in d["rays"]}
    face = rays[max(rays)]                              # ~178 deg
    P = fit_face_P(face["r"], face["Y2"], WINDOW)

    # --- S1: sigma1 * r at probe radii inside the window, per ray ---
    lo, hi = WINDOW
    r_probe = np.logspace(np.log10(lo), np.log10(hi), 5)
    per_theta = {}
    for th, ray in rays.items():
        s1r = sigma1(ray["lam1"], ray["lam2"], c1, c2) * ray["r"]
        m = np.isfinite(s1r) & (s1r > 0)
        if m.sum() < 4:
            continue
        val = np.exp(np.interp(np.log(r_probe),
                               np.log(ray["r"][m]), np.log(s1r[m])))
        per_theta[th] = float(np.mean(val))
    vals = np.array(list(per_theta.values()))
    s1r_mean = float(vals.mean())
    s1r_spread = float((vals.max() - vals.min()) / abs(s1r_mean))
    pred = c1 * P ** 2 / 2.0
    s1r_err = float(abs(s1r_mean / pred - 1.0))

    # --- S2: tip-opening shape exponent, re-fitted from stored raw rays ---
    near = rays[min(rays)]
    m_face = (face["r"] >= lo) & (face["r"] <= hi)
    p_open = loglog_slope(face["r"][m_face], face["Y2"][m_face])
    p_inplane = derivative_exponent(near["r"], near["Y1"], WINDOW)
    shape_from_exps = p_open / p_inplane
    inplane_exp = 1.25 if c2 != 0 else 1.0
    c0, _ = fit_offset(face["r"], face["Y1"], WINDOW, inplane_exp)
    m = (face["r"] >= lo) & (face["r"] <= hi)
    shape_direct = loglog_slope(np.abs(face["Y1"][m] - c0), face["Y2"][m])

    return {
        "tag": d["tag"], "material": d["material"], "c2": c2,
        "P_two_term": P,
        "s1r_mean": s1r_mean, "s1r_pred": pred,
        "s1r_err": s1r_err, "s1r_spread": s1r_spread,
        "shape_from_exps": float(shape_from_exps),
        "shape_direct": float(shape_direct),
    }


def main():
    cases = sorted(OUT.glob("fem_case_*.json"))
    if not cases:
        raise SystemExit(f"no fem_case_*.json under {OUT}")
    print("New-signature check against stored FEM cases "
          f"(window r in [{WINDOW[0]:.0e}, {WINDOW[1]:.0e}])")
    print("\n  S1: sigma1*r vs c1 P^2/2 (= G/pi)      S2: tip-shape exponent (MR->2/5, NH->1/2)")
    print(f"  {'case':<10} {'P(2term)':>9} {'s1*r mean':>10} {'pred':>8} "
          f"{'err':>7} {'spread':>8}   {'p2/p1':>6} {'direct':>7}")
    results = []
    for path in cases:
        t = analyze(path)
        results.append(t)
        print(f"  {t['tag']:<10} {t['P_two_term']:>9.4f} {t['s1r_mean']:>10.4f} "
              f"{t['s1r_pred']:>8.4f} {t['s1r_err']*100:>6.1f}% {t['s1r_spread']*100:>7.1f}%"
              f"   {t['shape_from_exps']:>6.3f} {t['shape_direct']:>7.3f}")

    mr = [t for t in results if t["c2"] != 0]
    nh = [t for t in results if t["c2"] == 0]
    checks = {
        "S1 (class-universal): sigma1*r flat in theta for ALL cases (< 12%)":
            all(t["s1r_spread"] < 0.12 for t in results),
        "S1 (class-universal): sigma1*r = c1 P^2/2 = G/pi within 10%, ALL cases":
            all(t["s1r_err"] < 0.10 for t in results),
        "S2 MR: shape exponent p_open/p_inplane ~ 2/5 (0.34..0.46)":
            all(0.34 < t["shape_from_exps"] < 0.46 for t in mr),
        "S2 control: shape exponent p_open/p_inplane ~ 1/2 (0.44..0.60)":
            all(0.44 < t["shape_from_exps"] < 0.60 for t in nh),
        "S2 direct-image route separates MR from control (gap > 0.08)":
            (min(t["shape_direct"] for t in nh)
             - max(t["shape_direct"] for t in mr)) > 0.08,
    }
    print("\n  checks:")
    for k, v in checks.items():
        print(f"    [{'PASS' if v else 'FAIL'}] {k}")
    if not all(checks.values()):
        raise SystemExit("new-signature checks FAILED")
    print("\nNew-signature checks passed: sigma22*r = G/pi is class-universal "
          "(both materials flat and on-value); the 2/5 vs 1/2 tip shape "
          "separates MR from the control.")


if __name__ == "__main__":
    main()
