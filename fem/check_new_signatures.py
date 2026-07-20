"""Check the leading-stress relation against the stored disk FEM cases.

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
      fit Y2 = P r^{1/2} + C r).  The FEM confirms that relation in the stored
      window.  It is not an I2 discriminator.

The former S2 raw-tip-shape gate was invalid: it omitted the physical
``C_s s`` mode and selected a tangent near a target slope.  It has been
removed.  ``analysis/profile_mode_audit.py`` provides the corrected
shared-tip-coordinate, C_s-aware finite-window diagnostic.

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

    return {
        "tag": d["tag"], "material": d["material"], "c2": c2,
        "P_two_term": P,
        "s1r_mean": s1r_mean, "s1r_pred": pred,
        "s1r_err": s1r_err, "s1r_spread": s1r_spread,
    }


def main():
    cases = sorted(OUT.glob("fem_case_*.json"))
    if not cases:
        raise SystemExit(f"no fem_case_*.json under {OUT}")
    print("Leading-stress check against stored disk FEM cases "
          f"(window r in [{WINDOW[0]:.0e}, {WINDOW[1]:.0e}])")
    print("\n  sigma1*r vs c1 P^2/2 (= G/pi)")
    print(f"  {'case':<10} {'P(2term)':>9} {'s1*r mean':>10} {'pred':>8} "
          f"{'err':>7} {'spread':>8}")
    results = []
    for path in cases:
        t = analyze(path)
        results.append(t)
        print(f"  {t['tag']:<10} {t['P_two_term']:>9.4f} {t['s1r_mean']:>10.4f} "
              f"{t['s1r_pred']:>8.4f} {t['s1r_err']*100:>6.1f}% "
              f"{t['s1r_spread']*100:>7.1f}%")

    checks = {
        "S1 (class-universal): sigma1*r flat in theta for ALL cases (< 12%)":
            all(t["s1r_spread"] < 0.12 for t in results),
        "S1 (class-universal): sigma1*r = c1 P^2/2 = G/pi within 10%, ALL cases":
            all(t["s1r_err"] < 0.10 for t in results),
    }
    print("\n  checks:")
    for k, v in checks.items():
        print(f"    [{'PASS' if v else 'FAIL'}] {k}")
    if not all(checks.values()):
        raise SystemExit("new-signature checks FAILED")
    print("\nLeading-stress checks passed: sigma22*r = G/pi is "
          "class-universal in the stored window.")


if __name__ == "__main__":
    main()
