#!/usr/bin/env python3
"""Reproduce the ESI finite-window strip table with the regular ``C_s s`` mode.

The discarded raw-tip-shape routine omitted the linear term in the deformed
horizontal coordinate and selected a tangent by proximity to the desired
slope.  This replacement selects no target shape slope.

For the two curated Mooney--Rivlin strip loads reported in the ESI, the
standard annulus is fitted jointly across all five rays with

    Y1(r, theta) = c0 + b(theta) r + a(theta) r**(5/4),

where the physical tip coordinate ``c0`` is shared.  Independent per-ray fits
are retained as sensitivity diagnostics.  The audit reports:

1. the raw local face-proxy slope using the shared intercept, with a band from
   shared and independent intercept fits on five nested windows;
2. whether ``b(theta)`` follows the regular null-mode shape
   ``C_s sin(theta/2)**2``; and
3. a target-free grid fit of ``Y1=c0+b*r+a*r**q`` on the face proxy and the
   same nested windows.

The inputs are tracked public strip data; no FEM solve is run. The output is a
structured JSON record only. This is a finite-window table-reproduction audit,
not an asymptotic matching calculation for ``C_s`` and not evidence that the
residual exponent is universal or asymptotically resolved.

Run:
    python analysis/profile_mode_audit.py
    python analysis/profile_mode_audit.py --write
    python analysis/profile_mode_audit.py --check-stored
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
STRIP_OUT = ROOT / "data" / "fem" / "strip"
RESULTS = ROOT / "data" / "derived" / "profile_mode_audit.json"

THETAS = (2, 45, 90, 135, 178)
LOCAL_POINTS = 7
FREE_Q_GRID = np.linspace(1.05, 1.55, 2001)
NESTED_FACTORS = (
    (1.00, 1.00, "full"),
    (1.25, 1.00, "drop inner"),
    (1.00, 0.80, "drop outer"),
    (1.25, 0.80, "drop both"),
    (1.50, 0.75, "central"),
)


@dataclass(frozen=True)
class Case:
    key: str
    label: str
    tag: str
    window: tuple[float, float]


CASES = (
    Case("strip_lam16", r"strip, $\lambda=1.6$", "MR_lam16",
         (3.0e-4, 8.0e-3)),
    Case("strip_lam22", r"strip, $\lambda=2.2$", "MR_lam22",
         (3.0e-4, 8.0e-3)),
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_strip(tag: str) -> tuple[dict[int, dict[str, np.ndarray]], list[Path]]:
    rays: dict[int, dict[str, np.ndarray]] = {}
    paths: list[Path] = []
    for theta in THETAS:
        path = STRIP_OUT / f"rays_{tag}_theta{theta}.csv"
        with path.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        rays[theta] = {
            key: np.asarray([float(row[key]) for row in rows])
            for key in ("r", "Y1", "Y2")
        }
        paths.append(path)
    return rays, paths


def load_case(case: Case) -> tuple[dict[int, dict[str, np.ndarray]], list[Path]]:
    return load_strip(case.tag)


def window_mask(r: np.ndarray, window: tuple[float, float]) -> np.ndarray:
    lo, hi = window
    return np.isfinite(r) & (r >= lo) & (r <= hi)


def scaled_linear_fit(
    r: np.ndarray,
    y: np.ndarray,
    window: tuple[float, float],
    q: float,
) -> tuple[np.ndarray, float, float, int]:
    """Fit ``y=c0+b*r+a*r**q`` after column scaling."""
    mask = window_mask(r, window) & np.isfinite(y)
    rr, yy = r[mask], y[mask]
    if rr.size < 12:
        raise ValueError(f"only {rr.size} points in fit window {window}")
    design = np.column_stack((np.ones(rr.size), rr, rr ** q))
    scale = np.linalg.norm(design, axis=0)
    design_scaled = design / scale
    coef_scaled, *_ = np.linalg.lstsq(design_scaled, yy, rcond=None)
    coef = coef_scaled / scale
    residual = yy - design @ coef
    rms = float(np.sqrt(np.mean(residual ** 2)))
    return coef, float(np.linalg.cond(design_scaled)), rms, int(rr.size)


def joint_fixed_q_fit(
    rays: dict[int, dict[str, np.ndarray]],
    window: tuple[float, float],
    q: float = 1.25,
) -> tuple[float, np.ndarray, np.ndarray, float, float, dict[int, float], int]:
    """Fit all rays with one c0 and independent b(theta), a(theta)."""
    n_theta = len(THETAS)
    rows: list[np.ndarray] = []
    values: list[float] = []
    row_theta: list[int] = []
    for index, theta in enumerate(THETAS):
        ray = rays[theta]
        mask = window_mask(ray["r"], window) & np.isfinite(ray["Y1"])
        rr, yy = ray["r"][mask], ray["Y1"][mask]
        if rr.size < 12:
            raise ValueError(
                f"theta={theta}: only {rr.size} points in joint window {window}"
            )
        for radius, value in zip(rr, yy):
            row = np.zeros(1 + 2 * n_theta)
            row[0] = 1.0
            row[1 + index] = radius
            row[1 + n_theta + index] = radius ** q
            rows.append(row)
            values.append(float(value))
            row_theta.append(theta)

    design = np.asarray(rows)
    response = np.asarray(values)
    scale = np.linalg.norm(design, axis=0)
    design_scaled = design / scale
    coef_scaled, *_ = np.linalg.lstsq(design_scaled, response, rcond=None)
    coef = coef_scaled / scale
    residual = response - design @ coef
    row_theta_array = np.asarray(row_theta)
    per_theta_rms = {
        theta: float(np.sqrt(np.mean(residual[row_theta_array == theta] ** 2)))
        for theta in THETAS
    }
    return (
        float(coef[0]),
        coef[1:1 + n_theta],
        coef[1 + n_theta:],
        float(np.linalg.cond(design_scaled)),
        float(np.sqrt(np.mean(residual ** 2))),
        per_theta_rms,
        int(response.size),
    )


def free_q_fit(
    r: np.ndarray,
    y: np.ndarray,
    window: tuple[float, float],
) -> tuple[float, np.ndarray, float, float, int]:
    """Grid-fit ``y=c0+b*r+a*r**q`` without a target or initial guess."""
    records = []
    for q in FREE_Q_GRID:
        coef, condition, rms, count = scaled_linear_fit(r, y, window, float(q))
        records.append((rms, float(q), coef, condition, count))
    rms, q, coef, condition, count = min(records, key=lambda item: item[0])
    if q in (float(FREE_Q_GRID[0]), float(FREE_Q_GRID[-1])):
        raise RuntimeError(f"free-q optimum {q} lies on the search boundary")
    return q, coef, condition, rms, count


def log_slope(x: np.ndarray, y: np.ndarray) -> float:
    valid = np.isfinite(x) & np.isfinite(y) & (x > 0.0) & (y > 0.0)
    if np.count_nonzero(valid) < 3:
        return float("nan")
    return float(np.polyfit(np.log(x[valid]), np.log(y[valid]), 1)[0])


def local_raw_shape_slope(
    r: np.ndarray,
    y1: np.ndarray,
    y2: np.ndarray,
    c0: float,
    window: tuple[float, float],
) -> tuple[np.ndarray, np.ndarray]:
    """Centered local estimate of d log(Y2) / d log|Y1-c0|."""
    indices = np.flatnonzero(
        window_mask(r, window)
        & np.isfinite(y1)
        & np.isfinite(y2)
        & (np.abs(y1 - c0) > 0.0)
        & (y2 > 0.0)
    )
    half = LOCAL_POINTS // 2
    radii, slopes = [], []
    for center in range(half, indices.size - half):
        use = indices[center - half:center + half + 1]
        p1 = log_slope(r[use], np.abs(y1[use] - c0))
        p2 = log_slope(r[use], y2[use])
        radii.append(r[indices[center]])
        slopes.append(p2 / p1)
    return np.asarray(radii), np.asarray(slopes)


def nested_windows(base: tuple[float, float]) -> list[tuple[str, tuple[float, float]]]:
    lo, hi = base
    return [(label, (lo * inner, hi * outer))
            for inner, outer, label in NESTED_FACTORS]


def analyse(case: Case) -> tuple[dict, dict[int, dict[str, np.ndarray]]]:
    rays, paths = load_case(case)
    missing = sorted(set(THETAS) - set(rays))
    if missing:
        raise FileNotFoundError(f"{case.key}: missing rays {missing}")

    face = rays[178]
    face_fixed, face_cond, face_rms, face_count = scaled_linear_fit(
        face["r"], face["Y1"], case.window, 1.25
    )
    face_c0, face_b, face_a = map(float, face_fixed)
    (
        joint_c0, joint_b, joint_a, joint_cond, joint_rms,
        joint_per_theta_rms, joint_count,
    ) = joint_fixed_q_fit(rays, case.window)

    r_local, slope_local = local_raw_shape_slope(
        face["r"], face["Y1"], face["Y2"], joint_c0, case.window
    )
    face_mask = window_mask(face["r"], case.window)
    raw_global = log_slope(
        np.abs(face["Y1"][face_mask] - joint_c0), face["Y2"][face_mask]
    )
    _, independent_slope_local = local_raw_shape_slope(
        face["r"], face["Y1"], face["Y2"], face_c0, case.window
    )
    independent_raw_global = log_slope(
        np.abs(face["Y1"][face_mask] - face_c0), face["Y2"][face_mask]
    )

    nested = []
    c0_ensemble = []
    for label, window in nested_windows(case.window):
        (
            nested_joint_c0, nested_joint_b, nested_joint_a,
            nested_joint_cond, nested_joint_rms,
            nested_joint_per_theta_rms, nested_joint_count,
        ) = joint_fixed_q_fit(rays, window)
        coef_fixed, cond_fixed, rms_fixed, n_fixed = scaled_linear_fit(
            face["r"], face["Y1"], window, 1.25
        )
        q_free, coef_free, cond_free, rms_free, n_free = free_q_fit(
            face["r"], face["Y1"], window
        )
        c0_ensemble.extend((nested_joint_c0, float(coef_fixed[0]),
                            float(coef_free[0])))
        nested.append({
            "label": label,
            "window": list(window),
            "joint_fixed_q": {
                "q": 1.25,
                "c0_shared": nested_joint_c0,
                "b_per_theta": nested_joint_b.tolist(),
                "a_per_theta": nested_joint_a.tolist(),
                "scaled_condition_number": nested_joint_cond,
                "rms": nested_joint_rms,
                "rms_per_theta": {
                    str(theta): nested_joint_per_theta_rms[theta]
                    for theta in THETAS
                },
                "n_points_all_rays": nested_joint_count,
            },
            "fixed_q": 1.25,
            "fixed_coefficients_c0_b_a": coef_fixed.tolist(),
            "fixed_scaled_condition_number": cond_fixed,
            "fixed_rms": rms_fixed,
            "free_q": q_free,
            "free_coefficients_c0_b_a": coef_free.tolist(),
            "free_scaled_condition_number": cond_free,
            "free_rms": rms_free,
            "n_points": min(n_fixed, n_free),
        })

    slope_ensemble = []
    for intercept in c0_ensemble:
        r_check, slope_check = local_raw_shape_slope(
            face["r"], face["Y1"], face["Y2"], intercept, case.window
        )
        if not np.allclose(r_check, r_local):
            raise RuntimeError("local-slope radii changed across intercept fits")
        slope_ensemble.append(slope_check)
    slope_ensemble_array = np.asarray(slope_ensemble)

    independent_theta_coefficients = []
    x_mode = np.sin(np.deg2rad(np.asarray(THETAS, dtype=float)) / 2.0) ** 2
    independent_b_values = []
    for theta in THETAS:
        ray = rays[theta]
        coef, theta_cond, theta_rms, theta_count = scaled_linear_fit(
            ray["r"], ray["Y1"], case.window, 1.25
        )
        independent_b_values.append(float(coef[1]))
        independent_theta_coefficients.append({
            "theta_deg": theta,
            "sin2_half_theta": float(
                np.sin(np.deg2rad(float(theta)) / 2.0) ** 2
            ),
            "coefficients_c0_b_a": coef.tolist(),
            "scaled_condition_number": theta_cond,
            "rms": theta_rms,
            "n_points": theta_count,
        })
    independent_b_array = np.asarray(independent_b_values)
    independent_cs = float(
        np.dot(x_mode, independent_b_array) / np.dot(x_mode, x_mode)
    )
    independent_angular_residual = float(
        np.linalg.norm(independent_b_array - independent_cs * x_mode)
        / np.linalg.norm(independent_b_array)
    )

    joint_cs = float(np.dot(x_mode, joint_b) / np.dot(x_mode, x_mode))
    joint_angular_residual = float(
        np.linalg.norm(joint_b - joint_cs * x_mode) / np.linalg.norm(joint_b)
    )
    joint_theta_coefficients = [
        {
            "theta_deg": theta,
            "sin2_half_theta": float(x_mode[index]),
            "b": float(joint_b[index]),
            "a": float(joint_a[index]),
            "rms": joint_per_theta_rms[theta],
        }
        for index, theta in enumerate(THETAS)
    ]

    result = {
        "case": case.key,
        "label": case.label,
        "geometry": "strip",
        "tag": case.tag,
        "standard_window": list(case.window),
        "input_files": [
            {"path": path.relative_to(ROOT).as_posix(), "sha256": sha256(path)}
            for path in paths
        ],
        "face_proxy_theta_deg": 178,
        "joint_fixed_q_fit": {
            "q": 1.25,
            "model": "one shared c0; independent b(theta) and a(theta)",
            "c0_shared": joint_c0,
            "scaled_condition_number": joint_cond,
            "rms": joint_rms,
            "n_points_all_rays": joint_count,
            "per_theta": joint_theta_coefficients,
        },
        "face_proxy_raw_shape": {
            "c0_source": "joint_fixed_q_fit.c0_shared",
            "c0_shared": joint_c0,
            "b_from_joint_fit": float(joint_b[-1]),
            "a_from_joint_fit": float(joint_a[-1]),
            "raw_global_shape_slope": raw_global,
            "raw_local_shape_slope_min": float(np.min(slope_local)),
            "raw_local_shape_slope_max": float(np.max(slope_local)),
            "raw_local_shape_slope_mean": float(np.mean(slope_local)),
        },
        "raw_local_curve": {
            "r": r_local.tolist(),
            "slope": slope_local.tolist(),
            "intercept_ensemble_min": np.min(slope_ensemble_array, axis=0).tolist(),
            "intercept_ensemble_max": np.max(slope_ensemble_array, axis=0).tolist(),
            "intercept_ensemble_definition": (
                "shared-c0 joint fixed-q, independent face fixed-q, and "
                "independent face free-q fits on all five nested windows"
            ),
        },
        "angular_joint_fixed_q_fit": {
            "C_s_through_origin": joint_cs,
            "relative_residual": joint_angular_residual,
            "per_theta": joint_theta_coefficients,
        },
        "independent_per_ray_fixed_q_diagnostics": {
            "face_coefficients_c0_b_a": [face_c0, face_b, face_a],
            "face_scaled_condition_number": face_cond,
            "face_rms": face_rms,
            "face_n_points": face_count,
            "face_raw_global_shape_slope": independent_raw_global,
            "face_raw_local_shape_slope_min": float(
                np.min(independent_slope_local)
            ),
            "face_raw_local_shape_slope_max": float(
                np.max(independent_slope_local)
            ),
            "C_s_through_origin": independent_cs,
            "angular_relative_residual": independent_angular_residual,
            "per_theta": independent_theta_coefficients,
        },
        "joint_minus_independent_differences": {
            "c0_shared_minus_face_c0": joint_c0 - face_c0,
            "b_face_joint_minus_independent": float(joint_b[-1] - face_b),
            "C_s_joint_minus_independent": joint_cs - independent_cs,
            "raw_global_slope_joint_minus_independent": (
                raw_global - independent_raw_global
            ),
        },
        "nested_face_fits": nested,
    }
    return result, rays


def build_payload() -> dict:
    return {
        "schema": "mr-strip-finite-window-table-v3-public",
        "interpretation": {
            "finite_window_consistency": (
                "fits are consistent with a nonzero s-like O(r) background "
                "and raw face-proxy slopes near 1/2 on the reported strip "
                "annulus"
            ),
            "not_established": (
                "ultimate matched C_s, a universal residual 5/4 power, or a "
                "universal raw 2/5 camera profile"
            ),
            "circularity_control": (
                "fixed-5/4 residual slopes are not used as exponent evidence; "
                "the residual power is refitted freely on nested windows"
            ),
            "shared_tip_coordinate": (
                "the production fixed-q fit uses one c0 across all five rays; "
                "independent per-ray fits are retained only as diagnostics"
            ),
            "publication_role": (
                "reproduces the ESI strip table; produces no manuscript figure"
            ),
        },
        "local_regression_points": LOCAL_POINTS,
        "free_q_search_interval": [float(FREE_Q_GRID[0]),
                                   float(FREE_Q_GRID[-1])],
        "cases": [analyse(case)[0] for case in CASES],
    }


def encoded_payload(payload: dict) -> str:
    return json.dumps(payload, indent=2, sort_keys=True) + "\n"


def compare_payloads(stored: object, fresh: object,
                     path: str = "$") -> tuple[bool, str]:
    """Compare regenerated results while tolerating BLAS-level roundoff."""
    if isinstance(stored, bool) or isinstance(fresh, bool):
        return (stored is fresh, path)
    if isinstance(stored, (int, float)) and isinstance(fresh, (int, float)):
        matched = math.isclose(float(stored), float(fresh),
                               rel_tol=1.0e-10, abs_tol=1.0e-12)
        return (matched, path)
    if isinstance(stored, dict) and isinstance(fresh, dict):
        if stored.keys() != fresh.keys():
            return (False, f"{path} (key set)")
        for key in stored:
            matched, location = compare_payloads(
                stored[key], fresh[key], f"{path}.{key}"
            )
            if not matched:
                return (False, location)
        return (True, path)
    if isinstance(stored, list) and isinstance(fresh, list):
        if len(stored) != len(fresh):
            return (False, f"{path} (list length)")
        for index, (stored_item, fresh_item) in enumerate(zip(stored, fresh)):
            matched, location = compare_payloads(
                stored_item, fresh_item, f"{path}[{index}]"
            )
            if not matched:
                return (False, location)
        return (True, path)
    return (stored == fresh, path)


def audit_checks(payload: dict) -> dict[str, bool]:
    cases = payload["cases"]
    primary_raw = [case["face_proxy_raw_shape"]["raw_global_shape_slope"]
                   for case in cases]
    angular = [case["angular_joint_fixed_q_fit"] for case in cases]
    free_q = [[entry["free_q"] for entry in case["nested_face_fits"]]
              for case in cases]
    return {
        "two curated strip loads": (
            len(cases) == 2
            and {case["case"] for case in cases}
            == {"strip_lam16", "strip_lam22"}
            and all(case["geometry"] == "strip" for case in cases)
        ),
        "fitted regular coefficient is nonzero": all(
            abs(item["C_s_through_origin"]) > 0.5 for item in angular
        ),
        "s-like angular residual below 3%": all(
            item["relative_residual"] < 0.03 for item in angular
        ),
        "raw stored-window slope near 1/2": all(
            0.48 < value < 0.54 for value in primary_raw
        ),
        "free-q optima are not grid boundaries": all(
            FREE_Q_GRID[0] < value < FREE_Q_GRID[-1]
            for values in free_q for value in values
        ),
    }


def print_summary(payload: dict) -> None:
    print("Strip finite-window ESI table (one shared c0 across five rays)")
    for result in payload["cases"]:
        face = result["face_proxy_raw_shape"]
        angular = result["angular_joint_fixed_q_fit"]
        free_q = [entry["free_q"] for entry in result["nested_face_fits"]]
        print(
            f"  {result['case']}: c0={face['c0_shared']:.8g}, "
            f"b_face={face['b_from_joint_fit']:.6g}, "
            f"raw={face['raw_global_shape_slope']:.4f}, "
            f"C_s={angular['C_s_through_origin']:.6g}, "
            f"angular residual={100*angular['relative_residual']:.2f}%, "
            f"free-q=[{min(free_q):.4f}, {max(free_q):.4f}]"
        )
    checks = audit_checks(payload)
    print("\n  checks:")
    for name, passed in checks.items():
        print(f"    [{'PASS' if passed else 'FAIL'}] {name}")
    if not all(checks.values()):
        raise SystemExit("profile-mode audit FAILED")
    print("\nStrip table audit passed; the residual exponent remains unresolved.")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--write", action="store_true",
                        help="write the strip-only derived JSON")
    parser.add_argument("--check-stored", action="store_true",
                        help="require stored JSON to match fresh analysis "
                             "up to numerical roundoff")
    args = parser.parse_args()

    payload = build_payload()
    encoded = encoded_payload(payload)
    if args.write:
        RESULTS.parent.mkdir(parents=True, exist_ok=True)
        RESULTS.write_text(encoded, encoding="utf-8")
        print(f"wrote {RESULTS.relative_to(ROOT)}")
    if args.check_stored:
        if not RESULTS.exists():
            raise SystemExit(
                "stored profile-mode JSON is missing; run with --write"
            )
        try:
            stored_payload = json.loads(RESULTS.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise SystemExit(f"cannot read stored profile-mode JSON: {error}")
        matched, location = compare_payloads(stored_payload, payload)
        if not matched:
            raise SystemExit(
                "stored profile-mode JSON differs materially at "
                f"{location}; run with --write and review"
            )
        print("stored JSON matches within numerical roundoff: "
              f"{RESULTS.relative_to(ROOT)}")
    print_summary(payload)


if __name__ == "__main__":
    main()
