#!/usr/bin/env python3
"""Stage-0 estimator for a bounded r^(5/4) strip-FEM campaign.

This program reads a live-P2 polar profile written by ``run_ps.py`` and its
companion JSON metadata.  It does not solve the finite-element problem.

The estimator keeps three logically different questions separate:

1. Exact-axis discovery.  At theta=0 the determinant-null ``C_s`` motion and
   the homogeneous ``C_h f^(5/2)`` direction both vanish.  We fit

       y1 = c0 + A_ax r^q + D r^(7/4),

   where the next-order amplitude D is free and q is discovered.

2. Full-angle discovery and branch adjudication.  We first fit

       y1 = c0 + b(theta) r + h(theta) r^q

   with one common q.  We then test, rather than impose,
   ``b = C_s sin(theta/2)^2`` and
   ``sqrt(P) h = g + C_h sin(theta/2)^(5/2)``.

3. Opening diagnostic.  On the exact crack face we fit the coefficient of
   the order-r opening term.  It is reported as a diagnostic only.  The
   available strip has nonzero C_s, whereas the closed weighted-7/4
   coefficient was derived on the C_s=0 representative.

All condition numbers are those of column-scaled design matrices X, never
of X.T X.  Linear standard errors and profile-curvature uncertainties are
conditional least-squares diagnostics.  FEM samples are correlated, so
these formal uncertainties are not statistical confidence statements.

Example
-------
python fit_r54_campaign_stage0.py \
  --profile outputs/p2profile_pilot54_nr144.npz \
  --metadata outputs/ps_pilot54_nr144.json \
  --output ../../analysis/STAGE0_R54_CAMPAIGN_PILOT54_NR144.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Callable

import numpy as np
from scipy.special import hyp2f1


PROTOCOL_DATE = "2026-07-23"
SCHEMA = "r54-campaign-stage0-v1"
CONDITION_LIMIT = 1.0e8
REPO_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_WINDOWS = (
    (1.5e-4, 1.6e-3),
    (3.0e-4, 3.0e-3),
    (1.5e-4, 3.0e-3),
    (6.0e-4, 6.0e-3),
    (1.6e-3, 1.6e-2),
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def portable_path(path: Path) -> str:
    """Prefer a repository-relative record while accepting arbitrary inputs."""
    try:
        return str(path.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(path.resolve())


def parse_window(text: str) -> tuple[float, float]:
    try:
        left, right = text.replace(",", ":").split(":")
        lo, hi = float(left), float(right)
    except Exception as exc:
        raise argparse.ArgumentTypeError(
            "window must have the form LO:HI") from exc
    if not (0.0 < lo < hi):
        raise argparse.ArgumentTypeError("window must satisfy 0 < LO < HI")
    return lo, hi


def rms(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    return float(np.sqrt(np.mean(values * values)))


def jsonable(value):
    if isinstance(value, dict):
        return {str(key): jsonable(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(val) for val in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


def scaled_linear_fit(
    design: np.ndarray,
    response: np.ndarray,
    *,
    covariance: bool = True,
) -> dict:
    """Least squares with transparent column scaling and rank diagnostics."""
    design = np.asarray(design, dtype=float)
    response = np.asarray(response, dtype=float)
    if design.ndim != 2 or response.ndim != 1:
        raise ValueError("design must be 2-D and response must be 1-D")
    if design.shape[0] != response.size:
        raise ValueError("design and response row counts differ")
    scale = np.linalg.norm(design, axis=0)
    if np.any(~np.isfinite(scale)) or np.any(scale == 0.0):
        raise ValueError("design contains a zero or non-finite column")

    scaled = design / scale
    beta_scaled, _, rank, singular = np.linalg.lstsq(
        scaled, response, rcond=None)
    beta = beta_scaled / scale
    residual = response - design @ beta
    rss = float(residual @ residual)
    nobs, npar = design.shape
    dof = int(nobs - rank)
    condition = (
        float(singular[0] / singular[-1])
        if singular.size and singular[-1] > 0.0
        else float("inf")
    )
    identifiable = bool(
        rank == npar and dof > 0
        and np.isfinite(condition) and condition <= CONDITION_LIMIT
    )

    cov = None
    se = None
    sigma2 = None
    if covariance and identifiable:
        sigma2 = rss / dof
        _, sval, vt = np.linalg.svd(scaled, full_matrices=False)
        cov_scaled = (vt.T * (sigma2 / (sval * sval))) @ vt
        inv_scale = 1.0 / scale
        cov = cov_scaled * np.outer(inv_scale, inv_scale)
        se = np.sqrt(np.maximum(np.diag(cov), 0.0))

    return {
        "coef": beta,
        "residual": residual,
        "rss": rss,
        "rms": rms(residual),
        "n_observations": int(nobs),
        "n_parameters": int(npar),
        "rank": int(rank),
        "degrees_of_freedom": dof,
        "condition_scaled_X": condition,
        "column_scales": scale,
        "identifiable_linear_fit": identifiable,
        "conditional_sigma2": sigma2,
        "conditional_covariance": cov,
        "conditional_standard_errors": se,
    }


def public_linear_fit(fit: dict) -> dict:
    """Remove large/private arrays while preserving auditable diagnostics."""
    return {
        "rss": fit["rss"],
        "rms": fit["rms"],
        "n_observations": fit["n_observations"],
        "n_parameters": fit["n_parameters"],
        "rank": fit["rank"],
        "degrees_of_freedom": fit["degrees_of_freedom"],
        "condition_scaled_X": fit["condition_scaled_X"],
        "identifiable_linear_fit": fit["identifiable_linear_fit"],
        "conditional_sigma2": fit["conditional_sigma2"],
        "uncertainty_scope": (
            "conditional iid least-squares diagnostic; correlated FEM "
            "samples make this non-probabilistic"
        ),
    }


def refine_grid_minimum(
    q_grid: np.ndarray,
    values: np.ndarray,
    objective: Callable[[float], float],
) -> tuple[float, float, bool]:
    """Refine an interior grid minimum with a bounded golden-section search."""
    index = int(np.argmin(values))
    if index == 0 or index == q_grid.size - 1:
        return float(q_grid[index]), float(values[index]), True

    left = float(q_grid[index - 1])
    right = float(q_grid[index + 1])
    phi = (1.0 + math.sqrt(5.0)) / 2.0
    c = right - (right - left) / phi
    d = left + (right - left) / phi
    fc, fd = objective(c), objective(d)
    for _ in range(28):
        if fc <= fd:
            right, d, fd = d, c, fc
            c = right - (right - left) / phi
            fc = objective(c)
        else:
            left, c, fc = c, d, fd
            d = left + (right - left) / phi
            fd = objective(d)
    q = 0.5 * (left + right)
    return q, float(objective(q)), False


def profile_uncertainty(
    objective: Callable[[float], float],
    q: float,
    rss_value: float,
    degrees_of_freedom: int,
    q_bounds: tuple[float, float],
) -> dict:
    """Local conditional uncertainty from the profiled RSS curvature."""
    span = q_bounds[1] - q_bounds[0]
    step = min(2.0e-3, span / 200.0)
    if (degrees_of_freedom <= 0 or q - step <= q_bounds[0]
            or q + step >= q_bounds[1]):
        return {
            "identifiable": False,
            "reason": "boundary optimum or nonpositive degrees of freedom",
            "conditional_se": None,
            "conditional_95_interval": None,
        }
    second = (
        objective(q + step) - 2.0 * rss_value + objective(q - step)
    ) / (step * step)
    sigma2 = rss_value / degrees_of_freedom
    if not np.isfinite(second) or second <= 0.0 or sigma2 < 0.0:
        return {
            "identifiable": False,
            "reason": "nonpositive or non-finite profile curvature",
            "conditional_se": None,
            "conditional_95_interval": None,
        }
    # RSS'' = 2 J_q^T J_q locally, hence Var(q)=2 sigma^2/RSS''.
    se = math.sqrt(2.0 * sigma2 / second)
    return {
        "identifiable": bool(np.isfinite(se)),
        "reason": None,
        "conditional_se": se,
        "conditional_95_interval": [q - 1.96 * se, q + 1.96 * se],
        "profile_second_derivative": second,
        "uncertainty_scope": (
            "local conditional least-squares curvature; mesh and sample "
            "correlation are not included"
        ),
    }


def window_indices(r: np.ndarray, window: tuple[float, float]) -> np.ndarray:
    indices = np.flatnonzero((r >= window[0]) & (r <= window[1]))
    if indices.size < 8:
        raise ValueError(
            f"only {indices.size} radii in window {window}; at least 8 required")
    return indices


def alternating_split(indices: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return indices[::2], indices[1::2]


def axis_fit_at(
    r: np.ndarray,
    y: np.ndarray,
    indices: np.ndarray,
    q: float,
    next_power: float,
    *,
    covariance: bool,
) -> dict:
    rr = r[indices]
    design = np.column_stack((
        np.ones(rr.size), rr ** q, rr ** next_power))
    return scaled_linear_fit(design, y[indices], covariance=covariance)


def exact_axis_analysis(
    r: np.ndarray,
    y_axis: np.ndarray,
    windows: tuple[tuple[float, float], ...],
    q_grid: np.ndarray,
    next_power: float,
    p_measured: float,
) -> list[dict]:
    results = []
    g0 = 4.0 * math.sqrt(2.0) / 5.0
    predicted_amplitude = g0 / math.sqrt(p_measured)
    for window in windows:
        indices = window_indices(r, window)
        train, hold = alternating_split(indices)
        if hold.size < 3:
            raise ValueError(f"too few held-out radii in window {window}")

        train_objective = lambda q: axis_fit_at(
            r, y_axis, train, q, next_power, covariance=False)["rss"]
        train_curve = np.array([train_objective(q) for q in q_grid])
        q_train, _, train_boundary = refine_grid_minimum(
            q_grid, train_curve, train_objective)
        train_fit = axis_fit_at(
            r, y_axis, train, q_train, next_power, covariance=False)
        c0_t, amp_t, nuisance_t = train_fit["coef"]
        prediction_hold = (
            c0_t + amp_t * r[hold] ** q_train
            + nuisance_t * r[hold] ** next_power)
        error_hold = y_axis[hold] - prediction_hold
        residual_scale = rms(amp_t * r[hold] ** q_train)

        full_objective = lambda q: axis_fit_at(
            r, y_axis, indices, q, next_power, covariance=False)["rss"]
        full_curve = np.array([full_objective(q) for q in q_grid])
        q_full, rss_full, full_boundary = refine_grid_minimum(
            q_grid, full_curve, full_objective)
        full_fit = axis_fit_at(
            r, y_axis, indices, q_full, next_power, covariance=True)
        c0, amplitude, nuisance = full_fit["coef"]
        se = full_fit["conditional_standard_errors"]
        q_unc = profile_uncertainty(
            full_objective, q_full, rss_full,
            full_fit["degrees_of_freedom"] - 1,
            (float(q_grid[0]), float(q_grid[-1])))

        results.append({
            "window": list(window),
            "n_radii": int(indices.size),
            "model": f"y1=c0+A*r^q+D*r^{next_power:g}",
            "q_train_selected": q_train,
            "q_train_on_search_boundary": train_boundary,
            "q_full": q_full,
            "q_full_on_search_boundary": full_boundary,
            "q_conditional_uncertainty": q_unc,
            "c0": c0,
            "A": amplitude,
            "D_next_order_nuisance": nuisance,
            "conditional_standard_errors": (
                {"c0": se[0], "A": se[1], "D": se[2]}
                if se is not None else None
            ),
            "A_predicted_from_measured_P": predicted_amplitude,
            "A_relative_error": (
                (amplitude - predicted_amplitude) / predicted_amplitude),
            "linear_fit": public_linear_fit(full_fit),
            "radial_holdout": {
                "scheme": "alternating radii; q and coefficients use train fold",
                "n_train": int(train.size),
                "n_holdout": int(hold.size),
                "rmse": rms(error_hold),
                "predicted_rq_component_rms": residual_scale,
                "rmse_over_predicted_rq_component": (
                    rms(error_hold) / residual_scale
                    if residual_scale > 0.0 else None),
            },
        })
    return results


def fast_unstructured_fit(
    r: np.ndarray,
    field: np.ndarray,
    q: float,
) -> tuple[float, np.ndarray, np.ndarray, float]:
    """Exact block elimination for c0 + b_theta r + h_theta r^q."""
    if field.ndim != 2 or field.shape[1] != r.size:
        raise ValueError("field must have shape (n_theta,n_r)")
    local = np.column_stack((r, r ** q))
    pinv = np.linalg.pinv(local)
    projector = np.eye(r.size) - local @ pinv
    ones = np.ones(r.size)
    projected_ones = projector @ ones
    denominator = field.shape[0] * float(ones @ projected_ones)
    if abs(denominator) < 1.0e-24:
        raise ValueError("global intercept is not identifiable")
    c0 = float(np.sum(field @ projected_ones) / denominator)
    coefficients = np.linalg.lstsq(
        local, (field - c0).T, rcond=None)[0]
    b, h = coefficients[0], coefficients[1]
    residual = field - (
        c0 + np.outer(b, r) + np.outer(h, r ** q))
    return c0, b, h, float(np.sum(residual * residual))


def unstructured_design(r: np.ndarray, n_theta: int, q: float) -> np.ndarray:
    n_r = r.size
    design = np.zeros((n_theta * n_r, 1 + 2 * n_theta))
    design[:, 0] = 1.0
    for index in range(n_theta):
        rows = slice(index * n_r, (index + 1) * n_r)
        design[rows, 1 + index] = r
        design[rows, 1 + n_theta + index] = r ** q
    return design


def final_unstructured_fit(
    r: np.ndarray,
    field: np.ndarray,
    q: float,
) -> dict:
    design = unstructured_design(r, field.shape[0], q)
    fit = scaled_linear_fit(design, field.ravel(), covariance=True)
    n_theta = field.shape[0]
    beta = fit["coef"]
    return {
        "fit": fit,
        "c0": float(beta[0]),
        "b": beta[1:1 + n_theta],
        "h": beta[1 + n_theta:1 + 2 * n_theta],
    }


def exact_g(theta: np.ndarray) -> np.ndarray:
    """Selected analytic-axis g(theta), evaluated without a slow endpoint sum."""
    x = np.sin(theta / 2.0) ** 2
    return (
        4.0 * math.sqrt(2.0) / 5.0
        * hyp2f1(0.5, -1.25, -0.25, x)
    )


def legacy_truncated_g_pi(terms: int = 200) -> float:
    """Value produced at the face by the old slowly convergent power sum."""
    central = 1.0
    total = 0.0
    for index in range(terms):
        if index:
            central *= (2.0 * index - 1.0) / (2.0 * index)
        total += -4.0 * math.sqrt(2.0) * central / (4.0 * index - 5.0)
    return total


def internal_checks() -> list[dict]:
    """Small deterministic checks independent of the FEM data."""
    checks = []
    theta = np.array([0.0, math.pi / 2.0, math.pi])
    values = exact_g(theta)
    expected = np.array([
        4.0 * math.sqrt(2.0) / 5.0,
        2.323823824248466,
        2.033311403505549807855597896,
    ])
    error = float(np.max(np.abs(values - expected)))
    checks.append({
        "name": (
            "exact hypergeometric g agrees with committed profile at "
            "theta=0,pi/2,pi"
        ),
        "passed": error < 1.0e-12,
        "maximum_error": error,
    })

    radii = np.geomspace(2.0e-4, 8.0e-3, 17)
    q = 1.273
    c0_true = 0.17
    b_true = np.array([-0.2, -0.8, -1.4])
    h_true = np.array([0.7, 1.1, 1.8])
    synthetic = (
        c0_true + np.outer(b_true, radii)
        + np.outer(h_true, radii ** q))
    c0, b, h, rss = fast_unstructured_fit(radii, synthetic, q)
    block_error = max(
        abs(c0 - c0_true),
        float(np.max(np.abs(b - b_true))),
        float(np.max(np.abs(h - h_true))),
        math.sqrt(rss),
    )
    checks.append({
        "name": "block-eliminated unstructured fit recovers synthetic field",
        "passed": block_error < 1.0e-10,
        "maximum_error": block_error,
    })

    design = unstructured_design(radii, synthetic.shape[0], q)
    direct = scaled_linear_fit(
        design, synthetic.ravel(), covariance=False)
    direct_beta = direct["coef"]
    direct_error = max(
        abs(direct_beta[0] - c0),
        float(np.max(np.abs(direct_beta[1:4] - b))),
        float(np.max(np.abs(direct_beta[4:7] - h))),
        abs(direct["rss"] - rss),
    )
    checks.append({
        "name": "block elimination agrees with direct scaled-X least squares",
        "passed": direct_error < 1.0e-10,
        "maximum_error": direct_error,
    })
    if not all(check["passed"] for check in checks):
        failures = [
            check["name"] for check in checks if not check["passed"]]
        raise AssertionError(f"internal estimator checks failed: {failures}")
    return checks


def project_cs(
    theta: np.ndarray,
    b: np.ndarray,
    covariance_b: np.ndarray | None,
) -> dict:
    shape = np.sin(theta / 2.0) ** 2
    denominator = float(shape @ shape)
    weights = shape / denominator
    coefficient = float(weights @ b)
    residual = b - coefficient * shape
    standard_error = None
    if covariance_b is not None:
        variance = float(weights @ covariance_b @ weights)
        standard_error = math.sqrt(max(variance, 0.0))
    return {
        "Cs": coefficient,
        "conditional_standard_error": standard_error,
        "relative_L2_shape_error": (
            float(np.linalg.norm(residual) / np.linalg.norm(b))
            if np.linalg.norm(b) > 0.0 else None),
        "tested_shape": "b(theta)=Cs*sin(theta/2)^2",
    }


def project_ch(
    theta: np.ndarray,
    h: np.ndarray,
    p_measured: float,
    covariance_h: np.ndarray | None,
) -> dict:
    f52 = np.sin(theta / 2.0) ** 2.5
    g = exact_g(theta)
    target = math.sqrt(p_measured) * h - g
    denominator = float(f52 @ f52)
    weights = f52 / denominator
    coefficient = float(weights @ target)
    model = (g + coefficient * f52) / math.sqrt(p_measured)
    standard_error = None
    if covariance_h is not None:
        covariance_target = p_measured * covariance_h
        variance = float(weights @ covariance_target @ weights)
        standard_error = math.sqrt(max(variance, 0.0))
    angular_residual = target - coefficient * f52
    angular_fit_se = (
        rms(angular_residual) / math.sqrt(denominator / theta.size)
        if theta.size > 1 and denominator > 0.0 else None
    )
    return {
        "Ch": coefficient,
        "conditional_linear_standard_error": standard_error,
        "angular_mismatch_scale": angular_fit_se,
        "relative_L2_profile_error": (
            float(np.linalg.norm(h - model) / np.linalg.norm(h))
            if np.linalg.norm(h) > 0.0 else None),
        "tested_family": "sqrt(P)*h=g_exact+Ch*sin(theta/2)^(5/2)",
        "g_evaluation": (
            "(4*sqrt(2)/5)*hyp2f1(1/2,-5/4;-1/4;sin(theta/2)^2)"
        ),
        "g_pi": float(g[-1]),
    }


def coefficient_covariance_blocks(
    fit: dict,
    n_theta: int,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    covariance = fit["conditional_covariance"]
    if covariance is None:
        return None, None
    b_start = 1
    h_start = 1 + n_theta
    return (
        covariance[b_start:h_start, b_start:h_start],
        covariance[h_start:h_start + n_theta, h_start:h_start + n_theta],
    )


def radial_holdout_unstructured(
    r: np.ndarray,
    field: np.ndarray,
    q_grid: np.ndarray,
) -> dict:
    all_indices = np.arange(r.size)
    train, hold = alternating_split(all_indices)
    objective = lambda q: fast_unstructured_fit(
        r[train], field[:, train], q)[3]
    curve = np.array([objective(q) for q in q_grid])
    q, _, boundary = refine_grid_minimum(q_grid, curve, objective)
    c0, b, h, _ = fast_unstructured_fit(r[train], field[:, train], q)
    prediction = (
        c0 + np.outer(b, r[hold]) + np.outer(h, r[hold] ** q))
    error = field[:, hold] - prediction
    residual_component = np.outer(h, r[hold] ** q)
    nonconstant_component = (
        np.outer(b, r[hold]) + residual_component)
    return {
        "scheme": "alternating radii; q and coefficients use train fold",
        "q_train_selected": q,
        "q_on_search_boundary": boundary,
        "n_train_radii": int(train.size),
        "n_holdout_radii": int(hold.size),
        "rmse": rms(error),
        "rmse_over_predicted_residual_component": (
            rms(error) / rms(residual_component)
            if rms(residual_component) > 0.0 else None),
        "rmse_over_predicted_nonconstant_field": (
            rms(error) / rms(nonconstant_component)
            if rms(nonconstant_component) > 0.0 else None),
    }


def angular_holdout_unstructured(
    theta: np.ndarray,
    r: np.ndarray,
    field: np.ndarray,
    q_grid: np.ndarray,
    stride: int,
) -> dict:
    candidates = np.arange(1, theta.size - 1)
    hold = candidates[(np.arange(candidates.size) % stride) == stride // 2]
    train = np.setdiff1d(np.arange(theta.size), hold)
    if hold.size < 2:
        raise ValueError("angular holdout has fewer than two angles")

    objective = lambda q: fast_unstructured_fit(
        r, field[train], q)[3]
    curve = np.array([objective(q) for q in q_grid])
    q, _, boundary = refine_grid_minimum(q_grid, curve, objective)
    c0, b_train, h_train, _ = fast_unstructured_fit(
        r, field[train], q)
    b_hold = np.interp(theta[hold], theta[train], b_train)
    h_hold = np.interp(theta[hold], theta[train], h_train)
    prediction = (
        c0 + np.outer(b_hold, r) + np.outer(h_hold, r ** q))
    error = field[hold] - prediction
    residual_component = np.outer(h_hold, r ** q)
    nonconstant_component = (
        np.outer(b_hold, r) + residual_component)
    return {
        "scheme": (
            f"every {stride}th interior angle held out; b(theta) and h(theta) "
            "linearly interpolated from retained angles"
        ),
        "q_train_selected": q,
        "q_on_search_boundary": boundary,
        "n_train_angles": int(train.size),
        "n_holdout_angles": int(hold.size),
        "holdout_angles_deg": np.rad2deg(theta[hold]),
        "rmse": rms(error),
        "rmse_over_predicted_residual_component": (
            rms(error) / rms(residual_component)
            if rms(residual_component) > 0.0 else None),
        "rmse_over_predicted_nonconstant_field": (
            rms(error) / rms(nonconstant_component)
            if rms(nonconstant_component) > 0.0 else None),
    }


def full_angle_discovery(
    theta: np.ndarray,
    r: np.ndarray,
    field: np.ndarray,
    windows: tuple[tuple[float, float], ...],
    q_grid: np.ndarray,
    p_measured: float,
    angular_stride: int,
) -> list[dict]:
    results = []
    for window in windows:
        indices = window_indices(r, window)
        rr, yy = r[indices], field[:, indices]
        objective = lambda q: fast_unstructured_fit(rr, yy, q)[3]
        curve = np.array([objective(q) for q in q_grid])
        q, rss_value, boundary = refine_grid_minimum(
            q_grid, curve, objective)
        final = final_unstructured_fit(rr, yy, q)
        fit, b, h = final["fit"], final["b"], final["h"]
        cov_b, cov_h = coefficient_covariance_blocks(fit, theta.size)
        q_unc = profile_uncertainty(
            objective, q, rss_value, fit["degrees_of_freedom"] - 1,
            (float(q_grid[0]), float(q_grid[-1])))

        results.append({
            "window": list(window),
            "n_radii": int(indices.size),
            "model": "y1=c0+b(theta)*r+h(theta)*r^q",
            "q_full": q,
            "q_on_search_boundary": boundary,
            "q_conditional_uncertainty": q_unc,
            "c0": final["c0"],
            "b": b,
            "h": h,
            "linear_fit": public_linear_fit(fit),
            "Cs_projection": project_cs(theta, b, cov_b),
            "Ch_projection": project_ch(
                theta, h, p_measured, cov_h),
            "radial_holdout": radial_holdout_unstructured(
                rr, yy, q_grid),
            "angular_holdout": angular_holdout_unstructured(
                theta, rr, yy, q_grid, angular_stride),
        })
    return results


def pure_residual_fit(
    r: np.ndarray,
    residual_field: np.ndarray,
    q: float,
) -> tuple[np.ndarray, float]:
    basis = r ** q
    denominator = float(basis @ basis)
    h = residual_field @ basis / denominator
    residual = residual_field - np.outer(h, basis)
    return h, float(np.sum(residual * residual))


def background_subtracted_discovery(
    theta: np.ndarray,
    r: np.ndarray,
    field: np.ndarray,
    windows: tuple[tuple[float, float], ...],
    q_grid: np.ndarray,
    p_measured: float,
    background_window: tuple[float, float],
    background_result: dict,
) -> dict:
    """Freeze the outer-fit regular background, then rediscover residual q."""
    b = np.asarray(background_result["b"], dtype=float)
    c0 = float(background_result["c0"])
    background = c0 + np.outer(b, r)
    residual_field = field - background

    window_results = []
    for window in windows:
        indices = window_indices(r, window)
        train_local, hold_local = alternating_split(
            np.arange(indices.size))
        rr, residual = r[indices], residual_field[:, indices]

        train_objective = lambda q: pure_residual_fit(
            rr[train_local], residual[:, train_local], q)[1]
        train_curve = np.array([train_objective(q) for q in q_grid])
        q_train, _, train_boundary = refine_grid_minimum(
            q_grid, train_curve, train_objective)
        h_train, _ = pure_residual_fit(
            rr[train_local], residual[:, train_local], q_train)
        error_hold = (
            residual[:, hold_local]
            - np.outer(h_train, rr[hold_local] ** q_train))
        predicted_hold = np.outer(
            h_train, rr[hold_local] ** q_train)

        full_objective = lambda q: pure_residual_fit(rr, residual, q)[1]
        full_curve = np.array([full_objective(q) for q in q_grid])
        q, rss_value, boundary = refine_grid_minimum(
            q_grid, full_curve, full_objective)
        h, _ = pure_residual_fit(rr, residual, q)
        nobs = residual.size
        npar = theta.size
        dof = nobs - npar - 1
        q_unc = profile_uncertainty(
            full_objective, q, rss_value, dof,
            (float(q_grid[0]), float(q_grid[-1])))

        basis = rr ** q
        sigma2 = rss_value / max(nobs - npar, 1)
        h_se = np.full(
            theta.size, math.sqrt(sigma2 / float(basis @ basis)))
        covariance_h = np.diag(h_se * h_se)

        window_results.append({
            "window": list(window),
            "n_radii": int(indices.size),
            "model_after_subtraction": "y1-c0-b(theta)*r=h(theta)*r^q",
            "q_train_selected": q_train,
            "q_train_on_search_boundary": train_boundary,
            "q_full": q,
            "q_on_search_boundary": boundary,
            "q_conditional_uncertainty": q_unc,
            "h": h,
            "residual_fit": {
                "rss": rss_value,
                "rms": math.sqrt(rss_value / nobs),
                "n_observations": int(nobs),
                "n_parameters": int(npar),
                "degrees_of_freedom": int(nobs - npar),
                "condition_scaled_X": 1.0,
                "identifiable_linear_fit": bool(nobs > npar),
                "conditional_h_standard_errors": h_se,
            },
            "Ch_projection": project_ch(
                theta, h, p_measured, covariance_h),
            "radial_holdout": {
                "scheme": (
                    "alternating radii after frozen outer-background "
                    "subtraction"
                ),
                "n_train": int(train_local.size),
                "n_holdout": int(hold_local.size),
                "rmse": rms(error_hold),
                "rmse_over_predicted_residual_component": (
                    rms(error_hold) / rms(predicted_hold)
                    if rms(predicted_hold) > 0.0 else None),
            },
        })

    return {
        "method": (
            "The outer simultaneous discovery fit supplies c0 and b(theta). "
            "Only that regular background is frozen and subtracted; q is then "
            "rediscovered in each window. This is a diagnostic decomposition, "
            "not an independent data set."
        ),
        "background_window": list(background_window),
        "background_q": background_result["q_full"],
        "background_c0": c0,
        "background_b": b,
        "background_Cs_projection": background_result["Cs_projection"],
        "windows": window_results,
    }


def opening_model_fit(
    r: np.ndarray,
    y: np.ndarray,
    indices: np.ndarray,
    powers: tuple[float, ...],
    *,
    covariance: bool,
) -> dict:
    rr = r[indices]
    design = np.column_stack([rr ** power for power in powers])
    return scaled_linear_fit(design, y[indices], covariance=covariance)


def opening_diagnostics(
    r: np.ndarray,
    y_face: np.ndarray,
    windows: tuple[tuple[float, float], ...],
    rho: float,
) -> list[dict]:
    results = []
    for window in windows:
        indices = window_indices(r, window)
        train, hold = alternating_split(indices)
        models = {}
        for label, powers in (
            ("sqrt_plus_r", (0.5, 1.0)),
            ("sqrt_plus_r_plus_r3over2", (0.5, 1.0, 1.5)),
        ):
            fit = opening_model_fit(
                r, y_face, indices, powers, covariance=True)
            train_fit = opening_model_fit(
                r, y_face, train, powers, covariance=False)
            prediction = sum(
                coefficient * r[hold] ** power
                for coefficient, power in zip(train_fit["coef"], powers))
            error = y_face[hold] - prediction
            correction = sum(
                coefficient * r[hold] ** power
                for coefficient, power in zip(
                    train_fit["coef"][1:], powers[1:]))
            se = fit["conditional_standard_errors"]
            models[label] = {
                "powers": list(powers),
                "coefficients": fit["coef"],
                "conditional_standard_errors": se,
                "P_sqrt_coefficient": fit["coef"][0],
                "C_r_coefficient": fit["coef"][1],
                "linear_fit": public_linear_fit(fit),
                "radial_holdout": {
                    "scheme": "alternating radii",
                    "rmse": rms(error),
                    "rmse_over_predicted_correction": (
                        rms(error) / rms(correction)
                        if rms(correction) > 0.0 else None),
                },
            }
        formal_f0 = -(2.0 / 3.0) * rho
        models["sqrt_plus_r"]["formal_F0_difference"] = (
            models["sqrt_plus_r"]["C_r_coefficient"] - formal_f0)
        results.append({
            "window": list(window),
            "n_radii": int(indices.size),
            "scope": (
                "diagnostic only; not a direct validation of the weighted-7/4 "
                "F=Cs=0 rung in this nonzero-Cs strip"
            ),
            "formal_F0_C_r": formal_f0,
            "models": models,
        })
    return results


def face_admissibility(y1: np.ndarray, y2: np.ndarray) -> dict:
    points = np.column_stack((y1, y2))

    def orientation(a, b, c):
        return (
            (b[0] - a[0]) * (c[1] - a[1])
            - (b[1] - a[1]) * (c[0] - a[0])
        )

    crossings = 0
    for left in range(points.shape[0] - 1):
        for right in range(left + 2, points.shape[0] - 1):
            a, b = points[left], points[left + 1]
            c, d = points[right], points[right + 1]
            p = orientation(c, d, a)
            q = orientation(c, d, b)
            s = orientation(a, b, c)
            t = orientation(a, b, d)
            if p * q < 0.0 and s * t < 0.0:
                crossings += 1
    monotone_opening = bool(np.all(np.diff(y2) > 0.0))
    return {
        "sampled_face_opening_monotone": monotone_opening,
        "sampled_face_self_intersections": crossings,
        "sampled_face_pass": bool(monotone_opening and crossings == 0),
        "scope": (
            "sampled P2 face trace only; not a global orientation, contact, "
            "or injectivity certificate"
        ),
    }


def load_inputs(profile_path: Path, metadata_path: Path) -> dict:
    with np.load(profile_path) as stored:
        profile = {key: np.asarray(stored[key]) for key in stored.files}
    metadata = json.loads(metadata_path.read_text())

    required = ("theta_deg", "r", "Y1", "Y2", "valid")
    missing = [key for key in required if key not in profile]
    if missing:
        raise ValueError(f"profile is missing keys: {missing}")
    theta_deg = np.asarray(profile["theta_deg"], dtype=float)
    r = np.asarray(profile["r"], dtype=float)
    y1 = np.asarray(profile["Y1"], dtype=float)
    y2 = np.asarray(profile["Y2"], dtype=float)
    valid = np.asarray(profile["valid"], dtype=bool)
    expected_shape = (theta_deg.size, r.size)
    if y1.shape != expected_shape or y2.shape != expected_shape:
        raise ValueError("Y1/Y2 shapes do not match theta and radius arrays")
    if valid.shape != expected_shape or not np.all(valid):
        raise ValueError("profile contains invalid or missing live-P2 samples")
    if not np.all(np.diff(theta_deg) > 0.0):
        raise ValueError("theta array is not strictly increasing")
    if not np.all(np.diff(r) > 0.0):
        raise ValueError("radius array is not strictly increasing")
    if not (np.isclose(theta_deg[0], 0.0)
            and np.isclose(theta_deg[-1], 180.0)):
        raise ValueError("profile must include exact 0 and 180 degree traces")

    p_measured = float(metadata["energy_release"]["P_measured"])
    c1 = float(profile.get("c1", metadata["case"]["c1"]))
    c2 = float(profile.get("c2", metadata["case"]["c2"]))
    if not (p_measured > 0.0 and c1 > 0.0):
        raise ValueError("P and c1 must be positive")

    return {
        "theta_deg": theta_deg,
        "theta": np.deg2rad(theta_deg),
        "r": r,
        "Y1": y1,
        "Y2": y2,
        "metadata": metadata,
        "P_measured": p_measured,
        "c1": c1,
        "c2": c2,
        "profile_metadata": {
            key: jsonable(profile[key])
            for key in (
                "lam_target", "lam_reached", "r_min", "n_r_mesh",
                "n_theta_mesh", "displacement_degree",
                "core_treatment", "sampling_scheme")
            if key in profile
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--window", type=parse_window, action="append",
        help="analysis window LO:HI; repeat for multiple windows")
    parser.add_argument(
        "--background-window", type=parse_window,
        default=(1.6e-3, 1.6e-2))
    parser.add_argument("--q-min", type=float, default=0.8)
    parser.add_argument("--q-max", type=float, default=1.55)
    parser.add_argument("--q-step", type=float, default=0.005)
    parser.add_argument("--axis-next-power", type=float, default=1.75)
    parser.add_argument("--angular-holdout-stride", type=int, default=5)
    args = parser.parse_args()

    profile_path = args.profile.resolve()
    metadata_path = args.metadata.resolve()
    output_path = args.output.resolve()
    if not profile_path.is_file() or not metadata_path.is_file():
        raise SystemExit("profile and metadata paths must be existing files")
    if not (0.0 < args.q_min < args.q_max < args.axis_next_power):
        raise SystemExit(
            "require 0 < q-min < q-max < axis-next-power")
    if args.q_step <= 0.0:
        raise SystemExit("q-step must be positive")
    if args.angular_holdout_stride < 3:
        raise SystemExit("angular-holdout-stride must be at least 3")

    loaded = load_inputs(profile_path, metadata_path)
    checks = internal_checks()
    r = loaded["r"]
    windows = tuple(args.window or DEFAULT_WINDOWS)
    for window in (*windows, args.background_window):
        window_indices(r, window)
    q_grid = np.arange(
        args.q_min, args.q_max + 0.5 * args.q_step, args.q_step)

    axis = exact_axis_analysis(
        r, loaded["Y1"][0], windows, q_grid,
        args.axis_next_power, loaded["P_measured"])
    discovery = full_angle_discovery(
        loaded["theta"], r, loaded["Y1"], windows, q_grid,
        loaded["P_measured"], args.angular_holdout_stride)

    background_index = next(
        (
            index for index, item in enumerate(discovery)
            if np.allclose(item["window"], args.background_window)
        ),
        None,
    )
    if background_index is None:
        background_extra = full_angle_discovery(
            loaded["theta"], r, loaded["Y1"],
            (args.background_window,), q_grid, loaded["P_measured"],
            args.angular_holdout_stride)[0]
    else:
        background_extra = discovery[background_index]

    background = background_subtracted_discovery(
        loaded["theta"], r, loaded["Y1"], windows, q_grid,
        loaded["P_measured"], args.background_window, background_extra)
    opening = opening_diagnostics(
        r, loaded["Y2"][-1], windows, loaded["c2"] / loaded["c1"])

    record = {
        "analysis": {
            "schema": SCHEMA,
            "protocol_date": PROTOCOL_DATE,
            "generated_by": "fem/pure_shear/fit_r54_campaign_stage0.py",
            "purpose": (
                "Deterministic exact-axis and full-angle estimator for the "
                "bounded FEM convergence campaign"
            ),
            "condition_number_definition": (
                "condition number of column-scaled X, not X.T X"
            ),
            "formal_uncertainty_scope": (
                "conditional least-squares diagnostics only; mesh, model, and "
                "correlated-sample uncertainty require the planned convergence "
                "campaign"
            ),
            "q_search": {
                "lower": args.q_min,
                "upper": args.q_max,
                "step": args.q_step,
            },
            "windows": [list(window) for window in windows],
            "background_window": list(args.background_window),
            "axis_next_power": args.axis_next_power,
            "angular_holdout_stride": args.angular_holdout_stride,
            "g_evaluation_audit": {
                "exact_hypergeometric_g_pi": float(
                    exact_g(np.array([math.pi]))[0]),
                "legacy_200_term_series_g_pi": legacy_truncated_g_pi(200),
                "legacy_relative_face_error": (
                    legacy_truncated_g_pi(200)
                    / float(exact_g(np.array([math.pi]))[0]) - 1.0
                ),
                "finding": (
                    "The former 200-term endpoint series is not converged. "
                    "All Stage-0 branch projections use the exact "
                    "hypergeometric evaluation."
                ),
            },
        },
        "verification": {
            "internal_checks": checks,
            "all_internal_checks_passed": True,
        },
        "inputs": {
            "profile": portable_path(profile_path),
            "profile_sha256": sha256(profile_path),
            "metadata": portable_path(metadata_path),
            "metadata_sha256": sha256(metadata_path),
            "profile_metadata": loaded["profile_metadata"],
            "P_measured": loaded["P_measured"],
            "c1": loaded["c1"],
            "c2": loaded["c2"],
            "n_theta_samples": int(loaded["theta"].size),
            "n_radius_samples": int(r.size),
            "radius_range": [float(r[0]), float(r[-1])],
        },
        "admissibility": face_admissibility(
            loaded["Y1"][-1], loaded["Y2"][-1]),
        "exact_axis": axis,
        "simultaneous_full_angle_discovery": discovery,
        "background_subtracted_discovery": background,
        "opening_r_coefficient_diagnostic": opening,
        "interpretation_contract": {
            "exponent": (
                "Convergence of exact-axis and target-free full-angle q toward "
                "5/4 tests realization of the analytical asymptotic class."
            ),
            "branch": (
                "A converged nonzero Ch rejects the selected Ch=0 analytic-axis "
                "member, not the 5/4 exponent or general angular ODE family."
            ),
            "opening": (
                "The fitted order-r opening coefficient is diagnostic only. "
                "It is not labeled a direct validation of the closed F=Cs=0 "
                "weighted-7/4 rung."
            ),
        },
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(jsonable(record), indent=2, sort_keys=True) + "\n")

    print(f"wrote {output_path}")
    print("exact-axis q:")
    for item in axis:
        print(
            f"  {item['window'][0]:.2e}..{item['window'][1]:.2e}: "
            f"{item['q_full']:.5f} "
            f"(holdout/residual "
            f"{item['radial_holdout']['rmse_over_predicted_rq_component']:.3g})"
        )
    print("full-angle q / Cs / Ch:")
    for item in discovery:
        print(
            f"  {item['window'][0]:.2e}..{item['window'][1]:.2e}: "
            f"q={item['q_full']:.5f}, "
            f"Cs={item['Cs_projection']['Cs']:.5f}, "
            f"Ch={item['Ch_projection']['Ch']:.5f}, "
            f"cond(X)={item['linear_fit']['condition_scaled_X']:.3e}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
