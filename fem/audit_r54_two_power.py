"""Free two-power exact-axis audit for the finest global--local profile.

The production estimator represents the next analytical nuisance grade by a
fixed ``r^(7/4)`` column. This independent check lets both the target exponent
and the nuisance exponent vary. It uses alternating radii for fitting and
holdout, so the observed ``5/4`` value is not imposed by fixing the next
power.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.optimize import differential_evolution


DEFAULT_WINDOWS = (
    (6.0e-4, 4.5e-3),
    (6.0e-4, 5.0e-3),
    (6.0e-4, 6.0e-3),
    (8.0e-4, 6.0e-3),
    (1.0e-3, 6.0e-3),
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def portable(path: Path) -> str:
    resolved = path.resolve()
    repo = Path(__file__).resolve().parents[1]
    try:
        return str(resolved.relative_to(repo))
    except ValueError:
        return str(resolved)


def fit_window(
        r: np.ndarray,
        y: np.ndarray,
        window: tuple[float, float],
        amplitude_prediction: float) -> dict:
    indices = np.flatnonzero((r >= window[0]) & (r <= window[1]))
    train = indices[::2]
    holdout = indices[1::2]
    if train.size < 8 or holdout.size < 8:
        raise ValueError(f"too few samples in window {window}")

    def linear_coefficients(q: float, p_next: float):
        X = np.column_stack((
            np.ones(train.size),
            r[train] ** q,
            r[train] ** p_next,
        ))
        coefficients, _, _, _ = np.linalg.lstsq(
            X, y[train], rcond=None)
        return coefficients

    def objective(exponents):
        q, p_next = exponents
        if p_next < q + 0.08:
            return 1.0
        coefficients = linear_coefficients(q, p_next)
        prediction = (
            coefficients[0]
            + coefficients[1] * r[train] ** q
            + coefficients[2] * r[train] ** p_next
        )
        return float(np.sum((y[train] - prediction) ** 2))

    optimized = differential_evolution(
        objective,
        bounds=((1.0, 1.4), (1.4, 2.5)),
        seed=4,
        tol=1.0e-9,
        polish=True,
        workers=1,
        updating="immediate",
    )
    if not optimized.success:
        raise RuntimeError(
            f"two-power optimization failed in {window}: "
            f"{optimized.message}")
    q, p_next = (float(value) for value in optimized.x)
    c0, amplitude, nuisance = (
        float(value) for value in linear_coefficients(q, p_next))
    prediction = (
        c0
        + amplitude * r[holdout] ** q
        + nuisance * r[holdout] ** p_next
    )
    holdout_rmse = float(np.sqrt(np.mean(
        (y[holdout] - prediction) ** 2)))
    target_component_rms = float(np.sqrt(np.mean(
        (amplitude * r[holdout] ** q) ** 2)))

    return {
        "window": list(window),
        "n_train": int(train.size),
        "n_holdout": int(holdout.size),
        "q": q,
        "p_next": p_next,
        "c0": c0,
        "A": amplitude,
        "D": nuisance,
        "A_predicted_from_measured_P": amplitude_prediction,
        "A_relative_error":
            float(amplitude / amplitude_prediction - 1.0),
        "holdout_rmse": holdout_rmse,
        "holdout_rmse_over_target_component":
            float(holdout_rmse / target_component_rms),
        "optimizer_train_rss": float(optimized.fun),
        "optimizer_iterations": int(optimized.nit),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    with np.load(args.profile, allow_pickle=False) as profile:
        r = np.asarray(profile["r"], dtype=float)
        y_axis = np.asarray(profile["Y1"][0], dtype=float)
    metadata = json.loads(args.metadata.read_text())
    measured_P = float(metadata["polar_profile"]["P_measured"])
    amplitude_prediction = (
        4.0 * np.sqrt(2.0) / 5.0 / np.sqrt(measured_P))

    fits = [
        fit_window(r, y_axis, window, amplitude_prediction)
        for window in DEFAULT_WINDOWS
    ]
    record = {
        "schema": "r54-axis-free-two-power-v1",
        "protocol_date": "2026-07-23",
        "generated_by": "fem/audit_r54_two_power.py",
        "purpose": (
            "Test whether the exact-axis 5/4 exponent survives when the "
            "next radial power is inferred rather than fixed at 7/4."
        ),
        "inputs": {
            "profile": portable(args.profile),
            "profile_sha256": sha256(args.profile),
            "metadata": portable(args.metadata),
            "metadata_sha256": sha256(args.metadata),
            "P_measured": measured_P,
        },
        "model": "y_axis=c0+A*r^q+D*r^p_next",
        "exponent_bounds": {
            "q": [1.0, 1.4],
            "p_next": [1.4, 2.5],
            "minimum_separation": 0.08,
        },
        "split": (
            "alternating radii within each window; exponents and "
            "coefficients fitted on the first fold and evaluated on the "
            "second"
        ),
        "optimizer": {
            "method": "scipy.optimize.differential_evolution",
            "seed": 4,
            "tol": 1.0e-9,
            "polish": True,
        },
        "fits": fits,
        "checks": {
            "all_q_within_0.002_of_1.25":
                all(abs(item["q"] - 1.25) < 0.002 for item in fits),
            "all_next_powers_between_1.68_and_1.76":
                all(1.68 < item["p_next"] < 1.76 for item in fits),
            "all_holdout_relative_errors_below_5e-5":
                all(
                    item["holdout_rmse_over_target_component"] < 5.0e-5
                    for item in fits),
        },
        "interpretation": (
            "The data recover q near 5/4 and a next power near 7/4 without "
            "fixing either exponent. This supports the exact-axis asymptotic "
            "class but does not select the full-angle Ch=0 branch."
        ),
    }
    if not all(record["checks"].values()):
        raise RuntimeError(f"two-power audit failed: {record['checks']}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(record, indent=2, sort_keys=True) + "\n")
    print(f"wrote {args.output.resolve()}")
    for item in fits:
        print(
            f"{item['window'][0]:.1e}..{item['window'][1]:.1e}: "
            f"q={item['q']:.8f}, p={item['p_next']:.8f}, "
            f"A_ax/A_ax_pred-1={item['A_relative_error']:.3e}, "
            "holdout/target="
            f"{item['holdout_rmse_over_target_component']:.3e}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
