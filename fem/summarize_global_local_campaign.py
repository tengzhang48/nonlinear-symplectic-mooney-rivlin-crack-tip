"""Build the compact audit summary for the bounded global--local campaign."""
from __future__ import annotations

import argparse
from datetime import date
import hashlib
import json
from pathlib import Path
import subprocess


def _read(path: Path):
    return json.loads(path.read_text())


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _git_provenance(repo: Path) -> dict:
    """Record the repository state of this summary, not the archived run."""
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo,
            check=True,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=repo,
            check=True,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        ).stdout
    except (OSError, subprocess.CalledProcessError):
        return {
            "summary_git_commit": None,
            "summary_worktree_was_dirty": None,
        }
    return {
        "summary_git_commit": commit,
        "summary_worktree_was_dirty": bool(status.strip()),
    }


def _solver_record(metadata: dict) -> dict:
    return {
        "r_min": metadata["strip"]["r_min"],
        "matching_radius": metadata["strip"]["matching_radius"],
        "n_r": metadata["strip"]["n_r"],
        "n_theta": metadata["strip"]["n_theta"],
        "mpi_ranks": metadata["mpi_ranks"],
        "strip_cells": metadata["strip"]["cells"],
        "local_cells": metadata["local"]["cells"],
        "local_direct": metadata["local"]["direct"]["converged"],
        "local_direct_iterations":
            metadata["local"]["direct"]["iterations"],
        "local_used_fallback":
            metadata["local"]["used_continuation_fallback"],
        "minimum_J": metadata["local"]["final_min_J"],
        "trace_max_error": metadata["local"]["outer_trace_max_error"],
        "manufactured_P2_transfer_max_error":
            metadata["manufactured_P2_transfer_max_error"],
        "relative_weak_reaction_defect":
            metadata["interface_reaction"]["relative_reaction_defect"],
        "reaction_norm_scope":
            metadata["interface_reaction"]["norm_scope"],
        "P_face": metadata["polar_profile"]["P_measured"],
        "P_from_strip_energy":
            metadata["energy_release"]["P_pred_from_G_spec"],
        "profile_samples": (
            metadata["polar_profile"]["n_r"]
            * metadata["polar_profile"]["n_theta"]),
        "all_profile_samples_valid":
            metadata["polar_profile"]["all_points_valid"],
    }


def _estimator_record(estimator: dict) -> dict:
    return {
        "windows": [item["window"] for item in estimator["exact_axis"]],
        "axis_q": [item["q_full"] for item in estimator["exact_axis"]],
        "axis_amplitude_relative_error": [
            item["A_relative_error"] for item in estimator["exact_axis"]
        ],
        "axis_holdout_relative_error": [
            item["radial_holdout"][
                "rmse_over_predicted_rq_component"]
            for item in estimator["exact_axis"]
        ],
        "background_Cs":
            estimator["background_subtracted_discovery"][
                "background_Cs_projection"]["Cs"],
        "background_Cs_shape_relative_L2_error":
            estimator["background_subtracted_discovery"][
                "background_Cs_projection"]["relative_L2_shape_error"],
        "background_subtracted_q": [
            item["q_full"]
            for item in estimator[
                "background_subtracted_discovery"]["windows"]
        ],
        "background_subtracted_Ch": [
            item["Ch_projection"]["Ch"]
            for item in estimator[
                "background_subtracted_discovery"]["windows"]
        ],
        "background_subtracted_angular_relative_L2_error": [
            item["Ch_projection"]["relative_L2_profile_error"]
            for item in estimator[
                "background_subtracted_discovery"]["windows"]
        ],
        "background_subtracted_radial_holdout_relative_error": [
            item["radial_holdout"][
                "rmse_over_predicted_residual_component"]
            for item in estimator[
                "background_subtracted_discovery"]["windows"]
        ],
        "simultaneous_full_angle_q": [
            item["q_full"]
            for item in estimator["simultaneous_full_angle_discovery"]
        ],
        "simultaneous_full_angle_Cs": [
            item["Cs_projection"]["Cs"]
            for item in estimator["simultaneous_full_angle_discovery"]
        ],
        "simultaneous_full_angle_Ch": [
            item["Ch_projection"]["Ch"]
            for item in estimator["simultaneous_full_angle_discovery"]
        ],
    }


def _case(input_dir: Path, metadata_name: str, estimator_name: str) -> dict:
    metadata_path = input_dir / metadata_name
    estimator_path = input_dir / estimator_name
    estimator = _read(estimator_path)
    metadata_hash = _sha256(metadata_path)
    recorded_metadata_hash = estimator["inputs"]["metadata_sha256"]
    if metadata_hash != recorded_metadata_hash:
        raise ValueError(
            f"metadata hash mismatch for {metadata_path}: "
            f"{metadata_hash} != {recorded_metadata_hash}")
    profile_name = Path(estimator["inputs"]["profile"]).name
    profile_path = input_dir / profile_name
    recorded_profile_hash = estimator["inputs"]["profile_sha256"]
    actual_profile_hash = _sha256(profile_path)
    if actual_profile_hash != recorded_profile_hash:
        raise ValueError(
            f"profile hash mismatch for {profile_path}: "
            f"{actual_profile_hash} != {recorded_profile_hash}")
    return {
        "solver": _solver_record(_read(metadata_path)),
        "estimator": _estimator_record(estimator),
        "source_files": {
            "metadata": metadata_name,
            "metadata_sha256": metadata_hash,
            "profile": profile_name,
            "profile_sha256": actual_profile_hash,
            "estimator": estimator_name,
            "estimator_sha256": _sha256(estimator_path),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-dir", type=Path, default=Path("fem/global_local_outputs"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source = args.input_dir.resolve()

    matching_cases = {
        "Rm_0.005": (
            "global_local_lam16_rm5e3_restricted_profile.json",
            "r54_lam16_rm5e3_restricted_profile.json",
        ),
        "Rm_0.01": (
            "global_local_lam16_rm1e2_restricted_profile.json",
            "r54_lam16_rm1e2_restricted_common_windows.json",
        ),
        "Rm_0.02": (
            "global_local_lam16_rm2e2_restricted_profile.json",
            "r54_lam16_rm2e2_restricted_common_windows.json",
        ),
    }
    core_cases = {
        "core_1e-5_ntheta60": matching_cases["Rm_0.01"],
        "core_5e-6_ntheta60": (
            "global_local_lam16_core5e6_rm1e2.json",
            "r54_lam16_core5e6_rm1e2.json",
        ),
        "core_2.5e-6_ntheta60": (
            "global_local_lam16_core2p5e6_rm1e2.json",
            "r54_lam16_core2p5e6_rm1e2.json",
        ),
        "core_2.5e-6_ntheta120": (
            "global_local_lam16_core2p5e6_nt120_rm1e2.json",
            "r54_lam16_core2p5e6_nt120_rm1e2.json",
        ),
    }
    refined_metadata_path = (
        source / "global_local_lam16_rm1e2_refine1_profile.json")
    refined = _read(refined_metadata_path)
    two_power_path = source / "r54_axis_two_power_sensitivity.json"
    two_power = _read(two_power_path)
    for source_kind in ("profile", "metadata"):
        source_path = source / Path(
            two_power["inputs"][source_kind]).name
        actual_hash = _sha256(source_path)
        recorded_hash = two_power["inputs"][
            f"{source_kind}_sha256"]
        if actual_hash != recorded_hash:
            raise ValueError(
                f"two-power {source_kind} hash mismatch for {source_path}: "
                f"{actual_hash} != {recorded_hash}")
    matching_records = {
        name: _case(source, *filenames)
        for name, filenames in matching_cases.items()
    }
    core_records = {
        name: _case(source, *filenames)
        for name, filenames in core_cases.items()
    }
    matching_q = [
        record["estimator"]["axis_q"]
        for record in matching_records.values()
    ]
    matching_q_span = [
        max(values) - min(values)
        for values in zip(*matching_q)
    ]
    core_q = [
        record["estimator"]["axis_q"]
        for record in core_records.values()
    ]
    core_q_error = [
        [abs(q - 1.25) for q in values]
        for values in core_q
    ]
    angular_delta = [
        abs(q120 - q60)
        for q60, q120 in zip(
            core_records["core_2.5e-6_ntheta60"]["estimator"]["axis_q"],
            core_records["core_2.5e-6_ntheta120"]["estimator"]["axis_q"],
        )
    ]

    repo = Path(__file__).resolve().parents[1]
    finest_metadata = _read(
        source / "global_local_lam16_core2p5e6_nt120_rm1e2.json")
    report = {
        "schema": "global-local-campaign-summary-v1",
        "generated_by": "fem/summarize_global_local_campaign.py",
        "generated_on": date.today().isoformat(),
        "problem": {
            "material": "reduced incompressible plane-stress Mooney-Rivlin",
            "c1": 1.0,
            "c2": 1.0,
            "pure_shear_lambda": 1.6,
            "method": (
                "one-way global strip to exact/nested local crack-tip "
                "submodel with a shared P2 interface"),
        },
        "run_provenance": {
            **_git_provenance(repo),
            "source_records_have_runtime_metadata":
                "runtime" in finest_metadata,
            "solver_runtime": finest_metadata.get("runtime"),
            "scope": (
                "This block describes the current summary invocation and "
                "the runtime stored by the input solver records. It does not "
                "reuse provenance from the archived 2026-07-23 campaign."
            ),
        },
        "matching_radius_cases": matching_records,
        "core_and_angular_cases": core_records,
        "derived_robustness_checks": {
            "matching_radius_axis_q_span_by_window": matching_q_span,
            "axis_absolute_error_from_1.25_by_core_case": {
                name: values
                for name, values in zip(core_records, core_q_error)
            },
            "axis_q_change_ntheta60_to_ntheta120_at_smallest_core":
                angular_delta,
            "smallest_core_widest_window": {
                "axis_q":
                    core_records["core_2.5e-6_ntheta120"][
                        "estimator"]["axis_q"][-1],
                "axis_amplitude_relative_error":
                    core_records["core_2.5e-6_ntheta120"][
                        "estimator"][
                            "axis_amplitude_relative_error"][-1],
                "background_subtracted_q":
                    core_records["core_2.5e-6_ntheta120"][
                        "estimator"]["background_subtracted_q"][-1],
                "background_subtracted_angular_relative_L2_error":
                    core_records["core_2.5e-6_ntheta120"][
                        "estimator"][
                            "background_subtracted_angular_relative_L2_error"
                        ][-1],
            },
        },
        "free_two_power_axis_check": {
            "model": two_power["model"],
            "q": [item["q"] for item in two_power["fits"]],
            "p_next": [
                item["p_next"] for item in two_power["fits"]],
            "windows": [
                item["window"] for item in two_power["fits"]],
            "amplitude_relative_error": [
                item["A_relative_error"]
                for item in two_power["fits"]],
            "holdout_relative_error": [
                item["holdout_rmse_over_target_component"]
                for item in two_power["fits"]],
            "checks": two_power["checks"],
            "source_file": two_power_path.name,
            "source_sha256": _sha256(two_power_path),
        },
        "separately_refined_local_case": {
            "metadata": refined_metadata_path.name,
            "metadata_sha256": _sha256(refined_metadata_path),
            "restricted_reaction_defect":
                refined["interface_reaction"][
                    "relative_reaction_defect"],
            "refined_cells": refined["fine_local"]["cells"],
            "refined_minimum_J": refined["fine_local"]["final_min_J"],
            "refined_used_fallback":
                refined["fine_local"]["used_continuation_fallback"],
            "refined_reaction_defect":
                refined["fine_local"]["interface_reaction"][
                    "relative_reaction_defect"],
        },
        "findings": {
            "five_quarters": (
                "Supported numerically as an asymptotic class by matching-"
                "radius independence, systematic core convergence, angular "
                "convergence, and exact-axis amplitude convergence. The "
                "free two-power audit recovers both q near 5/4 and the next "
                "power near 7/4 without fixing either exponent. The "
                "background-subtracted full-angle exponent approaches the "
                "same value, and its angular profile is consistent with the "
                "ODE family after admitting the homogeneous Ch member."
            ),
            "Cs": (
                "Specimen-specific and approximately -3.2 to -3.3 at the "
                "smallest core, but a strict radial-window plateau is not "
                "established."
            ),
            "Ch": (
                "Fits consistently favor a nonzero value near 1.2 in the "
                "broader windows, but core/window convergence is insufficient "
                "for a selected-coefficient claim."
            ),
            "one_way_scope": (
                "The exact restricted submodel balances weak interface "
                "reactions to solver tolerance. Independent local refinement "
                "has a roughly two-percent P2 reaction-coefficient defect in "
                "the stated Euclidean norm and is not a two-way coupled "
                "equilibrium."
            ),
        },
        "stopping_decision": (
            "No further solve is required for the numerical support of the "
            "5/4 exponent. Further work is needed only if the paper elects to "
            "claim selected Cs or Ch rather than report them as provisional "
            "specimen-level estimates."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(f"wrote {args.output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
