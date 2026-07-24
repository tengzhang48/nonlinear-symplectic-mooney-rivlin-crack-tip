#!/usr/bin/env python3
"""Deterministic gates for the principal claims backed by stored FEM data."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import subprocess
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
STRIP = ROOT / "data" / "fem" / "strip"
SUMMARY = STRIP / "summary.csv"
LEDGER = ROOT / "data" / "claims" / "principal_claims.json"
STRIP_MANIFEST = STRIP / "ARTIFACTS.sha256"
FIGURES = ROOT / "figures" / "rendered"
GLOBAL_LOCAL = ROOT / "data" / "fem" / "global_local"
GLOBAL_LOCAL_SUMMARY = (
    GLOBAL_LOCAL / "global_local_campaign_summary_2026-07-23.json"
)


def gate(name: str, condition: bool, detail: str) -> None:
    print(f"[{'PASS' if condition else 'FAIL'}] {name}: {detail}")
    if not condition:
        raise AssertionError(name)


def corner_aligned_geometry(a: float, b: float, H: float, r_min: float,
                            n_r: int, n_sectors: int,
                            corners: list[float]) -> tuple[np.ndarray, np.ndarray]:
    """Rebuild the stored corner-snapped geometry without importing FEniCSx."""
    theta = np.linspace(0.0, np.pi, n_sectors + 1)
    available = set(range(1, n_sectors))
    for corner in corners:
        index = min(available,
                    key=lambda candidate: (abs(theta[candidate] - corner),
                                           candidate))
        theta[index] = corner
        available.remove(index)
    theta.sort()

    radial_extent = []
    for angle in theta:
        cosine, sine = math.cos(angle), math.sin(angle)
        candidates = []
        if sine > 1e-12:
            candidates.append(H / sine)
        if cosine > 1e-12:
            candidates.append(b / cosine)
        elif cosine < -1e-12:
            candidates.append(a / -cosine)
        radial_extent.append(min(candidates))
    radial_extent = np.asarray(radial_extent)
    fraction = np.arange(n_r + 1)[:, None] / n_r
    radius = r_min * (radial_extent[None, :] / r_min) ** fraction
    coords = np.column_stack([
        (radius * np.cos(theta)[None, :]).ravel(),
        (radius * np.sin(theta)[None, :]).ravel(),
    ])
    return theta, coords


def radial_connectivity(n_r: int, n_sectors: int) -> np.ndarray:
    """Connectivity written by ``ps_mesh.build_strip`` before DOLFINx reorders."""
    n_angles = n_sectors + 1
    cells = []
    for ring in range(n_r):
        for sector in range(n_sectors):
            p00 = ring * n_angles + sector
            p10 = (ring + 1) * n_angles + sector
            p11 = (ring + 1) * n_angles + sector + 1
            p01 = ring * n_angles + sector + 1
            cells.extend(((p00, p10, p11), (p00, p11, p01)))
    return np.asarray(cells, dtype=np.int64)


def canonical_triangles(cells: np.ndarray) -> np.ndarray:
    """Sort vertices and rows so connectivity comparisons ignore ordering."""
    canonical = np.sort(cells, axis=1)
    order = np.lexsort((canonical[:, 2], canonical[:, 1], canonical[:, 0]))
    return canonical[order]


def main() -> None:
    ledger = json.loads(LEDGER.read_text(encoding="utf-8"))
    gate("claims ledger schema", ledger.get("schema_version") == 1,
         f"{len(ledger.get('claims', []))} claims")
    claims = {claim["id"]: claim for claim in ledger.get("claims", [])}
    leading_claim = claims.get("leading-map", {})
    leading_statement = leading_claim.get("statement", "")
    gate(
        "leading-map scope",
        (leading_claim.get("status") == "formal-leading-constrained-map"
         and "formal leading constrained/null-family map" in leading_statement
         and "full retained-C_s finite-compliance equilibrium branch" in leading_statement
         and "remain open" in leading_statement),
        "formal constrained/null-family map; retained-C_s branch remains open",
    )
    energy_claim = claims.get("energy-release", {})
    energy_statement = energy_claim.get("statement", "")
    gate(
        "C_s-inclusive energy-flux scope",
        (energy_claim.get("status")
         == "established-on-superposed-truncated-map"
         and "superposed truncated map" in energy_statement
         and "leading contour flux is exactly independent of arbitrary C_s"
         in energy_statement
         and "not a completed retained-C_s finite-compliance equilibrium branch"
         in energy_statement
         and "including arbitrary C_s in the retained constrained map"
         not in energy_statement),
        "exact leading-flux result on truncated map, not a completed branch",
    )
    expected_figure_stems = {
        "fig_master",
        "fig_asymap",
        "fig_chain",
        "fig_ps_portrait",
        "fig_plateau",
        "fig_cratio",
        "fig_r54_axis_convergence",
        "fig_esi_mesh",
    }
    figure_pdfs = sorted(FIGURES.glob("*.pdf"))
    figure_pngs = sorted(FIGURES.glob("*.png"))
    type3_pdfs = [
        path.name for path in figure_pdfs
        if b"/Subtype /Type3" in path.read_bytes()
    ]
    gate(
        "current figure inventory and publisher font preflight",
        ({path.stem for path in figure_pdfs} == expected_figure_stems
         and {path.stem for path in figure_pngs} == expected_figure_stems
         and not type3_pdfs),
        f"{len(figure_pdfs)} PDF/PNG pairs; no embedded Type 3 fonts",
    )

    residual_claim = claims.get("five-quarters-fem", {})
    residual_statement = residual_claim.get("statement", "")
    gate(
        "five-quarters FEM scope",
        (residual_claim.get("status")
         == "supported-for-tested-strip"
         and "c1=c2=1" in residual_statement
         and "lambda=1.6" in residual_statement
         and "does not select C_s or C_h" in residual_statement
         and "is not a two-way coupled global-local calculation"
         in residual_statement),
        "tested-strip support without coefficient or two-way claims",
    )
    first_material_claim = claims.get("first-material-rung", {})
    first_material_statement = first_material_claim.get("statement", "")
    gate(
        "first-material rung scope",
        (first_material_claim.get("status") == "closed-on-selected-base"
         and "chosen F=0 representative" in first_material_statement
         and "Lambda=7/4" in first_material_statement
         and "total face traction closes" in first_material_statement
         and "no new local amplitude" in first_material_statement),
        "closed Lambda=7/4 rung on the selected base",
    )
    resonance_claim = claims.get("restricted-opening-resonance", {})
    resonance_statement = resonance_claim.get("statement", "")
    gate(
        "restricted opening-resonance scope",
        (resonance_claim.get("status") == "restricted-formal-coefficient"
         and "chosen F=0 (C_s=0) analytic-axis representative"
         in resonance_statement
         and "restricted interaction" in resonance_statement
         and "Lambda=13/4" in resonance_statement
         and "logarithmic companion" in resonance_statement
         and "complete same-grade source, coupled response, and "
             "specimen-selected net logarithmic amplitude remain open"
         in resonance_statement),
        "restricted Lambda=13/4 coefficient on the selected base",
    )
    global_local = json.loads(
        GLOBAL_LOCAL_SUMMARY.read_text(encoding="utf-8"))
    finest = global_local["derived_robustness_checks"][
        "smallest_core_widest_window"]
    gate(
        "five-quarters archived campaign values",
        (math.isclose(finest["axis_q"], 1.2515292964993656,
                      rel_tol=0.0, abs_tol=5e-13)
         and math.isclose(
             finest["axis_amplitude_relative_error"],
             0.01241967751933517,
             rel_tol=0.0, abs_tol=5e-13)
         and math.isclose(
             finest["background_subtracted_q"],
             1.2534925101011027,
             rel_tol=0.0, abs_tol=5e-13)),
        "q=1.251529, amplitude error=1.242%, full-angle q=1.253493",
    )

    with SUMMARY.open(newline="", encoding="utf-8") as stream:
        rows = {row["tag"]: row for row in csv.DictReader(stream)}
    gate("stored strip sweep count", len(rows) == 14, f"{len(rows)} cases")

    records = {
        path.stem.removeprefix("ps_"): json.loads(path.read_text(encoding="utf-8"))
        for path in sorted(STRIP.glob("ps_*.json"))
    }
    expected_specs = {
        "MR_lam13": (1.0, 1.0, 1.3, 3.0, 6.0, 64, 18),
        "MR_lam15": (1.0, 1.0, 1.5, 3.0, 6.0, 64, 18),
        "MR_lam16": (1.0, 1.0, 1.6, 3.0, 6.0, 64, 18),
        "MR_lam18": (1.0, 1.0, 1.8, 3.0, 6.0, 64, 18),
        "MR_lam22": (1.0, 1.0, 2.2, 3.0, 6.0, 64, 18),
        "MR_a2": (1.0, 1.0, 1.6, 2.0, 6.0, 64, 18),
        "MR_a4": (1.0, 1.0, 1.6, 4.0, 6.0, 64, 18),
        "MR_a5": (1.0, 1.0, 1.6, 5.0, 6.0, 64, 18),
        "MESH_nr48": (1.0, 1.0, 1.6, 3.0, 6.0, 48, 18),
        "MESH_nr96": (1.0, 1.0, 1.6, 3.0, 6.0, 96, 18),
        "NH_lam15": (1.0, 0.0, 1.5, 3.0, 6.0, 64, 18),
        "NH_lam18": (1.0, 0.0, 1.8, 3.0, 6.0, 64, 18),
        "MR_lam16_c2_third": (1.0, 1.0 / 3.0, 1.6, 3.0, 6.0, 64, 32),
        "MR_lam16_c2_3": (1.0, 3.0, 1.6, 3.0, 6.0, 64, 18),
    }
    gate("strip JSON/summary inventory",
         set(records) == set(rows) == set(expected_specs),
         f"{len(records)} JSON records")
    case_ok = True
    for tag, spec in expected_specs.items():
        c1, c2, lam, a, b, n_r, n_steps = spec
        record = records[tag]
        case = record["case"]
        protocol = record.get("protocol", {})
        mesh = protocol.get("mesh", {})
        case_ok &= all(math.isclose(float(case[key]), value,
                                    rel_tol=0.0, abs_tol=2e-14)
                       for key, value in (("c1", c1), ("c2", c2),
                                          ("lam", lam), ("a", a), ("b", b),
                                          ("h0", 1.0)))
        case_ok &= mesh.get("n_r") == n_r and protocol.get("n_steps") == n_steps
        case_ok &= record.get("tag") == tag
        case_ok &= record.get("material") == ("MR" if c2 != 0.0 else "NeoHookean")
    gate("strip case/tag provenance", case_ok,
         "loads, cracks, meshes, controls, and material ratios match their tags")
    summary_ok = True
    for tag, record in records.items():
        summary = rows[tag]
        case = record["case"]
        signatures = record["signatures"]
        energy = record["energy_release"]
        expected = {
            "lam": case["lam"], "a": case["a"],
            "G_J": energy["G_domain_J"], "G_spec": energy["G_spec_theory"],
            "GJ_err": energy["rel_err_GJ_vs_spec"],
            "P_meas": energy["P_measured"],
            "P_pred": energy["P_pred_from_G_spec"],
            "P_err": energy["rel_err_P_vs_pred"],
            "J_exp": signatures["J_exp"], "open_exp": signatures["open_exp"],
            "plateau": signatures["Jr14_plateau"],
            "spread": signatures["Jr14_spread"],
        }
        summary_ok &= summary["material"] == record["material"]
        summary_ok &= int(summary["ncells"]) == record["n_cells"]
        summary_ok &= summary["dmin_tag"] == str("MESH" in tag)
        summary_ok &= all(
            math.isclose(float(summary[key]), float(value),
                         rel_tol=1e-13, abs_tol=1e-13)
            for key, value in expected.items()
        )
    gate("strip JSON/summary numerical parity", summary_ok,
         "all scalar columns agree with their source JSON")
    protocol_ok = True
    mesh_geometry_ok = True
    for tag, record in records.items():
        protocol = record.get("protocol", {})
        mesh = protocol.get("mesh", {})
        expected_steps = expected_specs[tag][-1]
        case = record["case"]
        expected_corners = [math.atan2(0.5, case["b"]),
                            math.pi - math.atan2(0.5, case["a"])]
        corners = mesh.get("corner_angles", [])
        protocol_ok &= (
            protocol.get("n_steps") == expected_steps
            and len(protocol.get("fit_window", [])) == 2
            and all(math.isclose(value, expected, rel_tol=0.0, abs_tol=1e-15)
                    for value, expected in zip(protocol.get("fit_window", []),
                                               (3e-4, 8e-3)))
            and mesh.get("n_theta_base") == 120
            and mesh.get("n_sectors") == 120
            and math.isclose(mesh.get("r_min", math.nan), 1e-5,
                             rel_tol=0.0, abs_tol=1e-15)
            and mesh.get("angular_scheme") == "corner-snapped-v1"
            and len(corners) == 2
            and all(math.isclose(value, expected, rel_tol=0.0, abs_tol=1e-14)
                    for value, expected in zip(corners, expected_corners))
            and record.get("n_cells")
            == 2 * mesh.get("n_r", -1) * mesh.get("n_sectors", -1)
        )
        if len(corners) == 2 and mesh.get("n_sectors") == 120:
            theta, expected_coords = corner_aligned_geometry(
                case["a"], case["b"], 0.5, mesh.get("r_min", math.nan),
                mesh.get("n_r", -1), mesh["n_sectors"], corners)
            angular_gaps = np.diff(theta)
            outer = expected_coords[-(mesh["n_sectors"] + 1):]
            outer_area = 0.5 * abs(np.sum(
                outer[:-1, 0] * outer[1:, 1]
                - outer[1:, 0] * outer[:-1, 1]
            ))
            mesh_geometry_ok &= (
                angular_gaps.min() > math.radians(0.9)
                and angular_gaps.max() < math.radians(2.1)
                and math.isclose(outer_area, (case["a"] + case["b"]) * 0.5,
                                 rel_tol=0.0, abs_tol=2e-13)
            )
    gate("strip JSON mesh/continuation provenance", protocol_ok,
         "120 corner-aligned sectors; initial n_steps=18 (32 for c2/c1=1/3)")
    gate("strip mesh geometry", mesh_geometry_ok,
         "exact rectangular outer boundary without sub-degree angular slivers")

    ray_ok = True
    ray_values: dict[str, dict[int, dict[str, np.ndarray]]] = {}
    expected_ray_names = set()
    for tag, record in records.items():
        protocol = record["protocol"]
        mesh = protocol["mesh"]
        for angle in (2, 45, 90, 135, 178):
            path = STRIP / f"rays_{tag}_theta{angle}.csv"
            expected_ray_names.add(path.name)
            if not path.exists():
                ray_ok = False
                continue
            with path.open(newline="", encoding="utf-8") as stream:
                samples = list(csv.DictReader(stream))
            ray_ok &= len(samples) == 40
            physical = {"r", "theta_deg", "Y1", "Y2", "J", "lam1", "lam2"}
            required = physical | {
                "n_cells", "n_steps", "n_r", "n_theta_base", "n_sectors",
                "angular_scheme", "corner_angle_right", "corner_angle_left",
            }
            ray_ok &= bool(samples) and required.issubset(samples[0])
            if samples and required.issubset(samples[0]):
                ray_ok &= all(
                    int(sample["n_cells"]) == record["n_cells"]
                    and int(sample["n_steps"]) == protocol["n_steps"]
                    and int(sample["n_r"]) == mesh["n_r"]
                    and int(sample["n_theta_base"]) == mesh["n_theta_base"]
                    and int(sample["n_sectors"]) == mesh["n_sectors"]
                    and sample["angular_scheme"] == mesh["angular_scheme"]
                    and math.isclose(float(sample["corner_angle_right"]),
                                     mesh["corner_angles"][0], abs_tol=1e-14)
                    and math.isclose(float(sample["corner_angle_left"]),
                                     mesh["corner_angles"][1], abs_tol=1e-14)
                    and math.isclose(float(sample["theta_deg"]), angle,
                                     rel_tol=0.0, abs_tol=1e-14)
                    and all(math.isfinite(float(sample[key])) for key in physical)
                    for sample in samples
                )
                radii = [float(sample["r"]) for sample in samples]
                ray_ok &= radii[0] > 0.0 and all(
                    later > earlier for earlier, later in zip(radii, radii[1:])
                )
                ray_values.setdefault(tag, {})[angle] = {
                    key: np.asarray([float(sample[key]) for sample in samples])
                    for key in physical
                }
    actual_ray_names = {path.name for path in STRIP.glob("rays_*_theta*.csv")}
    ray_ok &= actual_ray_names == expected_ray_names
    gate("strip ray inventory/provenance", ray_ok,
         f"{len(actual_ray_names)} ray files linked to their solve metadata")

    ray_signature_ok = set(ray_values) == set(records)
    for tag, record in records.items():
        rays = ray_values.get(tag, {})
        if set(rays) != {2, 45, 90, 135, 178}:
            ray_signature_ok = False
            continue
        lo, hi = record["protocol"]["fit_window"]

        def power_slope(ray: dict[str, np.ndarray], key: str) -> float:
            radius, value = ray["r"], ray[key]
            mask = ((radius >= lo) & (radius <= hi)
                    & np.isfinite(value) & (np.abs(value) > 0.0))
            design = np.column_stack([
                np.ones(mask.sum()), np.log(radius[mask])
            ])
            return float(np.linalg.lstsq(
                design, np.log(np.abs(value[mask])), rcond=None
            )[0][1])

        face = rays[178]
        face_mask = ((face["r"] >= lo) & (face["r"] <= hi)
                     & np.isfinite(face["Y2"]))
        face_design = np.column_stack([
            np.sqrt(face["r"][face_mask]), face["r"][face_mask]
        ])
        fitted_P = float(np.linalg.lstsq(
            face_design, face["Y2"][face_mask], rcond=None
        )[0][0])
        fitted_open = power_slope(face, "Y2")

        fitted_J = float(np.mean([
            power_slope(ray, "J") for ray in rays.values()
        ]))
        probe_r = np.logspace(np.log10(lo), np.log10(hi), 5)
        plateau_by_angle = []
        for ray in rays.values():
            mask = np.isfinite(ray["J"]) & (ray["J"] > 0.0)
            interpolated_J = np.exp(np.interp(
                np.log(probe_r), np.log(ray["r"][mask]),
                np.log(ray["J"][mask]),
            ))
            plateau_by_angle.append(float(np.mean(
                interpolated_J * probe_r ** 0.25
            )))
        fitted_plateau = float(np.mean(plateau_by_angle))
        fitted_spread = ((max(plateau_by_angle) - min(plateau_by_angle))
                         / abs(fitted_plateau))

        signatures = record["signatures"]
        expected_from_rays = {
            "P_measured": fitted_P,
            "open_exp": fitted_open,
            "J_exp": fitted_J,
            "Jr14_plateau": fitted_plateau,
            "Jr14_spread": fitted_spread,
        }
        ray_signature_ok &= all(
            math.isclose(float(signatures[key]), value,
                         rel_tol=1e-9, abs_tol=1e-10)
            for key, value in expected_from_rays.items()
        )
        ray_signature_ok &= math.isclose(
            float(record["energy_release"]["P_measured"]), fitted_P,
            rel_tol=1e-9, abs_tol=1e-10,
        )
    gate("strip ray/scalar numerical binding", ray_signature_ok,
         "P, opening/J powers, and angular plateau recompute from each case's rays")

    field_specs = {
        "MR_lam13": (1.0, 1.0, 1.3, 18),
        "MR_lam16": (1.0, 1.0, 1.6, 18),
        "MR_lam22": (1.0, 1.0, 2.2, 18),
        "NH_lam16": (1.0, 0.0, 1.6, 18),
        "MR_lam16_c2_third": (1.0, 1.0 / 3.0, 1.6, 32),
        "MR_lam16_c2_3": (1.0, 3.0, 1.6, 18),
    }
    field_paths = sorted(STRIP.glob("psfield_*.npz"))
    field_ok = {path.stem.removeprefix("psfield_") for path in field_paths} == set(field_specs)
    for path in field_paths:
        tag = path.stem.removeprefix("psfield_")
        if tag not in field_specs:
            continue
        c1, c2, lam, expected_steps = field_specs[tag]
        with np.load(path) as field:
            required = {"coords", "disp", "tris", "lam_reached", "c1", "c2",
                        "a", "b", "H", "r_min", "n_steps", "n_r",
                        "n_theta_base", "n_sectors", "angular_scheme",
                        "corner_angles"}
            field_ok &= required.issubset(field.files)
            if required.issubset(field.files):
                n_r = int(field["n_r"].item())
                n_sectors = int(field["n_sectors"].item())
                n_points = (n_r + 1) * (n_sectors + 1)
                coords, disp, tris = field["coords"], field["disp"], field["tris"]
                corner_angles = field["corner_angles"]
                _, expected_coords = corner_aligned_geometry(
                    float(field["a"].item()), float(field["b"].item()),
                    float(field["H"].item()), float(field["r_min"].item()),
                    n_r, n_sectors, corner_angles.tolist())
                coord_order = np.lexsort((coords[:, 1], coords[:, 0]))
                expected_order = np.lexsort(
                    (expected_coords[:, 1], expected_coords[:, 0]))
                expected_to_actual = np.empty(n_points, dtype=np.int64)
                expected_to_actual[expected_order] = coord_order
                expected_tris = expected_to_actual[
                    radial_connectivity(n_r, n_sectors)
                ]
                triangle_points = coords[tris]
                edge_one = triangle_points[:, 1] - triangle_points[:, 0]
                edge_two = triangle_points[:, 2] - triangle_points[:, 0]
                twice_area = (edge_one[:, 0] * edge_two[:, 1]
                              - edge_one[:, 1] * edge_two[:, 0])
                top = np.isclose(coords[:, 1], float(field["H"].item()),
                                 rtol=0.0, atol=1e-12)
                ligament = (np.isclose(coords[:, 1], 0.0, rtol=0.0,
                                        atol=1e-12)
                            & (coords[:, 0] > 1e-9))
                field_ok &= (
                    int(field["n_steps"].item()) == expected_steps
                    and int(field["n_theta_base"].item()) == 120
                    and n_sectors == 120 and n_r == 64
                    and field["angular_scheme"].item() == "corner-snapped-v1"
                    and corner_angles.shape == (2,)
                    and np.allclose(
                        corner_angles,
                        [math.atan2(0.5, 6.0), math.pi - math.atan2(0.5, 3.0)],
                        rtol=0.0, atol=1e-14)
                    and coords.shape == (n_points, 2)
                    and disp.shape == coords.shape
                    and tris.shape == (2 * n_r * n_sectors, 3)
                    and np.isfinite(coords).all() and np.isfinite(disp).all()
                    and tris.min() >= 0 and tris.max() < n_points
                    and np.allclose(coords[coord_order],
                                    expected_coords[expected_order],
                                    rtol=0.0, atol=2e-14)
                    and np.array_equal(canonical_triangles(tris),
                                       canonical_triangles(expected_tris))
                    and np.all(np.abs(twice_area) > 1e-18)
                    and np.linalg.norm(disp) > 0.0
                    and top.any() and ligament.any()
                    and np.allclose(disp[top, 0], 0.0,
                                    rtol=0.0, atol=2e-11)
                    and np.allclose(
                        disp[top, 1],
                        (float(field["lam_reached"].item()) - 1.0)
                        * float(field["H"].item()),
                        rtol=0.0, atol=2e-11)
                    and np.allclose(disp[ligament, 1], 0.0,
                                    rtol=0.0, atol=2e-11)
                    and math.isclose(float(field["c1"].item()), c1)
                    and math.isclose(float(field["c2"].item()), c2)
                    and math.isclose(float(field["lam_reached"].item()), lam,
                                     rel_tol=0.0, abs_tol=2e-14)
                    and math.isclose(float(field["a"].item()), 3.0)
                    and math.isclose(float(field["b"].item()), 6.0)
                    and math.isclose(float(field["H"].item()), 0.5)
                    and math.isclose(float(field["r_min"].item()), 1e-5)
                )
                for corner in ((-3.0, 0.0), (-3.0, 0.5),
                               (6.0, 0.0), (6.0, 0.5)):
                    distance = np.linalg.norm(coords - np.asarray(corner), axis=1)
                    field_ok &= float(distance.min()) < 1e-12
                if tag in records:
                    record = records[tag]
                    field_ok &= (
                        int(field["n_steps"].item()) == record["protocol"]["n_steps"]
                        and n_sectors == record["protocol"]["mesh"]["n_sectors"]
                        and tris.shape[0] == record["n_cells"]
                    )
    gate("strip full-field provenance", field_ok,
         f"{len(field_paths)} snapshots on the same mesh/protocol")

    artifact_names = ({SUMMARY.name}
                      | {path.name for path in STRIP.glob("ps_*.json")}
                      | {path.name for path in STRIP.glob("rays_*.csv")}
                      | {path.name for path in STRIP.glob("psfield_*.npz")})
    manifest_ok = STRIP_MANIFEST.exists()
    manifest_entries: dict[str, str] = {}
    if manifest_ok:
        for line in STRIP_MANIFEST.read_text(encoding="utf-8").splitlines():
            fields = line.split(maxsplit=1)
            if len(fields) != 2 or fields[1] in manifest_entries:
                manifest_ok = False
                continue
            manifest_entries[fields[1]] = fields[0]
        manifest_ok &= set(manifest_entries) == artifact_names
        manifest_ok &= all(
            hashlib.sha256((STRIP / name).read_bytes()).hexdigest() == digest
            for name, digest in manifest_entries.items()
        )
    gate("strip artifact content lock", manifest_ok,
         f"{len(manifest_entries)} filename/content bindings")

    all_g_errors = [float(row["GJ_err"]) for row in rows.values()]
    gate("pure-shear energy balance", max(all_g_errors) <= 0.0015,
         f"maximum relative error {100 * max(all_g_errors):.3f}%")

    load_tags = ["MR_lam13", "MR_lam15", "MR_lam16", "MR_lam18", "MR_lam22"]
    mr = [rows[tag] for tag in load_tags]
    p_errors = [float(row["P_err"]) for row in mr]
    gate("parameter-free P(lambda)", max(p_errors) <= 0.0301,
         f"relative-error range {100 * min(p_errors):.2f}–{100 * max(p_errors):.2f}%")

    j_errors = [abs(float(row["J_exp"]) + 0.25) for row in mr]
    open_errors = [abs(float(row["open_exp"]) - 0.5) for row in mr]
    gate("MR opening/J finite-window powers",
         max(j_errors) < 0.004 and max(open_errors) < 0.055,
         f"max |Δp|: J={max(j_errors):.4f}, opening={max(open_errors):.4f}")

    mr_spreads = [float(row["spread"]) for row in mr]
    control_spreads = [float(rows[tag]["spread"])
                       for tag in ("NH_lam15", "NH_lam18")]
    gate("I2-specific angular plateau contrast",
         max(mr_spreads) < 0.10 and min(control_spreads) > 1.0,
         f"MR {100 * min(mr_spreads):.1f}–{100 * max(mr_spreads):.1f}% "
         f"versus controls {100 * min(control_spreads):.1f}–"
         f"{100 * max(control_spreads):.1f}%")

    ratio_tags = ["MR_lam16_c2_third", "MR_lam16", "MR_lam16_c2_3"]
    ratio = [rows[tag] for tag in ratio_tags]
    ratio_ok = (
        max(float(row["GJ_err"]) for row in ratio) <= 0.001
        and max(float(row["P_err"]) for row in ratio) <= 0.0301
        and max(float(row["spread"]) for row in ratio) < 0.10
        and max(abs(float(row["open_exp"]) - 0.5) for row in ratio) < 0.055
    )
    gate("material-ratio study", ratio_ok,
         "c2/c1 = 1/3, 1, 3 satisfy energy, amplitude, angular-flatness, "
         "and opening-power gates")

    mesh_p = [float(rows[tag]["P_meas"])
              for tag in ("MESH_nr48", "MESH_nr96", "MR_lam16")]
    mesh_spread = (max(mesh_p) - min(mesh_p)) / sum(mesh_p) * 3
    gate("strip mesh sensitivity", mesh_spread < 0.002,
         f"relative P range {100 * mesh_spread:.3f}%")

    profile_table = subprocess.run(
        [sys.executable, str(ROOT / "analysis" / "profile_mode_audit.py"),
         "--check-stored"],
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    print(profile_table.stdout, end="")
    gate("strip-only ESI finite-window table reproduction",
         profile_table.returncode == 0,
         "b, raw slopes, and nested q values reproduce; q remains unresolved")

    print("All principal stored-data claims passed.")


if __name__ == "__main__":
    main()
