"""Run the one-way pure-shear-strip to refined crack-tip-disk submodel.

MPI example
-----------
PY="$(command -v python)"
MPIEXEC="$(command -v mpiexec)"
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
NUMEXPR_NUM_THREADS=1 \
  $MPIEXEC -n 2 $PY fem/run_global_local.py --tag baseline
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import sys
from pathlib import Path

import basix
import dolfinx
import mpi4py
import numpy as np
import petsc4py
import scipy
import ufl
from mpi4py import MPI
from petsc4py import PETSc

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "pure_shear"))

from ps_mesh import StripConfig  # noqa: E402
from ps_solve import SolveConfig, solve  # noqa: E402
from mr_fem_mesh import MeshConfig  # noqa: E402
from global_local_submodel import (  # noqa: E402
    LocalSolveConfig,
    boundary_reaction_trace,
    compare_interface_reactions,
    extract_strip_inner_mesh,
    extract_strip_outer_field,
    manufactured_transfer_error,
    minimum_jacobian,
    sample_polar_displacement,
    solve_refined_local,
    solve_restricted_local,
    solve_local_from_strip,
)


def _jsonable(value):
    if isinstance(value, dict):
        return {k: _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    return value


def _thread_environment() -> dict:
    names = (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    )
    return {name: os.environ.get(name) for name in names}


def _parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--out", default="fem/global_local_outputs")
    ap.add_argument("--c1", type=float, default=1.0)
    ap.add_argument("--c2", type=float, default=1.0)
    ap.add_argument("--lam", type=float, default=1.6)
    ap.add_argument("--a", type=float, default=3.0)
    ap.add_argument("--b", type=float, default=6.0)
    ap.add_argument("--H", type=float, default=0.5)
    ap.add_argument("--r-min", type=float, default=1.0e-5)
    ap.add_argument("--matching-radius", type=float, default=1.0e-2)
    ap.add_argument("--strip-n-r", type=int, default=64)
    ap.add_argument("--strip-n-theta", type=int, default=120)
    ap.add_argument("--matching-n-inner", type=int, default=None)
    ap.add_argument("--strip-load-steps", type=int, default=18)
    ap.add_argument("--local-n-r", type=int, default=80)
    ap.add_argument(
        "--fine-local-n-r", type=int, default=None,
        help=("optional second local radial resolution; the converged first "
              "local solution is prolonged to this mesh and solved directly "
              "at full load"))
    ap.add_argument(
        "--local-mode", choices=("nested", "polar"), default="nested",
        help=("nested extracts the exact strip submesh and hierarchically "
              "refines it; polar retains the nonmatching diagnostic path"))
    ap.add_argument(
        "--local-refinement-levels", type=int, default=1,
        help="number of hierarchical refinements after the restricted mesh")
    ap.add_argument("--local-fallback-steps", type=int, default=16)
    ap.add_argument(
        "--factor-solver", choices=("auto", "petsc", "mumps"),
        default="auto")
    ap.add_argument(
        "--skip-manufactured-test", action="store_true",
        help="skip the exact P2 cross-mesh transfer preflight")
    ap.add_argument(
        "--export-profile", action="store_true",
        help="export an MPI-gathered live-P2 polar displacement profile")
    ap.add_argument("--profile-r-lo", type=float, default=None)
    ap.add_argument("--profile-r-hi", type=float, default=None)
    ap.add_argument("--profile-n-r", type=int, default=160)
    ap.add_argument("--profile-n-theta", type=int, default=181)
    return ap.parse_args()


def main():
    args = _parse_args()
    comm = MPI.COMM_WORLD
    factor_solver = None if args.factor_solver == "auto" else args.factor_solver
    if comm.size > 1 and factor_solver == "petsc":
        raise ValueError(
            "PETSc's built-in LU is not a distributed factorization; "
            "use --factor-solver mumps or auto under MPI")
    thread_env = _thread_environment()
    if comm.size > 1:
        unsafe = {
            key: value for key, value in thread_env.items()
            if value not in ("1", None)
        }
        if unsafe:
            raise RuntimeError(
                "MPI run would oversubscribe threaded math libraries. "
                "Set OMP_NUM_THREADS=OPENBLAS_NUM_THREADS="
                f"MKL_NUM_THREADS=NUMEXPR_NUM_THREADS=1; got {unsafe}")

    strip_mesh_cfg = StripConfig(
        a=args.a,
        b=args.b,
        H=args.H,
        r_min=args.r_min,
        n_r=args.strip_n_r,
        n_theta=args.strip_n_theta,
        matching_radius=args.matching_radius,
        matching_n_inner=args.matching_n_inner,
    )
    strip_solve_cfg = SolveConfig(
        c1=args.c1,
        c2=args.c2,
        lam=args.lam,
        n_steps=args.strip_load_steps,
        factor_solver=factor_solver,
    )
    comm.barrier()
    t0 = MPI.Wtime()
    strip_res = solve(strip_solve_cfg, strip_mesh_cfg)
    strip_elapsed_local = MPI.Wtime() - t0
    strip_elapsed = comm.allreduce(strip_elapsed_local, op=MPI.MAX)
    strip_min_j = minimum_jacobian(strip_res["u"], quadrature_degree=6)

    local_cfg = LocalSolveConfig(
        c1=args.c1,
        c2=args.c2,
        fallback_steps=args.local_fallback_steps,
        factor_solver=factor_solver,
        interpolation_padding=max(
            1.0e-12, 1.0e-8 * args.matching_radius),
    )

    transfer_error = None
    if not args.skip_manufactured_test:
        # Build/transfer inside solve_local_from_strip later as well.  This
        # separate preflight uses the actual source mesh and an independently
        # generated local mesh, and therefore catches MPI ownership errors.
        if args.local_mode == "nested":
            test_mesh, _ = extract_strip_inner_mesh(strip_res)
        else:
            from mr_fem_mesh import build_mesh
            test_mesh, _ = build_mesh(
                MeshConfig(
                    r_min=args.r_min,
                    R=args.matching_radius,
                    n_r=args.local_n_r,
                    theta_nodes=np.asarray(
                        strip_res["info"]["theta_nodes"])),
                comm=comm)
        transfer_error = manufactured_transfer_error(
            strip_res["msh"], test_mesh, degree=2,
            padding=local_cfg.interpolation_padding)
        if transfer_error >= 1.0e-12:
            raise RuntimeError(
                f"manufactured P2 transfer failed: {transfer_error:.3e}")
        del test_mesh

    comm.barrier()
    t0 = MPI.Wtime()
    if args.local_mode == "nested":
        local_res = solve_restricted_local(strip_res, local_cfg)
    else:
        local_res = solve_local_from_strip(
            strip_res, local_cfg,
            MeshConfig(
                r_min=args.r_min,
                R=args.matching_radius,
                n_r=args.local_n_r,
                theta_nodes=np.asarray(
                    strip_res["info"]["theta_nodes"])))
    local_elapsed_local = MPI.Wtime() - t0
    local_elapsed = comm.allreduce(local_elapsed_local, op=MPI.MAX)

    fine_res = None
    fine_elapsed = None
    refinement_history = []
    if args.local_mode == "nested":
        if args.local_refinement_levels < 0:
            raise ValueError("local-refinement-levels cannot be negative")
        source = local_res
        for level in range(args.local_refinement_levels):
            comm.barrier()
            t0 = MPI.Wtime()
            candidate = solve_refined_local(source, local_cfg)
            elapsed_local = MPI.Wtime() - t0
            elapsed = comm.allreduce(elapsed_local, op=MPI.MAX)
            refinement_history.append({
                "level": level + 1,
                "cells": candidate["info"]["n_cells"],
                "wall_time_s": elapsed,
                "direct": candidate["direct"],
                "used_continuation_fallback":
                    candidate["used_continuation_fallback"],
                "raw_P2_seed_min_J": candidate["raw_P2_seed_min_J"],
                "seed_strategy": candidate["seed_strategy"],
                "seed_min_J": candidate["seed_min_J"],
                "final_min_J": candidate["final_min_J"],
                "relative_change_from_seed":
                    candidate["relative_change_from_seed"],
            })
            source = candidate
        if refinement_history:
            fine_res = source
            fine_elapsed = sum(
                item["wall_time_s"] for item in refinement_history)
    elif args.fine_local_n_r is not None:
        if args.fine_local_n_r <= args.local_n_r:
            raise ValueError("fine-local-n-r must exceed local-n-r")
        fine_mesh_cfg = MeshConfig(
            r_min=args.r_min, R=args.matching_radius,
            n_r=args.fine_local_n_r,
            theta_nodes=np.asarray(strip_res["info"]["theta_nodes"]))
        comm.barrier()
        t0 = MPI.Wtime()
        fine_res = solve_local_from_strip(
            local_res, local_cfg, fine_mesh_cfg)
        fine_elapsed_local = MPI.Wtime() - t0
        fine_elapsed = comm.allreduce(fine_elapsed_local, op=MPI.MAX)

    # Weak reactions provide the equilibrium diagnostic that a one-way
    # displacement transfer does not enforce.  The strip-outer reaction is
    # assembled once and compared with each local level on the identical P2
    # trace.
    outer_field = extract_strip_outer_field(
        strip_res, padding=local_cfg.interpolation_padding)
    outer_reaction = boundary_reaction_trace(
        outer_field, c1=args.c1, c2=args.c2,
        radius=args.matching_radius)
    local_reaction = boundary_reaction_trace(
        local_res, c1=args.c1, c2=args.c2,
        radius=args.matching_radius)
    local_reaction_comparison = compare_interface_reactions(
        local_reaction, outer_reaction)
    fine_reaction_comparison = None
    if fine_res is not None:
        fine_reaction = boundary_reaction_trace(
            fine_res, c1=args.c1, c2=args.c2,
            radius=args.matching_radius)
        fine_reaction_comparison = compare_interface_reactions(
            fine_reaction, outer_reaction)

    profile = None
    profile_summary = None
    if args.export_profile:
        profile_source = fine_res if fine_res is not None else local_res
        r_lo = (
            max(5.0 * args.r_min, 0.01 * args.matching_radius)
            if args.profile_r_lo is None else args.profile_r_lo)
        r_hi = (
            0.8 * args.matching_radius
            if args.profile_r_hi is None else args.profile_r_hi)
        if not (args.r_min < r_lo < r_hi < args.matching_radius):
            raise ValueError(
                "profile radii must satisfy r_min < r_lo < r_hi < Rm")
        radii = np.geomspace(r_lo, r_hi, args.profile_n_r)
        theta_deg = np.linspace(0.0, 180.0, args.profile_n_theta)
        profile = sample_polar_displacement(
            profile_source, theta_deg, radii,
            collision_padding=max(
                1.0e-13, 1.0e-9 * args.matching_radius))
        if not np.all(profile["valid"]):
            n_invalid = int(
                profile["valid"].size - np.count_nonzero(profile["valid"]))
            raise RuntimeError(
                f"polar profile contains {n_invalid} unowned sample points")
        from mr_fem_extract import fit_face_opening
        fit_window = (
            max(3.0 * r_lo, 0.03 * args.matching_radius),
            min(0.5 * args.matching_radius, 0.8 * r_hi),
        )
        P_measured, C_face, n_face, face_rms = fit_face_opening(
            radii, profile["Y2"][-1], fit_window)
        profile_summary = {
            "source": (
                "fine_local" if fine_res is not None else "restricted_local"),
            "r_range": [r_lo, r_hi],
            "n_r": args.profile_n_r,
            "n_theta": args.profile_n_theta,
            "all_points_valid": True,
            "owner_multiplicity_max":
                profile["owner_multiplicity_max"],
            "face_fit_window": fit_window,
            "face_fit_points": n_face,
            "P_measured": P_measured,
            "C_face": C_face,
            "face_fit_rms": face_rms,
        }

    W_inf = (
        (args.c1 + args.c2)
        * (strip_res["lam_reached"] ** 2
           + strip_res["lam_reached"] ** -2 - 2.0)
    )
    G_spec = 2.0 * args.H * W_inf
    P_from_energy = float(np.sqrt(2.0 * G_spec / (np.pi * args.c1)))

    report = {
        "tag": args.tag,
        "method": "one-way-global-local-submodel",
        "command_arguments": sys.argv[1:],
        "runtime": {
            "python": platform.python_version(),
            "dolfinx": dolfinx.__version__,
            "basix": basix.__version__,
            "ufl": ufl.__version__,
            "petsc": ".".join(str(value) for value in PETSc.Sys.getVersion()),
            "petsc4py": petsc4py.__version__,
            "mpi4py": mpi4py.__version__,
            "numpy": np.__version__,
            "scipy": scipy.__version__,
        },
        "local_mesh_mode": args.local_mode,
        "mpi_ranks": comm.size,
        "thread_environment": thread_env,
        "material": {"c1": args.c1, "c2": args.c2},
        "case": {
            "c1": args.c1, "c2": args.c2,
            "lam": strip_res["lam_reached"],
        },
        "load": {
            "lambda_target": args.lam,
            "lambda_reached": strip_res["lam_reached"],
        },
        "strip": {
            "cells": strip_res["info"]["n_cells"],
            "n_r": strip_res["info"]["n_r"],
            "n_theta": strip_res["info"]["n_sectors"],
            "r_min": strip_res["info"]["r_min"],
            "matching_radius": strip_res["info"]["matching_radius"],
            "matching_ring": strip_res["info"]["matching_ring"],
            "matching_n_inner": strip_res["info"]["matching_n_inner"],
            "matching_n_outer": strip_res["info"]["matching_n_outer"],
            "load_steps_initial": args.strip_load_steps,
            "newton_iterations": strip_res["n_newton"],
            "factor_solver": strip_res["factor_solver"],
            "wall_time_s": strip_elapsed,
            "min_J": strip_min_j,
        },
        "local": {
            "cells": local_res["info"]["n_cells"],
            "n_r": local_res["info"]["n_r"],
            "n_theta": local_res["info"]["n_theta"],
            "factor_solver": local_res["factor_solver"],
            "wall_time_s": local_elapsed,
            "direct": local_res["direct"],
            "used_continuation_fallback":
                local_res["used_continuation_fallback"],
            "fallback_iterations": local_res["fallback_iterations"],
            "fallback_failures": local_res["fallback_failures"],
            "load_reached": local_res["load_reached"],
            "raw_P2_seed_min_J": local_res["raw_P2_seed_min_J"],
            "seed_strategy": local_res["seed_strategy"],
            "seed_min_J": local_res["seed_min_J"],
            "final_min_J": local_res["final_min_J"],
            "seed_energy": local_res["seed_energy"],
            "final_energy": local_res["final_energy"],
            "relative_change_from_seed":
                local_res["relative_change_from_seed"],
            "outer_trace_max_error":
                local_res["outer_trace_max_error"],
        },
        "interface": local_res["interface_audit"],
        "interface_reaction": local_reaction_comparison,
        "manufactured_P2_transfer_max_error": transfer_error,
        "energy_release": {
            "W_inf_theory": W_inf,
            "G_spec_theory": G_spec,
            "P_pred_from_G_spec": P_from_energy,
            "P_measured": (
                None if profile_summary is None
                else profile_summary["P_measured"]),
        },
        "polar_profile": profile_summary,
        "scope": {
            "one_way": True,
            "traction_feedback_to_strip": False,
            "same_core_radius_in_global_and_local": True,
            "coefficients_claimable_from_this_single_run": False,
        },
    }
    if refinement_history:
        report["refinement_history"] = refinement_history
    if fine_res is not None:
        report["fine_local"] = {
            "cells": fine_res["info"]["n_cells"],
            "n_r": fine_res["info"]["n_r"],
            "n_theta": fine_res["info"]["n_theta"],
            "factor_solver": fine_res["factor_solver"],
            "wall_time_s": fine_elapsed,
            "direct": fine_res["direct"],
            "used_continuation_fallback":
                fine_res["used_continuation_fallback"],
            "fallback_iterations": fine_res["fallback_iterations"],
            "fallback_failures": fine_res["fallback_failures"],
            "load_reached": fine_res["load_reached"],
            "raw_P2_seed_min_J": fine_res["raw_P2_seed_min_J"],
            "seed_strategy": fine_res["seed_strategy"],
            "seed_min_J": fine_res["seed_min_J"],
            "final_min_J": fine_res["final_min_J"],
            "seed_energy": fine_res["seed_energy"],
            "final_energy": fine_res["final_energy"],
            "relative_change_from_seed":
                fine_res["relative_change_from_seed"],
            "outer_trace_max_error":
                fine_res["outer_trace_max_error"],
            "interface": fine_res["interface_audit"],
            "interface_reaction": fine_reaction_comparison,
        }

    if comm.rank == 0:
        outdir = Path(args.out)
        outdir.mkdir(parents=True, exist_ok=True)
        path = outdir / f"global_local_{args.tag}.json"
        path.write_text(json.dumps(_jsonable(report), indent=2) + "\n")
        if profile is not None:
            profile_path = outdir / f"global_local_profile_{args.tag}.npz"
            np.savez_compressed(
                profile_path,
                **profile,
                c1=args.c1,
                c2=args.c2,
                lam_target=args.lam,
                lam_reached=strip_res["lam_reached"],
                r_min=args.r_min,
                matching_radius=args.matching_radius,
                n_r_mesh=(
                    -1 if (
                        fine_res is not None
                        and fine_res["info"]["n_r"] is None)
                    else (
                        fine_res["info"]["n_r"]
                        if fine_res is not None
                        else local_res["info"]["n_r"])),
                n_cells_mesh=(
                    fine_res["info"]["n_cells"]
                    if fine_res is not None
                    else local_res["info"]["n_cells"]),
                n_theta_mesh=local_res["info"]["n_theta"],
                displacement_degree=2,
                core_treatment="excised-traction-free-semicircle",
                sampling_scheme="MPI-owned-cell/log-r/equispaced-theta",
            )
        direct = report["local"]["direct"]
        print(
            f"[{args.tag}] strip lambda={strip_res['lam_reached']:.6f}, "
            f"{report['strip']['cells']} cells, {strip_elapsed:.2f}s")
        print(
            f"[{args.tag}] local {report['local']['cells']} cells, "
            f"direct={direct['converged']} ({direct['iterations']} Newton), "
            f"fallback={report['local']['used_continuation_fallback']}, "
            f"minJ={report['local']['final_min_J']:.6g}, "
            f"trace_err={report['local']['outer_trace_max_error']:.3e}")
        if fine_res is not None:
            fine_direct = report["fine_local"]["direct"]
            print(
                f"[{args.tag}] fine local "
                f"{report['fine_local']['cells']} cells, "
                f"direct={fine_direct['converged']} "
                f"({fine_direct['iterations']} Newton), "
                f"fallback="
                f"{report['fine_local']['used_continuation_fallback']}, "
                f"minJ={report['fine_local']['final_min_J']:.6g}")
        if profile is not None:
            print(
                f"[{args.tag}] profile P={profile_summary['P_measured']:.6g}, "
                f"all {profile['valid'].size} samples valid")
        print(f"[{args.tag}] wrote {path}")


if __name__ == "__main__":
    main()
