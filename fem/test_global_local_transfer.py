"""MPI-safe manufactured tests for the global--local mesh and transfer."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from mpi4py import MPI
from dolfinx import fem

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "pure_shear"))

from ps_mesh import StripConfig, build_strip  # noqa: E402
from mr_fem_mesh import MeshConfig, build_mesh  # noqa: E402
from global_local_submodel import (  # noqa: E402
    boundary_reaction_trace,
    compare_interface_reactions,
    extract_strip_inner_mesh,
    extract_strip_outer_field,
    interface_audit,
    manufactured_transfer_error,
    refine_local_mesh,
    transfer_function,
)


def main():
    comm = MPI.COMM_WORLD
    Rm = 0.05
    strip_cfg = StripConfig(
        r_min=1.0e-5, n_r=20, n_theta=24,
        matching_radius=Rm, matching_n_inner=14)
    strip_msh, strip_info = build_strip(strip_cfg, comm=comm)
    strip_res = {"msh": strip_msh, "info": strip_info}
    disk_cfg = MeshConfig(
        r_min=strip_cfg.r_min, R=Rm, n_r=30,
        theta_nodes=strip_info["theta_nodes"])
    disk_msh, disk_info = build_mesh(disk_cfg, comm=comm)

    audit = interface_audit(strip_res, disk_msh, disk_info)
    error = manufactured_transfer_error(
        strip_msh, disk_msh, degree=2, padding=1.0e-9)
    assert audit["same_theta_vertices"]
    assert audit["facet_counts_match"]
    assert audit["n_expected_facets"] == strip_cfg.n_theta
    assert error < 1.0e-12, error

    # Exact strip-cell restriction followed by a true hierarchical triangle
    # refinement.  The outer polygon remains unchanged at both levels.
    Vs = fem.functionspace(strip_msh, ("Lagrange", 2, (2,)))
    us = fem.Function(Vs)

    def quadratic(x):
        return np.vstack((
            0.2 + x[0] + 0.3 * x[1] ** 2,
            -0.1 + 0.4 * x[1] + 0.2 * x[0] * x[1],
        ))

    us.interpolate(quadratic)
    strip_solution = {"msh": strip_msh, "u": us, "info": strip_info}
    restricted_msh, restricted_info = extract_strip_inner_mesh(strip_solution)
    restricted_error = manufactured_transfer_error(
        strip_msh, restricted_msh, degree=2, padding=1.0e-9)
    Vr = fem.functionspace(restricted_msh, ("Lagrange", 2, (2,)))
    ur = transfer_function(us, Vr, padding=1.0e-9)
    restricted_result = {
        "msh": restricted_msh,
        "V": Vr,
        "u": ur,
        "info": restricted_info,
    }
    refined_msh, refined_info = refine_local_mesh(restricted_result)
    refined_error = manufactured_transfer_error(
        restricted_msh, refined_msh, degree=2, padding=1.0e-9)
    refined_audit = interface_audit(
        restricted_result, refined_msh, refined_info)
    assert restricted_error < 1.0e-12, restricted_error
    assert refined_error < 1.0e-12, refined_error
    assert refined_audit["facet_counts_match"]

    # Affine constant-stress patch test for reaction sign, endpoint removal,
    # blocked-P2 ownership, and MPI gathering. The exact inner and outer
    # subdomains share the same interface field, so their outward reactions
    # must cancel.
    u_affine = fem.Function(Vs)

    def affine(x):
        return np.vstack((
            0.08 * x[0] + 0.03 * x[1],
            -0.02 * x[0] + 0.11 * x[1],
        ))

    u_affine.interpolate(affine)
    u_affine.x.scatter_forward()
    affine_strip = {
        "msh": strip_msh,
        "u": u_affine,
        "info": strip_info,
    }
    affine_inner_msh, affine_inner_info = extract_strip_inner_mesh(
        affine_strip)
    affine_inner_space = fem.functionspace(
        affine_inner_msh, ("Lagrange", 2, (2,)))
    affine_inner_u = transfer_function(
        u_affine, affine_inner_space, padding=1.0e-9)
    affine_inner = {
        "msh": affine_inner_msh,
        "V": affine_inner_space,
        "u": affine_inner_u,
        "info": affine_inner_info,
    }
    affine_outer = extract_strip_outer_field(
        affine_strip, padding=1.0e-9)
    reaction_inner = boundary_reaction_trace(
        affine_inner, c1=1.0, c2=1.0, radius=Rm)
    reaction_outer = boundary_reaction_trace(
        affine_outer, c1=1.0, c2=1.0, radius=Rm)
    reaction = compare_interface_reactions(
        reaction_inner, reaction_outer)
    assert reaction["relative_reaction_defect"] < 1.0e-10, reaction

    if comm.rank == 0:
        print(
            "global-local transfer PASS: "
            f"{audit['n_expected_facets']} shared facets, "
            f"polar/restricted/refined P2 max errors="
            f"{error:.3e}/{restricted_error:.3e}/{refined_error:.3e}, "
            "affine reaction defect="
            f"{reaction['relative_reaction_defect']:.3e}, "
            f"max chord sag={audit['max_chord_sag']:.3e}")


if __name__ == "__main__":
    main()
