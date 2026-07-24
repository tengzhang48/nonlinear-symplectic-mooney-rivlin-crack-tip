"""One-way pure-shear-strip to crack-tip-disk submodel.

The global strip contains an explicit internal semicircle ``r=R_m``.  The
local half-disk uses the same polygonal outer boundary and is driven by the
complete quadratic displacement trace of the global solution.  The global
field is also interpolated onto the local mesh as the full-load Newton
initial guess.

This module deliberately implements a one-way submodel, not a two-way
deformation--traction coupling.  Interface reactions and matching-radius
convergence must be checked before the local coefficients are treated as
specimen-selected quantities.
"""
from __future__ import annotations

from dataclasses import dataclass

import basix
import numpy as np
from mpi4py import MPI
from petsc4py import PETSc

import ufl
from dolfinx import fem, geometry, mesh
from dolfinx.fem.petsc import NonlinearProblem, assemble_vector

from mr_fem_mesh import MeshConfig, build_mesh


@dataclass
class LocalSolveConfig:
    """Nonlinear and discretization settings for the local half-disk."""

    c1: float = 1.0
    c2: float = 1.0
    degree: int = 2
    quad_degree: int = 6
    rtol: float = 1.0e-8
    atol: float = 1.0e-10
    max_it: int = 40
    fallback_steps: int = 16
    min_step: float = 1.0e-7
    interpolation_padding: float = 1.0e-10
    factor_solver: str | None = None


def owned_cells(msh) -> np.ndarray:
    """Return the owned cell indices on this MPI rank."""
    imap = msh.topology.index_map(msh.topology.dim)
    return np.arange(imap.size_local, dtype=np.int32)


def transfer_function(
        u_from: fem.Function,
        V_to: fem.FunctionSpace,
        *,
        padding: float = 1.0e-10) -> fem.Function:
    """Interpolate a distributed finite-element function to another mesh."""
    cells = owned_cells(V_to.mesh)
    interpolation_data = fem.create_interpolation_data(
        V_to, u_from.function_space, cells, padding=padding)
    source_owners = np.asarray(interpolation_data.src_owner, dtype=np.int32)
    n_missing_local = int(np.count_nonzero(source_owners < 0))
    n_missing = V_to.mesh.comm.allreduce(n_missing_local, op=MPI.SUM)
    if n_missing:
        raise RuntimeError(
            f"nonmatching interpolation has {n_missing} unowned source "
            "points; the source mesh does not cover the target mesh")
    u_to = fem.Function(V_to)
    u_to.interpolate_nonmatching(u_from, cells, interpolation_data)
    u_to.x.scatter_forward()
    return u_to


def transfer_vertex_linearized(
        u_from: fem.Function,
        V_to: fem.FunctionSpace,
        *,
        padding: float = 1.0e-10) -> fem.Function:
    """Transfer the source vertex field as a safer full-load initial guess.

    The prescribed interface still uses the complete P2 trace.  Only the
    interior Newton seed is linearized elementwise, which avoids importing
    unresolved P2 subcell oscillations from a coarser mesh.
    """
    source_msh = u_from.function_space.mesh
    V1 = fem.functionspace(
        source_msh, ("Lagrange", 1, (source_msh.geometry.dim,)))
    u1 = fem.Function(V1)
    u1.interpolate(u_from)
    u1.x.scatter_forward()
    return transfer_function(u1, V_to, padding=padding)


def _radius_marker(radius: float, *, atol: float):
    def marker(x):
        return np.isclose(
            np.sqrt(x[0] ** 2 + x[1] ** 2),
            radius, rtol=1.0e-10, atol=atol)
    return marker


def strip_interface_facets(msh, radius: float) -> np.ndarray:
    """Locate the explicit internal strip facets at ``radius``."""
    atol = max(1.0e-13, 1.0e-10 * radius)
    return mesh.locate_entities(
        msh, msh.topology.dim - 1, _radius_marker(radius, atol=atol))


def disk_outer_facets(msh, radius: float) -> np.ndarray:
    """Locate exterior facets forming the local disk's outer polygon."""
    atol = max(1.0e-13, 1.0e-10 * radius)
    return mesh.locate_entities_boundary(
        msh, msh.topology.dim - 1, _radius_marker(radius, atol=atol))


def _ligament_facets(msh) -> np.ndarray:
    tol = 1.0e-12

    def marker(x):
        return np.isclose(x[1], 0.0, atol=tol) & (x[0] > tol)

    return mesh.locate_entities_boundary(
        msh, msh.topology.dim - 1, marker)


def global_entity_count(msh, entities: np.ndarray, dim: int) -> int:
    """Count owned tagged entities without double-counting MPI ghosts."""
    size_local = msh.topology.index_map(dim).size_local
    local = int(np.count_nonzero(entities < size_local))
    return int(msh.comm.allreduce(local, op=MPI.SUM))


def _source_interface(source_res):
    info = source_res["info"]
    if info.get("matching_radius") is not None:
        return float(info["matching_radius"]), "strip-internal"
    if info.get("r_max") is not None:
        return float(info["r_max"]), "local-outer"
    raise ValueError("source result has no matching-interface radius")


def interface_audit(source_res, disk_msh, disk_info) -> dict:
    """Check that the source interface and disk boundary share one trace."""
    Rm, source_kind = _source_interface(source_res)
    theta_strip = np.asarray(source_res["info"]["theta_nodes"], dtype=float)
    theta_disk = np.asarray(disk_info["theta_nodes"], dtype=float)
    same_theta = (
        theta_strip.shape == theta_disk.shape
        and np.array_equal(theta_strip, theta_disk)
    )
    fdim = disk_msh.topology.dim - 1
    if source_kind == "strip-internal":
        sf = strip_interface_facets(source_res["msh"], Rm)
    else:
        sf = disk_outer_facets(source_res["msh"], Rm)
    df = disk_outer_facets(disk_msh, Rm)
    n_strip = global_entity_count(source_res["msh"], sf, fdim)
    n_disk = global_entity_count(disk_msh, df, fdim)
    expected = int(theta_strip.size - 1)
    dtheta = np.diff(theta_strip)
    chord_sag = float(Rm * (1.0 - np.cos(0.5 * np.max(dtheta))))
    return {
        "same_theta_vertices": bool(same_theta),
        "source_interface_kind": source_kind,
        "n_strip_interface_facets": n_strip,
        "n_disk_outer_facets": n_disk,
        "n_expected_facets": expected,
        "facet_counts_match": bool(
            n_strip == expected and n_disk == expected),
        "geometry": "shared-p1-chord",
        "max_chord_sag": chord_sag,
        "normal_convention": {
            "strip_inner_subdomain_outward": "+e_r",
            "strip_outer_subdomain_outward": "-e_r",
            "local_disk_outward": "+e_r",
        },
    }


def manufactured_transfer_error(
        source_msh,
        target_msh,
        *,
        degree: int = 2,
        padding: float = 1.0e-10) -> float:
    """Return the MPI-global max error for an exactly representable P2 field."""
    Vs = fem.functionspace(
        source_msh, ("Lagrange", degree, (source_msh.geometry.dim,)))
    Vt = fem.functionspace(
        target_msh, ("Lagrange", degree, (target_msh.geometry.dim,)))
    us = fem.Function(Vs)
    exact = fem.Function(Vt)

    def quadratic(x):
        return np.vstack((
            0.2 + 0.7 * x[0] - 0.4 * x[1]
            + 0.3 * x[0] ** 2 + 0.2 * x[0] * x[1],
            -0.1 + 0.5 * x[0] + 0.8 * x[1]
            - 0.25 * x[0] * x[1] + 0.1 * x[1] ** 2,
        ))

    us.interpolate(quadratic)
    exact.interpolate(quadratic)
    transferred = transfer_function(us, Vt, padding=padding)
    n_owned = Vt.dofmap.index_map.size_local * Vt.dofmap.bs
    local_error = (
        float(np.max(np.abs(
            transferred.x.array[:n_owned] - exact.x.array[:n_owned])))
        if n_owned else 0.0
    )
    return float(source_msh.comm.allreduce(local_error, op=MPI.MAX))


def _energy_kinematics(u, c1, c2):
    d = u.function_space.mesh.geometry.dim
    F = ufl.Identity(d) + ufl.grad(u)
    C = F.T * F
    I1 = ufl.tr(C)
    J = ufl.det(F)
    W = (
        c1 * (I1 + J ** -2 - 3)
        + c2 * (J ** 2 + I1 * J ** -2 - 3)
    )
    return W, J


def total_energy(u: fem.Function, c1: float, c2: float) -> float:
    """Assemble the total reduced strain energy over all MPI ranks."""
    W, _ = _energy_kinematics(u, c1, c2)
    local = fem.assemble_scalar(fem.form(W * ufl.dx))
    return float(u.function_space.mesh.comm.allreduce(local, op=MPI.SUM))


def minimum_jacobian(
        u: fem.Function,
        *,
        quadrature_degree: int = 6) -> float:
    """Evaluate the minimum deformation Jacobian at quadrature points."""
    _, J = _energy_kinematics(u, 1.0, 1.0)
    qpts, _ = basix.make_quadrature(
        basix.CellType.triangle, quadrature_degree)
    expr = fem.Expression(J, qpts)
    cells = owned_cells(u.function_space.mesh)
    values = np.asarray(expr.eval(u.function_space.mesh, cells), dtype=float)
    local_min = float(np.min(values)) if values.size else np.inf
    return float(
        u.function_space.mesh.comm.allreduce(local_min, op=MPI.MIN))


def _outer_trace_max_error(
        u: fem.Function,
        u_bc: fem.Function,
        bc_outer) -> float:
    dofs, _ = bc_outer.dof_indices()
    local = (
        float(np.max(np.abs(u.x.array[dofs] - u_bc.x.array[dofs])))
        if dofs.size else 0.0
    )
    return float(
        u.function_space.mesh.comm.allreduce(local, op=MPI.MAX))


def _factor_solver(cfg: LocalSolveConfig, comm: MPI.Comm) -> str:
    if cfg.factor_solver is not None:
        return cfg.factor_solver
    return "mumps" if comm.size > 1 else "petsc"


def extract_strip_inner_mesh(strip_res):
    """Extract the existing strip cells inside its explicit interface."""
    Rm, source_kind = _source_interface(strip_res)
    if source_kind != "strip-internal":
        raise ValueError("inner-cell extraction requires a strip source")
    source_msh = strip_res["msh"]
    tdim = source_msh.topology.dim
    tol = max(1.0e-13, 1.0e-10 * Rm)
    cells = mesh.locate_entities(
        source_msh, tdim,
        lambda x: np.sqrt(x[0] ** 2 + x[1] ** 2) <= Rm + tol)
    cells = cells[
        cells < source_msh.topology.index_map(tdim).size_local]
    submesh, _, _, _ = mesh.create_submesh(source_msh, tdim, cells)
    n_cells = int(submesh.comm.allreduce(
        submesh.topology.index_map(tdim).size_local, op=MPI.SUM))
    info = {
        "n_r": strip_res["info"]["matching_n_inner"],
        "n_theta": strip_res["info"]["n_sectors"],
        "n_cells": n_cells,
        "r_min": strip_res["info"]["r_min"],
        "r_max": Rm,
        "theta_nodes": np.asarray(
            strip_res["info"]["theta_nodes"], dtype=float).copy(),
        "mesh_relation": "exact-strip-cell-submesh",
        "refinement_level": 0,
    }
    return submesh, info


def extract_strip_outer_field(strip_res, *, padding: float = 1.0e-10):
    """Return the global strip field restricted to cells outside ``R_m``."""
    Rm, source_kind = _source_interface(strip_res)
    if source_kind != "strip-internal":
        raise ValueError("outer-cell extraction requires a strip source")
    source_msh = strip_res["msh"]
    tdim = source_msh.topology.dim
    tol = max(1.0e-13, 1.0e-10 * Rm)
    cells = mesh.locate_entities(
        source_msh, tdim,
        lambda x: np.sqrt(x[0] ** 2 + x[1] ** 2) >= Rm - tol)
    cells = cells[
        cells < source_msh.topology.index_map(tdim).size_local]
    submesh, _, _, _ = mesh.create_submesh(source_msh, tdim, cells)
    V = fem.functionspace(
        submesh, ("Lagrange", 2, (submesh.geometry.dim,)))
    u = transfer_function(strip_res["u"], V, padding=padding)
    n_cells = int(submesh.comm.allreduce(
        submesh.topology.index_map(tdim).size_local, op=MPI.SUM))
    info = {
        "n_cells": n_cells,
        "r_min": Rm,
        "r_max": None,
        "matching_radius": Rm,
        "theta_nodes": np.asarray(
            strip_res["info"]["theta_nodes"], dtype=float).copy(),
        "mesh_relation": "exact-strip-outer-cell-submesh",
    }
    return {"msh": submesh, "V": V, "u": u, "info": info}


def refine_local_mesh(local_res):
    """Hierarchically refine a local mesh while preserving its outer facets.

    All owned edges except the prescribed outer interface are marked.  This
    creates a true child mesh of the existing triangles and retains exactly
    the same polygonal interface and P2 boundary trace.
    """
    source_msh = local_res["msh"]
    Rm, source_kind = _source_interface(local_res)
    if source_kind != "local-outer":
        raise ValueError("local refinement requires a local-disk source")
    source_msh.topology.create_entities(1)
    outer = disk_outer_facets(source_msh, Rm)
    n_owned_edges = source_msh.topology.index_map(1).size_local
    is_outer = np.zeros(n_owned_edges, dtype=bool)
    owned_outer = outer[outer < n_owned_edges]
    is_outer[owned_outer] = True
    marked_edges = np.flatnonzero(~is_outer).astype(np.int32)
    refined, _, _ = mesh.refine(
        source_msh,
        edges=marked_edges,
        option=mesh.RefinementOption.parent_cell_and_facet)
    n_cells = int(refined.comm.allreduce(
        refined.topology.index_map(refined.topology.dim).size_local,
        op=MPI.SUM))
    info = {
        "n_r": None,
        "n_theta": local_res["info"]["n_theta"],
        "n_cells": n_cells,
        "r_min": local_res["info"]["r_min"],
        "r_max": Rm,
        "theta_nodes": np.asarray(
            local_res["info"]["theta_nodes"], dtype=float).copy(),
        "mesh_relation": "hierarchical-triangle-refinement",
        "refinement_level": int(
            local_res["info"].get("refinement_level", 0) + 1),
    }
    return refined, info


def solve_local_on_mesh(source_res, cfg: LocalSolveConfig, msh, info):
    """Solve one local level using a supplied nested or nonmatching mesh."""
    Rm, _ = _source_interface(source_res)
    if not np.isclose(info["r_max"], Rm):
        raise ValueError("local outer radius must equal source interface")
    audit = interface_audit(source_res, msh, info)
    if not audit["same_theta_vertices"] or not audit["facet_counts_match"]:
        raise RuntimeError(f"interface audit failed: {audit}")

    V = fem.functionspace(
        msh, ("Lagrange", cfg.degree, (msh.geometry.dim,)))
    p2_seed = transfer_function(
        source_res["u"], V, padding=cfg.interpolation_padding)
    u_target = fem.Function(V, name="u_boundary_target")
    u_target.x.array[:] = p2_seed.x.array
    u_target.x.scatter_forward()
    u_bc = fem.Function(V, name="u_boundary_active")
    u_bc.x.array[:] = u_target.x.array
    u_bc.x.scatter_forward()

    u = fem.Function(V, name="u_local")
    u.x.array[:] = p2_seed.x.array
    u.x.scatter_forward()
    v = ufl.TestFunction(V)

    c1 = fem.Constant(msh, PETSc.ScalarType(cfg.c1))
    c2 = fem.Constant(msh, PETSc.ScalarType(cfg.c2))
    W, _ = _energy_kinematics(u, c1, c2)
    residual = ufl.derivative(W * ufl.dx, u, v)

    fdim = msh.topology.dim - 1
    outer_facets = disk_outer_facets(msh, Rm)
    outer_dofs = fem.locate_dofs_topological(
        V, fdim, outer_facets, remote=True)
    bc_outer = fem.dirichletbc(u_bc, outer_dofs)

    ligament_facets = _ligament_facets(msh)
    Vy, _ = V.sub(1).collapse()
    ligament_dofs = fem.locate_dofs_topological(
        (V.sub(1), Vy), fdim, ligament_facets, remote=True)
    zero_y = fem.Function(Vy)
    zero_y.x.array[:] = 0.0
    bc_ligament = fem.dirichletbc(zero_y, ligament_dofs, V.sub(1))
    bcs = [bc_ligament, bc_outer]

    factor_solver = _factor_solver(cfg, msh.comm)
    petsc_options = {
        "snes_type": "newtonls",
        "snes_linesearch_type": "bt",
        "snes_rtol": cfg.rtol,
        "snes_atol": cfg.atol,
        "snes_stol": 1.0e-10,
        "snes_max_it": cfg.max_it,
        "ksp_type": "preonly",
        "pc_type": "lu",
        "pc_factor_mat_solver_type": factor_solver,
    }
    problem = NonlinearProblem(
        residual, u, bcs=bcs, petsc_options_prefix="gl_local_",
        petsc_options=petsc_options,
        form_compiler_options={"quadrature_degree": cfg.quad_degree})
    snes = problem.solver

    # Make the seed exactly satisfy both essential boundary conditions.
    for bc in bcs:
        bc.set(u.x.array)
    u.x.scatter_forward()
    raw_p2_seed_min_j = minimum_jacobian(
        u, quadrature_degree=cfg.quad_degree)
    seed_strategy = "full-P2-prolongation"
    if raw_p2_seed_min_j <= 0.0:
        safe_seed = transfer_vertex_linearized(
            source_res["u"], V, padding=cfg.interpolation_padding)
        u.x.array[:] = safe_seed.x.array
        for bc in bcs:
            bc.set(u.x.array)
        u.x.scatter_forward()
        seed_strategy = "P1-volume/P2-interface"

    seed_energy = total_energy(u, cfg.c1, cfg.c2)
    seed_min_j = minimum_jacobian(
        u, quadrature_degree=cfg.quad_degree)
    seed = fem.Function(V, name="u_initial")
    seed.x.array[:] = u.x.array
    seed.x.scatter_forward()
    seed_array = u.x.array.copy()

    direct_exception = None
    if seed_min_j <= 0.0:
        direct_exception = (
            "skipped: transferred full-load seed has "
            f"nonpositive minimum J ({seed_min_j:.6g})")
        direct_reason = 0
        direct_iterations = 0
        direct_min_j = seed_min_j
        direct_ok = False
    else:
        try:
            problem.solve()
            direct_reason = int(snes.getConvergedReason())
            direct_iterations = int(snes.getIterationNumber())
        except Exception as exc:  # retain a reproducible continuation fallback
            direct_exception = f"{type(exc).__name__}: {exc}"
            direct_reason = int(snes.getConvergedReason())
            direct_iterations = int(snes.getIterationNumber())

        direct_min_j = minimum_jacobian(
            u, quadrature_degree=cfg.quad_degree)
        direct_ok = bool(direct_reason > 0 and direct_min_j > 0.0)
    used_fallback = not direct_ok
    fallback_iterations = 0
    fallback_failures = 0
    load_reached = 1.0 if direct_ok else 0.0

    if used_fallback:
        # Restore a clean unloaded state and ramp only as a numerical
        # safeguard.  The physical boundary data remain the final strip trace.
        u.x.array[:] = 0.0
        u.x.scatter_forward()
        u_prev = u.x.array.copy()
        t = 0.0
        dt = 1.0 / max(1, cfg.fallback_steps)
        while t < 1.0 - 1.0e-12:
            t_try = min(1.0, t + dt)
            u_bc.x.array[:] = t_try * u_target.x.array
            u_bc.x.scatter_forward()
            try:
                problem.solve()
                reason = int(snes.getConvergedReason())
                its = int(snes.getIterationNumber())
                converged = reason > 0
            except Exception:
                converged = False
                its = int(snes.getIterationNumber())
            fallback_iterations += its
            if converged:
                t = t_try
                u_prev = u.x.array.copy()
            else:
                fallback_failures += 1
                dt *= 0.5
                u.x.array[:] = u_prev
                u.x.scatter_forward()
                u_bc.x.array[:] = t * u_target.x.array
                u_bc.x.scatter_forward()
                if dt < cfg.min_step:
                    raise RuntimeError(
                        "local continuation failed "
                        f"at transferred-load fraction {t:.6f}")
        load_reached = float(t)
        # Restore the exact final trace object for diagnostics and output.
        u_bc.x.array[:] = u_target.x.array
        u_bc.x.scatter_forward()

    final_min_j = minimum_jacobian(
        u, quadrature_degree=cfg.quad_degree)
    if final_min_j <= 0.0:
        raise RuntimeError(
            f"local solution is not orientation preserving: min J={final_min_j}")
    final_energy = total_energy(u, cfg.c1, cfg.c2)
    outer_error = _outer_trace_max_error(u, u_target, bc_outer)
    n_owned = V.dofmap.index_map.size_local * V.dofmap.bs
    local_seed_change = float(np.linalg.norm(
        u.x.array[:n_owned] - seed_array[:n_owned]) ** 2)
    local_seed_norm = float(np.linalg.norm(seed_array[:n_owned]) ** 2)
    seed_change = msh.comm.allreduce(local_seed_change, op=MPI.SUM)
    seed_norm = msh.comm.allreduce(local_seed_norm, op=MPI.SUM)

    return {
        "msh": msh,
        "V": V,
        "u": u,
        "u_seed": seed,
        "u_boundary": u_target,
        "info": info,
        "interface_audit": audit,
        "factor_solver": factor_solver,
        "direct": {
            "converged": direct_ok,
            "reason": direct_reason,
            "iterations": direct_iterations,
            "exception": direct_exception,
            "min_J": direct_min_j,
        },
        "used_continuation_fallback": used_fallback,
        "fallback_iterations": fallback_iterations,
        "fallback_failures": fallback_failures,
        "load_reached": load_reached,
        "seed_energy": seed_energy,
        "final_energy": final_energy,
        "raw_P2_seed_min_J": raw_p2_seed_min_j,
        "seed_strategy": seed_strategy,
        "seed_min_J": seed_min_j,
        "final_min_J": final_min_j,
        "outer_trace_max_error": outer_error,
        "relative_change_from_seed": float(
            np.sqrt(seed_change / seed_norm) if seed_norm > 0.0 else 0.0),
    }


def solve_local_from_strip(
        strip_res,
        cfg: LocalSolveConfig,
        mesh_cfg: MeshConfig):
    """Solve a separately generated local polar mesh (nonmatching option)."""
    Rm, _ = _source_interface(strip_res)
    if not np.isclose(mesh_cfg.R, Rm):
        raise ValueError("local outer radius must equal strip matching radius")
    if mesh_cfg.theta_nodes is None:
        raise ValueError(
            "local mesh must consume the strip interface theta_nodes")
    msh, info = build_mesh(mesh_cfg, comm=strip_res["msh"].comm)
    return solve_local_on_mesh(strip_res, cfg, msh, info)


def solve_restricted_local(strip_res, cfg: LocalSolveConfig):
    """Solve on the exact restriction of the strip mesh inside ``R_m``."""
    msh, info = extract_strip_inner_mesh(strip_res)
    return solve_local_on_mesh(strip_res, cfg, msh, info)


def solve_refined_local(local_res, cfg: LocalSolveConfig):
    """Refine a converged local mesh hierarchically and solve at full load."""
    msh, info = refine_local_mesh(local_res)
    return solve_local_on_mesh(local_res, cfg, msh, info)


def boundary_reaction_trace(
        result,
        *,
        c1: float,
        c2: float,
        radius: float,
        quadrature_degree: int = 6,
        exclude_endpoints: bool = True) -> dict:
    """Assemble the weak reaction functional on a P2 interface trace.

    Residual entries are gathered only for owned trace degrees of freedom.
    Their sign is the traction exerted by the subdomain through its outward
    normal.  No Dirichlet elimination or lifting is applied.
    """
    msh = result["msh"]
    u = result["u"]
    V = result.get("V", u.function_space)
    v = ufl.TestFunction(V)
    c1c = fem.Constant(msh, PETSc.ScalarType(c1))
    c2c = fem.Constant(msh, PETSc.ScalarType(c2))
    W, _ = _energy_kinematics(u, c1c, c2c)
    residual = ufl.derivative(W * ufl.dx, u, v)
    vector = assemble_vector(fem.form(
        residual,
        form_compiler_options={"quadrature_degree": quadrature_degree}))
    vector.ghostUpdate(
        addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)

    fdim = msh.topology.dim - 1
    facets = disk_outer_facets(msh, radius)
    block_dofs = fem.locate_dofs_topological(
        V, fdim, facets, remote=True)
    bs = V.dofmap.bs
    if bs != 2:
        raise ValueError(f"expected a two-component blocked space, got bs={bs}")
    scalar_dofs = (
        block_dofs[:, None] * bs + np.arange(bs, dtype=np.int32)[None, :]
    ).reshape(-1)
    n_owned_scalar = V.dofmap.index_map.size_local * bs
    scalar_dofs = scalar_dofs[scalar_dofs < n_owned_scalar]
    block = scalar_dofs // bs
    component = scalar_dofs % bs
    coordinates = V.tabulate_dof_coordinates()[block, :2]
    values = np.asarray(vector.array[scalar_dofs], dtype=float)

    keep = np.ones(values.size, dtype=bool)
    if exclude_endpoints:
        at_endpoint = (
            np.isclose(coordinates[:, 1], 0.0, atol=1.0e-12)
            & np.isclose(
                np.abs(coordinates[:, 0]), radius,
                rtol=1.0e-10, atol=max(1.0e-13, 1.0e-10 * radius))
        )
        keep &= ~at_endpoint
    local = np.column_stack((
        coordinates[keep, 0],
        coordinates[keep, 1],
        component[keep].astype(float),
        values[keep],
    ))
    gathered = msh.comm.gather(local, root=0)
    if msh.comm.rank == 0:
        entries = (
            np.vstack(gathered)
            if gathered else np.empty((0, 4), dtype=float)
        )
        order = np.lexsort((
            entries[:, 2], entries[:, 1], entries[:, 0]))
        entries = entries[order]
    else:
        entries = None
    entries = msh.comm.bcast(entries, root=0)
    return {
        "entries": entries,
        "n_scalar_dofs": int(entries.shape[0]),
        "excluded_endpoint_blocks": bool(exclude_endpoints),
        "sign": "subdomain-outward weak reaction",
        "quadrature_degree": int(quadrature_degree),
    }


def compare_interface_reactions(local_trace: dict, outer_trace: dict) -> dict:
    """Compare local ``+e_r`` and strip-outer ``-e_r`` weak reactions."""
    local = np.asarray(local_trace["entries"], dtype=float)
    outer = np.asarray(outer_trace["entries"], dtype=float)
    if local.shape != outer.shape:
        raise ValueError(
            f"reaction trace shapes differ: {local.shape} versus {outer.shape}")
    coordinate_error = float(np.max(np.abs(
        local[:, :3] - outer[:, :3]))) if local.size else 0.0
    if coordinate_error > 1.0e-11:
        raise ValueError(
            "reaction traces do not share the same coordinate/component keys; "
            f"max difference={coordinate_error:.3e}")
    local_force = local[:, 3]
    outer_force = outer[:, 3]
    defect = local_force + outer_force
    local_norm = float(np.linalg.norm(local_force))
    outer_norm = float(np.linalg.norm(outer_force))
    scale = 0.5 * (local_norm + outer_norm)
    relative = float(np.linalg.norm(defect) / scale) if scale > 0.0 else 0.0

    def resultant(entries):
        out = np.zeros(2)
        for component in (0, 1):
            out[component] = np.sum(
                entries[entries[:, 2] == component, 3])
        return out

    result_local = resultant(local)
    result_outer = resultant(outer)
    return {
        "n_scalar_trace_dofs": int(local.shape[0]),
        "coordinate_key_max_error": coordinate_error,
        "reaction_l2_local": local_norm,
        "reaction_l2_strip_outer": outer_norm,
        "reaction_l2_defect": float(np.linalg.norm(defect)),
        "relative_reaction_defect": relative,
        "resultant_local": result_local,
        "resultant_strip_outer": result_outer,
        "resultant_defect": result_local + result_outer,
        "resultant_scope": (
            "sum over the same endpoint-excluded coefficient vector; "
            "not the complete physical interface resultant"),
        "norm_scope": (
            "Euclidean norm of matching P2 weak-reaction coefficients; "
            "not yet the trace-mass dual norm"),
        "endpoint_scope": (
            "both vector blocks at theta=0 and theta=pi are excluded to "
            "remove the symmetry/crack-corner support ambiguity"),
    }


def sample_polar_displacement(
        result,
        theta_deg,
        radii,
        *,
        collision_padding: float = 1.0e-12) -> dict:
    """Evaluate a distributed P2 displacement on a common polar grid."""
    theta_deg = np.asarray(theta_deg, dtype=float)
    radii = np.asarray(radii, dtype=float)
    if theta_deg.ndim != 1 or radii.ndim != 1:
        raise ValueError("theta_deg and radii must be one-dimensional")
    if np.any(np.diff(theta_deg) <= 0.0) or np.any(np.diff(radii) <= 0.0):
        raise ValueError("theta_deg and radii must be strictly increasing")
    if theta_deg[0] < 0.0 or theta_deg[-1] > 180.0:
        raise ValueError("upper-half angles must lie in [0, 180] degrees")

    msh = result["msh"]
    u = result["u"]
    theta = np.deg2rad(theta_deg)
    rr, tt = np.meshgrid(radii, theta)
    x = rr * np.cos(tt)
    y = rr * np.sin(tt)
    points = np.column_stack((
        x.ravel(), y.ravel(), np.zeros(x.size, dtype=float)))
    tree = geometry.bb_tree(
        msh, msh.topology.dim, padding=collision_padding)
    candidates = geometry.compute_collisions_points(tree, points)
    colliding = geometry.compute_colliding_cells(msh, candidates, points)
    n_owned_cells = msh.topology.index_map(msh.topology.dim).size_local
    local_values = np.zeros((points.shape[0], 2), dtype=float)
    local_count = np.zeros(points.shape[0], dtype=np.int32)
    selected_indices = []
    selected_points = []
    selected_cells = []
    for index in range(points.shape[0]):
        cells = colliding.links(index)
        owned = cells[cells < n_owned_cells]
        if owned.size:
            selected_indices.append(index)
            selected_points.append(points[index])
            selected_cells.append(int(owned[0]))
    if selected_indices:
        evaluated = np.asarray(
            u.eval(
                np.asarray(selected_points, dtype=float),
                np.asarray(selected_cells, dtype=np.int32)),
            dtype=float).reshape(-1, 2)
        indices = np.asarray(selected_indices, dtype=np.int64)
        local_values[indices] = evaluated
        local_count[indices] = 1

    global_values = np.zeros_like(local_values)
    global_count = np.zeros_like(local_count)
    msh.comm.Allreduce(local_values, global_values, op=MPI.SUM)
    msh.comm.Allreduce(local_count, global_count, op=MPI.SUM)
    valid = global_count > 0
    global_values[valid] /= global_count[valid, None]
    global_values[~valid] = np.nan
    values = global_values.reshape(theta_deg.size, radii.size, 2)
    valid_grid = valid.reshape(theta_deg.size, radii.size)
    return {
        "theta_deg": theta_deg,
        "r": radii,
        "X": x,
        "Y": y,
        "ux": values[:, :, 0],
        "uy": values[:, :, 1],
        "Y1": x + values[:, :, 0],
        "Y2": y + values[:, :, 1],
        "valid": valid_grid,
        "owner_multiplicity_max": int(np.max(global_count)),
    }
