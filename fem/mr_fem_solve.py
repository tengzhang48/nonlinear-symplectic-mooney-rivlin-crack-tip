"""Solve the quarantined focused-disk auxiliary boundary-value problem.

The full-arc remote condition below imposes crack-parallel compression and is
not equivalent to the paper's Rivlin--Thomas pure-shear strip. The solver also
has no contact or global-injectivity constraint. Outputs are negative
provenance, not paper validation.

Reduced energy (thickness eliminated, lambda3 = 1/J, no pressure DOF):

    F  = I + grad(u)          (2x2 in-plane deformation gradient)
    C  = F^T F,  I1_2d = tr(C),  J = det F
    W  = c1*(I1_2d + J**-2 - 3) + c2*(J**2 + I1_2d*J**-2 - 3)

This is the reduced incompressible plane-stress functional used throughout.

Loading: remote isochoric stretch on the outer boundary,
    F_far(t) = diag(lambda(t)^-1, lambda(t)),  lambda(t) = lambda_target^t,
applied as Dirichlet u = (F_far - I) X, ramped t: 0 -> 1 (continuation).
Mode-I symmetry u_y = 0 on the ligament (theta = 0); crack face and core hole
are traction free.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from mpi4py import MPI
from petsc4py import PETSc

import ufl
from dolfinx import fem
from dolfinx.fem.petsc import NonlinearProblem

from mr_fem_mesh import MeshConfig, build_mesh


@dataclass
class SolveConfig:
    c1: float = 1.0
    c2: float = 1.0
    lam_target: float = 1.5
    degree: int = 2
    n_steps: int = 12           # initial number of load increments
    rtol: float = 1e-8
    atol: float = 1e-10
    max_it: int = 60
    quad_degree: int = 6
    verbose: bool = False


def _boundary_dofs(V, msh):
    """Return (ligament_y_dofs, outer_dofs) by geometric location."""
    tdim = msh.topology.dim
    msh.topology.create_connectivity(tdim - 1, tdim)

    # coordinates: ligament is Y ~ 0 and X > 0
    def on_ligament(x):
        return np.isclose(x[1], 0.0, atol=1e-12) & (x[0] > 0.0)

    R = float(np.max(np.linalg.norm(np.column_stack(
        [msh.geometry.x[:, 0], msh.geometry.x[:, 1]]), axis=1)))

    def on_outer(x):
        r = np.sqrt(x[0] ** 2 + x[1] ** 2)
        return r > 0.999 * R

    Vy = V.sub(1)
    lig_y = fem.locate_dofs_geometrical((Vy, Vy.collapse()[0]), on_ligament)
    outer = fem.locate_dofs_geometrical(V, on_outer)
    return lig_y, outer, R


def solve(scfg: SolveConfig, mcfg: MeshConfig | None = None):
    msh, info = build_mesh(mcfg)
    V = fem.functionspace(msh, ("Lagrange", scfg.degree, (msh.geometry.dim,)))
    u = fem.Function(V, name="u")
    u.x.array[:] = 0.0
    v = ufl.TestFunction(V)

    # --- kinematics / energy ---
    d = msh.geometry.dim
    I = ufl.Identity(d)
    F = I + ufl.grad(u)
    C = F.T * F
    I1 = ufl.tr(C)
    J = ufl.det(F)
    c1 = fem.Constant(msh, PETSc.ScalarType(scfg.c1))
    c2 = fem.Constant(msh, PETSc.ScalarType(scfg.c2))
    W = c1 * (I1 + J ** -2 - 3) + c2 * (J ** 2 + I1 * J ** -2 - 3)
    Pi = W * ufl.dx
    Res = ufl.derivative(Pi, u, v)

    # --- boundary conditions ---
    lig_y, outer, R = _boundary_dofs(V, msh)

    # ligament symmetry: u_y = 0
    Vy, _ = V.sub(1).collapse()
    zero_y = fem.Function(Vy)
    zero_y.x.array[:] = 0.0
    bc_lig = fem.dirichletbc(zero_y, lig_y, V.sub(1))

    # outer remote stretch, ramped via load factor t in [0, 1]
    t_load = fem.Constant(msh, PETSc.ScalarType(0.0))
    x = ufl.SpatialCoordinate(msh)
    lam_t = scfg.lam_target ** t_load
    Ffar = ufl.as_matrix([[1.0 / lam_t, 0.0], [0.0, lam_t]])
    u_far_expr = (Ffar - I) * ufl.as_vector([x[0], x[1]])
    u_far = fem.Function(V)
    expr = fem.Expression(u_far_expr, V.element.interpolation_points)
    bc_outer = fem.dirichletbc(u_far, outer)

    petsc_options = {
        "snes_type": "newtonls",
        "snes_linesearch_type": "bt",
        "snes_rtol": scfg.rtol,
        "snes_atol": scfg.atol,
        "snes_stol": 1e-10,
        "snes_max_it": scfg.max_it,
        "ksp_type": "preonly",
        "pc_type": "lu",
        "pc_factor_mat_solver_type": "petsc",  # MUMPS is ~25x slower in this env
    }
    if scfg.verbose:
        petsc_options["snes_monitor"] = None
    problem = NonlinearProblem(
        Res, u, bcs=[bc_lig, bc_outer],
        petsc_options_prefix="mr_",
        petsc_options=petsc_options,
        form_compiler_options={"quadrature_degree": scfg.quad_degree},
    )
    snes = problem.solver

    # --- incremental loading with adaptive step halving ---
    u_prev = u.x.array.copy()
    t = 0.0
    dt = 1.0 / scfg.n_steps
    n_newton = 0
    while t < 1.0 - 1e-12:
        t_try = min(1.0, t + dt)
        t_load.value = t_try
        u_far.interpolate(expr)
        try:
            problem.solve()
            reason = snes.getConvergedReason()
            its = snes.getIterationNumber()
            converged = reason > 0
        except Exception:
            converged = False
            its = scfg.max_it
        if converged:
            t = t_try
            n_newton += its
            u_prev = u.x.array.copy()
            if its <= 4 and dt < 1.0 / scfg.n_steps:
                dt = min(1.0 / scfg.n_steps, dt * 2.0)
        else:
            dt *= 0.5
            # restore last converged state for the retry
            u.x.array[:] = u_prev
            u.x.scatter_forward()
            t_load.value = t
            u_far.interpolate(expr)
            if dt < 1e-4 and t > 0.99:
                # load applied to within <1%: accept near-full load and stop
                break
            if dt < 1e-6:
                raise RuntimeError(
                    f"SNES failed to converge; t={t:.4f}, lam={scfg.lam_target}")
    # record the load level actually reached (lambda(t))
    lam_reached = scfg.lam_target ** t
    result = {
        "msh": msh,
        "u": u,
        "V": V,
        "info": info,
        "R": R,
        "lam_target": scfg.lam_target,
        "lam_reached": float(lam_reached),
        "t_reached": float(t),
        "c1": scfg.c1,
        "c2": scfg.c2,
        "n_newton": n_newton,
    }
    return result


if __name__ == "__main__":
    import time
    # quick smoke test on a coarse mesh, modest load
    mcfg = MeshConfig(r_min=1e-4, R=1.0, ratio=1.25, n_theta=48)
    scfg = SolveConfig(c1=1.0, c2=1.0, lam_target=1.3, n_steps=6, verbose=False)
    t0 = time.time()
    res = solve(scfg, mcfg)
    dt = time.time() - t0
    u = res["u"]
    msh = res["msh"]
    # report J range and max displacement
    Vdg = fem.functionspace(msh, ("DG", 0))
    d = msh.geometry.dim
    Fexpr = ufl.Identity(d) + ufl.grad(u)
    Jexpr = fem.Expression(ufl.det(Fexpr), Vdg.element.interpolation_points)
    Jfun = fem.Function(Vdg)
    Jfun.interpolate(Jexpr)
    print(f"solve time {dt:.1f}s, newton iters {res['n_newton']}, cells {res['info']['n_cells']}")
    print(f"J: min {Jfun.x.array.min():.4f}  max {Jfun.x.array.max():.4f}")
    umag = np.linalg.norm(u.x.array.reshape(-1, d), axis=1)
    print(f"|u|: max {umag.max():.4f}")
