"""Nonlinear solve of the pure-shear (planar-tension) MR strip.

Reduced incompressible plane-stress Mooney-Rivlin energy
    W = c1(I1_2d + J^-2 - 3) + c2(J^2 + I1_2d J^-2 - 3),   F = I + grad u,
solved on the upper-half strip (ps_mesh) with

    grip  y=H :  u_y = delta (ramped),  u_x = 0     (clamped -> lateral constraint)
    ligament y=0, x>0 :  u_y = 0                    (Mode-I symmetry)
    crack y=0 x<0, and x=-a, x=b :  traction free.

The clamp u_x=0 on the wide grips is what makes the bulk planar tension
(pure shear): lambda = (H+delta)/H, state (lambda_x, lambda_y, lambda_z) ~
(1, lambda, 1/lambda), so I1 = I2 and W_inf = (c1+c2)(lambda^2+lambda^-2-2).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from mpi4py import MPI
from petsc4py import PETSc

import ufl
from dolfinx import fem
from dolfinx.fem.petsc import NonlinearProblem

from ps_mesh import StripConfig, build_strip


@dataclass
class SolveConfig:
    c1: float = 1.0
    c2: float = 1.0
    lam: float = 1.5           # target vertical stretch (H+delta)/H
    degree: int = 2
    n_steps: int = 16
    rtol: float = 1e-8
    atol: float = 1e-10
    max_it: int = 40
    quad_degree: int = 6


def W_inf(lam, c1, c2):
    """Pure-shear strain energy density (state (1, lam, 1/lam), I1=I2)."""
    return (c1 + c2) * (lam ** 2 + lam ** -2 - 2.0)


def solve(scfg: SolveConfig, mcfg: StripConfig | None = None):
    msh, info = build_strip(mcfg)
    H = info["H"]
    a, b = info["a"], info["b"]
    V = fem.functionspace(msh, ("Lagrange", scfg.degree, (msh.geometry.dim,)))
    u = fem.Function(V, name="u")
    v = ufl.TestFunction(V)

    d = msh.geometry.dim
    I = ufl.Identity(d)
    F = I + ufl.grad(u)
    C = F.T * F
    I1 = ufl.tr(C)
    J = ufl.det(F)
    c1 = fem.Constant(msh, PETSc.ScalarType(scfg.c1))
    c2 = fem.Constant(msh, PETSc.ScalarType(scfg.c2))
    W = c1 * (I1 + J ** -2 - 3) + c2 * (J ** 2 + I1 * J ** -2 - 3)
    Res = ufl.derivative(W * ufl.dx, u, v)

    # --- boundary dofs ---
    tol = 1e-9

    def on_grip(x):
        return np.isclose(x[1], H, atol=1e-9)

    def on_ligament(x):
        return np.isclose(x[1], 0.0, atol=1e-12) & (x[0] > tol)

    Vx, _ = V.sub(0).collapse()
    Vy, _ = V.sub(1).collapse()
    grip_x = fem.locate_dofs_geometrical((V.sub(0), Vx), on_grip)
    grip_y = fem.locate_dofs_geometrical((V.sub(1), Vy), on_grip)
    lig_y = fem.locate_dofs_geometrical((V.sub(1), Vy), on_ligament)

    zero_x = fem.Function(Vx)
    zero_y = fem.Function(Vy)
    delta_f = fem.Function(Vy)                            # ramped grip u_y

    bc_grip_x = fem.dirichletbc(zero_x, grip_x, V.sub(0))         # u_x = 0 on grip
    bc_grip_y = fem.dirichletbc(delta_f, grip_y, V.sub(1))        # u_y = delta on grip
    bc_lig = fem.dirichletbc(zero_y, lig_y, V.sub(1))            # u_y = 0 on ligament
    bcs = [bc_grip_x, bc_grip_y, bc_lig]

    petsc_options = {
        "snes_type": "newtonls", "snes_linesearch_type": "bt",
        "snes_rtol": scfg.rtol, "snes_atol": scfg.atol, "snes_stol": 1e-10,
        "snes_max_it": scfg.max_it,
        "ksp_type": "preonly", "pc_type": "lu",
        "pc_factor_mat_solver_type": "petsc",
    }
    problem = NonlinearProblem(
        Res, u, bcs=bcs, petsc_options_prefix="ps_",
        petsc_options=petsc_options,
        form_compiler_options={"quadrature_degree": scfg.quad_degree})
    snes = problem.solver

    delta_target = (scfg.lam - 1.0) * H
    u_prev = u.x.array.copy()
    t = 0.0
    dt = 1.0 / scfg.n_steps
    n_newton = 0
    while t < 1.0 - 1e-12:
        t_try = min(1.0, t + dt)
        delta_f.x.array[:] = t_try * delta_target; delta_f.x.scatter_forward()
        try:
            problem.solve()
            converged = snes.getConvergedReason() > 0
            its = snes.getIterationNumber()
        except Exception:
            converged, its = False, scfg.max_it
        n_newton += its
        if converged:
            t = t_try
            u_prev = u.x.array.copy()
        else:
            dt *= 0.5
            u.x.array[:] = u_prev
            u.x.scatter_forward()
            delta_f.x.array[:] = t * delta_target; delta_f.x.scatter_forward()
            if dt < 1e-4 and t > 0.99:
                break
            if dt < 1e-7:
                raise RuntimeError(f"SNES failed; t={t:.4f}, lam={scfg.lam}")

    lam_reached = 1.0 + (t * delta_target) / H
    return {
        "msh": msh, "u": u, "V": V, "info": info,
        "lam_target": scfg.lam, "lam_reached": float(lam_reached),
        "t_reached": float(t), "c1": scfg.c1, "c2": scfg.c2,
        "n_newton": n_newton, "H": H, "a": a, "b": b, "h0": info["h0"],
        "W_inf": float(W_inf(lam_reached, scfg.c1, scfg.c2)),
    }


if __name__ == "__main__":
    import time
    t0 = time.time()
    res = solve(SolveConfig(lam=1.5, n_steps=16),
                StripConfig(a=3.0, b=6.0, H=0.5, r_min=1e-5, n_r=48, n_theta=90))
    dt = time.time() - t0
    Vdg = fem.functionspace(res["msh"], ("DG", 0))
    Jf = fem.Function(Vdg)
    Jf.interpolate(fem.Expression(ufl.det(ufl.Identity(2) + ufl.grad(res["u"])),
                                  Vdg.element.interpolation_points))
    print(f"solve {dt:.1f}s  newton {res['n_newton']}  cells {res['info']['n_cells']}  "
          f"lam {res['lam_reached']:.3f}")
    print(f"J range [{Jf.x.array.min():.3f}, {Jf.x.array.max():.3f}]  "
          f"W_inf(theory) = {res['W_inf']:.5f}")
