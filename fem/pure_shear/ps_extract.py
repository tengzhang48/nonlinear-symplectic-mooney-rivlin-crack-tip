"""Extraction for the pure-shear MR strip: near-tip signatures + energy release.

Reuses the near-tip ray extraction (opening/J powers, a near-axis residual
power, J r^1/4 plateau, principal stretches, and opening intensity P) from
../mr_fem_extract, and adds the
energy-release-rate machinery specific to the pure-shear specimen:

  * far-field strain-energy density  W_ff  measured ahead of the tip, checked
    against the pure-shear value  W_inf = (c1+c2)(lam^2+lam^-2-2);
  * specimen energy release rate     G_spec = W_inf * h0   (Rivlin-Thomas);
  * domain (EDI) J-integral           G_J    (path-independent, finite strain);
  * the near-tip identity check       G = (pi/2) c1 P^2, i.e. the measured P
    should equal  P_pred = sqrt( 2 G / (pi c1) ).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import ufl
from dolfinx import fem, geometry

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # ../ (fem/)
import mr_fem_extract as mfx  # noqa: E402


# ------------------------------------------------------------- far-field W
def _energy_density_expr(u, c1, c2):
    d = 2
    F = ufl.Identity(d) + ufl.grad(u)
    Jc = ufl.det(F)
    I1 = ufl.tr(F.T * F)
    return c1 * (I1 + Jc ** -2 - 3) + c2 * (Jc ** 2 + I1 * Jc ** -2 - 3)


def far_field_W(res, x_frac=0.6):
    """Sample W(F) on a vertical line well ahead of the tip (central, pure-shear)."""
    msh, u = res["msh"], res["u"]
    c1, c2 = res["c1"], res["c2"]
    Q = fem.functionspace(msh, ("DG", 1))
    Wf = fem.Function(Q)
    Wf.interpolate(fem.Expression(_energy_density_expr(u, c1, c2),
                                  Q.element.interpolation_points))
    bb = geometry.bb_tree(msh, msh.topology.dim)
    x0 = x_frac * res["b"]                      # ahead of tip, away from right edge
    ys = np.linspace(0.05 * res["H"], 0.95 * res["H"], 15)
    pts = np.column_stack([np.full_like(ys, x0), ys, np.zeros_like(ys)])
    cand = geometry.compute_collisions_points(bb, pts)
    coll = geometry.compute_colliding_cells(msh, cand, pts)
    vals = []
    for i in range(len(pts)):
        links = coll.links(i)
        if len(links):
            vals.append(Wf.eval(pts[i:i + 1], [links[0]])[0])
    vals = np.array(vals, float)
    return float(np.mean(vals)), float(np.std(vals))


# ------------------------------------------------------------- domain J-integral
def domain_J_integral(res, r1=0.04, r2=0.18):
    """Finite-strain reference EDI J-integral (half model). G_full = 2 * J_half.

    q = 1 for r<r1, linearly to 0 at r2.  Crack-growth direction = +x (index 0).
    Eshelby momentum Sigma_{J,0} = W delta_{J,0} - sum_i P_{iJ} F_{i0};
    J = -int Sigma_{J,0} q_{,J} dx.
    """
    msh, u = res["msh"], res["u"]
    c1, c2 = res["c1"], res["c2"]
    Vq = fem.functionspace(msh, ("Lagrange", 1))
    q = fem.Function(Vq)

    def qvals(x):
        r = np.sqrt(x[0] ** 2 + x[1] ** 2)
        return np.clip((r2 - r) / (r2 - r1), 0.0, 1.0)

    q.interpolate(qvals)

    d = 2
    F = ufl.variable(ufl.Identity(d) + ufl.grad(u))
    Jc = ufl.det(F)
    I1 = ufl.tr(F.T * F)
    W = c1 * (I1 + Jc ** -2 - 3) + c2 * (Jc ** 2 + I1 * Jc ** -2 - 3)
    P = ufl.diff(W, F)                                   # PK1 = dW/dF
    # Sigma_{J,0} = W [J==0] - sum_i P_{iJ} F_{i0}
    PF = ufl.dot(ufl.transpose(P), F)                    # (PF)_{J,0} = sum_i P_{iJ} F_{i0}
    Sig0 = W * ufl.as_vector([1.0, 0.0]) - ufl.as_vector([PF[0, 0], PF[1, 0]])
    integrand = -ufl.dot(Sig0, ufl.grad(q))
    Jhalf = fem.assemble_scalar(fem.form(integrand * ufl.dx))
    Jhalf = msh.comm.allreduce(Jhalf, op=__import__("mpi4py").MPI.SUM)
    return 2.0 * float(Jhalf)                            # full G


# ------------------------------------------------------------- top-level
def analyze(res, window=(3e-3, 6e-2)):
    c1 = res["c1"]
    # near-tip signatures + opening intensity P (reuse the ray machinery)
    t = mfx.run_tests(res, window, n_r=40)
    P = t["P"]

    W_ff, W_ff_std = far_field_W(res)
    W_inf = res["W_inf"]
    h0 = res["h0"]
    G_spec = W_inf * h0
    G_spec_meas = W_ff * h0
    G_J = domain_J_integral(res)
    P_pred = float(np.sqrt(2.0 * G_spec / (np.pi * c1)))
    P_pred_meas = float(np.sqrt(2.0 * G_spec_meas / (np.pi * c1)))
    P_pred_GJ = float(np.sqrt(2.0 * G_J / (np.pi * c1)))     # from the true J-integral G

    t.pop("rays", None)
    out = {
        "case": {"c1": c1, "c2": res["c2"], "lam": res["lam_reached"],
                 "a": res["a"], "b": res["b"], "h0": h0},
        "signatures": {
            "open_exp": t["p_open"], "J_exp": t["J_exp_mean"],
            "inplane_exp": t["p_inplane"],
            "Jr14_plateau": t["plateau_mean"], "Jr14_spread": t["plateau_rel_spread"],
            "P_measured": P,
        },
        "energy_release": {
            "W_inf_theory": W_inf, "W_ff_measured": W_ff, "W_ff_std": W_ff_std,
            "G_spec_theory": G_spec, "G_spec_measured": G_spec_meas,
            "G_domain_J": G_J,
            "P_pred_from_G_spec": P_pred,
            "P_pred_from_GJ": P_pred_GJ,
            "P_pred_from_measured_W": P_pred_meas,
            "P_measured": P,
            "rel_err_P_vs_pred": abs(P - P_pred) / P_pred if P_pred else None,
            "rel_err_P_vs_predGJ": abs(P - P_pred_GJ) / P_pred_GJ if P_pred_GJ else None,
            "rel_err_GJ_vs_spec": abs(G_J - G_spec) / G_spec if G_spec else None,
        },
    }
    return out


if __name__ == "__main__":
    import json
    from ps_mesh import StripConfig
    from ps_solve import SolveConfig, solve
    res = solve(SolveConfig(lam=1.6, n_steps=18),
                StripConfig(a=3.0, b=6.0, H=0.5, r_min=1e-5, n_r=64, n_theta=120))
    out = analyze(res, window=(3e-4, 8e-3))
    print(json.dumps(out, indent=2))
