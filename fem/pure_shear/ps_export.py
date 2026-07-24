"""Solve pure-shear cases and export full fields (npz) for the paper figures.

The compressed arrays contain coordinates, displacement, cells, achieved
stretch, material coefficients, and strip dimensions—the fields consumed by
the specimen and material-ratio figures.

Run in the repository's FEniCSx environment:
  python ps_export.py            # batch
  python ps_export.py 1.6 1.0    # one case
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from dolfinx import fem

from ps_mesh import StripConfig
from ps_solve import SolveConfig, solve

OUT = Path(__file__).resolve().parent / "outputs"


def write_snapshot(res, tag: str, out_dir: Path = OUT):
    """Write a solved field with the mesh and continuation metadata."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    msh, u = res["msh"], res["u"]
    info = res["info"]
    coords = msh.geometry.x[:, :2].copy()
    tris = msh.geometry.dofmap.reshape(-1, 3).astype(np.int64)
    V1 = fem.functionspace(msh, ("Lagrange", 1, (msh.geometry.dim,)))
    u1 = fem.Function(V1)
    u1.interpolate(u)
    dofc = V1.tabulate_dof_coordinates()[:, :2]
    disp = u1.x.array.reshape(-1, 2)[cKDTree(dofc).query(coords)[1]]
    np.savez_compressed(
        out_dir / f"psfield_{tag}.npz", coords=coords, disp=disp, tris=tris,
        lam_reached=res["lam_reached"], c1=res["c1"], c2=res["c2"],
        a=info["a"], b=info["b"], H=info["H"], r_min=info["r_min"],
        n_steps=res["n_steps"], n_r=info["n_r"],
        n_theta_base=info["n_theta_base"], n_sectors=info["n_sectors"],
        angular_scheme=info["angular_scheme"],
        corner_angles=np.asarray(info["corner_angles"]))
    print(f"exported psfield_{tag}.npz  (N={coords.shape[0]}, "
          f"lam={res['lam_reached']:.4f})", flush=True)


def export(lam: float, c2: float, tag: str, n_steps: int | None = None):
    mcfg = StripConfig()
    if n_steps is None:
        n_steps = 32 if np.isclose(c2, 1.0 / 3.0) else 18
    scfg = SolveConfig(c1=1.0, c2=c2, lam=lam, n_steps=n_steps)
    res = solve(scfg, mcfg)
    write_snapshot(res, tag, OUT)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        lam, c2 = float(sys.argv[1]), float(sys.argv[2])
        export(lam, c2, f"lam{int(round(10*lam)):02d}_c2_{c2:g}".replace(".", "p"))
    else:
        for lam, c2, tag in [(1.3, 1.0, "MR_lam13"), (1.6, 1.0, "MR_lam16"),
                             (2.2, 1.0, "MR_lam22"),
                             (1.6, 0.0, "NH_lam16"),
                             (1.6, 1.0 / 3.0, "MR_lam16_c2_third"),
                             (1.6, 3.0, "MR_lam16_c2_3")]:
            export(lam, c2, tag)
        print("BATCH DONE", flush=True)
