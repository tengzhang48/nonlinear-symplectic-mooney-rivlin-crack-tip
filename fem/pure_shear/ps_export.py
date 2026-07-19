"""Solve pure-shear cases and export full fields (npz) for the paper figures.

The compressed arrays contain coordinates, displacement, cells, achieved
stretch, material coefficients, and strip dimensions—the fields consumed by
the specimen and material-ratio figures.

Run in the environment defined by ``environment-fem.yml``:
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


def export(lam: float, c2: float, tag: str):
    OUT.mkdir(parents=True, exist_ok=True)
    mcfg = StripConfig()
    scfg = SolveConfig(c1=1.0, c2=c2, lam=lam)
    res = solve(scfg, mcfg)
    msh, u = res["msh"], res["u"]
    coords = msh.geometry.x[:, :2].copy()
    tris = msh.geometry.dofmap.reshape(-1, 3).astype(np.int64)
    V1 = fem.functionspace(msh, ("Lagrange", 1, (msh.geometry.dim,)))
    u1 = fem.Function(V1)
    u1.interpolate(u)
    dofc = V1.tabulate_dof_coordinates()[:, :2]
    disp = u1.x.array.reshape(-1, 2)[cKDTree(dofc).query(coords)[1]]
    np.savez_compressed(
        OUT / f"psfield_{tag}.npz", coords=coords, disp=disp, tris=tris,
        lam_reached=res["lam_reached"], c1=1.0, c2=c2,
        a=mcfg.a, b=mcfg.b, H=mcfg.H, r_min=mcfg.r_min)
    print(f"exported psfield_{tag}.npz  (N={coords.shape[0]}, "
          f"lam={res['lam_reached']:.4f})", flush=True)


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
