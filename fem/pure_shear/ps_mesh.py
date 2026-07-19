"""Rivlin-Thomas pure-shear (planar-tension) strip with an edge crack.

Upper-half model (Mode-I symmetry), crack TIP AT THE ORIGIN so the near-tip ray
extraction (theta from +x) is reused verbatim:

    reference domain  x in [-a, b],  y in [0, H]         (H = h0/2)
    crack face        y = 0,  x < 0        (traction free)
    ligament          y = 0,  x > 0        (symmetry u_y = 0)   -- tip at (0,0)
    grip              y = H                (u_y = delta, u_x = 0)
    left / right edge x = -a, x = b        (traction free)

Mesh: a TIP-FOCUSED RADIAL mesh (crack-tip rosette) that fills the strip.  For
each wedge angle theta_j in [0,pi] a ray is shot from the tip to the specimen
boundary at distance R_max(theta); graded rings r_min..R_max give fine, nearly
isotropic elements at the tip and a deep near-tip window, coarsening outward.
This distributes elements radially about the tip (unlike a tensor grid, which is
anisotropic off the axes).  A tiny core r_min keeps the singular point out.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from mpi4py import MPI

import basix.ufl
from dolfinx import mesh as dmesh


@dataclass
class StripConfig:
    a: float = 3.0        # crack length   (x in [-a, 0])
    b: float = 6.0        # ligament ahead (x in [0, b])
    H: float = 0.5        # half-height (full height h0 = 2H)
    r_min: float = 1.0e-5  # core radius at the tip
    n_r: int = 64          # graded rings per ray
    n_theta: int = 120     # angular wedges over [0, pi]

    @property
    def h0(self) -> float:
        return 2.0 * self.H

    @property
    def w(self) -> float:
        return self.a + self.b


def _R_max(theta, a, b, H):
    """Distance from the tip to the rectangle boundary along direction theta."""
    c, s = np.cos(theta), np.sin(theta)
    cand = []
    if s > 1e-12:
        cand.append(H / s)                 # top edge y = H
    if c > 1e-12:
        cand.append(b / c)                 # right edge x = b
    elif c < -1e-12:
        cand.append(a / (-c))              # left edge x = -a
    return min(cand)


def build_strip(cfg: StripConfig | None = None):
    if cfg is None:
        cfg = StripConfig()
    theta = np.linspace(0.0, np.pi, cfg.n_theta + 1)
    # insert rays exactly through the two far corners so the mesh boundary
    # follows the rectangle there (otherwise the grip effectively ends at
    # the last ray endpoint on y = H and the free margin relaxes)
    th_corners = np.array([np.arctan2(cfg.H, cfg.b),
                           np.pi - np.arctan2(cfg.H, cfg.a)])
    theta = np.unique(np.concatenate([theta, th_corners]))
    krow = np.arange(cfg.n_r + 1) / cfg.n_r
    nth = theta.size
    nr = cfg.n_r + 1

    X = np.empty((nr, nth))
    Y = np.empty((nr, nth))
    for j, th in enumerate(theta):
        Rm = _R_max(th, cfg.a, cfg.b, cfg.H)
        r = cfg.r_min * (Rm / cfg.r_min) ** krow          # geometric r_min..R_max
        X[:, j] = r * np.cos(th)
        Y[:, j] = r * np.sin(th)
    points = np.column_stack([X.ravel(), Y.ravel()]).astype(np.float64)

    def idx(k, j):
        return k * nth + j

    cells = []
    for k in range(cfg.n_r):
        for j in range(nth - 1):
            p00, p10 = idx(k, j), idx(k + 1, j)
            p11, p01 = idx(k + 1, j + 1), idx(k, j + 1)
            cells.append([p00, p10, p11])
            cells.append([p00, p11, p01])
    cells = np.array(cells, dtype=np.int64)

    elem = basix.ufl.element("Lagrange", "triangle", 1, shape=(2,))
    msh = dmesh.create_mesh(MPI.COMM_WORLD, cells, elem, points)

    info = {
        "a": cfg.a, "b": cfg.b, "H": cfg.H, "h0": cfg.h0, "w": cfg.w,
        "n_r": cfg.n_r, "n_theta": cfg.n_theta, "r_min": cfg.r_min,
        "n_points": int(points.shape[0]), "n_cells": int(cells.shape[0]),
        "w_over_h0": cfg.w / cfg.h0,
    }
    return msh, info


if __name__ == "__main__":
    cfg = StripConfig()
    msh, info = build_strip(cfg)
    for k, v in info.items():
        print(f"{k:12s}: {v}")
    # near-tip radial spacing at the shortest (theta=pi/2) and longest (theta=0) rays
    for th in (np.pi / 2, 0.0):
        Rm = _R_max(th, cfg.a, cfg.b, cfg.H)
        r = cfg.r_min * (Rm / cfg.r_min) ** (np.arange(3) / cfg.n_r)
        print(f"  theta={np.degrees(th):5.1f}: R_max={Rm:.3f}  first radii {r}")
