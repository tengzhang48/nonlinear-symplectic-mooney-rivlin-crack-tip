"""Mesh for the quarantined focused-disk auxiliary boundary-value problem.

This geometry is retained as negative provenance. It is not the paper's
Rivlin--Thomas strip and is excluded from standard figures and tests.

Built directly in dolfinx (no gmsh): a graded half-disk "spider web" polar
mesh centred on the crack tip.  The tip sits at the origin; the upper half
plane theta in [0, pi] is modelled (Mode I symmetry).

Geometry / conventions
----------------------
- Reference polar radius r in [r_min, R], graded geometrically (ratio ~1.15).
- Angle theta in [0, pi], Ntheta uniform wedges.
- theta = 0  -> ligament ahead of the tip (X > 0, Y = 0): symmetry  u_y = 0.
- theta = pi -> crack face          (X < 0, Y = 0): traction free.
- r = R      -> remote boundary: Dirichlet (prescribed Mode-I stretch).
- r = r_min  -> tiny core hole: traction free (keeps the singular point out).

A small core hole at r_min is standard for finite-strain tip problems: the
leading stretch lambda1 ~ r^{-1/2} is integrable but the pointwise singularity
is removed, which both regularises Newton and lets us sample cleanly outside
the core.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from mpi4py import MPI

import basix.ufl
from dolfinx import mesh as dmesh


@dataclass
class MeshConfig:
    r_min: float = 2.0e-6
    R: float = 1.0
    ratio: float = 1.15
    n_theta: int = 72  # wedges over [0, pi]  -> dtheta = 2.5 deg
    n_r: int | None = None
    theta_nodes: np.ndarray | None = None


def _radii(cfg: MeshConfig) -> np.ndarray:
    if not (0.0 < cfg.r_min < cfg.R):
        raise ValueError("mesh radii must satisfy 0 < r_min < R")
    if cfg.n_r is None:
        if cfg.ratio <= 1.0:
            raise ValueError("ratio must exceed one")
        n_r = int(np.ceil(np.log(cfg.R / cfg.r_min) / np.log(cfg.ratio)))
    else:
        n_r = int(cfg.n_r)
        if n_r < 1:
            raise ValueError("n_r must be positive")
    # geometric grading from r_min to R inclusive
    return np.geomspace(cfg.r_min, cfg.R, n_r + 1)


def _theta_nodes(cfg: MeshConfig) -> np.ndarray:
    if cfg.theta_nodes is None:
        if cfg.n_theta < 1:
            raise ValueError("n_theta must be positive")
        return np.linspace(0.0, np.pi, cfg.n_theta + 1)
    theta = np.asarray(cfg.theta_nodes, dtype=float)
    if theta.ndim != 1 or theta.size < 2:
        raise ValueError("theta_nodes must be a one-dimensional array")
    if not np.isclose(theta[0], 0.0) or not np.isclose(theta[-1], np.pi):
        raise ValueError("theta_nodes must span [0, pi]")
    if np.any(np.diff(theta) <= 0.0):
        raise ValueError("theta_nodes must be strictly increasing")
    return theta.copy()


def build_mesh(
        cfg: MeshConfig | None = None,
        comm: MPI.Comm = MPI.COMM_WORLD):
    """Return (msh, info) with a graded half-disk polar mesh."""
    if cfg is None:
        cfg = MeshConfig()

    r = _radii(cfg)
    theta = _theta_nodes(cfg)
    n_r = r.size - 1
    n_th = theta.size - 1

    # node grid: index(k, j) = k * (n_th + 1) + j, k over radii, j over angles
    RR, TT = np.meshgrid(r, theta, indexing="ij")
    X = (RR * np.cos(TT)).ravel()
    Y = (RR * np.sin(TT)).ravel()
    def idx(k, j):
        return k * (n_th + 1) + j

    if comm.rank == 0:
        points = np.column_stack([X, Y]).astype(np.float64)
        cells = []
        for k in range(n_r):
            for j in range(n_th):
                a = idx(k, j)
                b = idx(k + 1, j)
                c = idx(k + 1, j + 1)
                d = idx(k, j + 1)
                # split quad (a,b,c,d) into two triangles
                cells.append([a, b, c])
                cells.append([a, c, d])
        cells = np.asarray(cells, dtype=np.int64)
    else:
        points = np.empty((0, 2), dtype=np.float64)
        cells = np.empty((0, 3), dtype=np.int64)

    elem = basix.ufl.element("Lagrange", "triangle", 1, shape=(2,))
    msh = dmesh.create_mesh(comm, cells, elem, points)

    info = {
        "n_r": int(n_r),
        "n_theta": int(n_th),
        "n_points": int((n_r + 1) * (n_th + 1)),
        "n_cells": int(2 * n_r * n_th),
        "r_min": float(r[0]),
        "r_max": float(r[-1]),
        "radii": r,
        "theta_nodes": theta,
    }
    return msh, info


if __name__ == "__main__":
    cfg = MeshConfig()
    msh, info = build_mesh(cfg)
    for k, v in info.items():
        if k != "radii":
            print(f"{k:10s}: {v}")
    print("radii[:5] :", info["radii"][:5])
    print("radii[-3:]:", info["radii"][-3:])
