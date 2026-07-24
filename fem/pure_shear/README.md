# Pure-shear strip

The upper-half reference domain has height $H=h/2$, an edge crack of length
$a$, a ligament ahead of the tip, and clamped wide grips. The far-ahead state
is homogeneous pure shear $(1,\lambda,1/\lambda)$, giving

$$G=h(c_1+c_2)(\lambda^2+\lambda^{-2}-2).$$

`run_ps.py` solves one case and exports scalar JSON plus five angular ray CSVs.
With `--export-p2-profile`, it also samples the live quadratic finite-element
field on a polar grid. `ps_export.py` exports the six full-field snapshots
required by the figures. `ps_report.py` aggregates a locally generated sweep.
Output defaults to the local `outputs/` directory and is ignored by Git; the
immutable publication inputs are under `data/fem/strip/`.

The production rosette has 120 angular sectors and 15,360 triangles at
`n_r=64`. The two nearest angular rays are aligned exactly to the far rectangle
corners; this preserves the stated strip without either clipping the corners or
creating narrow extra sectors. The archived JSON, ray CSV, and NPZ files record
the angular and radial counts and continuation setting. The production initial
subdivision is 18, except `c2/c1=1/3`, which uses 32; the solver halves a step
adaptively after a failed Newton attempt.

The reported load sweep uses $\lambda=1.3$–$2.2$ for Mooney–Rivlin, with
crack-length, mesh, material-ratio, and $c_2=0$ controls. See
`tests/check_claims.py` for the exact stored-data gates.

The stored JSON retains a legacy near-axis in-plane exponent for provenance,
but it is not a current paper gate. The strip-only
`analysis/profile_mode_audit.py` instead reproduces the ESI finite-window
table with a regular $O(r)$ term and explicitly leaves the residual exponent
unresolved.

## Matching-circle campaign

`ps_mesh.py` can place an exact internal semicircular row at a chosen matching
radius. This is a mesh interface inside the strip, not a hole or a boundary
condition. The nonlinear strip solve remains unchanged. The complete P2
displacement on that row can then drive the inner calculations in
`../run_global_local.py`.

From the repository root, the retained campaign is reproduced with:

```bash
conda activate mr-crack-tip-fem
bash fem/run_global_local_campaign.sh
```

The campaign varies matching radius, core size, and angular resolution. Its
accepted $r^{5/4}$ estimate comes from exact-restriction profiles driven by
the globally solved, tip-refined strip sequence. The same-cell restriction
also checks P2 transfer and discrete equilibrium. A separately refined inner
solve demonstrates one-way displacement submodeling, but its reaction is not
returned to the strip and its field is not used in Figure 7.

The code path, interface-node convention, MPI checks, estimators, and precise
scope are documented in
[`../GLOBAL_LOCAL_WORKFLOW.md`](../GLOBAL_LOCAL_WORKFLOW.md). Results and
claim limits are in
[`../GLOBAL_LOCAL_RESULTS.md`](../GLOBAL_LOCAL_RESULTS.md).
