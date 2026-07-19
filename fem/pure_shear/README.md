# Pure-shear strip

The upper-half reference domain has height $H=h/2$, an edge crack of length
$a$, a ligament ahead of the tip, and clamped wide grips. The far-ahead state
is homogeneous pure shear $(1,\lambda,1/\lambda)$, giving

$$G=h(c_1+c_2)(\lambda^2+\lambda^{-2}-2).$$

`run_ps.py` solves one case and exports scalar JSON plus five angular ray CSVs.
`ps_export.py` exports the six full-field snapshots required by the figures.
`ps_report.py` aggregates a locally generated sweep. Output defaults to the
local `outputs/` directory and is ignored by Git; the immutable publication
inputs are under `data/fem/strip/`.

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
