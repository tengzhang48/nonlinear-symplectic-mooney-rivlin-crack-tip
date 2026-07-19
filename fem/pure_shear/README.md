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

The reported load sweep uses $\lambda=1.3$–$2.2$ for Mooney–Rivlin, with
crack-length, mesh, material-ratio, and $c_2=0$ controls. See
`tests/check_claims.py` for the exact stored-data gates.
