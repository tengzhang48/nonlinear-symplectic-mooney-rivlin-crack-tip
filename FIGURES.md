# Figure reproduction map

`python figures/make_figures.py` rebuilds the five current paper figure pairs
in `figures/rendered/`. It fails before rendering if any required input is
missing. `python figures/make_esi_mesh.py` separately rebuilds the strip-only
ESI mesh figure in the pinned FEniCSx environment.

| Placement | Output | Generator | Exact inputs beyond Python dependencies |
|---|---|---|---|
| Paper Figure 1 | `fig_master.{png,pdf}` | `figures/make_figures.py` | analytic schematic; no stored data |
| Paper Figure 2 | `fig_chain.{png,pdf}` | `figures/make_figures.py` | `data/fem/strip/summary.csv` |
| Paper Figure 3 | `fig_ps_portrait.{png,pdf}` | `figures/make_figures.py` | `psfield_MR_lam13.npz`, `psfield_MR_lam16.npz`, `psfield_MR_lam22.npz` |
| Paper Figure 4 | `fig_plateau.{png,pdf}` | `figures/make_figures.py` | `summary.csv`; five ray CSVs each for `MR_lam15`, `MR_lam18`, `NH_lam15`, `NH_lam18` |
| Paper Figure 5 | `fig_cratio.{png,pdf}` | `figures/make_figures.py` | `psfield_NH_lam16.npz`, `psfield_MR_lam16_c2_third.npz`, `psfield_MR_lam16.npz`, `psfield_MR_lam16_c2_3.npz` |
| ESI Figure S1 | `fig_esi_mesh.{png,pdf}` | `figures/make_esi_mesh.py` | fresh strip mesh from `fem/pure_shear/ps_mesh.py` |

The standard rendered inventory contains exactly these six PDF/PNG pairs.
The disk-derived comparison, stress, and profile figures and the historical
hierarchy rendering are retired and are not regenerated.

Additional strip JSON and ray files support the machine-readable validation
checks and document the full load, crack-length, mesh, and material-ratio
sweeps. The ESI finite-window table is reproduced separately by
`analysis/profile_mode_audit.py` into
`data/derived/profile_mode_audit.json`; it is not a figure.

SHA-256 hashes for every tracked input, script, and rendering are recorded in
`MANIFEST.sha256`. The manifest deliberately excludes itself.
