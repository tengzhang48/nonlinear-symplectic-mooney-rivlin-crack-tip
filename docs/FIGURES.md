# Figure reproduction map

`python figures/make_figures.py` rebuilds the seven current paper figure pairs
in `figures/rendered/`. It fails before rendering if any required input is
missing. `python figures/make_esi_mesh.py` separately rebuilds the strip-only
ESI mesh figure in the pinned FEniCSx environment.

| Placement | Output | Generator | Exact inputs beyond Python dependencies |
|---|---|---|---|
| Paper Figure 1 | `fig_master.{png,pdf}` | `figures/make_figures.py` | analytic schematic; no stored data |
| Paper Figure 2 | `fig_asymap.{png,pdf}` | `figures/make_fig_asymap.py`, called by `make_figures.py` | conceptual physics map; no fitted plotting data |
| Paper Figure 3 | `fig_chain.{png,pdf}` | `figures/make_figures.py` | `data/fem/strip/summary.csv` |
| Paper Figure 4 | `fig_ps_portrait.{png,pdf}` | `figures/make_figures.py` | `psfield_MR_lam13.npz`, `psfield_MR_lam16.npz`, `psfield_MR_lam22.npz` |
| Paper Figure 5 | `fig_plateau.{png,pdf}` | `figures/make_figures.py` | `summary.csv`; five ray CSVs each for `MR_lam15`, `MR_lam18`, `NH_lam15`, `NH_lam18` |
| Paper Figure 6 | `fig_cratio.{png,pdf}` | `figures/make_figures.py` | `psfield_NH_lam16.npz`, `psfield_MR_lam16_c2_third.npz`, `psfield_MR_lam16.npz`, `psfield_MR_lam16_c2_3.npz` |
| Paper Figure 7 | `fig_r54_axis_convergence.{png,pdf}` | `figures/make_fig_r54_axis_convergence.py`, called by `make_figures.py` | `data/fem/global_local/global_local_campaign_summary_2026-07-23.json`; `data/fem/global_local/global_local_profile_lam16_core2p5e6_nt120_rm1e2.npz` |
| ESI Figure S1 | `fig_esi_mesh.{png,pdf}` | `figures/make_esi_mesh.py` | fresh strip mesh from `fem/pure_shear/ps_mesh.py` |

The standard rendered inventory contains exactly these eight PDF/PNG pairs.
Auxiliary disk-derived comparison, stress, and profile figures and the
historical hierarchy rendering are not part of the current figure set and are
not regenerated.

Additional strip JSON and ray files support the machine-readable validation
checks and document the full load, crack-length, mesh, and material-ratio
sweeps. The ESI finite-window table is reproduced separately by
`analysis/profile_mode_audit.py` into
`data/derived/profile_mode_audit.json`; it is not a figure.

SHA-256 hashes for every tracked input, script, and rendering are recorded in
`MANIFEST.sha256`. The manifest deliberately excludes itself.
