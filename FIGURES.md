# Figure reproduction map

`python figures/make_figures.py` rebuilds the nine reproducibility figure pairs in
`figures/rendered/`. It fails before rendering if any required input is
missing. `python figures/make_esi_mesh.py` separately rebuilds the ESI mesh
figure in the pinned FEniCSx environment.

| Current placement | Output | Generator | Exact inputs beyond Python dependencies |
|---|---|---|---|
| Paper Figure 1 | `fig_master.{png,pdf}` | `figures/make_figures.py` | Analytic schematic; no stored data |
| Public explanatory figure; not in the current paper | `fig_hierarchy.{png,pdf}` | `figures/make_figures.py` | Live functions from `analysis/symplectic_dae.py` and `analysis/leading_field.py`; explicitly historical/scaffold, not a completed coupled operator |
| Paper Figure 2 | `fig_chain.{png,pdf}` | `figures/make_figures.py` | `data/fem/strip/summary.csv` |
| Paper Figure 3 | `fig_ps_portrait.{png,pdf}` | `figures/make_figures.py` | `psfield_MR_lam13.npz`, `psfield_MR_lam16.npz`, `psfield_MR_lam22.npz` |
| Paper Figure 5 | `fig_plateau.{png,pdf}` | `figures/make_figures.py` | `summary.csv`; five ray CSVs each for `MR_lam15`, `MR_lam18`, `NH_lam15`, `NH_lam18` |
| Paper Figure 6 | `fig_cratio.{png,pdf}` | `figures/make_figures.py` | `psfield_NH_lam16.npz`, `psfield_MR_lam16_c2_third.npz`, `psfield_MR_lam16.npz`, `psfield_MR_lam16_c2_3.npz` |
| Paper Figure 4 | `fig_solution_compare.{png,pdf}` | `figures/make_figures.py` | `data/fem/disk/fem_case_MR_lam20.json`; `data/analytic/mr_leading_profile.npz`; the in-plane curve is explicitly detrended by its regular $O(r)$ term |
| ESI Figure S2 | `fig_profile_correction.{png,pdf}` | `analysis/profile_mode_audit.py` (invoked by `figures/make_figures.py`) | `fem_case_MR_lam15.json`, `fem_case_MR_lam20.json`; five public strip rays each for `MR_lam16`, `MR_lam22` |
| Paper Figure 7 | `fig_sigma_G.{png,pdf}` | `figures/make_figures.py` | all four `data/fem/disk/fem_case_*.json` files |
| ESI Figure S1 | `fig_esi_mesh.{png,pdf}` | `figures/make_esi_mesh.py` | fresh meshes from `fem/mr_fem_mesh.py` and `fem/pure_shear/ps_mesh.py` |

The strip directory contains a few additional JSON and ray files used by the
machine-readable validation checks and to document the full load, crack-length,
mesh, and material-ratio sweeps. The figure preflight enumerates only the
subset it reads.

The former `fig_tip_shape` rendering is retired. It omitted the $C_s r$ face
term and chose a tangent by closeness to a target slope, so it is not retained
as evidence or as a tracked output. The replacement records one shared
physical $c_0$, per-ray sensitivity fits, and target-free nested face fits in
`data/derived/profile_mode_audit.json`.

SHA-256 hashes for every tracked input, script, and rendering are recorded in
`MANIFEST.sha256`. The manifest deliberately excludes itself.

The analytic NPZ is itself reproducible with
`python data/analytic/build_profile.py`; its ZIP metadata are fixed so an
unchanged environment and solver produce identical bytes.
