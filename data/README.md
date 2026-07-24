# Curated data

- `analytic/mr_leading_profile.npz`: deterministic sample of the formal
  regular-axis outer `g(theta)` representative. Regenerate it with
  `python data/analytic/build_profile.py`. It supports analytic checks, not a
  FEM-validated residual-exponent claim.

- `fem/strip/`: the specimen-scale pure-shear records used by Figures 3--6.
  It contains scalar summaries, angular rays, and six full-field snapshots.
  Every JSON, CSV, and NPZ records continuation and corner-aligned mesh
  settings. `ARTIFACTS.sha256` locks every curated strip artifact to its
  filename.

- `fem/global_local/`: the matching-circle records used by Figure 7. These
  are live-P2 exact-restriction profiles driven by newly solved pure-shear
  strip configurations, together with solver metadata, deterministic
  residual estimators, and a hashed campaign summary. They support the tested
  \(r^{5/4}\) class but do not select the specimen coefficients \(C_s\) or
  \(C_h\), and they are not outputs of a fully coupled two-way submodel.
  Estimator JSON files preserve the input paths recorded in the original
  production staging directory. In this public layout, the same input
  basenames live beside the estimators in `data/fem/global_local/`; the
  summarizer resolves those basenames and verifies their recorded SHA-256
  hashes.

- `derived/profile_mode_audit.json`: strip-only reproduction record for the
  ESI finite-window table at `lambda=1.6` and `2.2`. It stores the fitted
  regular coefficient, raw face slope, and nested target-free residual-power
  diagnostics. Regenerate it with
  `python analysis/profile_mode_audit.py --write`. These finite-window values
  do not establish an asymptotic residual exponent and produce no figure.

- `fem/disk/`: quarantined negative provenance for an auxiliary focused-disk
  BVP. The outer condition is not equivalent to Rivlin–Thomas pure shear, and
  the high-load branch is globally inadmissible. These files are excluded from
  paper claims, figures, and standard tests. See `fem/disk/README.md`.

- `claims/principal_claims.json`: compact ledger linking current paper claims
  to equations, code, and stored-data checks.

Fresh solver output is written beneath ignored `fem/` output directories and
is never consumed implicitly by the publication figure script.
