# Curated data

- `analytic/mr_leading_profile.npz`: deterministic sample of the formal
  regular-axis outer `g(theta)` representative. Regenerate it with
  `python data/analytic/build_profile.py`. It supports analytic checks, not a
  FEM-validated residual-exponent claim.

- `fem/strip/`: the sole paper-facing FEM evidence. It contains scalar
  summaries, angular rays, and six full-field snapshots used by the five paper
  figures. Every JSON, CSV, and NPZ records continuation and corner-aligned
  mesh settings. `ARTIFACTS.sha256` locks every curated strip artifact to its
  filename.

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
  to equations, code, and strip stored-data checks.

Fresh solver output is written beneath ignored `fem/` output directories and
is never consumed implicitly by the publication figure script.
