# Curated data

- `analytic/mr_leading_profile.npz`: selected leading angular profile used by
  the verifier and the disk comparison figure. Regenerate it with
  `python data/analytic/build_profile.py`.
- `fem/strip/`: primary pure-shear strip summaries, ray samples, and the six
  field snapshots required by the specimen and material-ratio figures. Every
  JSON, ray CSV, and NPZ records the continuation and corner-aligned mesh
  settings; the claims check rejects mixed mesh/protocol provenance. The
  directory-level `ARTIFACTS.sha256` separately locks every curated scalar,
  ray, summary, and field file to its filename; update that lock only after a
  complete, validated solver regeneration. From the repository root,
  `python make_manifest.py` refreshes this lock first and then the repository
  manifest.
- `fem/disk/`: four secondary deep-window cross-check cases. These JSON files
  retain solve/mesh metadata and raw sampled rays; derived fits are recomputed
  by the current check and figure code rather than stored as estimator-dependent
  summaries.
- `claims/principal_claims.json`: compact ledger linking principal claims to
  equations, code, and stored-data checks.

These are the exact arrays used by the tracked renderings. Newly generated FEM
output is written beneath `fem/` and is not consumed implicitly by the figure
script.
