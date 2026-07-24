# Curated global--local campaign data

This directory contains the accepted source records for the bounded
pure-shear global--local \(r^{5/4}\) campaign. It excludes smoke tests,
superseded windows, logs, and the large independently refined profile.

## Contents

- six compressed live-P2 polar profiles (`global_local_profile_*.npz`);
- six matching solver metadata records (`global_local_*.json`);
- one solver metadata record for the independently refined one-way submodel;
- six deterministic exact-axis/full-angle estimator records
  (`r54_lam16_*.json`); and
- one free two-power exact-axis sensitivity record
  (`r54_axis_two_power_sensitivity.json`).

The file
`global_local_campaign_summary_2026-07-23.json` records the SHA-256 hash of
every source used in the campaign conclusion. Each estimator record also
contains the hashes of its profile and solver metadata.

## Scope

The profiles come from same-cell exact restrictions of newly solved
pure-shear strips. Each restriction is re-solved with the complete P2 strip
trace before sampling. The profiles support the \(5/4\) exact-axis asymptotic
class and permit a window-sensitive full-angle decomposition. They are not
outputs of a fully coupled two-way hybrid algorithm.

The refined-submodel metadata is retained because it records the exact
prescribed P2 trace, the positive-\(J\) local solution, and the change in the
P2 reaction-coefficient vector when inner refinement is not fed back to the
outer strip. Its profile and estimator are excluded because the accepted
\(q\)-convergence sequence uses the explicit core and angular matrix instead.

## Reproduce

From the repository root:

```bash
conda env create -f environment-fem.yml
conda activate mr-crack-tip-fem
bash fem/run_global_local_campaign.sh
```

The script writes regenerated files under the ignored
`fem/global_local_outputs/` directory. The exact case matrix and estimator
windows are encoded in the script rather than inferred from filenames.

The tracked records were produced on 2026-07-23. Their archived summary
retains the original production provenance and source hashes. A fresh run
records its own command, software versions, MPI size, and summary-time Git
state instead of inheriting the archived provenance.
