# Code and evidence map

This map distinguishes direct derivations, numerical checks, finite-element
evidence, and formal/scaffold calculations. A passing script means that its
encoded gates pass at the evidence level stated here; it is not a blanket
proof of every nearby sentence in the manuscript.

| Path | Role | Evidence level |
|---|---|---|
| `verification/verify_equations.py` | 58 encoded equation relations, including the $C_s s$ null mode and conditional profile powers | Mixed symbolic derivations, numerical consistency checks, and transcription reductions |
| `analysis/leading_field.py` | Leading exponents, regular null-family member, selected outer $g$ branch, stretches, stress | Exact identities plus numerical ODE evaluation |
| `analysis/check_g_selection.py` | Regular-axis outer-branch selection of $g(\pi)$ and a $C_s=0$ row-one diagnostic | Quadrature, endpoint asymptotics, and symbolic substitution; not a full axis-layer matching proof |
| `analysis/check_row1_flux.py` | Row-one orders and carrier of the finite-radius $J$ excess on the selected $C_s=0$ map | Constraint-consistent numerical scaling and flux decomposition in that stated scope |
| `analysis/energy_release_rate.py` | Exact $C_s$-independence of the leading flux on the superposed truncated map, $G=(\pi/2)c_1P^2$, and the pure-shear loading relation | Exact Laurent-coefficient flux extraction plus separate numerical quadrature; not a completed retained-$C_s$ finite-compliance branch |
| `analysis/profile_mode_audit.py` | Shared-$c_0$, $C_s$-aware stored-data profile audit | Fixed-$5/4$ background extraction, per-ray/intercept sensitivity, and target-free nested face fits |
| `analysis/derive_constraint_row.py` | Linearized constraint row | Symbolic derivation plus randomized numerical comparison |
| `analysis/second_variation_reduction.py` | Interior constrained second variation | 14 symbolic/sampled leading-order gates |
| `analysis/reduction_steps234.py` | Constraint reduction, endpoint series, base reaction | Exact symbolic identities and endpoint series |
| `analysis/coupled_shear_completion.py` | Coupled response of the integer-shear family | 22 symbolic gates in the stated constrained-action scope |
| `analysis/qk_full_mr_ordering.py` | First material-order power ordering | 7 symbolic gates |
| `analysis/qk_full_mr_completion.py` | First material-order completion | 31 symbolic gates |
| `analysis/qk_later_companions.py` | Later slaved companions and residues | 28 symbolic/general-family gates plus exact representative endpoint checks |
| `analysis/symplectic_dae.py` | Five-row eigenvalue catalogue used in the hierarchy figure | Historical numerical scaffold; not the completed coupled spectrum |
| `data/analytic/build_profile.py` | Selected leading-profile archive | Deterministic sampling of `analysis/leading_field.py` |
| `fem/pure_shear/` | Primary specimen-scale validation | Separate numerical solution of the reduced plane-stress boundary-value problem |
| `fem/` | Secondary disk deep-window consistency check | Separate numerical solution with homogeneous remote-stretch boundary data |
| `fem/check_new_signatures.py` | Stored-data leading-stress check | Re-analysis of the four curated disk cases; no re-solve; the invalid raw-shape gate is removed |
| `tests/check_claims.py` | Principal stored-data assertions | Deterministic checks against the curated CSV/JSON data |
| `figures/make_figures.py` | Nine reproducibility figures | Deterministic rendering from tracked inputs plus live scaffold computation and the corrected profile audit |
| `figures/make_esi_mesh.py` | ESI mesh figure | Fresh mesh construction in the pinned FEniCSx environment |

## Established versus open

Established by the analytic and computational evidence in this repository:

- the formal leading constrained opening map, its exact
  $y_1\mapsto y_1+F(y_2)$ null family, and the $1/2$ and $-1/4$
  opening/Jacobian powers;
- the selected regular-axis outer branch of the $r^{5/4}g(\theta)$ residual,
  within its stated matching assumption;
- the locally uniaxial stretch magnitudes and angularly constant leading $J$;
- exact $C_s$-independence of the leading flux
  $G=(\pi/2)c_1P^2$ on the superposed truncated map;
- the pure-shear relation fixing $P(\lambda)$;
- the tested $I_2$-specific kinematic contrast against the $c_2=0$ control;
- a nonzero $s$-like regular background and raw slopes near $1/2$ on the
  stored finite annuli (not a universal asymptotic profile theorem);
- the specifically gated higher-order blocks identified by the consolidated
  scripts.

Not established here as completed results:

- a full five-block canonical pencil and conserved pairing;
- normalized extraction integrals for $B$ or $Q_k$;
- finite-compliance matching that selects the candidate higher amplitudes;
- a proof or matched computation that forces $C_s=0$;
- a full retained-$C_s$ finite-compliance equilibrium branch and its matching;
- a universal raw $2/5$ face profile or direct-camera discriminator;
- geometry-independent selection of the residual $5/4$ power from the
  current finite-window data;
- closure of the inner axis layer or the generated $k+3$ rung;
- transfer of the leading orbit to finite-extensibility constitutive laws.
