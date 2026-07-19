# Code and evidence map

This map distinguishes direct derivations, numerical checks, finite-element
evidence, and formal/scaffold calculations. A passing script means that its
encoded gates pass at the evidence level stated here; it is not a blanket
proof of every nearby sentence in the manuscript.

| Path | Role | Evidence level |
|---|---|---|
| `verification/verify_equations.py` | 57 encoded manuscript relations | Mixed symbolic derivations, numerical consistency checks, and transcription reductions |
| `analysis/leading_field.py` | Leading exponents, smooth $g$ branch, stretches, stress | Exact identities plus numerical ODE evaluation |
| `analysis/check_g_selection.py` | Finite-part selection of $g(\pi)$ and row-one cancellation | Quadrature, endpoint asymptotics, and symbolic substitution |
| `analysis/check_row1_flux.py` | Row-one orders and carrier of the finite-radius $J$ excess | Constraint-consistent numerical scaling and flux decomposition |
| `analysis/energy_release_rate.py` | $G=(\pi/2)c_1P^2$ and pure-shear loading relation | Exact symbolic limit plus independent numerical quadrature |
| `analysis/derive_constraint_row.py` | Linearized constraint row | Symbolic derivation plus randomized numerical comparison |
| `analysis/second_variation_reduction.py` | Interior constrained second variation | 14 symbolic/sampled leading-order gates |
| `analysis/reduction_steps234.py` | Constraint reduction, endpoint series, base reaction | Exact symbolic identities and endpoint series |
| `analysis/coupled_shear_completion.py` | Coupled response of the integer-shear family | 22 symbolic gates in the stated constrained-action scope |
| `analysis/qk_full_mr_ordering.py` | First material-order power ordering | 7 symbolic gates |
| `analysis/qk_full_mr_completion.py` | First material-order completion | 31 symbolic gates |
| `analysis/qk_later_companions.py` | Later slaved companions and residues | 28 symbolic/general-family gates plus exact representative endpoint checks |
| `analysis/symplectic_dae.py` | Five-row eigenvalue catalogue used in the hierarchy figure | Historical numerical scaffold; not the completed coupled spectrum |
| `data/analytic/build_profile.py` | Selected leading-profile archive | Deterministic sampling of `analysis/leading_field.py` |
| `fem/pure_shear/` | Primary specimen-scale validation | Independent numerical solution of the reduced plane-stress boundary-value problem |
| `fem/` | Secondary disk deep-window cross-check | Independent numerical solution with homogeneous remote-stretch boundary data |
| `fem/check_new_signatures.py` | Stored-data stress and tip-shape checks | Re-analysis of the four curated disk cases; no re-solve |
| `tests/check_claims.py` | Principal stored-data assertions | Deterministic checks against the curated CSV/JSON data |
| `figures/make_figures.py` | Nine main figures | Deterministic rendering from tracked inputs plus live scaffold computation |
| `figures/make_esi_mesh.py` | ESI mesh figure | Fresh mesh construction in the pinned FEniCSx environment |

## Established versus open

Established by the analytic and computational evidence in this repository:

- the leading constrained orbit and its $1/2$, $5/4$, and $-1/4$ powers;
- smooth-branch selection of the slave profile $g$;
- the locally uniaxial stretch magnitudes and angularly constant leading $J$;
- the limiting energy flux $G=(\pi/2)c_1P^2$;
- the pure-shear relation fixing $P(\lambda)$;
- the tested $I_2$-specific kinematic contrast against the $c_2=0$ control;
- the specifically gated higher-order blocks identified by the consolidated
  scripts.

Not established here as completed results:

- a full five-block canonical pencil and conserved pairing;
- normalized extraction integrals for $B$ or $Q_k$;
- finite-compliance matching that selects the candidate higher amplitudes;
- closure of the inner axis layer or the generated $k+3$ rung;
- transfer of the leading orbit to finite-extensibility constitutive laws.
