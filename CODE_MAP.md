# Code and evidence map

This map distinguishes paper-scope derivations, stored-data checks, fresh FEM
solves, and retained non-paper research calculations. Passing a script means
that its encoded gates pass at the evidence level stated here; it is not a
blanket proof of nearby scientific claims.

## Standard paper-scope lane

| Path | Role | Evidence level |
|---|---|---|
| `verification/verify_equations.py` | plane-stress identities, leading constrained map, flux, and closed opening-sector pairing | mixed symbolic derivations and numerical consistency checks |
| `analysis/leading_field.py` | leading exponents, null-family member, formal outer `g` branch, stretches, and stress | exact identities plus numerical ODE evaluation |
| `analysis/check_g_selection.py` | regular-axis outer-branch selection for formal `g` | endpoint asymptotics and quadrature; not full finite-compliance matching |
| `analysis/check_row1_flux.py` | leading flux decomposition on the selected representative | constraint-consistent numerical scaling |
| `analysis/energy_release_rate.py` | `Cs`-independent leading flux, `G=(pi/2)c1 P^2`, and pure-shear loading relation | exact Laurent-coefficient extraction plus numerical quadrature |
| `analysis/derive_constraint_row.py` | linearized constraint row | symbolic derivation plus randomized numerical comparison |
| `analysis/second_variation_reduction.py` | interior constrained second variation | symbolic and sampled leading-order gates |
| `analysis/reduction_steps234.py` | constraint reduction, endpoint series, and base reaction | exact symbolic identities and endpoint series |
| `analysis/gate4_first_material_background.py` | first finite material correction on the selected $F=0$ representative | 35 gates for the closed $\Lambda=7/4$ rung and total face traction |
| `analysis/verify_lambda13_restricted_log.py` | restricted $\Lambda=13/4$ scalar opening compatibility and log normalization on the chosen $F=0$ ($C_s=0$) analytic-axis representative | exact identities plus independent RK4/Cash–Karp convergence; not the complete same-grade source or coupled response |
| `analysis/profile_mode_audit.py` | reproduce the ESI strip finite-window table | stored strip rays only; no figure and no asymptotic residual-exponent claim |
| `fem/pure_shear/` | sole paper-facing FEM boundary-value problem | independent reduced plane-stress strip solve |
| `fem/global_local_submodel.py` | matching-circle restriction, refinement, P2 transfer, reactions, and MPI sampling | one-way strip/local displacement submodel; the independently refined reaction is not fed back |
| `fem/run_global_local.py` | one MPI matching-circle case | fresh strip FEM plus exact restriction or independently refined local solve |
| `fem/pure_shear/fit_r54_campaign_stage0.py` | exact-axis and full-angle residual estimator | deterministic discovery/holdout fits with explicit $C_s,C_h$ scope |
| `fem/audit_r54_two_power.py` | free two-power sensitivity check | recovers target and nuisance exponents without fixing either |
| `fem/summarize_global_local_campaign.py` | hashed campaign aggregation | matching-radius/core/angular/amplitude convergence summary |
| `tests/check_claims.py` | strip provenance and principal stored-data assertions | deterministic checks against curated CSV/JSON/NPZ and matching-circle summary data |
| `figures/make_figures.py` | seven current paper figures | deterministic rendering from tracked analytic and strip inputs |
| `figures/make_fig_asymap.py` | Figure 2 physics map | conceptual fork-and-merge map; no fitted plotting data |
| `figures/make_fig_r54_axis_convergence.py` | Figure 7 residual convergence | reads only the compact hashed matching-circle summary |
| `figures/make_esi_mesh.py` | strip-only ESI mesh figure | fresh mesh construction in the pinned FEniCSx environment |
| `figures/make_esi_matching_circle.py` | ESI matching-circle field and radial-window map | reads the final retained exact-restriction P2 profile; no derivative reconstruction |

## Quarantined auxiliary disk calculation

| Path | Role | Evidence boundary |
|---|---|---|
| `fem/mr_fem_mesh.py`, `fem/mr_fem_solve.py`, `fem/run_one_case.py` | preserve the focused-disk BVP and solver provenance | not pure shear; excluded from paper claims, figures, and standard tests |
| `fem/check_new_signatures.py` | archival postprocessing of stored disk fields | diagnostic only; not validation |
| `data/fem/disk/` | four stored disk cases | negative provenance, integrity-locked but scientifically quarantined |

The disk outer condition prescribes
`F_far=diag(lambda^(-1),lambda)` around the full arc. It imposes strong
crack-parallel compression, pins the outer mouth, and is not equivalent to the
Rivlin–Thomas strip. The high-load branch also develops a same-upper-face
self-intersection without a contact or global-injectivity model.

## Retained non-paper research calculations

| Path | Role | Publication status |
|---|---|---|
| `analysis/symplectic_dae.py` | historical five-row spectral scaffold | not a completed coupled canonical operator |
| `analysis/coupled_shear_completion.py` | formal integer-shear response calculation | research provenance; outside the paper claim ledger and standard runner |
| `analysis/qk_full_mr_ordering.py` | formal first material-order power audit | research provenance |
| `analysis/qk_full_mr_completion.py` | formal first material-order completion | research provenance |
| `analysis/qk_later_companions.py` | formal later companion calculations | research provenance |

These scripts are retained so the research path is inspectable. They do not
turn the historical scaffold into a completed coupled spectrum and are not
used to reproduce a manuscript figure or claim.

## Established versus open

Established at the declared evidence levels:

- the leading constrained opening map and exact
  `y1 -> y1 + F(y2)` null family;
- the opening and Jacobian powers and angularly constant leading
  `J r^(1/4)`;
- exact `Cs`-independence of the leading flux on the superposed truncated map;
- the pure-shear relation fixing `P(lambda)`;
- the tested `I2`-specific compensated-Jacobian contrast against `c2=0`;
- the closed scalar opening sector and its conserved pairing;
- the $r^{5/4}$ residual class and parameter-free exact-axis amplitude for the
  tested $c_1=c_2=1,\lambda=1.6$ strip, with full-angle consistency after the
  homogeneous $C_h$ member is admitted;
- the closed $\Lambda=7/4$ first-material rung on the chosen $F=0$
  representative; and
- the nonzero restricted $\Lambda=13/4$ scalar opening projection and its
  conditional log normalization.

Not established as completed paper results:

- finite-compliance matching that selects `Cs`, `Ch`, or higher amplitudes;
- two-way equilibrium between an independently refined local mesh and the
  outer strip;
- the complete same-grade $\Lambda=13/4$ source, coupled response, and
  specimen-level logarithmic amplitude;
- a full reaction-carrying coupled spectrum and endpoint operator;
- normalized higher-order extraction integrals;
- closure of the inner axis layer; or
- any disk-based physical validation.
