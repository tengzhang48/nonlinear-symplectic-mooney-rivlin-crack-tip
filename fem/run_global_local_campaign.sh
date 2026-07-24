#!/usr/bin/env bash
set -euo pipefail

# Reproduce the bounded one-way global--local convergence campaign.
# Run from the repository root after activating environment-fem.yml.

PY="${PY:-$(command -v python)}"
MPIEXEC="${MPIEXEC:-$(command -v mpiexec)}"
OUT_DIR="${OUT_DIR:-fem/global_local_outputs}"

if [[ -z "$PY" || -z "$MPIEXEC" ]]; then
  echo "python and mpiexec must both be available in PATH" >&2
  exit 2
fi

"$PY" -c 'import dolfinx, mpi4py, petsc4py'
mkdir -p "$OUT_DIR"

run_case() {
  local ranks="$1"
  local tag="$2"
  local core="$3"
  local matching_radius="$4"
  local matching_inner="$5"
  local radial_rows="$6"
  local angular_sectors="$7"
  local refinement_levels="$8"
  local profile_hi="$9"

  OMP_NUM_THREADS=1 \
  OPENBLAS_NUM_THREADS=1 \
  MKL_NUM_THREADS=1 \
  NUMEXPR_NUM_THREADS=1 \
    "$MPIEXEC" -n "$ranks" "$PY" fem/run_global_local.py \
    --tag "$tag" --out "$OUT_DIR" \
    --c1 1 --c2 1 --lam 1.6 --a 3 --b 6 --H 0.5 \
    --r-min "$core" --matching-radius "$matching_radius" \
    --matching-n-inner "$matching_inner" \
    --strip-n-r "$radial_rows" --strip-n-theta "$angular_sectors" \
    --strip-load-steps 18 --local-mode nested \
    --local-refinement-levels "$refinement_levels" \
    --local-fallback-steps 16 --factor-solver mumps \
    --export-profile --profile-r-lo 1e-4 --profile-r-hi "$profile_hi" \
    --profile-n-r 160 --profile-n-theta 181
}

fit_case() {
  local tag="$1"
  local output_name="$2"
  "$PY" fem/pure_shear/fit_r54_campaign_stage0.py \
    --profile "$OUT_DIR/global_local_profile_${tag}.npz" \
    --metadata "$OUT_DIR/global_local_${tag}.json" \
    --output "$OUT_DIR/$output_name" \
    --window 1.5e-4:1.6e-3 \
    --window 3e-4:3e-3 \
    --window 6e-4:3.8e-3 \
    --background-window 1.5e-3:3.8e-3
}

# Matching-radius check at a common core and angular resolution.
run_case 8 lam16_rm5e3_restricted_profile 1e-5 5e-3 28 48 60 0 4e-3
run_case 8 lam16_rm1e2_restricted_profile 1e-5 1e-2 31 48 60 0 8e-3
run_case 8 lam16_rm2e2_restricted_profile 1e-5 2e-2 34 48 60 0 1.6e-2

# Core and angular convergence at Rm=0.01.
run_case 8 lam16_core5e6_rm1e2 5e-6 1e-2 37 56 60 0 8e-3
run_case 8 lam16_core2p5e6_rm1e2 2.5e-6 1e-2 43 64 60 0 8e-3
run_case 16 lam16_core2p5e6_nt120_rm1e2 2.5e-6 1e-2 43 64 120 0 8e-3

# Independently refined one-way submodel at fixed core: the complete P2
# displacement is shared, while the changed local reaction is not fed back to
# the outer strip.
run_case 8 lam16_rm1e2_refine1_profile 1e-5 1e-2 31 48 60 1 8e-3

fit_case lam16_rm5e3_restricted_profile \
  r54_lam16_rm5e3_restricted_profile.json
fit_case lam16_rm1e2_restricted_profile \
  r54_lam16_rm1e2_restricted_common_windows.json
fit_case lam16_rm2e2_restricted_profile \
  r54_lam16_rm2e2_restricted_common_windows.json
fit_case lam16_core5e6_rm1e2 \
  r54_lam16_core5e6_rm1e2.json
fit_case lam16_core2p5e6_rm1e2 \
  r54_lam16_core2p5e6_rm1e2.json
fit_case lam16_core2p5e6_nt120_rm1e2 \
  r54_lam16_core2p5e6_nt120_rm1e2.json

"$PY" fem/audit_r54_two_power.py \
  --profile "$OUT_DIR/global_local_profile_lam16_core2p5e6_nt120_rm1e2.npz" \
  --metadata "$OUT_DIR/global_local_lam16_core2p5e6_nt120_rm1e2.json" \
  --output "$OUT_DIR/r54_axis_two_power_sensitivity.json"

"$PY" fem/summarize_global_local_campaign.py \
  --input-dir "$OUT_DIR" \
  --output "$OUT_DIR/global_local_campaign_summary.json"

echo "campaign complete: $OUT_DIR/global_local_campaign_summary.json"
