#!/usr/bin/env bash
set -euo pipefail

# E3a: DAPO-style GRPO using the components supported natively by TRL:
#   - token-level DAPO loss
#   - asymmetric Clip-Higher
#   - repeated policy updates per generation batch
#   - masking of truncated completions
#
# Dynamic sampling is intentionally not claimed here because TRL does not
# provide the original DAPO oversample-and-filter loop. That is reserved for
# the separate E3b trainer extension.

export OUTPUT_DIR="${OUTPUT_DIR:-outputs/medreason-e3a-dapo-qwen25-3b}"
export RUN_NAME="${RUN_NAME:-medreason-e3a-dapo}"
export LOSS_TYPE="${LOSS_TYPE:-dapo}"
export NUM_ITERATIONS="${NUM_ITERATIONS:-2}"
export EPSILON="${EPSILON:-0.2}"
export EPSILON_HIGH="${EPSILON_HIGH:-0.28}"
export MASK_TRUNCATED_COMPLETIONS="${MASK_TRUNCATED_COMPLETIONS:-True}"
export SCALE_REWARDS="${SCALE_REWARDS:-group}"

exec bash scripts/run_medreason_e2_grpo.sh
