#!/bin/bash
set -euo pipefail

BASE_MODEL="${BASE_MODEL:-/root/autodl-tmp/models/Qwen2.5-3B-Instruct}"
SFT_ADAPTER="${SFT_ADAPTER:-outputs/sft-smoke-qwen25-3b-v2}"
DATA_ROOT="${DATA_ROOT:-data/medreason/smoke}"
AUDIT_ROOT="${AUDIT_ROOT:-outputs/medreason-smoke/audit}"
HF_HOME="${HF_HOME:-/root/autodl-tmp/hf-cache}"

export HF_HOME

if [[ ! -d "${BASE_MODEL}" ]]; then
  echo "Base model directory not found: ${BASE_MODEL}" >&2
  exit 1
fi

if [[ ! -f "${SFT_ADAPTER}/adapter_config.json" ]]; then
  echo "SFT adapter not found: ${SFT_ADAPTER}/adapter_config.json" >&2
  exit 1
fi

python tools/prepare_medreason_data.py \
  --dataset_name FreedomIntelligence/medical-o1-reasoning-SFT \
  --subset_name en \
  --split train \
  --num_samples 100 \
  --validation_size 10 \
  --seed 42 \
  --streaming \
  --output_root "${DATA_ROOT}"

if python -c "import pytest" >/dev/null 2>&1; then
  python -m pytest -q \
    tests/test_medreason_rewards.py \
    tests/test_prepare_medreason_data.py \
    tests/test_medreason_audit.py
else
  echo "Warning: pytest is not installed; skipping lightweight tests." >&2
  echo "Install it with: python -m pip install pytest" >&2
fi

CUDA_VISIBLE_DEVICES=0 python tools/audit_medreason_rollouts.py \
  --model_name_or_path "${BASE_MODEL}" \
  --peft_path "${SFT_ADAPTER}" \
  --input_file "${DATA_ROOT}/rl/train/data.jsonl" \
  --num_samples 20 \
  --num_generations 4 \
  --max_new_tokens 256 \
  --temperature 0.8 \
  --top_p 0.95 \
  --seed 42 \
  --output_file "${AUDIT_ROOT}/rollouts.jsonl" \
  --hard_buffer_file "${AUDIT_ROOT}/hard_buffer.jsonl"

echo "MedReason smoke workflow completed."
echo "Summary: ${AUDIT_ROOT}/rollouts.summary.json"
