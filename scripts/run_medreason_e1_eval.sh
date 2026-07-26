#!/bin/bash
set -euo pipefail

BASE_MODEL="${BASE_MODEL:-/root/autodl-tmp/models/Qwen2.5-3B-Instruct}"
E1_ADAPTER="${E1_ADAPTER:-outputs/medreason-e1-qwen25-3b-lora-r16}"
INPUT_FILE="${INPUT_FILE:-data/medreason/rl_mcqa/validation/data.jsonl}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/medreason-e1-eval}"
BATCH_SIZE="${BATCH_SIZE:-4}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-1024}"
NUM_SAMPLES="${NUM_SAMPLES:-}"

mkdir -p "${OUTPUT_ROOT}"

common_args=(
  --model_name_or_path "${BASE_MODEL}"
  --tokenizer_name_or_path "${BASE_MODEL}"
  --input_file "${INPUT_FILE}"
  --batch_size "${BATCH_SIZE}"
  --max_new_tokens "${MAX_NEW_TOKENS}"
  --seed 42
  --trust_remote_code
  --overwrite
)
if [[ -n "${NUM_SAMPLES}" ]]; then
  common_args+=(--num_samples "${NUM_SAMPLES}")
fi

CUDA_VISIBLE_DEVICES=0 python tools/evaluate_medical_mcqa.py \
  --output_file "${OUTPUT_ROOT}/base.jsonl" \
  "${common_args[@]}"

CUDA_VISIBLE_DEVICES=0 python tools/evaluate_medical_mcqa.py \
  --peft_path "${E1_ADAPTER}" \
  --output_file "${OUTPUT_ROOT}/e1.jsonl" \
  "${common_args[@]}"

echo "Base summary: ${OUTPUT_ROOT}/base.summary.json"
echo "E1 summary:   ${OUTPUT_ROOT}/e1.summary.json"
