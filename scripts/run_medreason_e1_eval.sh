#!/bin/bash
set -euo pipefail

BASE_MODEL="${BASE_MODEL:-/root/autodl-tmp/models/Qwen2.5-3B-Instruct}"
E1_ADAPTER="${E1_ADAPTER:-outputs/medreason-e1-qwen25-3b-lora-r16}"
INPUT_FILE="${INPUT_FILE:-data/medreason/rl_mcqa/validation/data.jsonl}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/medreason-e1-eval}"
BATCH_SIZE="${BATCH_SIZE:-4}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-1024}"

mkdir -p "${OUTPUT_ROOT}"

CUDA_VISIBLE_DEVICES=0 python tools/evaluate_medical_mcqa.py \
  --model_name_or_path "${BASE_MODEL}" \
  --input_file "${INPUT_FILE}" \
  --output_file "${OUTPUT_ROOT}/base.jsonl" \
  --batch_size "${BATCH_SIZE}" \
  --max_new_tokens "${MAX_NEW_TOKENS}" \
  --seed 42 \
  --trust_remote_code \
  --overwrite

CUDA_VISIBLE_DEVICES=0 python tools/evaluate_medical_mcqa.py \
  --model_name_or_path "${BASE_MODEL}" \
  --peft_path "${E1_ADAPTER}" \
  --input_file "${INPUT_FILE}" \
  --output_file "${OUTPUT_ROOT}/e1.jsonl" \
  --batch_size "${BATCH_SIZE}" \
  --max_new_tokens "${MAX_NEW_TOKENS}" \
  --seed 42 \
  --trust_remote_code \
  --overwrite

echo "Base summary: ${OUTPUT_ROOT}/base.summary.json"
echo "E1 summary:   ${OUTPUT_ROOT}/e1.summary.json"
