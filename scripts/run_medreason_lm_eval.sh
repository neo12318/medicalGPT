#!/bin/bash
set -euo pipefail

BASE_MODEL="${BASE_MODEL:-/root/autodl-tmp/models/Qwen2.5-3B-Instruct}"
E1_ADAPTER="${E1_ADAPTER:-outputs/medreason-e1-qwen25-3b-lora-r16}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/medreason-eval/lm-eval}"
TASKS="${TASKS:-medqa_4options,medmcqa_explicit}"
BATCH_SIZE="${BATCH_SIZE:-auto}"
LIMIT="${LIMIT:-}"
LM_EVAL_VERSION="${LM_EVAL_VERSION:-0.4.12}"
TASK_PATH="${TASK_PATH:-evaluation/lm_eval_tasks}"

PYTHON_BIN="${PYTHON_BIN:-python}"
LM_EVAL_BIN="${LM_EVAL_BIN:-lm-eval}"

installed_version="$(
  "${PYTHON_BIN}" -c \
    "import importlib.metadata as m; print(m.version('lm_eval'))"
)"
if [[ "${installed_version}" != "${LM_EVAL_VERSION}" ]]; then
  echo "Expected lm-eval ${LM_EVAL_VERSION}, found ${installed_version}." >&2
  echo "Install with: ${PYTHON_BIN} -m pip install 'lm_eval[hf]==${LM_EVAL_VERSION}'" >&2
  exit 1
fi

if [[ ! -d "${BASE_MODEL}" ]]; then
  echo "Base model directory not found: ${BASE_MODEL}" >&2
  exit 1
fi
if [[ ! -f "${E1_ADAPTER}/adapter_config.json" ]]; then
  echo "E1 adapter not found: ${E1_ADAPTER}/adapter_config.json" >&2
  exit 1
fi

mkdir -p "${OUTPUT_ROOT}/base" "${OUTPUT_ROOT}/e1"

common_args=(
  run
  --model hf
  --tasks "${TASKS}"
  --include_path "${TASK_PATH}"
  --device cuda:0
  --batch_size "${BATCH_SIZE}"
  --apply_chat_template
  --seed 42
  --log_samples
)
if [[ -n "${LIMIT}" ]]; then
  common_args+=(--limit "${LIMIT}")
fi

echo "===== Base model ====="
"${LM_EVAL_BIN}" "${common_args[@]}" \
  --model_args "pretrained=${BASE_MODEL},dtype=bfloat16" \
  --output_path "${OUTPUT_ROOT}/base"

echo "===== E1 LoRA adapter ====="
"${LM_EVAL_BIN}" "${common_args[@]}" \
  --model_args "pretrained=${BASE_MODEL},peft=${E1_ADAPTER},dtype=bfloat16" \
  --output_path "${OUTPUT_ROOT}/e1"

echo "Base results: ${OUTPUT_ROOT}/base"
echo "E1 results:   ${OUTPUT_ROOT}/e1"
