#!/usr/bin/env bash
set -euo pipefail

# E2 is the clean Vanilla GRPO baseline:
#   E1 Complex-CoT LoRA -> outcome reward + 0.1 * format reward.
# DAPO/OPD components are intentionally excluded from this script.

BASE_MODEL="${BASE_MODEL:-/root/autodl-tmp/models/Qwen2.5-3B-Instruct}"
E1_ADAPTER="${E1_ADAPTER:-outputs/medreason-e1-qwen25-3b-lora-r16}"
TRAIN_DIR="${TRAIN_DIR:-data/medreason/rl_mcqa/train}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/medreason-e2-vanilla-grpo-qwen25-3b}"

TRAIN_SAMPLES="${TRAIN_SAMPLES:--1}"
MAX_STEPS="${MAX_STEPS:--1}"
NUM_TRAIN_EPOCHS="${NUM_TRAIN_EPOCHS:-1}"
PER_DEVICE_BATCH_SIZE="${PER_DEVICE_BATCH_SIZE:-8}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-4}"
GENERATION_BATCH_SIZE="${GENERATION_BATCH_SIZE:-32}"
NUM_GENERATIONS="${NUM_GENERATIONS:-4}"
MAX_COMPLETION_LENGTH="${MAX_COMPLETION_LENGTH:-768}"
LOGGING_STEPS="${LOGGING_STEPS:-5}"
SAVE_STEPS="${SAVE_STEPS:-100}"
LEARNING_RATE="${LEARNING_RATE:-5e-7}"
SEED="${SEED:-42}"

AUDIT_DIR="${OUTPUT_DIR}/audit"
mkdir -p "${AUDIT_DIR}"

if [[ ! -f "${BASE_MODEL}/config.json" ]]; then
  echo "Base model not found: ${BASE_MODEL}" >&2
  exit 1
fi

if [[ ! -f "${E1_ADAPTER}/adapter_config.json" ]]; then
  echo "E1 adapter not found: ${E1_ADAPTER}" >&2
  exit 1
fi

if [[ ! -f "${TRAIN_DIR}/data.jsonl" ]]; then
  echo "RL training data not found: ${TRAIN_DIR}/data.jsonl" >&2
  exit 1
fi

if (( GENERATION_BATCH_SIZE % NUM_GENERATIONS != 0 )); then
  echo "GENERATION_BATCH_SIZE must be divisible by NUM_GENERATIONS." >&2
  exit 1
fi

if (( GENERATION_BATCH_SIZE % PER_DEVICE_BATCH_SIZE != 0 )); then
  echo "GENERATION_BATCH_SIZE must be divisible by PER_DEVICE_BATCH_SIZE." >&2
  exit 1
fi

export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

CUDA_VISIBLE_DEVICES=0 python training/medical_mcqa_grpo_training.py \
  --model_name_or_path "${BASE_MODEL}" \
  --tokenizer_name_or_path "${BASE_MODEL}" \
  --train_file_dir "${TRAIN_DIR}" \
  --train_samples "${TRAIN_SAMPLES}" \
  --dataset_seed "${SEED}" \
  --preprocessing_num_workers 8 \
  --peft_path "${E1_ADAPTER}" \
  --rollout_audit_file "${AUDIT_DIR}/groups.jsonl" \
  --hard_buffer_file "${AUDIT_DIR}/hard_buffer.jsonl" \
  --output_dir "${OUTPUT_DIR}" \
  --max_steps "${MAX_STEPS}" \
  --num_train_epochs "${NUM_TRAIN_EPOCHS}" \
  --eval_strategy no \
  --save_strategy steps \
  --save_steps "${SAVE_STEPS}" \
  --save_total_limit 2 \
  --logging_strategy steps \
  --logging_steps "${LOGGING_STEPS}" \
  --logging_first_step True \
  --report_to tensorboard \
  --run_name medreason-e2-vanilla-grpo \
  --seed "${SEED}" \
  --data_seed "${SEED}" \
  --remove_unused_columns False \
  --dtype bfloat16 \
  --bf16 True \
  --tf32 True \
  --attn_implementation sdpa \
  --optim adamw_torch_fused \
  --learning_rate "${LEARNING_RATE}" \
  --lr_scheduler_type cosine \
  --warmup_ratio 0.03 \
  --weight_decay 0.0 \
  --max_grad_norm 1.0 \
  --loss_type grpo \
  --scale_rewards group \
  --beta 0.0 \
  --epsilon 0.2 \
  --num_iterations 1 \
  --use_vllm False \
  --temperature 0.8 \
  --top_p 0.95 \
  --num_generations "${NUM_GENERATIONS}" \
  --max_completion_length "${MAX_COMPLETION_LENGTH}" \
  --per_device_train_batch_size "${PER_DEVICE_BATCH_SIZE}" \
  --gradient_accumulation_steps "${GRADIENT_ACCUMULATION_STEPS}" \
  --generation_batch_size "${GENERATION_BATCH_SIZE}" \
  --gradient_checkpointing False \
  --use_peft True \
  --qlora False \
  --load_in_4bit False \
  --load_in_8bit False \
  --lora_target_modules q_proj k_proj v_proj o_proj gate_proj up_proj down_proj \
  --lora_r 16 \
  --lora_alpha 32 \
  --lora_dropout 0.05
