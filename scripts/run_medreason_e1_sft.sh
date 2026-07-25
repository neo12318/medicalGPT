#!/usr/bin/env bash
set -euo pipefail

BASE_MODEL="${BASE_MODEL:-/root/autodl-tmp/models/Qwen2.5-3B-Instruct}"
DATA_ROOT="${DATA_ROOT:-data/medreason/e1_complex_cot}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/medreason-e1-qwen25-3b-lora-r16}"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" \
python training/supervised_finetuning.py \
  --model_name_or_path "$BASE_MODEL" \
  --train_file_dir "$DATA_ROOT/train" \
  --validation_file_dir "$DATA_ROOT/validation" \
  --do_train \
  --do_eval \
  --use_peft True \
  --train_on_inputs False \
  --max_train_samples -1 \
  --max_eval_samples -1 \
  --model_max_length 4096 \
  --per_device_train_batch_size 1 \
  --per_device_eval_batch_size 1 \
  --gradient_accumulation_steps 16 \
  --num_train_epochs 2 \
  --learning_rate 5e-5 \
  --lr_scheduler_type cosine \
  --warmup_ratio 0.03 \
  --weight_decay 0.01 \
  --logging_strategy steps \
  --logging_steps 10 \
  --eval_strategy epoch \
  --save_strategy epoch \
  --save_total_limit 2 \
  --load_best_model_at_end True \
  --metric_for_best_model eval_loss \
  --greater_is_better False \
  --preprocessing_num_workers 4 \
  --target_modules all \
  --lora_rank 16 \
  --lora_alpha 32 \
  --lora_dropout 0.05 \
  --torch_dtype bfloat16 \
  --bf16 True \
  --gradient_checkpointing True \
  --flash_attn False \
  --report_to tensorboard \
  --logging_first_step True \
  --seed 42 \
  --data_seed 42 \
  --output_dir "$OUTPUT_DIR" \
  --cache_dir "${HF_HOME:-/root/autodl-tmp/hf-cache}"
