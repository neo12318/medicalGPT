# MedReason smoke-test workflow

This workflow validates the first experiment milestone:

```text
HuatuoGPT-o1 reference
  -> fixed 100-sample SFT/RL files
  -> existing SFT adapter inference
  -> four rollouts per prompt
  -> all-correct / mixed / all-wrong audit
  -> OPD-ready hard buffer
```

Run every command from the MedicalGPT repository root.

After activating the environment, steps 2-4 can also be run together:

```bash
BASE_MODEL=/root/autodl-tmp/models/Qwen2.5-3B-Instruct \
SFT_ADAPTER=outputs/sft-smoke-qwen25-3b-v2 \
bash scripts/run_medreason_smoke.sh
```

## 1. Pull the code and activate the existing environment

```bash
cd /root/autodl-tmp/MedicalGPT
git pull
conda activate medicalgpt

export HF_HOME=/root/autodl-tmp/hf-cache
export HF_ENDPOINT=https://hf-mirror.com
mkdir -p "$HF_HOME"
```

The preparation command uses Hugging Face streaming. It does not require a
manual dataset download, but Hugging Face will still cache streamed files under
`HF_HOME`. The mirror endpoint is required on AutoDL regions that cannot reach
the Hugging Face origin directly.

## 2. Stream and convert 100 HuatuoGPT-o1 samples

```bash
python tools/prepare_medreason_data.py \
  --dataset_name FreedomIntelligence/medical-o1-reasoning-SFT \
  --subset_name en \
  --split train \
  --num_samples 100 \
  --validation_size 10 \
  --seed 42 \
  --streaming \
  --output_root data/medreason/smoke
```

Generated files:

```text
data/medreason/smoke/
├── manifest.json
├── sft/
│   ├── train/data.jsonl
│   └── validation/data.jsonl
└── rl/
    ├── train/data.jsonl
    └── validation/data.jsonl
```

Quick checks:

```bash
wc -l data/medreason/smoke/sft/train/data.jsonl
wc -l data/medreason/smoke/sft/validation/data.jsonl
wc -l data/medreason/smoke/rl/train/data.jsonl
```

Expected counts are 90 training samples and 10 validation samples.

The sampled HuatuoGPT-o1 rows have open-ended answer text. They are appropriate
for validating data conversion and the rollout pipeline. Before formal GRPO,
use a canonical answer representation (for example a multiple-choice label and
option text) or a separately validated verifier; conservative exact text
matching is not a sufficient final reward for every semantically equivalent
free-text medical answer.

## 3. Run the lightweight tests

```bash
pytest -q \
  tests/test_medreason_rewards.py \
  tests/test_prepare_medreason_data.py \
  tests/test_medreason_audit.py
```

## 4. Audit the existing SFT smoke adapter

The command below performs inference only. It generates four completions for
each of 20 prompts and does not update the model.

```bash
CUDA_VISIBLE_DEVICES=0 python tools/audit_medreason_rollouts.py \
  --model_name_or_path /root/autodl-tmp/models/Qwen2.5-3B-Instruct \
  --peft_path outputs/sft-smoke-qwen25-3b-v2 \
  --input_file data/medreason/smoke/rl/train/data.jsonl \
  --num_samples 20 \
  --num_generations 4 \
  --max_new_tokens 256 \
  --temperature 0.8 \
  --top_p 0.95 \
  --seed 42 \
  --output_file outputs/medreason-smoke/audit/rollouts.jsonl \
  --hard_buffer_file outputs/medreason-smoke/audit/hard_buffer.jsonl
```

Outputs:

```text
outputs/medreason-smoke/audit/
├── rollouts.jsonl
├── rollouts.summary.json
└── hard_buffer.jsonl
```

`hard_buffer.jsonl` is already in ShareGPT form and can be consumed by
`training/opd_training.py` later.

## 5. Optional two-step medical GRPO smoke test

Do this only after the audit command succeeds. Use a new output directory on
every rerun because MedicalGPT automatically resumes an existing checkpoint.

```bash
CUDA_VISIBLE_DEVICES=0 python training/medical_grpo_training.py \
  --model_name_or_path /root/autodl-tmp/models/Qwen2.5-3B-Instruct \
  --peft_path outputs/sft-smoke-qwen25-3b-v2 \
  --train_file_dir data/medreason/smoke/rl/train \
  --validation_file_dir data/medreason/smoke/rl/validation \
  --train_samples 20 \
  --eval_samples 10 \
  --dataset_seed 42 \
  --use_peft True \
  --dtype bfloat16 \
  --bf16 True \
  --max_steps 2 \
  --learning_rate 5e-7 \
  --beta 0.0 \
  --loss_type grpo \
  --per_device_train_batch_size 4 \
  --gradient_accumulation_steps 1 \
  --num_generations 4 \
  --max_completion_length 256 \
  --gradient_checkpointing False \
  --use_vllm False \
  --remove_unused_columns False \
  --eval_strategy no \
  --save_strategy no \
  --logging_steps 1 \
  --report_to none \
  --preprocessing_num_workers 2 \
  --output_dir outputs/medical-grpo-smoke-qwen25-3b
```

Do not add `--overwrite_output_dir`; this repository's current argument parser
does not accept it.

## Acceptance criteria

The milestone is complete when:

1. the manifest reports 100 unique sample IDs;
2. all three lightweight tests pass;
3. 20 prompts produce 80 decoded completions;
4. every audit row is classified as `all_correct`, `mixed`, or `all_wrong`;
5. the summary counts add up to 20;
6. all-wrong prompts, if any, are exported to `hard_buffer.jsonl`;
7. the optional two-step GRPO run saves a trainable adapter without a new
   adapter/base-model mismatch.
