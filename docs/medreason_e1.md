# MedReason E1: formal Complex-CoT SFT and MCQA RL data

Run all commands from the MedicalGPT repository root.

## What this stage produces

Two independent datasets are built:

```text
HuatuoGPT-o1 Complex-CoT
  -> E1 SFT train/validation
  -> trains the Qwen2.5-3B LoRA adapter

MedQA + MedMCQA
  -> canonical A/B/C/D RL train/validation
  -> reserved for later GRPO; not used by E1 SFT
```

HuatuoGPT-o1 `Response` is open-ended medical text. It is suitable as an SFT
target but is not used as a formal exact-match GRPO label.

## 0. Environment

```bash
cd /root/autodl-tmp/MedicalGPT
git pull
conda activate medicalgpt

export HF_HOME=/root/autodl-tmp/hf-cache
export HF_ENDPOINT=https://hf-mirror.com
mkdir -p "$HF_HOME"

BASE_MODEL=/root/autodl-tmp/models/Qwen2.5-3B-Instruct
```

No manual dataset download is required. `datasets.load_dataset()` downloads or
streams the referenced datasets and caches the required files under `HF_HOME`.

## 1. Build the fixed 5,000-sample E1 dataset

```bash
python tools/prepare_complex_cot_sft.py \
  --dataset_name FreedomIntelligence/medical-o1-reasoning-SFT \
  --dataset_revision main \
  --subset_name en \
  --split train \
  --tokenizer_name_or_path "$BASE_MODEL" \
  --target_num_samples 5000 \
  --validation_size 500 \
  --max_candidates 20000 \
  --min_tokens 128 \
  --max_tokens 4096 \
  --seed 42 \
  --output_root data/medreason/e1_complex_cot
```

The script uses the Qwen2.5 tokenizer and its real chat template, including the
shared system prompt. It rejects:

- empty required fields;
- nested `<think>` or `<answer>` tags;
- normalized duplicate questions;
- samples below 128 tokens;
- samples above 4,096 tokens.

It keeps scanning candidates until exactly 5,000 valid samples remain, then
freezes 4,500 train and 500 validation rows.

Inspect the result:

```bash
wc -l \
  data/medreason/e1_complex_cot/train/data.jsonl \
  data/medreason/e1_complex_cot/validation/data.jsonl

python -m json.tool data/medreason/e1_complex_cot/manifest.json | less

python tools/validate_jsonl.py \
  --file_path data/medreason/e1_complex_cot/train/data.jsonl

python tools/validate_jsonl.py \
  --file_path data/medreason/e1_complex_cot/validation/data.jsonl
```

Acceptance criteria:

```text
train_samples = 4500
validation_samples = 500
accepted_token_length.max <= 4096
train_ids and validation_ids do not overlap
both output SHA256 values are present in manifest.json
```

Keep `manifest.json`; it records the split IDs, tokenizer, filter thresholds,
rejection counts, token percentiles, and output hashes.

## 2. Run tests before training

```bash
pytest -q \
  tests/test_prepare_complex_cot_sft.py \
  tests/test_prepare_medical_mcqa_rl.py \
  tests/test_prepare_medreason_data.py \
  tests/test_medreason_rewards.py
```

## 3. Train formal E1

Start a new output directory. Do not pass the smoke adapter as `--peft_path`;
formal E1 initializes a fresh rank-16 LoRA adapter from the same base model.

```bash
BASE_MODEL="$BASE_MODEL" \
DATA_ROOT=data/medreason/e1_complex_cot \
OUTPUT_DIR=outputs/medreason-e1-qwen25-3b-lora-r16 \
bash scripts/run_medreason_e1_sft.sh
```

Initial configuration:

```text
base model: Qwen2.5-3B-Instruct
context: 4096
LoRA: r=16, alpha=32, dropout=0.05, all linear layers
micro batch: 1
gradient accumulation: 16
epochs: 2
learning rate: 5e-5 with cosine decay
precision: bf16
gradient checkpointing: enabled
```

Monitor:

```bash
watch -n 2 nvidia-smi
tensorboard --logdir outputs/medreason-e1-qwen25-3b-lora-r16/runs --bind_all
```

E1 acceptance criteria:

- training loss decreases without NaN;
- validation loss is recorded at each epoch;
- the best adapter contains `adapter_config.json` and adapter weights;
- held-out prompts produce complete `<think>...</think><answer>...</answer>`;
- no validation data appears in the training directory.

If CUDA runs out of memory, keep `model_max_length=4096` aligned with the
filtered data and first reduce runtime memory (`gradient_checkpointing`, Flash
Attention compatibility, or filter/rebuild at 3072). Do not silently train
4,096-token data with a 2,048-token training limit.

## 4. Build formal MedQA/MedMCQA RL data

This can be done before or after E1 training. It does not use GPU.

```bash
python tools/prepare_medical_mcqa_rl.py \
  --medqa_dataset GBaker/MedQA-USMLE-4-options-hf \
  --medmcqa_dataset openlifescienceai/medmcqa \
  --medqa_train_samples 2500 \
  --medqa_validation_samples 250 \
  --medmcqa_train_samples 2500 \
  --medmcqa_validation_samples 250 \
  --medmcqa_cop_base 0 \
  --seed 42 \
  --output_root data/medreason/rl_mcqa
```

Each row has this form:

```json
{
  "id": "...",
  "source": "medqa",
  "source_split": "train",
  "question": "Clinical stem...\n\nA. ...\nB. ...\nC. ...\nD. ...",
  "options": {"A": "...", "B": "...", "C": "...", "D": "..."},
  "answer": "B",
  "answer_text": "..."
}
```

The GRPO reward uses only the canonical `answer` label. The separate
`medical_mcqa_grpo_training.py` entry accepts `<answer>B</answer>` and rejects
ambiguous forms such as `<answer>A or B</answer>` or open-ended answer text.

Validate:

```bash
wc -l \
  data/medreason/rl_mcqa/train/data.jsonl \
  data/medreason/rl_mcqa/validation/data.jsonl

python -m json.tool data/medreason/rl_mcqa/manifest.json | less

pytest -q \
  tests/test_prepare_medical_mcqa_rl.py \
  tests/test_medreason_rewards.py
```

Expected counts are 5,000 train and 500 validation. Official training and
development splits are preserved; test splits are not used for training.

## 5. Evaluate Base and E1 before GRPO

Use two complementary evaluations. They answer different questions.

### 5.1 Standard multiple-choice likelihood with lm-evaluation-harness

The repository pins the harness version so later runs use the same task
definitions and metrics:

```bash
python -m pip install "lm_eval[hf]==0.4.12"

bash scripts/run_medreason_lm_eval.sh
```

This runs `medqa_4options` and the repository's explicit
`medmcqa_explicit` task for both the Base model and the E1 PEFT adapter. It
records `acc`, `acc_norm`, and per-sample logs under
`outputs/medreason-eval/lm-eval`.

`lm-eval` scores the conditional likelihood of each option text. This is a
useful standard knowledge benchmark, but it does not exercise the trained
`<think>...</think><answer>...</answer>` generation format. Do not use a
limited smoke run as a reportable metric:

```bash
LIMIT=20 OUTPUT_ROOT=outputs/medreason-eval/lm-eval-smoke \
  bash scripts/run_medreason_lm_eval.sh
```

### 5.2 GRPO-aligned generative evaluation

Run the exact prompt format and strict answer parser that formal GRPO will use:

```bash
BATCH_SIZE=16 \
OUTPUT_ROOT=outputs/medreason-eval/generative-500 \
bash scripts/run_medreason_e1_eval.sh
```

For a quick diagnostic only:

```bash
NUM_SAMPLES=50 \
BATCH_SIZE=8 \
OUTPUT_ROOT=outputs/medreason-eval/generative-50 \
bash scripts/run_medreason_e1_eval.sh
```

The generated summaries include accuracy, format rate, truncation rate, output
length, and per-source metrics. Compare Base and E1 as paired predictions:

```bash
python tools/compare_medreason_evaluations.py \
  --base_file outputs/medreason-eval/generative-500/base.jsonl \
  --candidate_file outputs/medreason-eval/generative-500/e1.jsonl \
  --output_file outputs/medreason-eval/generative-500/comparison.json
```

The comparison reports Base-only and E1-only correct cases plus an exact
McNemar p-value. This avoids treating two correlated accuracy estimates as
independent runs.

## 6. Stop point before formal GRPO

Do not start formal GRPO immediately after building the MCQA files. First:

1. evaluate the E1 adapter on held-out MedQA/MedMCQA;
2. run 50-100 prompts with four rollouts each;
3. verify strict label parsing and reward values manually;
4. calculate all-correct, mixed, and all-wrong group ratios;
5. start GRPO only when the reward pipeline and group distribution are sane.

Formal GRPO will use:

```text
base model: the same Qwen2.5-3B-Instruct
peft_path: the selected E1 adapter
training entry: training/medical_mcqa_grpo_training.py
RL data: data/medreason/rl_mcqa
```
