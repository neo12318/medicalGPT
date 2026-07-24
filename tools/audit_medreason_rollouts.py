#!/usr/bin/env python3
"""Generate repeated medical rollouts and export group audit/hard-buffer JSONL."""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from medreason.audit import GROUP_ALL_WRONG, build_audit_row, build_hard_buffer_row
from training.medical_grpo_training import MEDICAL_SYSTEM_PROMPT


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as error:
                raise ValueError(f"Invalid JSON at {path}:{line_number}: {error}") from error
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def model_input_device(model) -> torch.device:
    try:
        return model.get_input_embeddings().weight.device
    except (AttributeError, StopIteration):
        return next(model.parameters()).device


def generate_group(
    *,
    model,
    tokenizer,
    question: str,
    num_generations: int,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
) -> list[str]:
    messages = [
        {"role": "system", "content": MEDICAL_SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    inputs = tokenizer(prompt, return_tensors="pt")
    device = model_input_device(model)
    inputs = {key: value.to(device) for key, value in inputs.items()}
    prompt_length = inputs["input_ids"].shape[1]

    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            num_return_sequences=num_generations,
            max_new_tokens=max_new_tokens,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    return tokenizer.batch_decode(outputs[:, prompt_length:], skip_special_tokens=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model_name_or_path", required=True)
    parser.add_argument("--peft_path")
    parser.add_argument("--input_file", type=Path, required=True)
    parser.add_argument("--output_file", type=Path, required=True)
    parser.add_argument("--hard_buffer_file", type=Path, required=True)
    parser.add_argument("--num_samples", type=int, default=20)
    parser.add_argument("--num_generations", type=int, default=4)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--trust_remote_code", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.num_samples <= 0:
        raise ValueError("num_samples must be positive.")
    if args.num_generations < 2:
        raise ValueError("num_generations must be at least 2 for group auditing.")

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    rows = read_jsonl(args.input_file)
    if not rows:
        raise ValueError(f"No samples found in {args.input_file}")
    rows = rows[: min(args.num_samples, len(rows))]

    tokenizer_source = args.peft_path or args.model_name_or_path
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_source,
        trust_remote_code=args.trust_remote_code,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        dtype=dtype,
        device_map="auto" if torch.cuda.is_available() else None,
        trust_remote_code=args.trust_remote_code,
    )
    if args.peft_path:
        model = PeftModel.from_pretrained(model, args.peft_path, is_trainable=False)
    model.eval()

    audit_rows = []
    for index, sample in enumerate(rows, start=1):
        completions = generate_group(
            model=model,
            tokenizer=tokenizer,
            question=sample["question"],
            num_generations=args.num_generations,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
        )
        audit_row = build_audit_row(sample, completions)
        audit_rows.append(audit_row)
        print(
            f"[{index}/{len(rows)}] id={audit_row['id']} "
            f"group={audit_row['group_type']} correct="
            f"{audit_row['num_correct']}/{audit_row['num_generations']}"
        )

    hard_rows = [
        build_hard_buffer_row(row)
        for row in audit_rows
        if row["group_type"] == GROUP_ALL_WRONG
    ]
    write_jsonl(args.output_file, audit_rows)
    write_jsonl(args.hard_buffer_file, hard_rows)

    group_counts = Counter(row["group_type"] for row in audit_rows)
    summary = {
        "samples": len(audit_rows),
        "num_generations": args.num_generations,
        "group_counts": dict(group_counts),
        "effective_group_ratio": group_counts.get("mixed", 0) / len(audit_rows),
        "hard_buffer_samples": len(hard_rows),
        "audit_file": str(args.output_file),
        "hard_buffer_file": str(args.hard_buffer_file),
    }
    summary_path = args.output_file.with_suffix(".summary.json")
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
