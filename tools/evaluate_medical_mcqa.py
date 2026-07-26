#!/usr/bin/env python3
"""Evaluate Base/PEFT models with the exact MCQA prompt and reward parser."""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch
from peft import PeftModel
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed

from medreason.evaluation import (
    build_mcqa_evaluation_row,
    summarize_mcqa_evaluation,
)
from medreason.prompts import MEDICAL_MCQA_SYSTEM_PROMPT


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"Invalid JSON at {path}:{line_number}: {error}") from error
            if not row.get("question") or not row.get("answer"):
                raise ValueError(f"Missing question/answer at {path}:{line_number}")
            rows.append(row)
    return rows


def write_jsonl_row(handle, row: dict[str, Any]) -> None:
    handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    handle.flush()


def model_input_device(model) -> torch.device:
    return model.get_input_embeddings().weight.device


def completion_length(
    generated_ids: torch.Tensor,
    *,
    eos_token_id: int | list[int] | None,
    max_new_tokens: int,
) -> tuple[int, bool]:
    eos_ids = (
        set(eos_token_id)
        if isinstance(eos_token_id, (list, tuple))
        else {eos_token_id}
        if eos_token_id is not None
        else set()
    )
    for index, token_id in enumerate(generated_ids.tolist(), start=1):
        if token_id in eos_ids:
            return index, False
    return min(int(generated_ids.numel()), max_new_tokens), True


def batched(values: list[Any], size: int):
    for start in range(0, len(values), size):
        yield values[start : start + size]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model_name_or_path", required=True)
    parser.add_argument(
        "--tokenizer_name_or_path",
        help="Tokenizer source. Defaults to model_name_or_path.",
    )
    parser.add_argument("--peft_path")
    parser.add_argument("--input_file", type=Path, required=True)
    parser.add_argument("--output_file", type=Path, required=True)
    parser.add_argument("--summary_file", type=Path)
    parser.add_argument("--num_samples", type=int, default=-1)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_generations", type=int, default=1)
    parser.add_argument("--max_new_tokens", type=int, default=1024)
    parser.add_argument("--do_sample", action="store_true")
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--shuffle", action="store_true")
    parser.add_argument("--trust_remote_code", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.batch_size <= 0:
        raise ValueError("batch_size must be positive.")
    if args.num_generations <= 0:
        raise ValueError("num_generations must be positive.")
    if args.num_generations > 1 and not args.do_sample:
        raise ValueError("num_generations > 1 requires --do_sample.")
    if args.output_file.exists() and not args.overwrite:
        raise FileExistsError(f"{args.output_file} exists; pass --overwrite to replace it.")

    rows = read_jsonl(args.input_file)
    if args.shuffle:
        random.Random(args.seed).shuffle(rows)
    if args.num_samples > 0:
        rows = rows[: min(args.num_samples, len(rows))]
    if not rows:
        raise ValueError("No evaluation samples selected.")

    set_seed(args.seed)
    tokenizer_source = args.tokenizer_name_or_path or args.model_name_or_path
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_source,
        trust_remote_code=args.trust_remote_code,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

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

    config = {
        "model_name_or_path": args.model_name_or_path,
        "tokenizer_name_or_path": tokenizer_source,
        "peft_path": args.peft_path,
        "input_file": str(args.input_file),
        "samples": len(rows),
        "batch_size": args.batch_size,
        "num_generations": args.num_generations,
        "max_new_tokens": args.max_new_tokens,
        "do_sample": args.do_sample,
        "temperature": args.temperature if args.do_sample else None,
        "top_p": args.top_p if args.do_sample else None,
        "seed": args.seed,
        "system_prompt": MEDICAL_MCQA_SYSTEM_PROMPT,
    }
    args.output_file.parent.mkdir(parents=True, exist_ok=True)
    summary_file = args.summary_file or args.output_file.with_suffix(".summary.json")
    summary_file.parent.mkdir(parents=True, exist_ok=True)

    evaluation_rows: list[dict[str, Any]] = []
    started = time.monotonic()
    with args.output_file.open("w", encoding="utf-8", newline="\n") as output_handle:
        progress = tqdm(total=len(rows), desc="MCQA prompts")
        for batch_rows in batched(rows, args.batch_size):
            prompts = [
                tokenizer.apply_chat_template(
                    [
                        {"role": "system", "content": MEDICAL_MCQA_SYSTEM_PROMPT},
                        {"role": "user", "content": row["question"]},
                    ],
                    tokenize=False,
                    add_generation_prompt=True,
                )
                for row in batch_rows
            ]
            inputs = tokenizer(
                prompts,
                return_tensors="pt",
                padding=True,
                pad_to_multiple_of=8,
            )
            device = model_input_device(model)
            inputs = {key: value.to(device) for key, value in inputs.items()}
            prompt_width = inputs["input_ids"].shape[1]
            generation_kwargs: dict[str, Any] = {
                "do_sample": args.do_sample,
                "num_return_sequences": args.num_generations,
                "max_new_tokens": args.max_new_tokens,
                "pad_token_id": tokenizer.pad_token_id,
                "eos_token_id": tokenizer.eos_token_id,
            }
            if args.do_sample:
                generation_kwargs.update(
                    temperature=args.temperature,
                    top_p=args.top_p,
                )
            with torch.inference_mode():
                outputs = model.generate(**inputs, **generation_kwargs)
            generated = outputs[:, prompt_width:]
            decoded = tokenizer.batch_decode(generated, skip_special_tokens=True)

            for row_index, sample in enumerate(batch_rows):
                start = row_index * args.num_generations
                end = start + args.num_generations
                sample_ids = generated[start:end]
                lengths_and_limits = [
                    completion_length(
                        sequence,
                        eos_token_id=tokenizer.eos_token_id,
                        max_new_tokens=args.max_new_tokens,
                    )
                    for sequence in sample_ids
                ]
                evaluation_row = build_mcqa_evaluation_row(
                    sample,
                    decoded[start:end],
                    [item[0] for item in lengths_and_limits],
                    [item[1] for item in lengths_and_limits],
                )
                evaluation_rows.append(evaluation_row)
                write_jsonl_row(output_handle, evaluation_row)
            progress.update(len(batch_rows))
        progress.close()

    config["runtime_seconds"] = time.monotonic() - started
    summary = summarize_mcqa_evaluation(evaluation_rows, config=config)
    summary_file.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
