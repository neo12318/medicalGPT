#!/usr/bin/env python3
"""Build a fixed, token-filtered Complex-CoT SFT dataset for MedReason E1."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from prepare_medreason_data import (
    DEFAULT_DATASET,
    DEFAULT_SUBSET,
    convert_source_row,
    stable_sample_id,
    write_jsonl,
)

from medreason.prompts import MEDICAL_REASONING_SYSTEM_PROMPT


FORBIDDEN_TAG_RE = re.compile(r"</?(?:think|answer)>", flags=re.IGNORECASE)


def normalized_question(text: Any) -> str:
    return re.sub(r"\s+", " ", str(text or "").strip()).casefold()


def percentile(values: list[int], q: float) -> int:
    if not values:
        return 0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, math.ceil(q * len(ordered)) - 1))
    return ordered[index]


def length_summary(values: list[int]) -> dict[str, float | int]:
    if not values:
        return {"count": 0, "min": 0, "p50": 0, "p90": 0, "p95": 0, "p99": 0, "max": 0, "mean": 0.0}
    return {
        "count": len(values),
        "min": min(values),
        "p50": percentile(values, 0.50),
        "p90": percentile(values, 0.90),
        "p95": percentile(values, 0.95),
        "p99": percentile(values, 0.99),
        "max": max(values),
        "mean": round(sum(values) / len(values), 2),
    }


def count_chat_tokens(tokenizer: Any, sft_row: dict[str, Any]) -> int:
    messages = [{"role": "system", "content": sft_row["system_prompt"]}]
    messages.extend(
        {
            "role": "user" if turn["from"] in {"human", "user"} else "assistant",
            "content": turn["value"],
        }
        for turn in sft_row["conversations"]
    )
    token_ids = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=False,
    )
    return len(token_ids)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fixed_split(
    rows: list[dict[str, Any]],
    validation_size: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if validation_size <= 0 or validation_size >= len(rows):
        raise ValueError("validation_size must be positive and smaller than target_num_samples.")
    # Rows already come from a seeded shuffled stream. Keeping this order makes
    # the split deterministic and avoids a second, easy-to-forget randomization.
    return rows[:-validation_size], rows[-validation_size:]


def stream_filtered_rows(
    *,
    tokenizer: Any,
    dataset_name: str,
    subset_name: str | None,
    split: str,
    seed: int,
    shuffle_buffer_size: int,
    target_num_samples: int,
    max_candidates: int,
    min_tokens: int,
    max_tokens: int,
    question_field: str,
    cot_field: str,
    answer_field: str,
    dataset_revision: str | None = None,
) -> tuple[
    list[dict[str, Any]],
    list[int],
    list[int],
    Counter[str],
    list[dict[str, Any]],
    int,
]:
    from datasets import load_dataset

    dataset = load_dataset(
        dataset_name,
        subset_name or None,
        split=split,
        streaming=True,
        revision=dataset_revision,
    ).shuffle(seed=seed, buffer_size=shuffle_buffer_size)

    accepted: list[dict[str, Any]] = []
    accepted_lengths: list[int] = []
    candidate_lengths: list[int] = []
    rejected_counts: Counter[str] = Counter()
    rejected_examples: list[dict[str, Any]] = []
    seen_questions: set[str] = set()
    candidates_seen = 0

    for source_row in dataset:
        if candidates_seen >= max_candidates or len(accepted) >= target_num_samples:
            break
        candidates_seen += 1
        try:
            question = str(source_row.get(question_field) or "").strip()
            cot = str(source_row.get(cot_field) or "").strip()
            answer = str(source_row.get(answer_field) or "").strip()
            if not question or not cot or not answer:
                raise ValueError("missing_required_field")
            if FORBIDDEN_TAG_RE.search(cot) or FORBIDDEN_TAG_RE.search(answer):
                raise ValueError("nested_output_tag")

            question_key = normalized_question(question)
            if question_key in seen_questions:
                raise ValueError("duplicate_question")
            seen_questions.add(question_key)

            sft_row, _ = convert_source_row(
                dict(source_row),
                question_field=question_field,
                cot_field=cot_field,
                answer_field=answer_field,
            )
            sft_row["system_prompt"] = MEDICAL_REASONING_SYSTEM_PROMPT
            token_length = count_chat_tokens(tokenizer, sft_row)
            candidate_lengths.append(token_length)
            if token_length < min_tokens:
                raise ValueError("too_short")
            if token_length > max_tokens:
                raise ValueError("too_long")

            sft_row["_meta"] = {
                "id": stable_sample_id(question, answer),
                "token_length": token_length,
            }
            accepted.append(sft_row)
            accepted_lengths.append(token_length)
        except (TypeError, ValueError) as exc:
            reason = str(exc) or exc.__class__.__name__
            rejected_counts[reason] += 1
            if len(rejected_examples) < 200:
                rejected_examples.append(
                    {
                        "reason": reason,
                        "question": str(source_row.get(question_field) or "")[:500],
                    }
                )

    if len(accepted) < target_num_samples:
        raise RuntimeError(
            f"Only retained {len(accepted)} samples after scanning {candidates_seen}; "
            f"increase --max_candidates or relax token limits."
        )
    return (
        accepted,
        accepted_lengths,
        candidate_lengths,
        rejected_counts,
        rejected_examples,
        candidates_seen,
    )


def strip_meta(rows: Iterable[dict[str, Any]]) -> Iterable[dict[str, Any]]:
    for row in rows:
        yield {
            "conversations": row["conversations"],
            "system_prompt": row["system_prompt"],
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset_name", default=DEFAULT_DATASET)
    parser.add_argument("--dataset_revision", default="main")
    parser.add_argument("--subset_name", default=DEFAULT_SUBSET)
    parser.add_argument("--split", default="train")
    parser.add_argument("--tokenizer_name_or_path", required=True)
    parser.add_argument("--target_num_samples", type=int, default=5_000)
    parser.add_argument("--validation_size", type=int, default=500)
    parser.add_argument("--max_candidates", type=int, default=20_000)
    parser.add_argument("--min_tokens", type=int, default=128)
    parser.add_argument("--max_tokens", type=int, default=4_096)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--shuffle_buffer_size", type=int, default=20_000)
    parser.add_argument("--output_root", type=Path, default=Path("data/medreason/e1_complex_cot"))
    parser.add_argument("--question_field", default="Question")
    parser.add_argument("--cot_field", default="Complex_CoT")
    parser.add_argument("--answer_field", default="Response")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.target_num_samples <= 0:
        raise ValueError("target_num_samples must be positive.")
    if not 0 < args.min_tokens <= args.max_tokens:
        raise ValueError("Require 0 < min_tokens <= max_tokens.")

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer_name_or_path,
        trust_remote_code=True,
    )
    (
        rows,
        token_lengths,
        candidate_lengths,
        rejected_counts,
        rejected_examples,
        candidates_seen,
    ) = stream_filtered_rows(
        tokenizer=tokenizer,
        dataset_name=args.dataset_name,
        subset_name=args.subset_name,
        split=args.split,
        seed=args.seed,
        shuffle_buffer_size=args.shuffle_buffer_size,
        target_num_samples=args.target_num_samples,
        max_candidates=args.max_candidates,
        min_tokens=args.min_tokens,
        max_tokens=args.max_tokens,
        question_field=args.question_field,
        cot_field=args.cot_field,
        answer_field=args.answer_field,
        dataset_revision=args.dataset_revision,
    )
    train_rows, validation_rows = fixed_split(rows, args.validation_size)

    train_path = args.output_root / "train" / "data.jsonl"
    validation_path = args.output_root / "validation" / "data.jsonl"
    rejected_path = args.output_root / "rejected_examples.jsonl"
    write_jsonl(train_path, strip_meta(train_rows))
    write_jsonl(validation_path, strip_meta(validation_rows))
    write_jsonl(rejected_path, rejected_examples)

    manifest = {
        "purpose": "E1 Complex-CoT SFT only; do not use Response as an exact-match GRPO label.",
        "dataset_name": args.dataset_name,
        "dataset_revision": args.dataset_revision,
        "subset_name": args.subset_name,
        "source_split": args.split,
        "tokenizer_name_or_path": args.tokenizer_name_or_path,
        "seed": args.seed,
        "candidates_seen": candidates_seen,
        "accepted_samples": len(rows),
        "train_samples": len(train_rows),
        "validation_samples": len(validation_rows),
        "min_tokens": args.min_tokens,
        "max_tokens": args.max_tokens,
        "candidate_token_length": length_summary(candidate_lengths),
        "accepted_token_length": length_summary(token_lengths),
        "rejected_counts": dict(sorted(rejected_counts.items())),
        "train_ids": [row["_meta"]["id"] for row in train_rows],
        "validation_ids": [row["_meta"]["id"] for row in validation_rows],
        "files": {
            "train": {"path": str(train_path), "sha256": file_sha256(train_path)},
            "validation": {"path": str(validation_path), "sha256": file_sha256(validation_path)},
            "rejected_examples": {"path": str(rejected_path), "sha256": file_sha256(rejected_path)},
        },
    }
    args.output_root.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output_root / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
