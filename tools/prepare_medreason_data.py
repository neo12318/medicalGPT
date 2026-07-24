#!/usr/bin/env python3
"""Stream, sample, and convert HuatuoGPT-o1 data for MedicalGPT experiments."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path
from typing import Any, Iterable


DEFAULT_DATASET = "FreedomIntelligence/medical-o1-reasoning-SFT"
DEFAULT_SUBSET = "en"


def _clean_text(value: Any) -> str:
    return str(value or "").strip()


def stable_sample_id(question: str, answer: str) -> str:
    payload = f"{question}\n{answer}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:16]


def convert_source_row(
    row: dict[str, Any],
    *,
    question_field: str = "Question",
    cot_field: str = "Complex_CoT",
    answer_field: str = "Response",
) -> tuple[dict[str, Any], dict[str, Any]]:
    question = _clean_text(row.get(question_field))
    cot = _clean_text(row.get(cot_field))
    answer = _clean_text(row.get(answer_field))
    if not question or not cot or not answer:
        missing = [
            name
            for name, value in (
                (question_field, question),
                (cot_field, cot),
                (answer_field, answer),
            )
            if not value
        ]
        raise ValueError(f"Source row has empty required fields: {missing}")

    sample_id = stable_sample_id(question, answer)
    completion = f"<think>\n{cot}\n</think>\n<answer>\n{answer}\n</answer>"
    sft_row = {
        "conversations": [
            {"from": "human", "value": question},
            {"from": "gpt", "value": completion},
        ]
    }
    rl_row = {
        "id": sample_id,
        "question": question,
        "answer": answer,
        "reference_completion": completion,
    }
    return sft_row, rl_row


def load_reference_rows(
    *,
    dataset_name: str,
    subset_name: str | None,
    split: str,
    num_samples: int,
    seed: int,
    streaming: bool,
    shuffle_buffer_size: int,
) -> list[dict[str, Any]]:
    from datasets import load_dataset

    dataset = load_dataset(
        dataset_name,
        subset_name or None,
        split=split,
        streaming=streaming,
    )
    if streaming:
        dataset = dataset.shuffle(seed=seed, buffer_size=shuffle_buffer_size)
        return list(dataset.take(num_samples))

    if num_samples > len(dataset):
        raise ValueError(
            f"Requested {num_samples} samples, but split {split!r} contains only {len(dataset)}."
        )
    return [dict(row) for row in dataset.shuffle(seed=seed).select(range(num_samples))]


def split_rows(
    rows: list[tuple[dict[str, Any], dict[str, Any]]],
    validation_size: int,
    seed: int,
) -> tuple[list[tuple[dict[str, Any], dict[str, Any]]], list[tuple[dict[str, Any], dict[str, Any]]]]:
    if validation_size < 0 or validation_size >= len(rows):
        raise ValueError("validation_size must be >= 0 and smaller than num_samples.")
    shuffled = list(rows)
    random.Random(seed).shuffle(shuffled)
    if validation_size == 0:
        return shuffled, []
    return shuffled[:-validation_size], shuffled[-validation_size:]


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def output_path(root: Path, task: str, split: str) -> Path:
    return root / task / split / "data.jsonl"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset_name", default=DEFAULT_DATASET)
    parser.add_argument("--subset_name", default=DEFAULT_SUBSET)
    parser.add_argument("--split", default="train")
    parser.add_argument("--num_samples", type=int, default=100)
    parser.add_argument("--validation_size", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_root", type=Path, default=Path("data/medreason/smoke"))
    parser.add_argument("--question_field", default="Question")
    parser.add_argument("--cot_field", default="Complex_CoT")
    parser.add_argument("--answer_field", default="Response")
    parser.add_argument("--streaming", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--shuffle_buffer_size", type=int, default=10_000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.num_samples <= 0:
        raise ValueError("num_samples must be positive.")
    source_rows = load_reference_rows(
        dataset_name=args.dataset_name,
        subset_name=args.subset_name,
        split=args.split,
        num_samples=args.num_samples,
        seed=args.seed,
        streaming=args.streaming,
        shuffle_buffer_size=args.shuffle_buffer_size,
    )
    converted = [
        convert_source_row(
            row,
            question_field=args.question_field,
            cot_field=args.cot_field,
            answer_field=args.answer_field,
        )
        for row in source_rows
    ]
    sample_ids = [row[1]["id"] for row in converted]
    if len(set(sample_ids)) != len(sample_ids):
        duplicate_count = len(sample_ids) - len(set(sample_ids))
        raise ValueError(
            f"The sampled data contains {duplicate_count} duplicate question/answer IDs. "
            "Choose another seed or clean the source before training."
        )
    train_rows, validation_rows = split_rows(converted, args.validation_size, args.seed)

    write_jsonl(output_path(args.output_root, "sft", "train"), (row[0] for row in train_rows))
    write_jsonl(output_path(args.output_root, "sft", "validation"), (row[0] for row in validation_rows))
    write_jsonl(output_path(args.output_root, "rl", "train"), (row[1] for row in train_rows))
    write_jsonl(output_path(args.output_root, "rl", "validation"), (row[1] for row in validation_rows))

    manifest = {
        "dataset_name": args.dataset_name,
        "subset_name": args.subset_name,
        "source_split": args.split,
        "streaming": args.streaming,
        "seed": args.seed,
        "num_samples": len(converted),
        "train_samples": len(train_rows),
        "validation_samples": len(validation_rows),
        "question_field": args.question_field,
        "cot_field": args.cot_field,
        "answer_field": args.answer_field,
        "sample_ids": sample_ids,
    }
    args.output_root.mkdir(parents=True, exist_ok=True)
    (args.output_root / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
