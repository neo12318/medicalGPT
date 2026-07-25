#!/usr/bin/env python3
"""Build fixed MedQA/MedMCQA multiple-choice data for verifiable medical RL."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
from pathlib import Path
from typing import Any, Iterable, Mapping


LABELS = ("A", "B", "C", "D")


def clean_text(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "").strip())


def stable_id(source: str, source_id: Any, question: str) -> str:
    payload = f"{source}\n{source_id}\n{question}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:16]


def normalize_options(options: Any) -> dict[str, str]:
    if isinstance(options, Mapping):
        normalized = {str(key).upper(): clean_text(value) for key, value in options.items()}
    elif isinstance(options, (list, tuple)) and len(options) == 4:
        normalized = dict(zip(LABELS, map(clean_text, options)))
    else:
        raise ValueError("options must be a mapping or a four-item sequence")
    if set(normalized) != set(LABELS) or any(not normalized[label] for label in LABELS):
        raise ValueError("options must contain four non-empty A/B/C/D choices")
    return {label: normalized[label] for label in LABELS}


def normalize_label(value: Any) -> str:
    label = clean_text(value).upper()
    match = re.match(r"^([A-D])(?:\b|[\.\):])", label)
    if match:
        return match.group(1)
    if label in LABELS:
        return label
    raise ValueError(f"invalid answer label: {value!r}")


def format_mcqa_prompt(question: str, options: Mapping[str, str]) -> str:
    option_lines = "\n".join(f"{label}. {options[label]}" for label in LABELS)
    return (
        f"{clean_text(question)}\n\n{option_lines}\n\n"
        "Reason through the case, then put only the single best option label "
        "(A, B, C, or D) in <answer>."
    )


def convert_medqa_row(row: Mapping[str, Any], split: str) -> dict[str, Any]:
    question = clean_text(row.get("question"))
    if not question:
        question = clean_text(f"{row.get('sent1') or ''} {row.get('sent2') or ''}")

    raw_options = row.get("options")
    if raw_options is None:
        raw_options = [row.get(f"ending{index}") for index in range(4)]
    options = normalize_options(raw_options)

    if row.get("answer_idx") is not None:
        answer_label = normalize_label(row.get("answer_idx"))
    elif row.get("label") is not None:
        label_index = int(row["label"])
        if not 0 <= label_index < 4:
            raise ValueError(f"invalid zero-based MedQA label: {label_index}")
        answer_label = LABELS[label_index]
    else:
        raise ValueError("MedQA row has neither answer_idx nor label")
    answer_text = clean_text(row.get("answer")) or options[answer_label]
    if not question:
        raise ValueError("empty question")
    if clean_text(options[answer_label]).casefold() != answer_text.casefold():
        raise ValueError("MedQA answer_idx and answer text disagree")
    source_id = row.get("id") or row.get("question")
    return {
        "id": stable_id("medqa", source_id, question),
        "source": "medqa",
        "source_split": split,
        "question": format_mcqa_prompt(question, options),
        "raw_question": question,
        "options": options,
        "answer": answer_label,
        "answer_text": options[answer_label],
    }


def medmcqa_label(cop: Any, cop_base: int = 0) -> str:
    if isinstance(cop, str) and not cop.strip().isdigit():
        return normalize_label(cop)
    index = int(cop) - cop_base
    if not 0 <= index < 4:
        raise ValueError(f"invalid MedMCQA cop={cop!r} for cop_base={cop_base}")
    return LABELS[index]


def convert_medmcqa_row(
    row: Mapping[str, Any],
    split: str,
    *,
    cop_base: int = 0,
) -> dict[str, Any]:
    question = clean_text(row.get("question"))
    options = normalize_options([row.get("opa"), row.get("opb"), row.get("opc"), row.get("opd")])
    answer_label = medmcqa_label(row.get("cop"), cop_base=cop_base)
    if not question:
        raise ValueError("empty question")
    source_id = row.get("id") or row.get("question")
    return {
        "id": stable_id("medmcqa", source_id, question),
        "source": "medmcqa",
        "source_split": split,
        "question": format_mcqa_prompt(question, options),
        "raw_question": question,
        "options": options,
        "answer": answer_label,
        "answer_text": options[answer_label],
        "subject": clean_text(row.get("subject_name")),
        "topic": clean_text(row.get("topic_name")),
    }


def select_rows(dataset: Any, count: int, seed: int) -> list[dict[str, Any]]:
    if count <= 0:
        return []
    if count > len(dataset):
        raise ValueError(f"Requested {count} rows from a split containing {len(dataset)} rows.")
    return [dict(row) for row in dataset.shuffle(seed=seed).select(range(count))]


def pick_split(dataset_dict: Any, requested: str, aliases: tuple[str, ...] = ()) -> tuple[str, Any]:
    for name in (requested, *aliases):
        if name in dataset_dict:
            return name, dataset_dict[name]
    raise KeyError(f"None of the requested splits {(requested, *aliases)} exist; got {list(dataset_dict)}")


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256(path.read_bytes())
    return digest.hexdigest()


def assert_unique_and_disjoint(
    train_rows: list[dict[str, Any]],
    validation_rows: list[dict[str, Any]],
) -> None:
    train_ids = [row["id"] for row in train_rows]
    validation_ids = [row["id"] for row in validation_rows]
    if len(train_ids) != len(set(train_ids)):
        raise ValueError("Duplicate IDs found in RL training data.")
    if len(validation_ids) != len(set(validation_ids)):
        raise ValueError("Duplicate IDs found in RL validation data.")
    overlap = set(train_ids) & set(validation_ids)
    if overlap:
        raise ValueError(f"Train/validation overlap detected: {len(overlap)} IDs.")
    train_questions = [clean_text(row["raw_question"]).casefold() for row in train_rows]
    validation_questions = [clean_text(row["raw_question"]).casefold() for row in validation_rows]
    if len(train_questions) != len(set(train_questions)):
        raise ValueError("Duplicate normalized questions found in RL training data.")
    if len(validation_questions) != len(set(validation_questions)):
        raise ValueError("Duplicate normalized questions found in RL validation data.")
    question_overlap = set(train_questions) & set(validation_questions)
    if question_overlap:
        raise ValueError(
            f"Train/validation question leakage detected: {len(question_overlap)} normalized questions."
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--medqa_dataset", default="GBaker/MedQA-USMLE-4-options-hf")
    parser.add_argument("--medmcqa_dataset", default="openlifescienceai/medmcqa")
    parser.add_argument("--medqa_train_samples", type=int, default=2_500)
    parser.add_argument("--medqa_validation_samples", type=int, default=250)
    parser.add_argument("--medmcqa_train_samples", type=int, default=2_500)
    parser.add_argument("--medmcqa_validation_samples", type=int, default=250)
    parser.add_argument(
        "--medmcqa_cop_base",
        type=int,
        choices=(0, 1),
        default=0,
        help="HF ClassLabel uses 0; use 1 only for raw legacy MedMCQA JSON.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_root", type=Path, default=Path("data/medreason/rl_mcqa"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    from datasets import load_dataset

    medqa = load_dataset(args.medqa_dataset)
    medmcqa = load_dataset(args.medmcqa_dataset)
    medqa_train_name, medqa_train = pick_split(medqa, "train")
    medqa_validation_name, medqa_validation = pick_split(medqa, "validation", aliases=("dev",))
    medmcqa_train_name, medmcqa_train = pick_split(medmcqa, "train")
    medmcqa_validation_name, medmcqa_validation = pick_split(
        medmcqa, "validation", aliases=("dev",)
    )

    train_rows = [
        *(
            convert_medqa_row(row, medqa_train_name)
            for row in select_rows(medqa_train, args.medqa_train_samples, args.seed)
        ),
        *(
            convert_medmcqa_row(row, medmcqa_train_name, cop_base=args.medmcqa_cop_base)
            for row in select_rows(medmcqa_train, args.medmcqa_train_samples, args.seed)
        ),
    ]
    validation_rows = [
        *(
            convert_medqa_row(row, medqa_validation_name)
            for row in select_rows(medqa_validation, args.medqa_validation_samples, args.seed)
        ),
        *(
            convert_medmcqa_row(
                row,
                medmcqa_validation_name,
                cop_base=args.medmcqa_cop_base,
            )
            for row in select_rows(
                medmcqa_validation,
                args.medmcqa_validation_samples,
                args.seed,
            )
        ),
    ]
    random.Random(args.seed).shuffle(train_rows)
    random.Random(args.seed).shuffle(validation_rows)
    assert_unique_and_disjoint(train_rows, validation_rows)

    train_path = args.output_root / "train" / "data.jsonl"
    validation_path = args.output_root / "validation" / "data.jsonl"
    write_jsonl(train_path, train_rows)
    write_jsonl(validation_path, validation_rows)
    manifest = {
        "purpose": "Verifiable multiple-choice RL; gold reward is the canonical A/B/C/D label.",
        "seed": args.seed,
        "train_samples": len(train_rows),
        "validation_samples": len(validation_rows),
        "sources": {
            "medqa": {
                "dataset": args.medqa_dataset,
                "train_split": medqa_train_name,
                "validation_split": medqa_validation_name,
                "train_samples": args.medqa_train_samples,
                "validation_samples": args.medqa_validation_samples,
            },
            "medmcqa": {
                "dataset": args.medmcqa_dataset,
                "train_split": medmcqa_train_name,
                "validation_split": medmcqa_validation_name,
                "cop_base": args.medmcqa_cop_base,
                "train_samples": args.medmcqa_train_samples,
                "validation_samples": args.medmcqa_validation_samples,
            },
        },
        "files": {
            "train": {"path": str(train_path), "sha256": file_sha256(train_path)},
            "validation": {"path": str(validation_path), "sha256": file_sha256(validation_path)},
        },
    }
    args.output_root.mkdir(parents=True, exist_ok=True)
    (args.output_root / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
