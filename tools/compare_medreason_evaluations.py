#!/usr/bin/env python3
"""Compare two deterministic MedReason MCQA evaluation JSONL files."""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def read_rows(path: Path) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            sample_id = str(row.get("id", ""))
            if not sample_id:
                raise ValueError(f"Missing id at {path}:{line_number}")
            if sample_id in rows:
                raise ValueError(f"Duplicate id {sample_id!r} in {path}")
            if len(row.get("rollouts", [])) != 1:
                raise ValueError(
                    f"{path}:{line_number} must contain exactly one rollout."
                )
            rows[sample_id] = row
    return rows


def wilson_interval(successes: int, total: int, z: float = 1.96) -> list[float]:
    if total == 0:
        return [0.0, 0.0]
    probability = successes / total
    denominator = 1 + z * z / total
    center = (probability + z * z / (2 * total)) / denominator
    margin = (
        z
        * math.sqrt(
            probability * (1 - probability) / total
            + z * z / (4 * total * total)
        )
        / denominator
    )
    return [center - margin, center + margin]


def exact_mcnemar_p(base_only: int, candidate_only: int) -> float:
    discordant = base_only + candidate_only
    if discordant == 0:
        return 1.0
    lower = min(base_only, candidate_only)
    one_tail = sum(math.comb(discordant, i) for i in range(lower + 1))
    return min(1.0, 2 * one_tail / (2**discordant))


def summarize_pairs(
    pairs: list[tuple[dict[str, Any], dict[str, Any]]],
) -> dict[str, Any]:
    flips: Counter[str] = Counter()
    base_correct = 0
    candidate_correct = 0
    base_format = 0
    candidate_format = 0
    base_tokens: list[int] = []
    candidate_tokens: list[int] = []

    for base_row, candidate_row in pairs:
        base_rollout = base_row["rollouts"][0]
        candidate_rollout = candidate_row["rollouts"][0]
        base_is_correct = bool(base_rollout["correct"])
        candidate_is_correct = bool(candidate_rollout["correct"])
        base_correct += base_is_correct
        candidate_correct += candidate_is_correct
        base_format += bool(base_rollout["valid_format"])
        candidate_format += bool(candidate_rollout["valid_format"])
        base_tokens.append(int(base_rollout["generated_tokens"]))
        candidate_tokens.append(int(candidate_rollout["generated_tokens"]))

        if base_is_correct and candidate_is_correct:
            flips["both_correct"] += 1
        elif base_is_correct:
            flips["base_only"] += 1
        elif candidate_is_correct:
            flips["candidate_only"] += 1
        else:
            flips["both_wrong"] += 1

    total = len(pairs)
    base_accuracy = base_correct / total if total else 0.0
    candidate_accuracy = candidate_correct / total if total else 0.0
    return {
        "samples": total,
        "base": {
            "correct": base_correct,
            "accuracy": base_accuracy,
            "accuracy_ci95_wilson": wilson_interval(base_correct, total),
            "format_rate": base_format / total if total else 0.0,
            "mean_generated_tokens": (
                sum(base_tokens) / len(base_tokens) if base_tokens else 0.0
            ),
        },
        "candidate": {
            "correct": candidate_correct,
            "accuracy": candidate_accuracy,
            "accuracy_ci95_wilson": wilson_interval(candidate_correct, total),
            "format_rate": candidate_format / total if total else 0.0,
            "mean_generated_tokens": (
                sum(candidate_tokens) / len(candidate_tokens)
                if candidate_tokens
                else 0.0
            ),
        },
        "accuracy_delta": candidate_accuracy - base_accuracy,
        "pair_counts": dict(sorted(flips.items())),
        "mcnemar_exact_p": exact_mcnemar_p(
            flips["base_only"],
            flips["candidate_only"],
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base_file", type=Path, required=True)
    parser.add_argument("--candidate_file", type=Path, required=True)
    parser.add_argument("--output_file", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    base_rows = read_rows(args.base_file)
    candidate_rows = read_rows(args.candidate_file)
    if base_rows.keys() != candidate_rows.keys():
        missing_candidate = sorted(base_rows.keys() - candidate_rows.keys())
        missing_base = sorted(candidate_rows.keys() - base_rows.keys())
        raise ValueError(
            "Evaluation IDs differ: "
            f"missing_candidate={missing_candidate[:5]}, "
            f"missing_base={missing_base[:5]}"
        )

    pairs = [(base_rows[sample_id], candidate_rows[sample_id]) for sample_id in base_rows]
    by_source: dict[str, list[tuple[dict[str, Any], dict[str, Any]]]] = defaultdict(
        list
    )
    for pair in pairs:
        by_source[str(pair[0].get("source", "unknown"))].append(pair)

    report = {
        "base_file": str(args.base_file),
        "candidate_file": str(args.candidate_file),
        "overall": summarize_pairs(pairs),
        "by_source": {
            source: summarize_pairs(source_pairs)
            for source, source_pairs in sorted(by_source.items())
        },
    }
    text = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    if args.output_file:
        args.output_file.parent.mkdir(parents=True, exist_ok=True)
        args.output_file.write_text(text, encoding="utf-8")
    print(text, end="")


if __name__ == "__main__":
    main()
