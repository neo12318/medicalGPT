"""Pure helpers for summarizing generative medical MCQA evaluation."""

from __future__ import annotations

import statistics
from collections import Counter, defaultdict
from typing import Any, Iterable

from medreason.audit import classify_group
from medreason.rewards import extract_mcqa_label, is_valid_reasoning_format


def build_mcqa_evaluation_row(
    sample: dict[str, Any],
    completions: Iterable[str],
    generated_tokens: Iterable[int],
    hit_limits: Iterable[bool],
) -> dict[str, Any]:
    """Build one prompt-level record using the strict GRPO label parser."""
    rollout_rows = []
    gold_label = extract_mcqa_label(sample.get("answer"))
    if gold_label is None:
        raise ValueError(f"Sample {sample.get('id')!r} has no canonical A-D answer.")

    for completion, token_count, hit_limit in zip(
        completions,
        generated_tokens,
        hit_limits,
        strict=True,
    ):
        predicted_label = extract_mcqa_label(completion)
        rollout_rows.append(
            {
                "text": completion,
                "predicted_label": predicted_label,
                "correct": predicted_label == gold_label,
                "valid_format": is_valid_reasoning_format(completion),
                "generated_tokens": int(token_count),
                "hit_limit": bool(hit_limit),
            }
        )

    if not rollout_rows:
        raise ValueError("At least one completion is required.")
    correctness = [rollout["correct"] for rollout in rollout_rows]
    return {
        "id": sample.get("id"),
        "source": sample.get("source", "unknown"),
        "source_split": sample.get("source_split"),
        "question": sample["question"],
        "raw_question": sample.get("raw_question"),
        "options": sample.get("options"),
        "gold_label": gold_label,
        "answer_text": sample.get("answer_text"),
        "rollouts": rollout_rows,
        "num_correct": sum(correctness),
        "num_generations": len(rollout_rows),
        "group_type": classify_group(correctness),
    }


def _percentile(values: list[int], percentile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = (len(ordered) - 1) * percentile
    lower = int(index)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = index - lower
    return ordered[lower] * (1 - fraction) + ordered[upper] * fraction


def _summarize_subset(rows: list[dict[str, Any]]) -> dict[str, Any]:
    rollouts = [rollout for row in rows for rollout in row["rollouts"]]
    token_counts = [rollout["generated_tokens"] for rollout in rollouts]
    correct = sum(rollout["correct"] for rollout in rollouts)
    valid_format = sum(rollout["valid_format"] for rollout in rollouts)
    hit_limit = sum(rollout["hit_limit"] for rollout in rollouts)
    group_counts = Counter(row["group_type"] for row in rows)
    sample_any_correct = sum(row["num_correct"] > 0 for row in rows)
    sample_majority_correct = sum(
        row["num_correct"] > row["num_generations"] / 2 for row in rows
    )
    return {
        "samples": len(rows),
        "completions": len(rollouts),
        "accuracy": correct / len(rollouts) if rollouts else 0.0,
        "format_rate": valid_format / len(rollouts) if rollouts else 0.0,
        "truncation_rate": hit_limit / len(rollouts) if rollouts else 0.0,
        "any_correct_rate": sample_any_correct / len(rows) if rows else 0.0,
        "majority_correct_rate": sample_majority_correct / len(rows) if rows else 0.0,
        "group_counts": dict(sorted(group_counts.items())),
        "effective_group_ratio": group_counts.get("mixed", 0) / len(rows) if rows else 0.0,
        "generated_tokens": {
            "mean": statistics.fmean(token_counts) if token_counts else 0.0,
            "p50": _percentile(token_counts, 0.50),
            "p95": _percentile(token_counts, 0.95),
            "max": max(token_counts, default=0),
        },
    }


def summarize_mcqa_evaluation(
    rows: list[dict[str, Any]],
    *,
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Summarize overall and per-source accuracy, format, length, and groups."""
    by_source: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_source[str(row.get("source", "unknown"))].append(row)
    return {
        "config": config or {},
        "overall": _summarize_subset(rows),
        "by_source": {
            source: _summarize_subset(source_rows)
            for source, source_rows in sorted(by_source.items())
        },
    }
