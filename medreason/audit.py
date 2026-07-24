"""Pure helpers for classifying and exporting GRPO rollout groups."""

from __future__ import annotations

from typing import Any, Iterable

from medreason.rewards import answers_match


GROUP_ALL_CORRECT = "all_correct"
GROUP_MIXED = "mixed"
GROUP_ALL_WRONG = "all_wrong"


def classify_group(correctness: Iterable[bool]) -> str:
    values = [bool(value) for value in correctness]
    if not values:
        raise ValueError("A rollout group must contain at least one completion.")
    correct_count = sum(values)
    if correct_count == len(values):
        return GROUP_ALL_CORRECT
    if correct_count == 0:
        return GROUP_ALL_WRONG
    return GROUP_MIXED


def build_audit_row(sample: dict[str, Any], completions: list[str]) -> dict[str, Any]:
    reference = sample["answer"]
    correctness = [answers_match(completion, reference) for completion in completions]
    return {
        "id": sample.get("id"),
        "question": sample["question"],
        "answer": reference,
        "reference_completion": sample.get("reference_completion"),
        "rollouts": [
            {"text": completion, "correct": correct}
            for completion, correct in zip(completions, correctness)
        ],
        "num_correct": sum(correctness),
        "num_generations": len(completions),
        "group_type": classify_group(correctness),
    }


def build_hard_buffer_row(audit_row: dict[str, Any]) -> dict[str, Any]:
    if audit_row["group_type"] != GROUP_ALL_WRONG:
        raise ValueError("Only all-wrong groups can be exported to the hard buffer.")
    reference_completion = audit_row.get("reference_completion")
    if not reference_completion:
        raise ValueError("reference_completion is required for OPD hard-buffer data.")
    return {
        "conversations": [
            {"from": "human", "value": audit_row["question"]},
            {"from": "gpt", "value": reference_completion},
        ],
        "prompt_id": audit_row.get("id"),
        "gold_answer": audit_row["answer"],
        "student_rollouts": [rollout["text"] for rollout in audit_row["rollouts"]],
        "group_type": GROUP_ALL_WRONG,
    }

