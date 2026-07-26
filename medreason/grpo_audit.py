"""Training-time rollout group auditing for medical GRPO experiments.

The pure helpers in this module intentionally do not require a model or GPU.
``AuditedGRPOTrainer`` is a thin TRL integration that records the rewards
returned by the first two reward functions:

1. outcome reward
2. format reward

The trainer writes JSONL only from the main process.  It gathers lightweight
metadata from every process before grouping, so the same code also remains
well-defined if a later experiment is launched with more than one process.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Sequence

from medreason.audit import GROUP_ALL_WRONG, classify_group
from medreason.rewards import completion_content

try:
    from accelerate.utils import gather_object
except ModuleNotFoundError:  # Keep the pure helpers importable in lightweight test environments.

    def gather_object(value):
        return value


try:
    from trl import GRPOTrainer
except ModuleNotFoundError:  # Keep the pure helpers importable without the training stack.

    class GRPOTrainer:  # type: ignore[no-redef]
        def __init__(self, *args, **kwargs):
            raise ModuleNotFoundError(
                "AuditedGRPOTrainer requires TRL. Install the project requirements first."
            )


def _reward_rows(rewards_per_func: Any) -> list[list[float]]:
    """Convert a tensor-like reward matrix into JSON-serializable rows."""
    if hasattr(rewards_per_func, "detach"):
        rewards_per_func = rewards_per_func.detach()
    if hasattr(rewards_per_func, "cpu"):
        rewards_per_func = rewards_per_func.cpu()
    if hasattr(rewards_per_func, "tolist"):
        rewards_per_func = rewards_per_func.tolist()
    rows = [[float(value) for value in row] for row in rewards_per_func]
    if rows and any(len(row) < 2 for row in rows):
        raise ValueError(
            "GRPO audit requires at least two reward columns: outcome then format."
        )
    return rows


def build_training_audit_rows(
    *,
    inputs: Sequence[dict[str, Any]],
    completions: Sequence[Any],
    rewards_per_func: Any,
    num_generations: int,
    step: int,
) -> list[dict[str, Any]]:
    """Build one audit record per contiguous GRPO rollout group."""
    if num_generations < 2:
        raise ValueError("num_generations must be at least 2 for group auditing.")
    reward_rows = _reward_rows(rewards_per_func)
    if not (len(inputs) == len(completions) == len(reward_rows)):
        raise ValueError(
            "inputs, completions, and rewards_per_func must contain the same "
            "number of completion-level rows."
        )
    if len(inputs) % num_generations:
        raise ValueError(
            "The number of completion-level rows must be divisible by num_generations."
        )

    audit_rows: list[dict[str, Any]] = []
    for start in range(0, len(inputs), num_generations):
        end = start + num_generations
        group_inputs = inputs[start:end]
        group_completions = [
            completion_content(completion) for completion in completions[start:end]
        ]
        group_rewards = reward_rows[start:end]
        first = group_inputs[0]

        for field in ("id", "source", "question", "answer"):
            first_value = first.get(field)
            if any(item.get(field) != first_value for item in group_inputs[1:]):
                raise ValueError(
                    f"Rollout group rows disagree on {field!r}; completion ordering "
                    "no longer matches num_generations."
                )

        outcome_rewards = [row[0] for row in group_rewards]
        format_rewards = [row[1] for row in group_rewards]
        correctness = [reward > 0.0 for reward in outcome_rewards]
        audit_rows.append(
            {
                "step": int(step),
                "id": first.get("id"),
                "source": first.get("source"),
                "question": first.get("question"),
                "answer": first.get("answer"),
                "completions": group_completions,
                "rewards": {
                    "outcome": outcome_rewards,
                    "format": format_rewards,
                    "per_function": group_rewards,
                    "total": [sum(row) for row in group_rewards],
                },
                "num_correct": sum(correctness),
                "num_generations": num_generations,
                "group_type": classify_group(correctness),
            }
        )
    return audit_rows


def hard_buffer_rows(audit_rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """Select all-wrong audit records without discarding their failed rollouts."""
    return [
        dict(row)
        for row in audit_rows
        if row.get("group_type") == GROUP_ALL_WRONG
    ]


def append_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    """Append records durably enough for a long-running training process."""
    rows = list(rows)
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        handle.flush()


class AuditedGRPOTrainer(GRPOTrainer):
    """GRPOTrainer that exports outcome-based rollout groups during training."""

    def __init__(
        self,
        *args,
        audit_file: str | Path | None = None,
        hard_buffer_file: str | Path | None = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        output_dir = Path(self.args.output_dir)
        self.audit_file = Path(audit_file) if audit_file else output_dir / "rollout_audit.jsonl"
        self.hard_buffer_file = (
            Path(hard_buffer_file)
            if hard_buffer_file
            else output_dir / "hard_buffer.jsonl"
        )

    def _calculate_rewards(
        self,
        inputs,
        prompts,
        completions,
        completion_ids_list,
    ):
        """Delegate scoring to TRL, then persist the gathered training groups."""
        rewards_per_func = super()._calculate_rewards(
            inputs,
            prompts,
            completions,
            completion_ids_list,
        )
        if not self.model.training:
            return rewards_per_func

        local_payload = [
            {
                "id": sample.get("id"),
                "source": sample.get("source"),
                "question": sample.get("question"),
                "answer": sample.get("answer"),
                "completion": completion_content(completion),
            }
            for sample, completion in zip(inputs, completions, strict=True)
        ]
        gathered_payload = gather_object(local_payload)

        if self.accelerator.is_main_process:
            gathered_inputs = [
                {
                    "id": item.get("id"),
                    "source": item.get("source"),
                    "question": item.get("question"),
                    "answer": item.get("answer"),
                }
                for item in gathered_payload
            ]
            gathered_completions = [
                item.get("completion", "") for item in gathered_payload
            ]
            audit_rows = build_training_audit_rows(
                inputs=gathered_inputs,
                completions=gathered_completions,
                rewards_per_func=rewards_per_func,
                num_generations=self.num_generations,
                step=self.state.global_step,
            )
            append_jsonl(self.audit_file, audit_rows)
            append_jsonl(self.hard_buffer_file, hard_buffer_rows(audit_rows))

        return rewards_per_func
