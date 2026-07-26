import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from medreason.grpo_audit import build_training_audit_rows, hard_buffer_rows


def repeated_input(sample_id: str, answer: str = "B") -> dict:
    return {
        "id": sample_id,
        "source": "medqa",
        "question": f"Question {sample_id}",
        "answer": answer,
    }


def test_build_training_audit_rows_groups_outcome_column_only():
    inputs = [
        *[repeated_input("mixed") for _ in range(4)],
        *[repeated_input("wrong") for _ in range(4)],
    ]
    completions = [
        "<think>x</think><answer>B</answer>",
        "<think>x</think><answer>A</answer>",
        "<think>x</think><answer>B</answer>",
        "bad format",
        "<think>x</think><answer>A</answer>",
        "<think>x</think><answer>C</answer>",
        "bad format",
        "<think>x</think><answer>D</answer>",
    ]
    rewards = [
        [1.0, 0.1],
        [0.0, 0.1],
        [1.0, 0.1],
        [0.0, 0.0],
        [0.0, 0.1],
        [0.0, 0.1],
        [0.0, 0.0],
        [0.0, 0.1],
    ]

    rows = build_training_audit_rows(
        inputs=inputs,
        completions=completions,
        rewards_per_func=rewards,
        num_generations=4,
        step=7,
    )

    assert [row["group_type"] for row in rows] == ["mixed", "all_wrong"]
    assert rows[0]["step"] == 7
    assert rows[0]["id"] == "mixed"
    assert rows[0]["source"] == "medqa"
    assert rows[0]["question"] == "Question mixed"
    assert rows[0]["answer"] == "B"
    assert rows[0]["completions"] == completions[:4]
    assert rows[0]["rewards"]["outcome"] == [1.0, 0.0, 1.0, 0.0]
    assert rows[0]["rewards"]["format"] == [0.1, 0.1, 0.1, 0.0]
    assert rows[0]["rewards"]["per_function"] == rewards[:4]
    assert rows[0]["rewards"]["total"] == [1.1, 0.1, 1.1, 0.0]

    hard_rows = hard_buffer_rows(rows)
    assert len(hard_rows) == 1
    assert hard_rows[0]["id"] == "wrong"
    assert hard_rows[0]["group_type"] == "all_wrong"
    assert hard_rows[0]["completions"] == completions[4:]


def test_build_training_audit_rows_accepts_trl_conversation_completions():
    completions = [
        [{"role": "assistant", "content": f"completion-{index}"}]
        for index in range(4)
    ]
    rows = build_training_audit_rows(
        inputs=[repeated_input("all-correct") for _ in range(4)],
        completions=completions,
        rewards_per_func=[[1.0, 0.1] for _ in range(4)],
        num_generations=4,
        step=0,
    )
    assert rows[0]["group_type"] == "all_correct"
    assert rows[0]["completions"] == [
        "completion-0",
        "completion-1",
        "completion-2",
        "completion-3",
    ]


def test_build_training_audit_rows_rejects_broken_group_alignment():
    inputs = [repeated_input("first") for _ in range(3)]
    inputs.append(repeated_input("other"))
    with pytest.raises(ValueError, match="disagree"):
        build_training_audit_rows(
            inputs=inputs,
            completions=["x"] * 4,
            rewards_per_func=[[0.0, 0.1]] * 4,
            num_generations=4,
            step=0,
        )


def test_build_training_audit_rows_requires_outcome_and_format_columns():
    with pytest.raises(ValueError, match="at least two reward columns"):
        build_training_audit_rows(
            inputs=[repeated_input("sample") for _ in range(4)],
            completions=["x"] * 4,
            rewards_per_func=[[0.0]] * 4,
            num_generations=4,
            step=0,
        )
