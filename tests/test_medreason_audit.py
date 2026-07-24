import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from medreason.audit import (
    GROUP_ALL_CORRECT,
    GROUP_ALL_WRONG,
    GROUP_MIXED,
    build_audit_row,
    build_hard_buffer_row,
    classify_group,
)


def test_classify_group():
    assert classify_group([True, True, True, True]) == GROUP_ALL_CORRECT
    assert classify_group([True, False, True, False]) == GROUP_MIXED
    assert classify_group([False, False, False, False]) == GROUP_ALL_WRONG
    with pytest.raises(ValueError):
        classify_group([])


def test_build_audit_and_hard_buffer_rows():
    reference_completion = "<think>gold reasoning</think><answer>B</answer>"
    sample = {
        "id": "sample-1",
        "question": "Question?",
        "answer": "B",
        "reference_completion": reference_completion,
    }
    completions = [
        "<think>wrong</think><answer>A</answer>",
        "<think>wrong</think><answer>C</answer>",
    ]

    audit_row = build_audit_row(sample, completions)
    assert audit_row["group_type"] == GROUP_ALL_WRONG
    assert audit_row["num_correct"] == 0

    hard_row = build_hard_buffer_row(audit_row)
    assert hard_row["prompt_id"] == "sample-1"
    assert hard_row["conversations"][-1]["value"] == reference_completion
    assert hard_row["student_rollouts"] == completions

