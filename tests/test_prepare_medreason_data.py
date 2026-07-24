import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.prepare_medreason_data import convert_source_row, split_rows, stable_sample_id


def _source_row(index=1):
    return {
        "Question": f"Question {index}?",
        "Complex_CoT": f"Reasoning {index}",
        "Response": f"Answer {index}",
    }


def test_convert_source_row_builds_sharegpt_and_rl_records():
    sft_row, rl_row = convert_source_row(_source_row())

    assert sft_row["conversations"][0] == {"from": "human", "value": "Question 1?"}
    assert sft_row["conversations"][1]["value"] == (
        "<think>\nReasoning 1\n</think>\n<answer>\nAnswer 1\n</answer>"
    )
    assert rl_row["question"] == "Question 1?"
    assert rl_row["answer"] == "Answer 1"
    assert rl_row["reference_completion"] == sft_row["conversations"][1]["value"]
    assert rl_row["id"] == stable_sample_id("Question 1?", "Answer 1")


def test_split_rows_is_seeded_and_disjoint():
    converted = [convert_source_row(_source_row(index)) for index in range(10)]
    train_a, validation_a = split_rows(converted, validation_size=2, seed=42)
    train_b, validation_b = split_rows(converted, validation_size=2, seed=42)

    assert [row[1]["id"] for row in train_a] == [row[1]["id"] for row in train_b]
    assert [row[1]["id"] for row in validation_a] == [row[1]["id"] for row in validation_b]
    assert len(train_a) == 8
    assert len(validation_a) == 2
    assert {row[1]["id"] for row in train_a}.isdisjoint(
        {row[1]["id"] for row in validation_a}
    )

