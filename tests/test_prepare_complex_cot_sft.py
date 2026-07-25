import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))

from prepare_complex_cot_sft import (
    fixed_split,
    length_summary,
    normalized_question,
    token_id_length,
)


def test_length_summary_uses_nearest_rank_percentiles():
    assert length_summary([10, 20, 30, 40, 50]) == {
        "count": 5,
        "min": 10,
        "p50": 30,
        "p90": 50,
        "p95": 50,
        "p99": 50,
        "max": 50,
        "mean": 30.0,
    }


def test_fixed_split_and_question_normalization():
    rows = [{"index": index} for index in range(10)]
    train, validation = fixed_split(rows, 2)
    assert [row["index"] for row in train] == list(range(8))
    assert [row["index"] for row in validation] == [8, 9]
    assert normalized_question("  A   Medical QUESTION? ") == "a medical question?"


def test_token_id_length_supports_transformers_return_shapes():
    assert token_id_length([1, 2, 3]) == 3
    assert token_id_length([[1, 2, 3]]) == 3
    assert token_id_length({"input_ids": [1, 2, 3], "attention_mask": [1, 1, 1]}) == 3
    assert token_id_length(
        {"input_ids": [[1, 2, 3]], "attention_mask": [[1, 1, 1]]}
    ) == 3
