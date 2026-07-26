from medreason.evaluation import (
    build_mcqa_evaluation_row,
    summarize_mcqa_evaluation,
)


def sample(source="medqa", answer="B"):
    return {
        "id": "demo",
        "source": source,
        "source_split": "validation",
        "question": "Question",
        "answer": answer,
        "answer_text": "Beta",
    }


def test_build_mcqa_evaluation_row_uses_strict_label_parser():
    row = build_mcqa_evaluation_row(
        sample(),
        [
            "<think>valid reasoning</think><answer>B</answer>",
            "<think>ambiguous</think><answer>B or C</answer>",
        ],
        [12, 20],
        [False, True],
    )
    assert row["num_correct"] == 1
    assert row["group_type"] == "mixed"
    assert row["rollouts"][0]["valid_format"] is True
    assert row["rollouts"][1]["predicted_label"] is None
    assert row["rollouts"][1]["hit_limit"] is True


def test_summarize_mcqa_evaluation_reports_sources_and_groups():
    medqa = build_mcqa_evaluation_row(
        sample("medqa", "B"),
        ["<think>x</think><answer>B</answer>"],
        [10],
        [False],
    )
    medmcqa = build_mcqa_evaluation_row(
        sample("medmcqa", "A"),
        ["<think>x</think><answer>D</answer>"],
        [30],
        [True],
    )
    summary = summarize_mcqa_evaluation([medqa, medmcqa], config={"seed": 42})
    assert summary["config"]["seed"] == 42
    assert summary["overall"]["accuracy"] == 0.5
    assert summary["overall"]["format_rate"] == 1.0
    assert summary["overall"]["truncation_rate"] == 0.5
    assert summary["overall"]["generated_tokens"]["mean"] == 20.0
    assert summary["by_source"]["medqa"]["accuracy"] == 1.0
    assert summary["by_source"]["medmcqa"]["accuracy"] == 0.0
