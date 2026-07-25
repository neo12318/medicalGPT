import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))

from prepare_medical_mcqa_rl import (
    convert_medmcqa_row,
    convert_medqa_row,
    medmcqa_label,
)


def test_convert_medqa_uses_canonical_label_reward():
    row = {
        "id": "m1",
        "question": "Which treatment is best?",
        "options": {"A": "Alpha", "B": "Beta", "C": "Gamma", "D": "Delta"},
        "answer_idx": "B",
        "answer": "Beta",
    }
    converted = convert_medqa_row(row, "train")
    assert converted["answer"] == "B"
    assert converted["answer_text"] == "Beta"
    assert "A. Alpha" in converted["question"]
    assert "D. Delta" in converted["question"]


def test_convert_medqa_supports_hf_sent_ending_schema():
    row = {
        "id": "dev-1",
        "sent1": "Clinical stem.",
        "sent2": "Which is best?",
        "ending0": "Alpha",
        "ending1": "Beta",
        "ending2": "Gamma",
        "ending3": "Delta",
        "label": 2,
    }
    converted = convert_medqa_row(row, "validation")
    assert converted["answer"] == "C"
    assert converted["answer_text"] == "Gamma"
    assert converted["raw_question"] == "Clinical stem. Which is best?"


def test_convert_medmcqa_handles_hf_zero_based_class_label():
    row = {
        "id": "mc1",
        "question": "Select one.",
        "opa": "One",
        "opb": "Two",
        "opc": "Three",
        "opd": "Four",
        "cop": 2,
        "subject_name": "Medicine",
        "topic_name": "Demo",
    }
    converted = convert_medmcqa_row(row, "validation", cop_base=0)
    assert converted["answer"] == "C"
    assert converted["answer_text"] == "Three"
    assert medmcqa_label("d") == "D"
