import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from medreason.rewards import (
    answers_match,
    extract_answer,
    extract_mcqa_label,
    is_valid_reasoning_format,
    medical_accuracy_reward,
    medical_format_reward,
    medical_mcqa_accuracy_reward,
    normalize_answer,
)


def test_extract_and_normalize_tagged_answer():
    completion = "<think>Reasoning</think><answer>  Gastric Ulcer. </answer>"
    assert extract_answer(completion) == "Gastric Ulcer."
    assert normalize_answer(completion) == "gastric ulcer"


def test_format_requires_both_non_empty_blocks_and_no_trailing_text():
    assert is_valid_reasoning_format("<think>reason</think><answer>B</answer>")
    assert not is_valid_reasoning_format("<think></think><answer>B</answer>")
    assert not is_valid_reasoning_format("<think>reason</think><answer></answer>")
    assert not is_valid_reasoning_format("<think>reason</think><answer>B</answer> extra")


def test_answers_match_exact_text_and_choice_prefixes():
    assert answers_match("<answer>Gastric ulcer</answer>", "gastric ulcer")
    assert answers_match("<answer>B. Gastric ulcer</answer>", "Gastric ulcer")
    assert answers_match("<answer>B</answer>", "B. Gastric ulcer")
    assert not answers_match("<answer>ulcer</answer>", "Gastric ulcer")
    assert not answers_match("<answer>A. Gastric ulcer</answer>", "B. Gastric ulcer")


def test_medical_reward_functions_accept_trl_completion_shape():
    completions = [
        [{"role": "assistant", "content": "<think>x</think><answer>B</answer>"}],
        [{"role": "assistant", "content": "wrong"}],
    ]
    answers = ["B. Correct option", "Correct answer"]
    assert medical_accuracy_reward(completions, answers) == [1.0, 0.0]
    assert medical_format_reward(completions) == [0.1, 0.0]


def test_strict_mcqa_reward_accepts_only_one_option_label():
    assert extract_mcqa_label("<think>x</think><answer>B</answer>") == "B"
    assert extract_mcqa_label("<answer>b</answer>") == "B"
    assert extract_mcqa_label("<answer>B. Gastric ulcer</answer>") is None
    assert extract_mcqa_label("<answer>The answer is B</answer>") is None
    assert extract_mcqa_label("<answer>A or B</answer>") is None
    completions = [
        [{"role": "assistant", "content": "<think>x</think><answer>B</answer>"}],
        [{"role": "assistant", "content": "<think>x</think><answer>B. text</answer>"}],
    ]
    assert medical_mcqa_accuracy_reward(completions, ["B", "B"]) == [1.0, 0.0]
