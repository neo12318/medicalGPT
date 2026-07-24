"""Answer parsing and reward helpers for verifiable medical reasoning."""

from __future__ import annotations

import re
import unicodedata
from typing import Any, Iterable


_TAGGED_COMPLETION_RE = re.compile(
    r"^\s*<think>\s*(?P<think>.+?)\s*</think>\s*"
    r"<answer>\s*(?P<answer>.+?)\s*</answer>\s*$",
    flags=re.DOTALL | re.IGNORECASE,
)
_ANSWER_ONLY_RE = re.compile(
    r"<answer>\s*(?P<answer>.*?)\s*</answer>",
    flags=re.DOTALL | re.IGNORECASE,
)
_OPTION_PREFIX_RE = re.compile(r"^\s*([A-H])\s*[\.\):：、-]\s*(.+?)\s*$", flags=re.IGNORECASE)


def completion_content(completion: Any) -> str:
    """Return text from the completion shapes accepted by TRL reward functions."""
    if completion is None:
        return ""
    if isinstance(completion, str):
        return completion
    if isinstance(completion, dict):
        return str(completion.get("content", ""))
    if isinstance(completion, (list, tuple)) and completion:
        return completion_content(completion[0])
    return str(completion)


def extract_answer(text: Any) -> str:
    """Extract the content of the first ``<answer>`` block.

    Untagged text is returned unchanged so the parser can also be used to audit
    partially formatted generations.
    """
    content = completion_content(text).strip()
    match = _ANSWER_ONLY_RE.search(content)
    return match.group("answer").strip() if match else content


def is_valid_reasoning_format(text: Any) -> bool:
    """Whether a completion has one non-empty think block and answer block."""
    return _TAGGED_COMPLETION_RE.fullmatch(completion_content(text)) is not None


def normalize_answer(text: Any) -> str:
    """Conservatively normalize an answer for exact verification."""
    value = unicodedata.normalize("NFKC", extract_answer(text))
    value = value.strip().strip("\"'`")
    value = re.sub(r"\s+", " ", value).casefold()
    value = value.rstrip(" \t\r\n.,;:!?，。；：！？")
    return value


def _option_parts(text: Any) -> tuple[str | None, str]:
    normalized = normalize_answer(text)
    match = _OPTION_PREFIX_RE.fullmatch(normalized)
    if not match:
        return None, normalized
    return match.group(1).casefold(), normalize_answer(match.group(2))


def answers_match(prediction: Any, reference: Any) -> bool:
    """Compare medical answers without unsafe substring matching.

    Exact normalized text is preferred. A leading multiple-choice label is
    ignored only when the remaining option text matches exactly.
    """
    pred = normalize_answer(prediction)
    gold = normalize_answer(reference)
    if not pred or not gold:
        return False
    if pred == gold:
        return True

    pred_label, pred_text = _option_parts(pred)
    gold_label, gold_text = _option_parts(gold)
    if pred_label and gold_label:
        return pred_label == gold_label
    if pred_text and pred_text == gold_text:
        return True
    if pred_label and pred_label == gold:
        return True
    if gold_label and gold_label == pred:
        return True
    return False


def medical_accuracy_reward(
    completions: Iterable[Any],
    answer: Iterable[Any],
    **_: Any,
) -> list[float]:
    """Return a binary outcome reward for each medical completion."""
    return [
        1.0 if answers_match(completion_content(completion), reference) else 0.0
        for completion, reference in zip(completions, answer)
    ]


def medical_format_reward(
    completions: Iterable[Any],
    format_weight: float = 0.1,
    **_: Any,
) -> list[float]:
    """Return a small reward for the required reasoning/answer format."""
    return [
        float(format_weight) if is_valid_reasoning_format(completion) else 0.0
        for completion in completions
    ]
