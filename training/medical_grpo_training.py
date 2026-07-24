# -*- coding: utf-8 -*-
"""Train verifiable medical reasoning with MedicalGPT's GRPO backbone."""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from trl import GRPOConfig, ModelConfig, TrlParser

from medreason.rewards import medical_accuracy_reward, medical_format_reward
from training.grpo_training import ScriptArguments, grpo_train


MEDICAL_SYSTEM_PROMPT = (
    "You are a medical reasoning assistant. Analyze the question carefully. "
    "Return exactly one non-empty reasoning block and one concise final-answer block using: "
    "<think>reasoning process</think><answer>final answer</answer>. "
    "Do not place any text outside these tags."
)


def weighted_medical_format_reward(completions, **kwargs):
    return medical_format_reward(completions, format_weight=0.1, **kwargs)


def main() -> None:
    parser = TrlParser((ModelConfig, ScriptArguments, GRPOConfig))
    model_args, script_args, training_args = parser.parse_args_and_config()
    grpo_train(
        model_args,
        script_args,
        training_args,
        reward_funcs=[
            medical_accuracy_reward,
            weighted_medical_format_reward,
        ],
        system_prompt=MEDICAL_SYSTEM_PROMPT,
    )


if __name__ == "__main__":
    main()
