# -*- coding: utf-8 -*-
"""Train MedQA/MedMCQA with a strict canonical-option GRPO reward."""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from trl import GRPOConfig, ModelConfig, TrlParser

from medreason.prompts import MEDICAL_MCQA_SYSTEM_PROMPT
from medreason.rewards import medical_format_reward, medical_mcqa_accuracy_reward
from training.grpo_training import ScriptArguments, grpo_train


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
            medical_mcqa_accuracy_reward,
            weighted_medical_format_reward,
        ],
        system_prompt=MEDICAL_MCQA_SYSTEM_PROMPT,
    )


if __name__ == "__main__":
    main()
