# Copyright 2025 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# import debugpy
# try:
#     # 5678 is the default attach port in the VS Code debug configurations. Unless a host and port are specified, host defaults to 127.0.0.1
#     debugpy.listen(("localhost", 9501))
#     print("Waiting for debugger attach")
#     debugpy.wait_for_client()
# except Exception as e:
#     pass

import json
import math
import os
import random
from dataclasses import dataclass, field
from typing import Optional

import yaml
from open_r1.trainer import Qwen2VLGRPOTrainer
from PIL import Image
from torch.utils.data import Dataset
from trl import GRPOConfig, ModelConfig, ScriptArguments, TrlParser, get_peft_config

from reward_functions import *


@dataclass
class GRPOScriptArguments(ScriptArguments):
    """
    Script arguments for the GRPO training script.

    Args:
        reward_funcs (`list[str]`):
            List of reward functions. Possible values: 'accuracy', 'format'.
    """

    reward_funcs: list[str] = field(
        default_factory=lambda: [
            "soft_format",
            "strict_format",
            "xmlcount_reward_func",
            "action_selection_reward",
            "optimal_action_reward",
        ],
        metadata={
            "help": "List of reward functions. Possible values: 'accuracy', 'format'"
        },
    )
    max_pixels: Optional[int] = field(
        default=12845056,
        metadata={"help": "Maximum number of pixels for the image"},
    )
    min_pixels: Optional[int] = field(
        default=3136,
        metadata={"help": "Minimum number of pixels for the image"},
    )
    image_root: Optional[str] = field(
        default=None,
        metadata={"help": "Root directory of the image"},
    )


reward_funcs_registry = {
    "soft_format": soft_format_reward_func,
    "strict_format": strict_format_reward_func,
    "xmlcount_reward_func": xmlcount_reward_func,
    "action_selection_reward": action_selection_reward_func,
    "optimal_action_reward": optimal_action_reward_func,
}

SYSTEM_PROMPT = (
    "A conversation between User and Assistant. The user asks a question, and the"
    " Assistant solves it. The assistant first reasons about the reasoning process in"
    " the mind and then provides the user with the answer. The reasoning process and"
    " answer are enclosed within <reasoning> </reasoning> and <answer> </answer> tags,"
    " respectively, i.e., <reasoning>\nreasoning process here\n</reasoning>\n<answer>"
    "\nanswer here\n</answer>\n"
)


class LazySupervisedDataset(Dataset):
    def __init__(self, data_path: str, script_args: GRPOScriptArguments):
        super(LazySupervisedDataset, self).__init__()
        self.script_args = script_args
        self.list_data_dict = []

        if data_path.endswith(".yaml"):
            with open(data_path, "r") as file:
                yaml_data = yaml.safe_load(file)
                datasets = yaml_data.get("datasets")
                # file should be in the format of:
                # datasets:
                #   - json_path: xxxx1.json
                #     sampling_strategy: first:1000
                #   - json_path: xxxx2.json
                #     sampling_strategy: end:3000
                #   - json_path: xxxx3.json
                #     sampling_strategy: random:999

                for data in datasets:
                    json_path = data.get("json_path")
                    sampling_strategy = data.get("sampling_strategy", "all")
                    sampling_number = None

                    if json_path.endswith(".jsonl"):
                        cur_data_dict = []
                        with open(json_path, "r") as json_file:
                            for line in json_file:
                                cur_data_dict.append(json.loads(line.strip()))
                    elif json_path.endswith(".json"):
                        with open(json_path, "r") as json_file:
                            cur_data_dict = json.load(json_file)
                    else:
                        raise ValueError(f"Unsupported file type: {json_path}")

                    if ":" in sampling_strategy:
                        sampling_strategy, sampling_number = sampling_strategy.split(
                            ":"
                        )
                        if "%" in sampling_number:
                            sampling_number = math.ceil(
                                int(sampling_number.split("%")[0])
                                * len(cur_data_dict)
                                / 100
                            )
                        else:
                            sampling_number = int(sampling_number)

                    # Apply the sampling strategy
                    if sampling_strategy == "first" and sampling_number is not None:
                        cur_data_dict = cur_data_dict[:sampling_number]
                    elif sampling_strategy == "end" and sampling_number is not None:
                        cur_data_dict = cur_data_dict[-sampling_number:]
                    elif sampling_strategy == "random" and sampling_number is not None:
                        random.shuffle(cur_data_dict)
                        cur_data_dict = cur_data_dict[:sampling_number]
                    print(f"Loaded {len(cur_data_dict)} samples from {json_path}")
                    self.list_data_dict.extend(cur_data_dict)
        else:
            raise ValueError(f"Unsupported file type: {data_path}")

    def __len__(self):
        return len(self.list_data_dict)

    def __getitem__(self, i):
        # Format into conversation
        def make_conversation(example):
            return {
                "prompt": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": (
                            "Please select the next navigation action towards the"
                            f" target: {example['target_id'].split('|')[0]}."
                        ),
                    },
                ],
            }

        QUESTION_TEMPLATE = (
            "{Question} Please select one of the navigation actions: MoveAhead,"
            " RotateLeft, RotateRight, LookUp, LookDown, and Done. Output the reasoning"
            " process in <reasoning> </reasoning> and final answer in <answer> </answer> tags."
        )

        def make_conversation_image(example):
            return {
                "prompt": [
                    # {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
                    {
                        "role": "user",
                        "content": [
                            {"type": "image"},
                            {
                                "type": "text",
                                "text": QUESTION_TEMPLATE.format(
                                    Question=(
                                        "Please select the next navigation action"
                                        " towards the target:"
                                        f" {example['target_id'].split('|')[0]}."
                                    )
                                ),
                            },
                        ],
                    },
                ],
            }

        example = self.list_data_dict[i]
        image_root = self.script_args.image_root
        if "image" in example:
            image_path = os.path.join(image_root, example["image"])
            image = Image.open(image_path).convert("RGB")
        else:
            image = None

        return {
            "image": image,
            "problem": (
                "Please select the next navigation action towards the target:"
                f" {example['target_id'].split('|')[0]}."
            ),
            "solution": example["optimal_action"],
            "prompt": (
                make_conversation_image(example)["prompt"]
                if "image" in example
                else make_conversation(example)["prompt"]
            ),
        }


def main(script_args, training_args, model_args):
    reward_funcs = [reward_funcs_registry[func] for func in script_args.reward_funcs]
    print("reward_funcs:", reward_funcs)

    # Load the dataset
    dataset = LazySupervisedDataset(script_args.dataset_name, script_args)

    trainer_cls = Qwen2VLGRPOTrainer

    # Initialize the GRPO trainer
    trainer = trainer_cls(
        model=model_args.model_name_or_path,
        reward_funcs=reward_funcs,
        args=training_args,
        train_dataset=dataset,
        eval_dataset=None,
        peft_config=get_peft_config(model_args),
        attn_implementation=model_args.attn_implementation,
        max_pixels=script_args.max_pixels,
        min_pixels=script_args.min_pixels,
        torch_dtype=model_args.torch_dtype,
    )

    # Train and push the model to the Hub
    trainer.train()

    # Save and push to hub
    trainer.save_model(training_args.output_dir)
    if training_args.push_to_hub:
        trainer.push_to_hub(dataset_name=script_args.dataset_name)


if __name__ == "__main__":
    parser = TrlParser((GRPOScriptArguments, GRPOConfig, ModelConfig))
    script_args, training_args, model_args = parser.parse_args_and_config()
    main(script_args, training_args, model_args)
