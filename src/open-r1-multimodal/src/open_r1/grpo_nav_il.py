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
import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

import yaml
from math_verify import parse, verify
from open_r1.trainer import Qwen2VLGRPOTrainer
from PIL import Image
from torch.utils.data import Dataset
from transformers import Qwen2VLForConditionalGeneration, TrainingArguments
from trl import (
    GRPOConfig,
    GRPOTrainer,
    ModelConfig,
    ScriptArguments,
    TrlParser,
    get_peft_config,
)


@dataclass
class GRPOScriptArguments(ScriptArguments):
    """
    Script arguments for the GRPO training script.

    Args:
        reward_funcs (`list[str]`):
            List of reward functions. Possible values: 'accuracy', 'format'.
    """

    reward_funcs: list[str] = field(
        default_factory=lambda: ["accuracy", "format"],
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


SYSTEM_PROMPT = (
    "A conversation between User and Assistant. The user asks a question, and the"
    " Assistant solves it. The assistant first thinks about the reasoning process in"
    " the mind and then provides the user with the answer. The reasoning process and"
    " answer are enclosed within <think> </think> and <answer> </answer> tags,"
    " respectively, i.e., <think> reasoning process here </think><answer> answer here"
    " </answer>"
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
                    {"role": "user", "content": f"Please select the next navigation action towards the target: {example['target_id'].split('|')[0]}."},
                ],
            }

        QUESTION_TEMPLATE = (
            "{Question} Please select one of the navigation actions: MoveAhead, RotateLeft, RotateRight, LookUp, LookDown, and Done. Output the thinking process in <think> </think> and final"
            " answer in <answer> </answer> tags."
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
                                    Question=f"Please select the next navigation action towards the target: {example['target_id'].split('|')[0]}."
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
            "problem": f"Please select the next navigation action towards the target: {example['target_id'].split('|')[0]}.",
            "solution": example['optimal_action'],
            "prompt": (
                make_conversation_image(example)["prompt"]
                if "image" in example
                else make_conversation(example)["prompt"]
            ),
        }


def iou_reward(completions, solution, **kwargs):
    def iou(box1, box2):
        inter_x1 = max(box1[0], box2[0])
        inter_y1 = max(box1[1], box2[1])
        inter_x2 = min(box1[2] - 1, box2[2] - 1)
        inter_y2 = min(box1[3] - 1, box2[3] - 1)
        if inter_x1 < inter_x2 and inter_y1 < inter_y2:
            inter = (inter_x2 - inter_x1 + 1) * (inter_y2 - inter_y1 + 1)
        else:
            inter = 0
        union = (
            (box1[2] - box1[0]) * (box1[3] - box1[1])
            + (box2[2] - box2[0]) * (box2[3] - box2[1])
            - inter
        )
        return float(inter) / union

    contents = [completion[0]["content"] for completion in completions]
    rewards = []
    current_time = datetime.now().strftime("%d-%H-%M-%S-%f")
    answer_tag_pattern = r"<answer>(.*?)</answer>"
    number_pattern = r"[-+]?\d*\.\d+|\d+"
    for content, sol in zip(contents, solution):
        reward = 0.0
        # Try symbolic verification first
        try:
            content_answer_match = re.search(answer_tag_pattern, content)
            if content_answer_match:
                content_answer = content_answer_match.group(1).strip()
                number_match = re.findall(number_pattern, content_answer)

                x1, y1, x2, y2 = [float(number_match[i]) for i in range(4)]
                bbox = [int(x1), int(y1), int(x2), int(y2)]
                if iou(bbox, sol) > 0.5:
                    reward = 1.0
                if all(bbox[i] <= 1 for i in range(4)):
                    bbox = [
                        int(x1 * 1000),
                        int(y1 * 1000),
                        int(x2 * 1000),
                        int(y2 * 1000),
                    ]
                    if iou(bbox, sol) > 0.5:
                        reward = 1.0
        except Exception:
            pass  # Continue to next verification method if this fails

        rewards.append(reward)
        if os.getenv("DEBUG_MODE") == "true":
            log_path = os.getenv("LOG_PATH")
            # local_rank = int(os.getenv("LOCAL_RANK", 0))
            with open(log_path, "a") as f:
                f.write(
                    f"------------- {current_time} Accuracy reward:"
                    f" {reward} -------------\n"
                )
                f.write(f"Content: {content}\n")
                f.write(f"Solution: {sol}\n")
    return rewards


def format_reward(completions, **kwargs):
    """Reward function that checks if the completion has a specific format."""
    pattern = r"<think>.*?</think>\s*<answer>.*?</answer>"
    completion_contents = [completion[0]["content"] for completion in completions]
    matches = [
        re.fullmatch(pattern, content, re.DOTALL) for content in completion_contents
    ]
    return [1.0 if match else 0.0 for match in matches]


def action_selection_reward(completions, **kwargs):
    """Reward function that checks if the completion has chosen a valid navigation action in a specific format.
    
    The function expects the completion to contain a <think>...</think> section followed by an <answer>...</answer>
    section, where the answer is exactly one of the following: MoveAhead, RotateLeft, RotateRight, LookUp, LookDown, or Done.
    """
    # Regular expression pattern that matches the required structure and valid answer options
    pattern = r"<think>.*?</think>\s*<answer>\s*(MoveAhead|RotateLeft|RotateRight|LookUp|LookDown|Done)\s*</answer>"
    
    # Extract the content field from each completion
    completion_contents = [completion[0]["content"] for completion in completions]
    
    # Check each completion content against the pattern using re.fullmatch with DOTALL flag to include newlines
    matches = [re.fullmatch(pattern, content, re.DOTALL) for content in completion_contents]
    
    # Return a reward of 1.0 if the content matches the pattern exactly, otherwise 0.0
    return [1.0 if match else 0.0 for match in matches]

reward_funcs_registry = {
    "accuracy": iou_reward,
    "format": format_reward,
    "action_selection_reward": action_selection_reward,
}

# reward_funcs_registry_2_5 = {
#     "accuracy": iou_reward_2_5,
#     "format": format_reward,
# }


def main(script_args, training_args, model_args):
    # Get reward functions
    # if "Qwen2.5-VL" in model_args.model_name_or_path:
    #     reward_funcs = [reward_funcs_registry_2_5[func] for func in script_args.reward_funcs]
    # else:
    #     reward_funcs = [reward_funcs_registry[func] for func in script_args.reward_funcs]
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
