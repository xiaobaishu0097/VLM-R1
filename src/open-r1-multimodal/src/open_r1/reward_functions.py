import re


def extract_xml_answer(text: str) -> str:
    answer = text.split("<answer>")[-1]
    answer = answer.split("</answer>")[0]
    return answer.strip()


def strict_format_reward_func(completions, **kwargs) -> list[float]:
    """Reward function that checks if the completion has a specific format."""
    pattern = r"^<reasoning>\n.*?\n</reasoning>\n<answer>\n.*?\n</answer>\n$"
    responses = [completion[0]["content"] for completion in completions]
    matches = [re.match(pattern, r) for r in responses]
    return [0.5 if match else 0.0 for match in matches]


def soft_format_reward_func(completions, **kwargs) -> list[float]:
    """Reward function that checks if the completion has a specific format."""
    pattern = r"<reasoning>.*?</reasoning>\s*<answer>.*?</answer>"
    responses = [completion[0]["content"] for completion in completions]
    matches = [re.match(pattern, r) for r in responses]
    return [0.5 if match else 0.0 for match in matches]


def count_xml(text) -> float:
    count = 0.0
    if text.count("<reasoning>\n") == 1:
        count += 0.125
    if text.count("\n</reasoning>\n") == 1:
        count += 0.125
    if text.count("\n<answer>\n") == 1:
        count += 0.125
        count -= len(text.split("\n</answer>\n")[-1]) * 0.001
    if text.count("\n</answer>") == 1:
        count += 0.125
        count -= (len(text.split("\n</answer>")[-1]) - 1) * 0.001
    return count


def xmlcount_reward_func(completions, **kwargs) -> list[float]:
    contents = [completion[0]["content"] for completion in completions]
    return [count_xml(c) for c in contents]


def action_selection_reward_func(completions, **kwargs):
    """Reward function that checks if the completion has chosen a valid navigation action in a specific format.

    The function expects the completion to contain a <reasoning>...</reasoning> section followed by an <answer>...</answer>
    section, where the answer is exactly one of the following: MoveAhead, RotateLeft, RotateRight, LookUp, LookDown, or Done.
    """
    # Regular expression pattern that matches the required structure and valid answer options
    pattern = r"<reasoning>.*?</reasoning>\s*<answer>\s*(MoveAhead|RotateLeft|RotateRight|LookUp|LookDown|Done)\s*</answer>"

    # Extract the content field from each completion
    responses = [completion[0]["content"] for completion in completions]

    # Check each completion content against the pattern using re.fullmatch with DOTALL flag to include newlines
    matches = [re.match(pattern, r) for r in responses]

    # Return a reward of 1.0 if the content matches the pattern exactly, otherwise 0.0
    return [1.0 if match else 0.0 for match in matches]


def optimal_action_reward_func(prompts, completions, answer, **kwargs):
    responses = [completion[0]["content"] for completion in completions]
    q = prompts[0][-1]["content"]
    extracted_responses = [extract_xml_answer(r) for r in responses]
    print(
        "-" * 20,
        f"Question:\n{q}",
        f"\nAnswer:\n{answer[0]}",
        f"\nResponse:\n{responses[0]}",
        f"\nExtracted:\n{extracted_responses[0]}",
    )
    return [2.0 if r == a else 0.0 for r, a in zip(extracted_responses, answer)]
