"""
solving_business_problems_helpers.py
==========
Helper functions and classes extracted from
10_LJ_Solving_Your_Business_Problems_Example.py.
"""

import argparse
import json
import pathlib
import random
import re
import time

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, BaseMessage
from tqdm.auto import tqdm
from typing import Any, Dict, List

PROMPTS_FILE = pathlib.Path(__file__).parent.parent / "data" / "prompts.json"
GUIDES_FILE = pathlib.Path(__file__).parent.parent / "data" / "guides.json"


class RateLimitedIterator:
    def __init__(self, iterable, max_iterations_per_minute):
        self._iterable = iter(iterable)
        self._max_iterations_per_minute = max_iterations_per_minute
        self._min_interval = 1.0 / (max_iterations_per_minute / 60.)
        self._last_yield_time = None

    def __iter__(self):
        return self

    def __next__(self):
        current_time = time.time()

        if self._last_yield_time is not None:
            elapsed_time = current_time - self._last_yield_time
            if elapsed_time < self._min_interval:
                time.sleep(self._min_interval - elapsed_time)

        self._last_yield_time = time.time()
        return next(self._iterable)


def load_prompt(prompt_name: str) -> str:
    """Load a named prompt from prompts.json."""
    with PROMPTS_FILE.open(encoding="utf-8") as f:
        prompts: dict = json.load(f)
    if prompt_name not in prompts:
        available = ", ".join(prompts.keys())
        raise ValueError(f"Prompt '{prompt_name}' not found. Available prompts: {available}")
    return prompts[prompt_name]


def load_guide_placeholders(prompt: str) -> Dict[str, str]:
    """Return guide values from guides.json for any {{?guide_*}} placeholders present in *prompt*.

    Only keys whose corresponding placeholder actually appears in the prompt are
    included, so the returned dict can be unpacked directly into fill_placeholders
    without supplying values for placeholders that are not used.

    Raises:
        KeyError: if a {{?guide_*}} placeholder in the prompt has no matching
                  entry in guides.json.
    """
    with GUIDES_FILE.open(encoding="utf-8") as f:
        guides: Dict[str, str] = json.load(f)

    guide_keys = ["guide_urgency", "guide_sentiment", "guide_categories"]
    result: Dict[str, str] = {}
    for key in guide_keys:
        if f"{{{{?{key}}}}}" in prompt:
            if key not in guides:
                raise KeyError(f"Guide key '{key}' not found in guides.json.")
            result[key] = guides[key]
    return result


def fill_placeholders(prompt: str, **kwargs: str) -> str:
    """Replace {{?<key>}} placeholders in *prompt* with keyword argument values.

    Example:
        fill_placeholders(prompt, urgency='`low`, `high`', sentiment='`positive`')
        '{{?urgency}}' -> '`low`, `high`'
    """
    def _replace(match: re.Match) -> str:
        key = match.group(1)
        if key not in kwargs:
            raise KeyError(f"Placeholder '{{{{?{key}}}}}' has no matching keyword argument.")
        return kwargs[key]

    return re.sub(r"\{\{\?(\w+)\}\}", _replace, prompt)


def build_few_shot_examples(dev_set: List[Dict[str, Any]], k: int = 3) -> str:
    """Sample *k* examples from *dev_set* and format them with the example template.

    Each example renders as:
        <example>
        {message}

        ## Output

        {json ground_truth}
        </example>

    The returned string is intended to replace the {{?few_shot_examples}}
    placeholder inside a system prompt.
    """
    example_template = """<example>
                            {example_input}

                        ## Output

                            {example_output}
                        </example>"""

    samples = random.sample(dev_set, k)
    formatted = [
        example_template.format(
            example_input=example["message"],
            example_output=json.dumps(example["ground_truth"]),
        )
        for example in samples
    ]
    return "\n\n".join(formatted)


def transpose_list_of_dicts(list_of_dicts):
    keys = list_of_dicts[0].keys()
    transposed_dict = {key: [] for key in keys}
    for d in list_of_dicts:
        for key, value in d.items():
            transposed_dict[key].append(value)
    return transposed_dict


def model_evaluation(mail: Dict[str, Any], response: str, categories: set) -> Dict[str, Any]:
    """Evaluate the LLM response against the ground truth labels of a mail.

    Args:
        mail:       A single mail dict containing a 'ground_truth' key.
        response:   The raw LLM response string (expected to be valid JSON).
        categories: The full set of known categories (used to normalise the
                    category overlap score).

    Returns:
        A dict with four evaluation keys:
            is_valid_json      – whether the response could be parsed as JSON
            correct_categories – overlap score in [0, 1] (1 = perfect match)
            correct_sentiment  – True if sentiment matches ground truth
            correct_urgency    – True if urgency matches ground truth
    """
    result = {
        "is_valid_json": False,
        "correct_categories": False,
        "correct_sentiment": False,
        "correct_urgency": False,
    }
    try:
        pred = json.loads(response)
    except json.JSONDecodeError:
        result["is_valid_json"] = False
    else:
        result["is_valid_json"] = True
        result["correct_categories"] = 1 - (
            len(set(mail["ground_truth"]["categories"]) ^ set(pred["categories"]))
            / len(categories)
        )
        result["correct_sentiment"] = pred["sentiment"] == mail["ground_truth"]["sentiment"]
        result["correct_urgency"] = pred["urgency"] == mail["ground_truth"]["urgency"]
    return result


def pretty_print_table(data):
    # Get all row names (outer dict keys)
    row_names = list(data.keys())

    # Get all column names (inner dict keys)
    if row_names:
        column_names = list(data[row_names[0]].keys())
    else:
        column_names = []

    # Calculate column widths
    column_widths = [max(len(str(column_name)), max(len(f"{data[row][column_name]:.2f}") for row in row_names)) for column_name in column_names]
    row_name_width = max(len(str(row_name)) for row_name in row_names)

    # Print header
    header = f"{'':>{row_name_width}} " + " ".join([f"{column_name:>{width}}" for column_name, width in zip(column_names, column_widths)])
    print(header)
    print("=" * len(header))

    # Print rows
    for row_name in row_names:
        row = f"{row_name:>{row_name_width}} " + " ".join([f"{data[row_name][column_name]:>{width}.1%}" for column_name, width in zip(column_names, column_widths)])
        print(row)


def evaluation_full_dataset(
    dataset: List[Dict[str, Any]],
    rate_limit: int,
    llm,
    system_prompt: str,
    categories: set,
) -> Dict[str, Any]:
    """Run model_evaluation over every mail in *dataset* with rate limiting.

    Args:
        dataset:       List of mail dicts to evaluate (e.g. test_set_small).
        rate_limit:    Maximum LLM calls per minute.
        llm:           Initialised LangChain chat model.
        system_prompt: Resolved system prompt string (placeholders already filled).
        categories:    Full set of known categories for normalising the category score.

    Returns:
        A dict of averaged evaluation metrics across the whole dataset.
    """
    results = []

    for mail in tqdm(RateLimitedIterator(dataset, rate_limit), total=len(dataset)):
        messages: List[BaseMessage] = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=mail["message"]),
        ]
        response = llm.invoke(messages)
        results.append(model_evaluation(mail, response.content, categories))

    results = transpose_list_of_dicts(results)
    n = len(dataset)
    for k, v in results.items():
        results[k] = sum(v) / n
    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyse support mails using a named system prompt."
    )
    parser.add_argument(
        "prompt_name",
        help="Name of the prompt to use (e.g. prompt_1, prompt_2, …). "
             "Must match a key in prompts.json.",
    )
    parser.add_argument(
        "example_message_idx",
        type=int,
        nargs="?",
        default=None,
        help="Zero-based index of the message from the dev set to analyse. "
             "When provided, evaluates a single mail; otherwise evaluates the full test set.",
    )
    return parser.parse_args()
