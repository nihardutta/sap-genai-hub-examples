"""
10_LJ_Solving_Your_Business_Problems_Example.py
==============================================================================================================
Solving Your Business Problems Using Prompts and LLMs (Facility Management Example)
----------------------------------------------
About this project
This is NOT a replacement for SAP's Generative AI Hub. It is an educational re-creation of SAP's
learning journey example, for people who do not have access to SAP AI Core and its services.
It uses LangChain and freely available LLMs to walk through the same ideas and concepts, so you can learn
the techniques without an SAP landscape.

Original SAP notebook (the reference for this example):
https://github.com/SAP-samples/ai-core-samples/blob/main/10_Learning_Journeys/Solving_Your_Business_Problems_Example.ipynb
----------------------------------------------
Start with introducing the scenario:
    *Company wants to analyse support mail
    *Incoming mails should be assigned to the correct service category and assigned the correct urgency and sentiment

Goal: Analyse every message and extract:
    *urgency
    *sentiment
    *categories

LLM Initialisation
-------------------
The model provider defaults to "groq". To switch providers:
  1. Add the provider's credentials to your .env file following the naming convention:
       <PROVIDER>_API_KEY=...   e.g. NVIDIA_API_KEY=nvapi-...
       <PROVIDER>_MODEL=...     e.g. NVIDIA_MODEL=meta/llama-3.3-70b-instruct
  2. Pass the provider name explicitly as a global variable.:
       MODEL_PROVIDER = 'OpenRouter'
  3. Ensure the matching LangChain integration package is installed, e.g.:
       uv add langchain-groq                  # for Groq
       uv add langchain-openai                # for OpenAI
       uv add langchain-openrouter            # for OpenRouter  
       uv add langchain-nvidia-ai-endpoints   # for NVIDIA  
    If package is not installed, an error message will appear similar to the following.
    ImportError: Initializing ChatOpenRouter requires the langchain-openrouter package. Please install it with `pip install langchain-openrouter`

Author: Nihar Dutta
Repository: https://github.com/nihardutta/sap-genai-hub-examples.git
Disclaimer: For education purpose only. Not for productive use. As usual, check for vulnerabilities of any open-source 
codes before you hit run.
===============================================================================================================
"""

import json
import os
import pathlib

from langchain_core.messages import HumanMessage, SystemMessage, BaseMessage
from typing import Dict, List
from dotenv import load_dotenv

from helper.llm_factory import get_llm
from helper.solving_business_problems_helpers import (
    build_few_shot_examples,
    evaluation_full_dataset,
    fill_placeholders,
    load_guide_placeholders,
    load_prompt,
    model_evaluation,
    parse_args,
    pretty_print_table,
)

# Load variables from the .env file into the system environment
load_dotenv()

# Pass the provider name explicitly
MODEL_PROVIDER = 'OpenRouter'

def evaluate_single(llm, messages: List[BaseMessage], mail: dict, categories: set, eval_type: str) -> None:
    """Evaluate the LLM response for a single mail and print the result."""
    response = llm.invoke(messages)
    eval_result = model_evaluation(mail, response.content, categories)
    overall_result = {eval_type: eval_result}
    pretty_print_table(overall_result)


def evaluate_full(llm, test_set_small: list, system_prompt: str, categories: set, eval_type: str) -> None:
    """Evaluate the LLM response over the full (small) test set and print the result."""
    eval_result_full = evaluation_full_dataset(
        test_set_small, rate_limit=30, llm=llm,
        system_prompt=system_prompt, categories=categories,
    )
    overall_result = {eval_type: eval_result_full}
    pretty_print_table(overall_result)


def main():

    print("*********Solving Your Business Problems Using Prompts and LLMs*********")
    HERE = pathlib.Path.cwd()

    with (HERE / 'data' / 'filtered_mails-hardest.jsonl').open() as stream:
        mails = [json.loads(line) for line in stream if line.strip()]
    dev_set, test_set = mails[:int(len(mails)/2)], mails[int(len(mails)/2):]
    test_set_small = test_set[:20]

    categories = set()
    urgency = set()
    sentiment = set()
    for mail in mails:
        categories = categories.union(set(mail['ground_truth']['categories']))
        urgency.add(mail['ground_truth']['urgency'])
        sentiment.add(mail['ground_truth']['sentiment'])

    option_lists = {
        'urgency': ', '.join(f"`{entry}`" for entry in sorted(urgency)),
        'sentiment': ', '.join(f"`{entry}`" for entry in sorted(sentiment)),
        'categories': ', '.join(f"`{entry}`" for entry in sorted(categories)),
    }
    # print(option_lists)

    # Determine prefix 
    prefix = MODEL_PROVIDER.upper()

    eval_type = "basic--" + os.getenv(f"{prefix}_MODEL")

    # Parse CLI arguments
    args = parse_args()
    prompt_name: str = args.prompt_name
    example_message_idx: int = args.example_message_idx

    # Load raw prompt from file
    raw_prompt = load_prompt(prompt_name)

    # If the prompt uses few-shot examples, build and inject them first
    few_shot_kwargs: Dict[str, str] = {}
    if "{{?few_shot_examples}}" in raw_prompt:
        few_shot_kwargs["few_shot_examples"] = build_few_shot_examples(dev_set, k=3)
        eval_type = "few_shot--" + os.getenv(f"{prefix}_MODEL")

    # If the prompt uses guide placeholders, load their values from guides.json
    guide_kwargs: Dict[str, str] = {}
    if any(f"{{{{?guide_{key}}}}}" in raw_prompt for key in ("urgency", "sentiment", "categories")):
        guide_kwargs = load_guide_placeholders(raw_prompt)
        eval_type = "metaprompting--" + eval_type

    # Resolve all {{?<key>}} placeholders (option lists + optional few-shot + guide blocks)
    system_prompt = fill_placeholders(raw_prompt, **option_lists, **few_shot_kwargs, **guide_kwargs)

    # Initialise llm
    llm = get_llm(MODEL_PROVIDER)
    
    print(f"Using prompt '{prompt_name}' | message index {example_message_idx}")
    if example_message_idx is not None:
        # ******Evaluate response for single mail******
        mail = dev_set[example_message_idx]
        messages: List[BaseMessage] = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=mail["message"]),
        ]
        evaluate_single(llm, messages, mail, categories, eval_type)
    else:
        # ******Evaluate response for multiple mails******
        evaluate_full(llm, test_set_small, system_prompt, categories, eval_type)


if __name__ == "__main__":
    main()
