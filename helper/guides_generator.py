"""
guides_generator.py
==============================================================================================================
Guide Generator — Build Classification Guides from 25 Examples from Dev-Set Using an LLM
----------------------------------------------
Recreated with Langchain and open-source LLMs
----------------------------------------------
Purpose:
    *Load the dev set of support mails
    *For each label dimension (categories, urgency, sentiment), format a
     selection of dev-set examples into a meta-prompt
    *Ask the LLM to produce a step-by-step classification guide for that
     dimension based on the examples
    *Print the resulting guides
    *Write the resulting guides to a JSON file (guides.json)

Notes:
    *You need to run guides_generator.py only once if guides.json does not exist or empty
    *You can increase or decrease number of examples from Dev-Set based on LLM rate limit.
    Currently, it's set to 25.

Usage:
    python helper/guides_generator.py

Author: Nihar Dutta
Repository: https://github.com/nihardutta/sap-genai-hub-examples.git
Disclaimer: For education purpose only. Not for productive use. As usual, check for vulnerabilities of any open-source
codes before you hit run.
===============================================================================================================
"""

import re
import pathlib
import json
from dataclasses import dataclass, field
from typing import List, Dict, Any

from langchain_core.messages import SystemMessage, HumanMessage, BaseMessage
from dotenv import load_dotenv
from llm_factory import get_llm

# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------
load_dotenv()

# Pass the provider name explicitly
MODEL_PROVIDER = 'OpenRouter'

# ---------------------------------------------------------------------------
# Template helpers
# ---------------------------------------------------------------------------

@dataclass
class UserMessage:
    """A user-turn message whose content may contain {{?placeholder}} tokens."""
    content: str


@dataclass
class Template:
    """A prompt template composed of SystemMessage / UserMessage entries.

    Each entry's content may contain {{?key}} placeholders that are resolved
    by fill_placeholders() at render time.
    """
    messages: List[Any]

    def render(self, **kwargs: str) -> List[BaseMessage]:
        """Return a list of LangChain BaseMessage objects with all placeholders filled."""
        rendered: List[BaseMessage] = []
        for msg in self.messages:
            filled = fill_placeholders(msg.content, **kwargs)
            if isinstance(msg, SystemMessage):
                rendered.append(SystemMessage(content=filled))
            elif isinstance(msg, UserMessage):
                rendered.append(HumanMessage(content=filled))
        return rendered


def fill_placeholders(prompt: str, **kwargs: str) -> str:
    """Replace {{?<key>}} placeholders in *prompt* with keyword argument values."""
    def _replace(match: re.Match) -> str:
        key = match.group(1)
        if key not in kwargs:
            raise KeyError(f"Placeholder '{{{{?{key}}}}}' has no matching keyword argument.")
        return kwargs[key]

    return re.sub(r"\{\{\?(\w+)\}\}", _replace, prompt)


# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------

example_template_metaprompt = """<example>
{example_input}

## Output
{key}={example_output}
</example>"""

prompt_get_guide = Template(
    messages=[
        SystemMessage(
            content="""Here are some examples:
---
{{?examples}}
---
Use the examples above to come up with a guide on how to distinguish between {{?options}} {{?key}}.
Use the following format:
```
### **<category 1>**
- <instruction 1>
- <instruction 2>
- <instruction 3>
### **<category 2>**
- <instruction 1>
- <instruction 2>
- <instruction 3>
...
```
When creating the guide:
- make it step-by-step instructions
- Consider that some labels in the examples might be incorrect
- Avoid including explicit information from the examples in the guide
The guide has to cover: {{?options}}
"""
        ),
        UserMessage("{{?input}}"),
    ]
)

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("*********Guide Generator — Building Classification Guides*********")
    HERE = pathlib.Path.cwd()

    # Load mails and split into dev / test sets
    # Picks up 25 examples from Dev-Set
    with (HERE / "data" / "filtered_mails-hardest.jsonl").open() as stream:
        mails = [json.loads(line) for line in stream if line.strip()]
    dev_set = mails[: int(len(mails) / 2)][:25]

    # Collect all known option values per dimension
    option_lists: Dict[str, str] = {}
    for key in ["categories", "urgency", "sentiment"]:
        values: set = set()
        for mail in mails:
            gt = mail["ground_truth"][key]
            if isinstance(gt, list):
                values.update(gt)
            else:
                values.add(gt)
        option_lists[key] = ", ".join(f"`{v}`" for v in sorted(values))

    # LLM initialisation
    llm = get_llm(MODEL_PROVIDER)

    guides: Dict[str, str] = {}

    for key in ["categories", "urgency", "sentiment"]:
        options = option_lists[key]

        # Format every dev-set example for the meta-prompt
        selected_examples_txt = "\n---\n".join(
            example_template_metaprompt.format(
                example_input=example["message"],
                key=key,
                example_output=example["ground_truth"][key],
            )
            for example in dev_set
        )

        # Render the template and invoke the LLM
        messages = prompt_get_guide.render(
            examples=selected_examples_txt,
            options=options,
            key=key,
            input=f"Please write the guide for {key}.",
        )

        print(f"\nGenerating guide for '{key}' ...")
        response = llm.invoke(messages)
        guides[key] = response.content

        print(f"\n{'='*60}")
        print(f"  Guide: {key.upper()}")
        print(f"{'='*60}")
        print(guides[key])

    # Write guides to JSON file with prefixed keys
    output = {f"guide_{key}": guides[key] for key in guides}
    output_path = HERE / "data" / "guides.json"
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\nGuides written to '{output_path}'")

    return guides


if __name__ == "__main__":
    main()
