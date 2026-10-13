"""
prompt_editor.py
==============================================================================================================
Prompt Editor — Add or Update Named Prompts in prompts.json
----------------------------------------------
Utility script for managing system prompts used by the LLM pipeline.
----------------------------------------------
Purpose:
    *Accept a prompt name as a CLI argument
    *Accept a multi-line prompt text interactively via stdin
    *Convert the multi-line text to a single-line escaped string
    *Append or overwrite the named entry in prompts.json

Usage:
    python helper/prompt_editor.py <prompt_name>
    Example: python helper/prompt_editor.py prompt_3
    Then paste your multi-line prompt and press Ctrl+Z + Enter (Windows) or Ctrl+D (Unix) to finish.

Author: Nihar Dutta
Repository: https://github.com/nihardutta/sap-genai-hub-examples.git
Disclaimer: For education purpose only. Not for productive use. As usual, check for vulnerabilities of any open-source 
codes before you hit run.
===============================================================================================================
"""

import sys
import json
import pathlib


PROMPTS_FILE = pathlib.Path(__file__).parent.parent / "data" / "prompts.json"


def to_single_line(text: str) -> str:
    """Replace newlines with the literal two-character sequence \\n."""
    return "\\n".join(text.splitlines())


def main():
    if len(sys.argv) < 2:
        print(
            "Usage: python helper/prompt_editor.py <prompt_name>\n"
            "Example: python helper/prompt_editor.py prompt_3",
            file=sys.stderr,
        )
        sys.exit(1)

    prompt_name = sys.argv[1]

    # Read multi-line prompt from stdin
    print("Paste your multi-line prompt, then press Ctrl+Z + Enter (Windows) or Ctrl+D (Unix) to finish:")
    raw = sys.stdin.read()

    # Load existing prompts (or start fresh if file is missing / empty)
    if PROMPTS_FILE.exists() and PROMPTS_FILE.stat().st_size > 0:
        with PROMPTS_FILE.open(encoding="utf-8") as f:
            prompts: dict = json.load(f)
    else:
        prompts = {}

    # Add / overwrite the entry
    prompts[prompt_name] = to_single_line(raw)

    # Write back to prompts.json
    with PROMPTS_FILE.open("w", encoding="utf-8") as f:
        json.dump(prompts, f, indent=4, ensure_ascii=False)

    print(f"Saved '{prompt_name}' to {PROMPTS_FILE}")
    print(json.dumps({prompt_name: prompts[prompt_name]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
