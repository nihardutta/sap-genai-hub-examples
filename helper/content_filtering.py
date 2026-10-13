"""
content_filtering.py
Helpers for LLM-based content filtering, modelled on SAP's Azure content filter.
"""

import json
import pathlib
from enum import IntEnum
from typing import Any, Dict, List, Optional
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_core.runnables import Runnable, RunnableLambda
from pydantic import BaseModel, Field

SEVERITY_HELP = "Severity: 0 safe, 2 low, 4 medium, 6 high"
CONTENT_FILE = pathlib.Path(__file__).parent.parent / "data/contents.json"


class Threshold(IntEnum):
    """Same names as SAP's AzureThreshold. The value is the highest severity allowed."""
    ALLOW_SAFE = 0
    ALLOW_SAFE_LOW = 2
    ALLOW_SAFE_LOW_MEDIUM = 4
    ALLOW_ALL = 6


class SafetyAssessment(BaseModel):
    hate: int = Field(description=f"Hate or harassment. {SEVERITY_HELP}")
    violence: int = Field(description=f"Violence. {SEVERITY_HELP}")
    self_harm: int = Field(description=f"Self harm. {SEVERITY_HELP}")
    sexual: int = Field(description=f"Sexual content. {SEVERITY_HELP}")


SAFETY_PROMPT = SystemMessage(
    "You are a content safety classifier. Rate the user's text in four categories: "
    "hate, violence, self_harm and sexual. Use only these severity values: "
    "0 = safe, 2 = low, 4 = medium, 6 = high. Rate the text itself; do not answer it."
)


class ContentFilterError(Exception):
    """Raised when content goes over a threshold, like SAP's OrchestrationError."""

    def __init__(self, stage: str, violations: Dict[str, tuple]):
        details = ", ".join(
            f"{category} severity {severity} (allowed up to {int(limit)})"
            for category, (severity, limit) in violations.items()
        )
        self.stage = stage
        self.violations = violations
        self.message = f"Content blocked by {stage} filter: {details}"
        super().__init__(self.message)


class ContentFilter:
    def __init__(
        self,
        classifier: Runnable,
        hate: Threshold = Threshold.ALLOW_SAFE,
        violence: Threshold = Threshold.ALLOW_SAFE,
        self_harm: Threshold = Threshold.ALLOW_SAFE,
        sexual: Threshold = Threshold.ALLOW_SAFE,
    ):
        self.classifier = classifier
        self.thresholds = {"hate": hate, "violence": violence, "self_harm": self_harm, "sexual": sexual}

    def check(self, text: str, stage: str) -> SafetyAssessment:
        assessment = self.classifier.invoke([SAFETY_PROMPT, HumanMessage(text)])
        violations = {}
        for category, limit in self.thresholds.items():
            severity = max(0, min(6, int(getattr(assessment, category))))
            if severity > limit:
                violations[category] = (severity, limit)
        if violations:
            raise ContentFilterError(stage, violations)
        return assessment

    def as_input_filter(self, fields: Optional[List[str]] = None) -> Runnable:
        """Chain step placed before the template. Checks the text inputs, then passes them on."""
        def _check(values: Dict[str, Any]) -> Dict[str, Any]:
            for key, value in values.items():
                if isinstance(value, str) and (fields is None or key in fields):
                    self.check(value, stage="input")
            return values
        return RunnableLambda(_check, name="InputFilter")

    def as_output_filter(self) -> Runnable:
        """Chain step placed after the LLM. Checks the reply, then passes it on."""
        def _check(message: BaseMessage) -> BaseMessage:
            self.check(message.text, stage="output")
            return message
        return RunnableLambda(_check, name="OutputFilter")


def load_content(content_name: str) -> str:
    """Load a named message from data/contents.json.

    Before running the content filtering, add the messages/content to test to
    data/contents.json. If the file does not exist, copy
    data/contents.example.json to data/contents.json and fill in a message
    for each key (msg_clean, msg_h, msg_v, msg_sh, msg_s).
    """
    with CONTENT_FILE.open(encoding="utf-8") as f:
        contents: dict = json.load(f)
    if content_name not in contents:
        available = ", ".join(contents.keys())
        raise ValueError(f"Content '{content_name}' not found. Available prompts: {available}")
    return contents[content_name]
