"""
message_template.py
Message template in the style of SAP's Template class, built on LangChain messages.
"""

import re
from typing import Any, Dict, List, Optional, Sequence
from langchain_core.messages import BaseMessage, SystemMessage
from langchain_core.runnables import Runnable, RunnableLambda

PLACEHOLDER = re.compile(r"\{\{\s*\?(\w+)\s*\}\}")


class MessageTemplate:
    """A message template in the style of SAP's Template class.

    messages: list of SystemMessage, HumanMessage or AIMessage objects.
              Their text can contain {{?name}} placeholders.
    defaults: values used when the chain input does not supply a placeholder.
    """

    def __init__(self, messages: Sequence[BaseMessage], defaults: Optional[Dict[str, Any]] = None):
        self.messages: List[BaseMessage] = list(messages)
        self.defaults: Dict[str, Any] = dict(defaults or {})

    @staticmethod
    def _text(message: BaseMessage) -> str:
        if not isinstance(message.content, str):
            raise TypeError("MessageTemplate supports plain text message content only")
        return message.content

    @property
    def placeholders(self) -> List[str]:
        """Names of all placeholders, in the order they first appear."""
        names: List[str] = []
        for message in self.messages:
            for name in PLACEHOLDER.findall(self._text(message)):
                if name not in names:
                    names.append(name)
        return names

    def format_messages(self, values: Optional[Dict[str, Any]] = None) -> List[BaseMessage]:
        """Return a new list of messages with every placeholder filled in."""
        values = {**self.defaults, **(values or {})}
        history: List[BaseMessage] = list(values.pop("history", None) or [])

        missing = [name for name in self.placeholders if name not in values]
        if missing:
            raise KeyError(f"No value given for placeholder(s): {', '.join(missing)}")

        def fill(match: re.Match) -> str:
            return str(values[match.group(1)])

        # model_copy keeps the message type (system, human or AI) and only swaps the text
        rendered = [
            message.model_copy(update={"content": PLACEHOLDER.sub(fill, self._text(message))})
            for message in self.messages
        ]

        # Conversation history goes after the leading system messages
        split = 0
        while split < len(rendered) and isinstance(rendered[split], SystemMessage):
            split += 1
        return rendered[:split] + history + rendered[split:]

    def as_runnable(self) -> Runnable:
        """Use the template as a step in a LangChain chain."""
        return RunnableLambda(self.format_messages, name="MessageTemplate")
