"""
llm_factory.py
==============
LLM initialisation helpers.
"""

import os

from langchain.chat_models import init_chat_model


def get_llm(model_provider: str = None, temp: float = 0.2):
    """Initialise and return a chat model for the given provider.

    The provider is resolved (in order) from:
      1. the ``model_provider`` argument
      2. the ``MODEL_PROVIDER`` environment variable
      3. the hard-coded default ``"groq"``

    For the resolved provider the following env-var naming convention is used:
      - API key : ``<PROVIDER_UPPER>_API_KEY``  (e.g. ``NVIDIA_API_KEY``)
      - Model   : ``<PROVIDER_UPPER>_MODEL``    (e.g. ``NVIDIA_MODEL``)

    ``<PROVIDER_UPPER>_MODEL`` falls back to the generic ``MODEL`` env var when
    the provider-specific key is not set.
    """

    prefix = model_provider.upper()

    api_key = os.getenv(f"{prefix}_API_KEY")
    model = os.getenv(f"{prefix}_MODEL")

    print(f"********Using Model: {model} provided by: {model_provider} with temperature: {temp}********")
    return init_chat_model(
        model=model,
        model_provider=model_provider,
        api_key=api_key,
        temperature=temp,
    )
