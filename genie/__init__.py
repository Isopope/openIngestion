"""openingestion.genie — generative model backends (Genie interface).

A Genie wraps a generative model API and exposes a unified interface
used by agentic chunkers (e.g. SlumberChunker) to reason about text.

Available Genies
----------------
``OpenAIGenie``
    OpenAI chat completions API. Also compatible with any OpenAI-compatible
    provider (vLLM, LM Studio, Ollama, OpenRouter, …) via ``base_url``.
    Requires: ``pip install openai pydantic tenacity``

``BaseGenie``
    Abstract base class for custom Genie implementations.

Usage::

    from openingestion.genie import OpenAIGenie

    genie = OpenAIGenie(model="gpt-4o-mini")
    text = genie.generate("Summarise this paragraph...")
"""
from openingestion.genie.base import BaseGenie
from openingestion.genie.openai_genie import OpenAIGenie

__all__ = [
    "BaseGenie",
    "OpenAIGenie",
]
