"""OpenAI Genie — generative model backend using the OpenAI API.

Supports any OpenAI-compatible endpoint (OpenAI, Azure via base_url, local
servers such as LM Studio or vLLM, etc.) by setting ``base_url``.

Requires:
    pip install openai pydantic tenacity
"""
from __future__ import annotations

import importlib.util as importutil
import os
from typing import TYPE_CHECKING, Any, Optional, cast

from loguru import logger

try:
    from tenacity import (
        retry,
        retry_if_exception_type,
        stop_after_attempt,
        wait_exponential,
    )
    _TENACITY_AVAILABLE = True
except ImportError:
    _TENACITY_AVAILABLE = False

try:
    from openai import APIError, APITimeoutError, AsyncOpenAI, OpenAI, RateLimitError
    _OPENAI_AVAILABLE = True
except ImportError:
    _OPENAI_AVAILABLE = False

    class APIError(Exception):  # type: ignore[no-redef]
        """Placeholder when openai is not installed."""

    class APITimeoutError(Exception):  # type: ignore[no-redef]
        """Placeholder when openai is not installed."""

    class RateLimitError(Exception):  # type: ignore[no-redef]
        """Placeholder when openai is not installed."""

    OpenAI = None  # type: ignore[assignment,misc]
    AsyncOpenAI = None  # type: ignore[assignment,misc]

from openingestion.genie.base import BaseGenie

if TYPE_CHECKING:
    from pydantic import BaseModel

# ---------------------------------------------------------------------------
# Retry decorator factory — no-op when tenacity is absent
# ---------------------------------------------------------------------------

def _retry_on_openai_errors(func):  # type: ignore[no-untyped-def]
    if not _TENACITY_AVAILABLE:
        return func
    _exc_types = cast(
        "tuple[type[BaseException], ...]",
        (RateLimitError, APIError, APITimeoutError),
    )
    return retry(
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=2, max=60),
        retry=retry_if_exception_type(_exc_types),
    )(func)


# ---------------------------------------------------------------------------
# OpenAIGenie
# ---------------------------------------------------------------------------

class OpenAIGenie(BaseGenie):
    """Genie backed by the OpenAI chat completions API.

    Compatible with any OpenAI-compatible provider (vLLM, LM Studio,
    OpenRouter, …) by supplying a custom ``base_url``.

    Args:
        model: Model identifier (default ``"gpt-4o"``).
        base_url: Optional custom endpoint URL.
        api_key: API key. Defaults to the ``OPENAI_API_KEY`` environment
                 variable.

    Example::

        from openingestion.genie import OpenAIGenie

        genie = OpenAIGenie(model="gpt-4o-mini")
        text = genie.generate("What is the capital of France?")

    With a local OpenAI-compatible server::

        genie = OpenAIGenie(
            model="mistral",
            base_url="http://localhost:11434/v1",
            api_key="ollama",
        )
    """

    def __init__(
        self,
        model: str = "gpt-4o",
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
    ) -> None:
        super().__init__()

        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError(
                "OpenAIGenie requires an API key. Either pass `api_key` or "
                "set the OPENAI_API_KEY environment variable."
            )

        assert OpenAI is not None, "openai package is required but not installed"

        client_kwargs: dict[str, Any] = {"api_key": self.api_key}
        if base_url is not None:
            client_kwargs["base_url"] = base_url

        self.client = OpenAI(**client_kwargs)
        self.async_client = AsyncOpenAI(**client_kwargs)
        self.model = model
        logger.debug("OpenAIGenie initialised with model={}", model)

    # ------------------------------------------------------------------
    # Sync
    # ------------------------------------------------------------------

    @_retry_on_openai_errors
    def generate(self, prompt: str, max_tokens: int | None = None) -> str:
        """Generate a plain-text response for *prompt*."""
        kwargs: dict[str, Any] = {}
        if max_tokens is not None:
            kwargs["max_tokens"] = max_tokens
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            **kwargs,
        )
        content = response.choices[0].message.content
        if content is None:
            raise ValueError("OpenAI returned an empty response")
        return content

    @_retry_on_openai_errors
    def generate_vision(
        self, prompt: str, image_b64: str, detail: str = "auto", system: str = ""
    ) -> str:
        """Generate a description or text from a base64 encoded image.

        Args:
            prompt: Instructions for the model (e.g., "Extract Markdown table").
            image_b64: Base64 data URI (e.g., "data:image/jpeg;base64,...").
            detail: Image detail level: "low", "high", or "auto".
            system: Optional system prompt to set context/behavior.
        """
        messages: list[dict[str, Any]] = []
        if system:
            messages.append({"role": "system", "content": system})

        messages.append({
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": image_b64,
                        "detail": detail,
                    },
                },
            ],
        })

        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
        )
        content = response.choices[0].message.content
        if content is None:
            raise ValueError("OpenAI vision model returned an empty response")
        return content

    @_retry_on_openai_errors
    def generate_json(self, prompt: str, schema: "type[BaseModel]") -> dict[str, Any]:
        """Generate a structured JSON response validated against *schema*.

        Uses the OpenAI structured-output (``beta.chat.completions.parse``)
        endpoint which enforces the Pydantic schema server-side.
        """
        response = self.client.beta.chat.completions.parse(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            response_format=schema,  # type: ignore[arg-type]
        )
        content = response.choices[0].message.parsed
        if content is None:
            raise ValueError("OpenAI returned an empty structured response")
        return content.model_dump()

    # ------------------------------------------------------------------
    # Async
    # ------------------------------------------------------------------

    @_retry_on_openai_errors
    async def agenerate(self, prompt: str, max_tokens: int | None = None) -> str:
        """Async variant of :meth:`generate`."""
        kwargs: dict[str, Any] = {}
        if max_tokens is not None:
            kwargs["max_tokens"] = max_tokens
        response = await self.async_client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            **kwargs,
        )
        content = response.choices[0].message.content
        if content is None:
            raise ValueError("OpenAI returned an empty response")
        return content

    @_retry_on_openai_errors
    async def agenerate_json(self, prompt: str, schema: "type[BaseModel]") -> dict[str, Any]:
        """Async variant of :meth:`generate_json`."""
        response = await self.async_client.beta.chat.completions.parse(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            response_format=schema,  # type: ignore[arg-type]
        )
        content = response.choices[0].message.parsed
        if content is None:
            raise ValueError("OpenAI returned an empty structured response")
        return content.model_dump()

    # ------------------------------------------------------------------
    # Availability
    # ------------------------------------------------------------------

    @classmethod
    def _is_available(cls) -> bool:
        return (
            importutil.find_spec("pydantic") is not None
            and importutil.find_spec("openai") is not None
        )

    def __repr__(self) -> str:
        return f"OpenAIGenie(model={self.model!r})"
