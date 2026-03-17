"""BaseGenie — abstract interface for generative model backends.

All Genie implementations must provide at minimum:
  - ``generate(prompt)``       — synchronous text generation
  - ``agenerate(prompt)``      — async text generation
  - ``generate_json(prompt, schema)``   — structured output (sync)
  - ``agenerate_json(prompt, schema)``  — structured output (async)
  - ``_is_available()``        — class-method availability check
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pydantic import BaseModel


class BaseGenie(ABC):
    """Abstract base class for all Genie implementations."""

    def __init__(self) -> None:
        if not self._is_available():
            raise ImportError(
                f"{self.__class__.__name__} dependencies are not installed. "
                "Please install the required extras."
            )

    # ------------------------------------------------------------------
    # Synchronous interface
    # ------------------------------------------------------------------

    @abstractmethod
    def generate(self, prompt: str) -> str:
        """Generate a plain-text response for *prompt*."""
        raise NotImplementedError

    @abstractmethod
    def generate_json(self, prompt: str, schema: "type[BaseModel]") -> dict[str, Any]:
        """Generate a structured JSON response validated against *schema*."""
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Async interface
    # ------------------------------------------------------------------

    @abstractmethod
    async def agenerate(self, prompt: str) -> str:
        """Async variant of :meth:`generate`."""
        raise NotImplementedError

    @abstractmethod
    async def agenerate_json(self, prompt: str, schema: "type[BaseModel]") -> dict[str, Any]:
        """Async variant of :meth:`generate_json`."""
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Availability check
    # ------------------------------------------------------------------

    @classmethod
    @abstractmethod
    def _is_available(cls) -> bool:
        """Return ``True`` if all required dependencies are installed."""
        raise NotImplementedError

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"
