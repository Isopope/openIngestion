"""Embedder abstractions for the openingestion pipeline.

Design mirrors ``openingestion.utils.tokenizer``:
- Abstract base class :class:`BaseEmbedder`
- Optional backends loaded lazily (no hard import)
- :class:`AutoEmbedder` factory resolves ``str | BaseEmbedder | callable``

Supported backends
------------------
``sentence-transformers``  (``pip install sentence-transformers``)
    Any model name accepted by ``SentenceTransformer(model_name)``.
    Example: ``"sentence-transformers/all-MiniLM-L6-v2"``
              ``"minishlab/potion-base-32M"``

``callable``
    Any ``(list[str]) -> list[list[float]]`` function injected directly.

Usage
-----
::

    from openingestion.utils.embedder import AutoEmbedder

    # sentence-transformers model (lazy load)
    emb = AutoEmbedder("all-MiniLM-L6-v2")
    vecs = emb.embed_batch(["Hello world", "Foo bar"])

    # Inject your own embedder object (duck-typed)
    emb = AutoEmbedder(my_embedder_object)

    # Inject a callable
    emb = AutoEmbedder(lambda texts: my_api.embed(texts))

Cosine similarity is always available via :meth:`BaseEmbedder.similarity`.
"""
from __future__ import annotations

import importlib.util as importutil
from abc import ABC, abstractmethod
from typing import Any, Callable

import numpy as np

from loguru import logger


# ──────────────────────────────────────────────────────────────────────────────
# Protocol (structural typing, no inheritance required)
# ──────────────────────────────────────────────────────────────────────────────

from typing import Protocol, runtime_checkable


@runtime_checkable
class EmbedderProtocol(Protocol):
    """Structural protocol: any object with ``embed_batch`` qualifies."""

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Embed a list of texts into a list of float vectors."""
        ...


# ──────────────────────────────────────────────────────────────────────────────
# Base class
# ──────────────────────────────────────────────────────────────────────────────

class BaseEmbedder(ABC):
    """Abstract base class for all embedders.

    Subclasses must implement :meth:`embed_batch`.
    :meth:`embed` and :meth:`similarity` are provided for free.
    """

    @abstractmethod
    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Embed a list of texts.

        Args:
            texts: Texts to embed.

        Returns:
            List of float vectors, one per input text.
        """
        raise NotImplementedError

    def embed(self, text: str) -> list[float]:
        """Embed a single text.

        Args:
            text: Text to embed.

        Returns:
            Float vector.
        """
        return self.embed_batch([text])[0]

    @staticmethod
    def similarity(a: list[float] | np.ndarray, b: list[float] | np.ndarray) -> float:
        """Cosine similarity between two vectors.

        Args:
            a: First vector.
            b: Second vector.

        Returns:
            Cosine similarity in [-1, 1].
        """
        va = np.asarray(a, dtype=np.float64)
        vb = np.asarray(b, dtype=np.float64)
        denom = np.linalg.norm(va) * np.linalg.norm(vb)
        if denom == 0.0:
            return 0.0
        return float(np.dot(va, vb) / denom)

    @abstractmethod
    def __repr__(self) -> str: ...


# ──────────────────────────────────────────────────────────────────────────────
# sentence-transformers wrapper (lazy)
# ──────────────────────────────────────────────────────────────────────────────

class _SentenceTransformersEmbedder(BaseEmbedder):
    """Lazy wrapper around a ``SentenceTransformer`` model."""

    def __init__(self, model_name: str) -> None:
        if importutil.find_spec("sentence_transformers") is None:
            raise ImportError(
                "sentence-transformers is not installed. "
                "Run: pip install sentence-transformers"
            )
        from sentence_transformers import SentenceTransformer
        logger.info("Loading SentenceTransformer model: {}", model_name)
        self._model = SentenceTransformer(model_name)
        self._model_name = model_name

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        vecs = self._model.encode(texts, show_progress_bar=False, convert_to_numpy=True)
        return vecs.tolist()

    def __repr__(self) -> str:
        return f"_SentenceTransformersEmbedder(model={self._model_name!r})"


# ──────────────────────────────────────────────────────────────────────────────
# Callable wrapper
# ──────────────────────────────────────────────────────────────────────────────

class _CallableEmbedder(BaseEmbedder):
    """Wrap a callable ``(list[str]) -> list[list[float]]``."""

    def __init__(self, fn: Callable[[list[str]], list[list[float]]]) -> None:
        self._fn = fn

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return self._fn(texts)

    def __repr__(self) -> str:
        return f"_CallableEmbedder(fn={self._fn!r})"


# ──────────────────────────────────────────────────────────────────────────────
# Duck-type wrapper (any object with embed_batch)
# ──────────────────────────────────────────────────────────────────────────────

class _DuckEmbedder(BaseEmbedder):
    """Wrap any object that already exposes ``embed_batch``."""

    def __init__(self, obj: Any) -> None:
        self._obj = obj

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return self._obj.embed_batch(texts)

    def __repr__(self) -> str:
        return f"_DuckEmbedder(obj={self._obj!r})"


# ──────────────────────────────────────────────────────────────────────────────
# Error
# ──────────────────────────────────────────────────────────────────────────────

class InvalidEmbedderError(ValueError):
    """Raised when no backend can resolve the requested embedder."""

    def __init__(self, message: str, *, backend_errors: dict[str, str] | None = None) -> None:
        super().__init__(message)
        self.backend_errors = backend_errors or {}


# ──────────────────────────────────────────────────────────────────────────────
# AutoEmbedder factory
# ──────────────────────────────────────────────────────────────────────────────

class AutoEmbedder:
    """Factory that resolves any embedder specification to a :class:`BaseEmbedder`.

    Args:
        embedder: One of:

            * ``str``           — model name for ``sentence-transformers``
            * :class:`BaseEmbedder` — returned as-is
            * Any object with an ``embed_batch`` method — wrapped in :class:`_DuckEmbedder`
            * callable          — wrapped in :class:`_CallableEmbedder`

    Raises:
        :class:`InvalidEmbedderError` if resolution fails.
        :class:`ImportError` if the required backend library is not installed.

    Example::

        from openingestion.utils.embedder import AutoEmbedder

        emb = AutoEmbedder("all-MiniLM-L6-v2")
        vec = emb.embed("Hello world")
    """

    def __new__(cls, embedder: str | BaseEmbedder | Any) -> BaseEmbedder:  # type: ignore[misc]
        # Already a BaseEmbedder → passthrough
        if isinstance(embedder, BaseEmbedder):
            return embedder

        # String → try sentence-transformers
        if isinstance(embedder, str):
            return _SentenceTransformersEmbedder(embedder)

        # Duck-typed object with embed_batch
        if isinstance(embedder, EmbedderProtocol):
            return _DuckEmbedder(embedder)

        # Plain callable
        if callable(embedder):
            return _CallableEmbedder(embedder)

        raise InvalidEmbedderError(
            f"Cannot resolve embedder of type {type(embedder).__name__}. "
            "Expected: str | BaseEmbedder | callable | object with embed_batch()."
        )
