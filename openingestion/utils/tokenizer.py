"""Tokenizer abstractions for the openingestion pipeline.

Design inspired by ``chonkie.tokenizer`` — adapted with no external
dependencies in the base layer, and optional backends loaded lazily.

Built-in tokenizers (zero dependencies)
-----------------------------------------
``"heuristic"``   :class:`HeuristicTokenizer`  — ``len(text) // 4``  (fast default)
``"character"``   :class:`CharacterTokenizer`  — one token per character
``"word"``        :class:`WordTokenizer`        — one token per whitespace-delimited word
``"byte"``        :class:`ByteTokenizer`        — one token per UTF-8 byte

Optional backends (loaded only if the library is installed)
-------------------------------------------------------------
``tiktoken``        — any encoding name handled by ``tiktoken.get_encoding``
``tokenizers``      — any model name handled by ``tokenizers.Tokenizer.from_pretrained``
``transformers``    — any model name handled by ``AutoTokenizer.from_pretrained``

Usage
-----
::

    from openingestion.utils.tokenizer import AutoTokenizer

    # Built-in
    tok = AutoTokenizer("heuristic")
    tok = AutoTokenizer("word")

    # tiktoken (requires: pip install tiktoken)
    tok = AutoTokenizer("cl100k_base")

    # HuggingFace (requires: pip install transformers)
    tok = AutoTokenizer("bert-base-uncased")

    # Wrap an existing tokenizer object
    import tiktoken
    enc = tiktoken.get_encoding("cl100k_base")
    tok = AutoTokenizer(enc)

    # Wrap a plain callable  (text: str) -> int
    tok = AutoTokenizer(lambda text: len(text.split()))

    # All expose the same interface:
    tok.count_tokens("hello world")          # int
    tok.count_tokens_batch(["a", "b", "c"]) # list[int]
"""
from __future__ import annotations

import importlib.util as importutil
from abc import ABC, abstractmethod
from collections import defaultdict
from collections.abc import Callable, Sequence
from typing import Any

from loguru import logger


# ──────────────────────────────────────────────────────────────────────────────
# Protocol  (structural typing — no inheritance required)
# ──────────────────────────────────────────────────────────────────────────────

class TokenizerProtocol:
    """Structural protocol: any object with ``count_tokens`` can be used.

    Not enforced at runtime — it serves as documentation and type-hint target.
    """

    def count_tokens(self, text: str) -> int: ...
    def count_tokens_batch(self, texts: Sequence[str]) -> Sequence[int]: ...


# ──────────────────────────────────────────────────────────────────────────────
# Abstract base
# ──────────────────────────────────────────────────────────────────────────────

class BaseTokenizer(ABC):
    """Abstract base for all openingestion tokenizers.

    Subclasses must implement :meth:`count_tokens`.
    The batch variant delegates to it by default.
    """

    @abstractmethod
    def count_tokens(self, text: str) -> int:
        """Return the number of tokens in *text*."""
        ...

    def count_tokens_batch(self, texts: Sequence[str]) -> list[int]:
        """Count tokens for each text in *texts*.

        Default implementation is a plain loop; backends may override for
        native batch efficiency.
        """
        return [self.count_tokens(t) for t in texts]

    @abstractmethod
    def __repr__(self) -> str: ...


# ──────────────────────────────────────────────────────────────────────────────
# Built-in tokenizers
# ──────────────────────────────────────────────────────────────────────────────

class HeuristicTokenizer(BaseTokenizer):
    """Fast heuristic: 1 token ≈ 4 characters (GPT-family approximation).

    Suitable as a zero-dependency default when exact counts are not required.
    """

    def count_tokens(self, text: str) -> int:
        return max(1, len(text) // 4)

    def __repr__(self) -> str:
        return "HeuristicTokenizer()"


class CharacterTokenizer(BaseTokenizer):
    """One token per Unicode character."""

    def count_tokens(self, text: str) -> int:
        return len(text)

    def __repr__(self) -> str:
        return "CharacterTokenizer()"


class WordTokenizer(BaseTokenizer):
    """One token per whitespace-delimited word."""

    def count_tokens(self, text: str) -> int:
        return len(text.split())

    def __repr__(self) -> str:
        return "WordTokenizer()"


class ByteTokenizer(BaseTokenizer):
    """One token per UTF-8 byte."""

    def count_tokens(self, text: str) -> int:
        return len(text.encode("utf-8"))

    def __repr__(self) -> str:
        return "ByteTokenizer()"


# Registry of built-in string aliases  ──────────────────────────────────────
_BUILTIN: dict[str, type[BaseTokenizer]] = {
    "heuristic": HeuristicTokenizer,
    "character": CharacterTokenizer,
    "word":      WordTokenizer,
    "byte":      ByteTokenizer,
}


# ──────────────────────────────────────────────────────────────────────────────
# Backend wrappers
# ──────────────────────────────────────────────────────────────────────────────

class _TiktokenWrapper(BaseTokenizer):
    def __init__(self, enc: Any) -> None:
        self._enc = enc

    def count_tokens(self, text: str) -> int:
        return len(self._enc.encode(text))

    def count_tokens_batch(self, texts: Sequence[str]) -> list[int]:
        return [len(ids) for ids in self._enc.encode_batch(texts)]

    def __repr__(self) -> str:
        name = getattr(self._enc, "name", "?")
        return f"_TiktokenWrapper(encoding={name!r})"


class _TokenizersWrapper(BaseTokenizer):
    def __init__(self, tok: Any) -> None:
        self._tok = tok

    def count_tokens(self, text: str) -> int:
        return len(self._tok.encode(text, add_special_tokens=False).ids)

    def count_tokens_batch(self, texts: Sequence[str]) -> list[int]:
        encodings = self._tok.encode_batch(
            list(texts), add_special_tokens=False
        )
        return [len(e.ids) for e in encodings]

    def __repr__(self) -> str:
        return "_TokenizersWrapper()"


class _TransformersWrapper(BaseTokenizer):
    def __init__(self, tok: Any) -> None:
        self._tok = tok

    def count_tokens(self, text: str) -> int:
        return len(self._tok.encode(text, add_special_tokens=False))

    def count_tokens_batch(self, texts: Sequence[str]) -> list[int]:
        enc = self._tok(list(texts), add_special_tokens=False)
        return [len(ids) for ids in enc["input_ids"]]

    def __repr__(self) -> str:
        name = getattr(self._tok, "name_or_path", "?")
        return f"_TransformersWrapper(model={name!r})"


class _CallableWrapper(BaseTokenizer):
    """Wraps a plain ``(text: str) -> int`` callable."""

    def __init__(self, fn: Callable[[str], int]) -> None:
        self._fn = fn

    def count_tokens(self, text: str) -> int:
        return self._fn(text)

    def __repr__(self) -> str:
        return f"_CallableWrapper(fn={self._fn!r})"


# ──────────────────────────────────────────────────────────────────────────────
# Error type
# ──────────────────────────────────────────────────────────────────────────────

class InvalidTokenizerError(ValueError):
    """Raised when no backend can load the requested tokenizer."""

    def __init__(self, message: str, *, backend_errors: dict[str, str]) -> None:
        super().__init__(message)
        self.backend_errors = backend_errors


# ──────────────────────────────────────────────────────────────────────────────
# AutoTokenizer
# ──────────────────────────────────────────────────────────────────────────────

def _load_from_string(name: str) -> BaseTokenizer:
    """Resolve a string tokenizer identifier to a :class:`BaseTokenizer`."""
    # 1. Built-in aliases
    if name in _BUILTIN:
        logger.debug("AutoTokenizer: using built-in {!r}", name)
        return _BUILTIN[name]()

    backend_errors: dict[str, str] = {}

    # 2. tiktoken
    if importutil.find_spec("tiktoken") is not None:
        try:
            import tiktoken
            enc = tiktoken.get_encoding(name)
            logger.debug("AutoTokenizer: loaded via tiktoken ({})", name)
            return _TiktokenWrapper(enc)
        except Exception as exc:
            backend_errors["tiktoken"] = str(exc)
    else:
        backend_errors["tiktoken"] = "'tiktoken' not installed"

    # 3. HuggingFace tokenizers
    if importutil.find_spec("tokenizers") is not None:
        try:
            from tokenizers import Tokenizer as HFTokenizer
            tok = HFTokenizer.from_pretrained(name)
            logger.debug("AutoTokenizer: loaded via tokenizers ({})", name)
            return _TokenizersWrapper(tok)
        except Exception as exc:
            backend_errors["tokenizers"] = str(exc)
    else:
        backend_errors["tokenizers"] = "'tokenizers' not installed"

    # 4. HuggingFace transformers
    if importutil.find_spec("transformers") is not None:
        try:
            from transformers import AutoTokenizer as HFAuto
            tok = HFAuto.from_pretrained(name)
            logger.debug("AutoTokenizer: loaded via transformers ({})", name)
            return _TransformersWrapper(tok)
        except Exception as exc:
            backend_errors["transformers"] = str(exc)
    else:
        backend_errors["transformers"] = "'transformers' not installed"

    raise InvalidTokenizerError(
        f"Tokenizer {name!r} could not be loaded from any backend: {backend_errors}",
        backend_errors=backend_errors,
    )


def AutoTokenizer(
    tokenizer: str | Callable[[str], int] | Any = "heuristic",
) -> BaseTokenizer:
    """Resolve *tokenizer* to a :class:`BaseTokenizer` instance.

    Args:
        tokenizer: One of:

            * ``str`` — built-in alias (``"heuristic"``, ``"character"``,
              ``"word"``, ``"byte"``) or a model/encoding name tried against
              tiktoken → tokenizers → transformers in that order.
            * :class:`BaseTokenizer` — returned as-is.
            * ``callable`` ``(text: str) -> int`` — wrapped in
              :class:`_CallableWrapper`.
            * ``tiktoken.Encoding``, ``tokenizers.Tokenizer``, or
              ``transformers.PreTrainedTokenizer`` — wrapped automatically.

    Returns:
        A :class:`BaseTokenizer` instance.

    Raises:
        :class:`InvalidTokenizerError`: If a string identifier cannot be
            resolved by any available backend.
        :class:`ValueError`: If the type is not recognised.

    Examples::

        from openingestion.utils.tokenizer import AutoTokenizer

        tok = AutoTokenizer()                         # HeuristicTokenizer
        tok = AutoTokenizer("word")                   # WordTokenizer
        tok = AutoTokenizer("cl100k_base")            # TiktokenWrapper
        tok = AutoTokenizer(lambda t: len(t.split())) # CallableWrapper
    """
    # Already a BaseTokenizer — pass through
    if isinstance(tokenizer, BaseTokenizer):
        return tokenizer

    # String: built-in or backend lookup
    if isinstance(tokenizer, str):
        return _load_from_string(tokenizer)

    # Plain callable
    if callable(tokenizer):
        # Distinguish between known tokenizer objects (which are also callable
        # in some backends) and plain count functions by checking for
        # a 'encode' or 'count_tokens' attribute first.
        if hasattr(tokenizer, "count_tokens"):
            # Duck-type compatible with our protocol — wrap minimally
            class _DuckWrapper(BaseTokenizer):
                def count_tokens(self, text: str) -> int:
                    return tokenizer.count_tokens(text)
                def count_tokens_batch(self, texts):
                    if hasattr(tokenizer, "count_tokens_batch"):
                        return tokenizer.count_tokens_batch(texts)
                    return [tokenizer.count_tokens(t) for t in texts]
                def __repr__(self):
                    return f"_DuckWrapper({tokenizer!r})"
            return _DuckWrapper()

        if hasattr(tokenizer, "encode"):
            # Identify backend by module path
            mod = type(tokenizer).__module__
            if "tiktoken" in mod:
                return _TiktokenWrapper(tokenizer)
            if "tokenizers" in mod:
                return _TokenizersWrapper(tokenizer)
            if "transformers" in mod:
                return _TransformersWrapper(tokenizer)

        # Plain callable: (text) -> int
        return _CallableWrapper(tokenizer)

    raise ValueError(
        f"Unsupported tokenizer type: {type(tokenizer)!r}. "
        "Pass a string alias, a BaseTokenizer instance, a known tokenizer "
        "object, or a callable (text: str) -> int."
    )
