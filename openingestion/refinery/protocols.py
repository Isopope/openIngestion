"""Protocols and re-exports for the RagRefinery (CHOMP step 3).

Tokenizer
---------
The canonical tokenizer abstraction is now in :mod:`openingestion.utils.tokenizer`.
:class:`BaseTokenizer` and :func:`AutoTokenizer` are re-exported here for
backward compatibility.

Hasher
------
Minimal structural protocol — any object with a ``hash(text) -> str`` method
is accepted by the RagRefinery.

Example::

    from openingestion.refinery.protocols import AutoTokenizer

    tok = AutoTokenizer("cl100k_base")       # tiktoken
    tok = AutoTokenizer("word")              # built-in word tokenizer
    tok = AutoTokenizer(lambda t: len(t))    # plain callable
"""
from __future__ import annotations

from typing import Protocol, runtime_checkable

# Re-export from the canonical location
from openingestion.utils.tokenizer import AutoTokenizer, BaseTokenizer

# Backward-compat alias kept for code that imports `Tokenizer` from here
Tokenizer = BaseTokenizer


@runtime_checkable
class Hasher(Protocol):
    """Protocol for content hashing."""

    def hash(self, text: str) -> str:
        """Return a stable string hash of text."""
        ...
