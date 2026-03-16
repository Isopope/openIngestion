"""openingestion.chunker — CHOMP step 2: chunking strategies.

Available chunkers
------------------
``TokenChunker``
    Fixed-size token windows with configurable overlap.
    See :mod:`openingestion.chunker.by_token`.

``SentenceChunker``
    Sentence-boundary-preserving windows with configurable overlap.
    See :mod:`openingestion.chunker.by_sentence`.

``BaseChunker``
    Abstract base class for custom chunker implementations.

``SemanticChunker``
    Embedding-based semantic similarity chunker with Savitzky-Golay filtering.
    See :mod:`openingestion.chunker.by_semantic`.
"""
from openingestion.chunker.base import BaseChunker
from openingestion.chunker.by_block import BlockChunker
from openingestion.chunker.by_semantic import SemanticChunker
from openingestion.chunker.by_sentence import SentenceChunker
from openingestion.chunker.by_token import TokenChunker

__all__ = [
    "BaseChunker",
    "BlockChunker",
    "TokenChunker",
    "SentenceChunker",
    "SemanticChunker",
]
