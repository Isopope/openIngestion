"""openingestion.chunker — CHOMP step 2: chunking strategies.

Available chunkers
------------------
``BlockChunker``
    One RagChunk per ContentBlock — maximum positional fidelity.
    See :mod:`openingestion.chunker.by_block`.

``TokenChunker``
    Fixed-size token windows with configurable overlap.
    See :mod:`openingestion.chunker.by_token`.

``SentenceChunker``
    Sentence-boundary-preserving windows with configurable overlap.
    See :mod:`openingestion.chunker.by_sentence`.

``RecursiveChunker``
    Hierarchical recursive splitting (paragraphs → sentences → words → chars).
    Good for long, well-structured documents (books, research papers).
    See :mod:`openingestion.chunker.by_recursive`.

``SemanticChunker``
    Embedding-based semantic similarity chunker with Savitzky-Golay filtering.
    See :mod:`openingestion.chunker.by_semantic`.

``SlumberChunker``
    LLM-guided agentic chunker (inspired by LumberChunker / chonkie).
    Uses a Genie (LLM) to detect semantic topic boundaries.
    See :mod:`openingestion.chunker.by_slumber`.

``BaseChunker``
    Abstract base class for custom chunker implementations.
"""
from openingestion.chunker.base import BaseChunker
from openingestion.chunker.by_block import BlockChunker
from openingestion.chunker.by_recursive import RecursiveChunker, RecursiveLevel, RecursiveRules
from openingestion.chunker.by_sentence import SentenceChunker
from openingestion.chunker.by_token import TokenChunker
from openingestion.chunker.multipass import MultipassChunker

# SemanticChunker requires numpy + scipy + sentence-transformers
# SlumberChunker requires openai + pydantic + tqdm
# Both are loaded lazily — install the matching extra to use them.
_LAZY = {
    "SemanticChunker": "openingestion.chunker.by_semantic",
    "SlumberChunker": "openingestion.chunker.by_slumber",
}


def __getattr__(name: str):
    if name in _LAZY:
        import importlib
        mod = importlib.import_module(_LAZY[name])
        return getattr(mod, name)
    raise AttributeError(f"module 'openingestion.chunker' has no attribute {name!r}")


__all__ = [
    "BaseChunker",
    "BlockChunker",
    "MultipassChunker",
    "TokenChunker",
    "SentenceChunker",
    "RecursiveChunker",
    "RecursiveLevel",
    "RecursiveRules",
    "SemanticChunker",   # lazy — pip install openingestion[semantic]
    "SlumberChunker",    # lazy — pip install openingestion[slumber]
]
