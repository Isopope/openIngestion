"""openingestion.utils — shared utilities for the CHOMP pipeline."""
from openingestion.utils.tokenizer import (
    AutoTokenizer,
    BaseTokenizer,
    ByteTokenizer,
    CharacterTokenizer,
    HeuristicTokenizer,
    InvalidTokenizerError,
    TokenizerProtocol,
    WordTokenizer,
)

# Embedder symbols are loaded lazily to avoid a hard dependency on numpy
# at import time. Install with: pip install openingestion[semantic]
_EMBEDDER_NAMES = frozenset([
    "AutoEmbedder",
    "BaseEmbedder",
    "EmbedderProtocol",
    "InvalidEmbedderError",
])


def __getattr__(name: str):
    if name in _EMBEDDER_NAMES:
        from openingestion.utils.embedder import (
            AutoEmbedder,
            BaseEmbedder,
            EmbedderProtocol,
            InvalidEmbedderError,
        )
        _mapping = {
            "AutoEmbedder": AutoEmbedder,
            "BaseEmbedder": BaseEmbedder,
            "EmbedderProtocol": EmbedderProtocol,
            "InvalidEmbedderError": InvalidEmbedderError,
        }
        return _mapping[name]
    raise AttributeError(f"module 'openingestion.utils' has no attribute {name!r}")


__all__ = [
    # embedder (lazy)
    "AutoEmbedder",
    "BaseEmbedder",
    "EmbedderProtocol",
    "InvalidEmbedderError",
    # tokenizer (always available)
    "AutoTokenizer",
    "BaseTokenizer",
    "HeuristicTokenizer",
    "CharacterTokenizer",
    "WordTokenizer",
    "ByteTokenizer",
    "TokenizerProtocol",
    "InvalidTokenizerError",
]
