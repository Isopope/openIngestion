"""Core data models for openingestion.

Two immutable (or lightly mutable) dataclasses are defined here:

- ``ContentBlock`` — a single semantic unit produced by a Chef.
- ``RagChunk``     — a RAG-ready text chunk produced by a Chunker.

Both share the same ``BlockKind`` enum so that kind information is
preserved through the entire CHOMP pipeline.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


# ──────────────────────────────────────────────────────────────────────────────
# Block kind
# ──────────────────────────────────────────────────────────────────────────────

class BlockKind(str, Enum):
    """Semantic type of a content block.

    ``str`` mixin ensures JSON serialisability without extra converters.
    """

    TEXT = "text"
    TITLE = "title"
    IMAGE = "image"
    TABLE = "table"
    EQUATION = "equation"
    DISCARDED = "discarded"


# ──────────────────────────────────────────────────────────────────────────────
# ContentBlock — Chef output
# ──────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class ContentBlock:
    """Immutable representation of a single MinerU content block.

    Attributes:
        kind:          Semantic type of the block.
        text:          Plain-text content (may be empty for pure-image blocks).
        page_idx:      0-based page index in the source document.
        bbox:          Bounding box ``[x0, y0, x1, y1]`` normalised to [0, 1000].
        title_level:   Heading level (1-6); 0 for non-title blocks.
        html:          HTML representation (tables, equations).
        img_path:      Relative path to the image file produced by MinerU.
        captions:      Figure / table captions associated with the block.
        footnotes:     Footnotes attached to the block.
        block_index:   Position of the block within its page (0-based counter).
        reading_order: Global reading-order index across the whole document.
        raw:           Original raw dict from ``*_content_list.json``.
    """

    kind: BlockKind
    text: str
    page_idx: int
    bbox: list[int]               # [x0, y0, x1, y1] normalised [0–1000]
    title_level: int = 0
    html: str = ""
    img_path: str = ""            # relative path from MinerU output root
    captions: list[str] = field(default_factory=list)
    footnotes: list[str] = field(default_factory=list)
    block_index: int = 0
    reading_order: int = 0
    raw: dict = field(default_factory=dict)


# ──────────────────────────────────────────────────────────────────────────────
# RagChunk — Chunker output
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class RagChunk:
    """A RAG-ready text chunk with spatial and structural metadata.

    Attributes:
        page_content:      Text content of the chunk (fed to the LLM).
        source:            Absolute path to the source document.
        kind:              Dominant block kind of the chunk.
        title_path:        Breadcrumb of title ancestors (e.g. "Intro > 1.2").
        title_level:       Heading level of the enclosing section (0 if none).
        position_int:      Spatial provenance as ``[[page_idx, x0, y0, x1, y1], ...]``,
                           one entry per contributing ContentBlock.
        extras:            Supplementary data: ``html``, ``img_path`` (absolute),
                           ``captions``, ``footnotes``, etc.
        chunk_index:       Sequential index in the chunker output (0-based).
        block_indices:     ``block_index`` values of the ContentBlocks that
                           contributed to this chunk.
        reading_order:     Reading-order index of the first contributing block.
        prev_chunk_index:  ``chunk_index`` of the preceding chunk, or ``None``.
        next_chunk_index:  ``chunk_index`` of the following chunk, or ``None``.
        token_count:       Estimated token count (filled by RagRefinery).
        content_hash:      Stable hash of ``page_content`` (filled by RagRefinery).
    """

    page_content: str
    source: str                           # absolute path
    kind: BlockKind
    title_path: str = ""
    title_level: int = 0
    position_int: list[list[int]] = field(default_factory=list)
    extras: dict = field(default_factory=dict)
    chunk_index: int = 0
    block_indices: list[int] = field(default_factory=list)
    reading_order: int = 0
    prev_chunk_index: Optional[int] = None
    next_chunk_index: Optional[int] = None
    token_count: int = 0
    content_hash: str = ""
