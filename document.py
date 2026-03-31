"""Core data models for openingestion.

Three dataclasses are defined here:

- ``FetchedDocument`` — lightweight wrapper around a local path, produced by a Fetcher.
- ``ContentBlock``    — a single semantic unit produced by a Chef.
- ``RagChunk``        — a RAG-ready text chunk produced by a Chunker.

``BlockKind`` is shared across ``ContentBlock`` and ``RagChunk`` so that kind
information is preserved through the entire pipeline.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
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
    LIST = "list"
    TITLE = "title"
    IMAGE = "image"
    TABLE = "table"
    EQUATION = "equation"
    DISCARDED = "discarded"


# ──────────────────────────────────────────────────────────────────────────────
# FetchedDocument — Fetcher output
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class FetchedDocument:
    """Lightweight wrapper around a local file path, produced by a Fetcher.

    This is the first object in the pipeline.  A Fetcher yields
    ``FetchedDocument`` instances; a Chef consumes them to produce
    ``ContentBlock`` lists.

    Attributes:
        source:    Canonical identifier for the document (absolute path,
                   URL, S3 URI, …).  Always a ``str`` so it is
                   JSON-serialisable.
        path:      Local filesystem path.  ``None`` for cloud/stream sources
                   that have not yet been materialised.
        mime_type: MIME type detected at fetch time (e.g. ``"application/pdf"``).
                   Empty string when unknown.
        metadata:  Arbitrary key-value pairs injected by the Fetcher
                   (e.g. ``{"bucket": "…", "etag": "…", "db_id": 42}``).
    """

    source: str
    path: Optional[Path]
    mime_type: str = ""
    metadata: dict = field(default_factory=dict)


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
        doc_summary:       LLM-generated summary of the whole document (~100 tokens),
                           shared by every chunk of the same source document.
                           Filled by :class:`~openingestion.refinery.ContextualRagRefinery`.
        chunk_context:     LLM-generated sentence situating this chunk in the document,
                           filled by :class:`~openingestion.refinery.ContextualRagRefinery`.
        granularity:       Chunking granularity label (e.g. ``"mini"``, ``"normal"``,
                           ``"large"``).  Empty string for single-pass pipelines.
                           Set by :class:`~openingestion.chunker.MultipassChunker`.
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
    doc_summary: str = ""
    chunk_context: str = ""
    granularity: str = ""
