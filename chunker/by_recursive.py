"""Recursive chunker — hierarchical text splitting with token size limits.

Design inspired by ``chonkie.RecursiveChunker``, adapted for the openingestion
CHOMP pipeline (``ContentBlock`` -> ``RagChunk`` with spatial metadata).

Strategy
--------
Split text from ContentBlocks recursively using a hierarchy of delimiters:

  Level 0 — paragraph breaks (``\\n\\n``)
  Level 1 — sentence boundaries (``". "``, ``"! "``, ``"? "``, ``"\\n"``)
  Level 2 — whitespace word boundaries
  Level 3 — character-level (fixed token windows)

At each level the text is split by that level's delimiters.  Pieces that still
exceed ``chunk_size`` tokens are recursed into the next level.  Pieces that
fit are merged greedily until the next would overflow ``chunk_size``, then the
accumulated group is emitted.

Spatial metadata
----------------
When multiple ContentBlocks contribute text to one logical chunk, the resulting
RagChunk carries ``position_int`` and ``block_indices`` from *all* contributing
blocks — preserving multi-page / multi-region provenance.

TITLE, TABLE, IMAGE, and EQUATION blocks are always emitted individually and
act as **hard boundaries**: the running text buffer is flushed before them.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Literal, Optional, Union

from loguru import logger

from openingestion.chunker.base import BaseChunker, _page_content_for
from openingestion.document import BlockKind, ContentBlock, RagChunk
from openingestion.utils.tokenizer import AutoTokenizer, BaseTokenizer

# Block kinds that bypass the recursive logic and are emitted individually.
_STANDALONE = frozenset({BlockKind.TABLE, BlockKind.IMAGE, BlockKind.EQUATION})


# ---------------------------------------------------------------------------
# RecursiveLevel  —  describes one level of splitting
# ---------------------------------------------------------------------------

@dataclass
class RecursiveLevel:
    """One level of the recursive splitting hierarchy.

    Args:
        delimiters:  List of string delimiters to split on at this level.
                     ``None`` triggers a whitespace-based split (words).
        include_delim: Where the delimiter lands after splitting.
                     ``"prev"`` (default) appends it to the preceding piece;
                     ``"next"`` prepends it to the following piece;
                     ``None`` discards it.
    """

    delimiters: Optional[list[str]] = None
    include_delim: Optional[Literal["prev", "next"]] = "prev"


@dataclass
class RecursiveRules:
    """Ordered sequence of RecursiveLevels used by RecursiveChunker.

    Default hierarchy mirrors common document structure::

        Level 0 — paragraph breaks
        Level 1 — sentence boundaries
        Level 2 — comma / clause boundaries
        Level 3 — whitespace (words)  [delimiters=None triggers word split]
        Level 4 — character fallback  [both None → fixed-size token windows]

    You can pass your own ``levels`` list to override the default.
    """

    levels: list[RecursiveLevel] = field(default_factory=lambda: [
        RecursiveLevel(delimiters=["\n\n", "\n\n\n"], include_delim=None),
        RecursiveLevel(delimiters=[". ", "! ", "? ", ".\n", "!\n", "?\n"], include_delim="prev"),
        RecursiveLevel(delimiters=[", ", "; ", ": "], include_delim="prev"),
        RecursiveLevel(delimiters=None, include_delim="prev"),   # word split
        RecursiveLevel(delimiters=None, include_delim=None),     # char fallback
    ])

    def __len__(self) -> int:
        return len(self.levels)

    def __getitem__(self, idx: int) -> RecursiveLevel:
        return self.levels[idx]


# ---------------------------------------------------------------------------
# Internal types
# ---------------------------------------------------------------------------

@dataclass
class _TextSpan:
    """A span of text extracted from a ContentBlock, with character offsets
    relative to the aggregated text buffer."""

    text: str
    start: int          # offset in the aggregated string
    end: int            # exclusive end
    block: ContentBlock


# ---------------------------------------------------------------------------
# Low-level split helpers
# ---------------------------------------------------------------------------

def _split_by_delimiters(
    text: str,
    delimiters: list[str],
    include_delim: Optional[Literal["prev", "next"]],
    min_characters: int,
) -> list[str]:
    """Split *text* on any of *delimiters*, respecting *include_delim*."""
    sorted_delimiters = sorted(delimiters, key=len, reverse=True)
    pattern = re.compile("(" + "|".join(re.escape(d) for d in sorted_delimiters) + ")")

    parts = pattern.split(text)
    pieces: list[str] = []
    i = 0
    while i < len(parts):
        chunk = parts[i]
        i += 1
        if i < len(parts):
            delim_found = parts[i]
            i += 1
            if include_delim == "prev":
                chunk = chunk + delim_found
            elif include_delim == "next":
                if i < len(parts):
                    parts[i] = delim_found + parts[i]
            # None → delimiter discarded
        stripped = chunk.strip()
        if len(stripped) >= min_characters:
            pieces.append(stripped)
    return pieces or [text.strip()]


def _split_by_whitespace(
    text: str,
    include_delim: Optional[Literal["prev", "next"]],
    min_characters: int,
) -> list[str]:
    """Split on whitespace boundaries."""
    words = re.split(r"(\s+)", text)
    pieces: list[str] = []
    i = 0
    while i < len(words):
        word = words[i]
        i += 1
        if i < len(words):
            ws = words[i]
            i += 1
            if include_delim == "prev":
                word = word + ws
            elif include_delim == "next":
                if i < len(words):
                    words[i] = ws + words[i]
        stripped = word.strip()
        if len(stripped) >= min_characters:
            pieces.append(stripped)
    return pieces or [text.strip()]


# ---------------------------------------------------------------------------
# RecursiveChunker
# ---------------------------------------------------------------------------

class RecursiveChunker(BaseChunker):
    """Chunk ContentBlocks by recursively splitting text at multiple granularities.

    Text blocks are accumulated and then split hierarchically.  Each level of
    the hierarchy tries to split on progressively finer delimiters.  If a
    piece still exceeds ``chunk_size`` tokens after a split it is passed to the
    next level.  Short pieces are merged greedily until the token budget is
    exhausted before being emitted as a RagChunk.

    TITLE, TABLE, IMAGE, and EQUATION blocks are always emitted individually
    and act as hard text-buffer boundaries.

    Args:
        chunk_size:               Maximum number of tokens per chunk (default 512).
        rules:                    Ordered :class:`RecursiveRules` (default hierarchy
                                  covers paragraphs → sentences → words → chars).
        min_characters_per_chunk: Pieces shorter than this (chars) are discarded
                                  during splitting (default 24).
        tokenizer:                Tokenizer name or :class:`BaseTokenizer` instance.
                                  Defaults to the ``heuristic`` (``len//4``) tokenizer.
        include_discarded:        Keep ``DISCARDED`` blocks (default ``False``).

    Example::

        from openingestion.chunker.by_recursive import RecursiveChunker

        chunker = RecursiveChunker(chunk_size=512)
        chunks = chunker(blocks, source="/path/to/doc.pdf")

    Custom rules::

        from openingestion.chunker.by_recursive import RecursiveChunker, RecursiveLevel, RecursiveRules

        rules = RecursiveRules(levels=[
            RecursiveLevel(delimiters=["\\n\\n"], include_delim=None),
            RecursiveLevel(delimiters=[". ", "? ", "! "], include_delim="prev"),
            RecursiveLevel(delimiters=None),   # word split
        ])
        chunker = RecursiveChunker(chunk_size=256, rules=rules)
    """

    def __init__(
        self,
        chunk_size: int = 512,
        rules: RecursiveRules = None,
        min_characters_per_chunk: int = 24,
        tokenizer: Union[str, BaseTokenizer, None] = None,
        include_discarded: bool = False,
    ) -> None:
        if chunk_size <= 0:
            raise ValueError("chunk_size must be > 0")
        if min_characters_per_chunk < 1:
            raise ValueError("min_characters_per_chunk must be >= 1")

        self.chunk_size = chunk_size
        self.rules: RecursiveRules = rules if rules is not None else RecursiveRules()
        self.min_characters_per_chunk = min_characters_per_chunk
        self._tok: BaseTokenizer = AutoTokenizer(tokenizer if tokenizer is not None else "heuristic")
        self.include_discarded = include_discarded

    # -----------------------------------------------------------------------
    # Core recursive text splitting
    # -----------------------------------------------------------------------

    def _count(self, text: str) -> int:
        return self._tok.count_tokens(text)

    def _split_at_level(self, text: str, level: RecursiveLevel) -> list[str]:
        """Split *text* at the current *level*."""
        if level.delimiters is not None:
            return _split_by_delimiters(
                text, level.delimiters, level.include_delim, self.min_characters_per_chunk
            )
        else:
            # None delimiters → word-level split (or char fallback if already word)
            return _split_by_whitespace(text, level.include_delim, self.min_characters_per_chunk)

    def _merge_pieces(self, pieces: list[str]) -> list[str]:
        """Greedily merge *pieces* into groups that fit within ``chunk_size``."""
        merged: list[str] = []
        current_parts: list[str] = []
        current_tokens = 0

        for piece in pieces:
            piece_tokens = self._count(piece)
            if current_parts and current_tokens + piece_tokens > self.chunk_size:
                merged.append(" ".join(current_parts))
                current_parts = [piece]
                current_tokens = piece_tokens
            else:
                current_parts.append(piece)
                current_tokens += piece_tokens

        if current_parts:
            merged.append(" ".join(current_parts))

        return merged

    def _recursive_split(self, text: str, level: int = 0) -> list[str]:
        """Return a list of text chunks that each fit within ``chunk_size``."""
        if not text.strip():
            return []

        if self._count(text) <= self.chunk_size:
            return [text]

        # Exhausted all levels — emit as-is (oversized but unavoidable)
        if level >= len(self.rules):
            return [text]

        current_level = self.rules[level]
        pieces = self._split_at_level(text, current_level)

        # If splitting produced no useful subdivision, go deeper
        if len(pieces) <= 1 and pieces and pieces[0] == text.strip():
            return self._recursive_split(text, level + 1)

        # Recurse into pieces that are still too large, then merge the rest
        fine_pieces: list[str] = []
        for piece in pieces:
            if self._count(piece) > self.chunk_size:
                fine_pieces.extend(self._recursive_split(piece, level + 1))
            else:
                fine_pieces.append(piece)

        return self._merge_pieces(fine_pieces)

    # -----------------------------------------------------------------------
    # Spatial metadata helpers
    # -----------------------------------------------------------------------

    def _spans_for_text(
        self, text: str, spans: list[_TextSpan]
    ) -> list[_TextSpan]:
        """Find spans whose text range overlaps with a given chunk string.

        This is a best-effort, order-preserving match based on substring
        containment in the aggregated buffer.
        """
        # Rebuild a map: aggregate text → cumulative offsets (done once externally)
        return spans  # caller will filter by intersection

    def _blocks_for_chunk(
        self,
        chunk_text: str,
        spans: list[_TextSpan],
        agg_text: str,
    ) -> list[ContentBlock]:
        """Return all ContentBlocks that intersect with *chunk_text* in *agg_text*."""
        # Find the first occurrence of the chunk text in the aggregate
        pos = agg_text.find(chunk_text)
        if pos == -1:
            # Fall back: return all blocks
            return [s.block for s in spans]

        chunk_start = pos
        chunk_end = pos + len(chunk_text)

        contributing: list[ContentBlock] = []
        seen_ids: set[int] = set()
        for span in spans:
            # Check overlap: [span.start, span.end) ∩ [chunk_start, chunk_end)
            if span.end > chunk_start and span.start < chunk_end:
                bid = id(span.block)
                if bid not in seen_ids:
                    contributing.append(span.block)
                    seen_ids.add(bid)
        return contributing or [spans[0].block]

    # -----------------------------------------------------------------------
    # Buffer flush
    # -----------------------------------------------------------------------

    def _flush_buffer(
        self,
        spans: list[_TextSpan],
        source: str,
        title_path: str,
        title_level: int,
        chunks: list[RagChunk],
    ) -> None:
        """Flush all accumulated text spans → emit RagChunks."""
        if not spans:
            return

        agg_text = "".join(s.text for s in spans)
        if not agg_text.strip():
            return

        text_chunks = self._recursive_split(agg_text, level=0)

        for text in text_chunks:
            if not text.strip():
                continue
            contributing = self._blocks_for_chunk(text, spans, agg_text)
            chunks.append(RagChunk(
                page_content=text,
                source=source,
                kind=BlockKind.TEXT,
                title_path=title_path,
                title_level=title_level,
                position_int=[[b.page_idx, *b.bbox] for b in contributing],
                block_indices=[b.block_index for b in contributing],
                reading_order=contributing[0].reading_order,
                chunk_index=len(chunks),
                extras={},
            ))

    # -----------------------------------------------------------------------
    # Public API
    # -----------------------------------------------------------------------

    def chunk(self, blocks: list[ContentBlock], source: str) -> list[RagChunk]:
        """Recursively chunk ContentBlocks into token-bounded RagChunks.

        Args:
            blocks: ContentBlocks in reading order (Chef output).
            source: Absolute path to the source document.

        Returns:
            List of RagChunks with ``prev_chunk_index`` / ``next_chunk_index``
            already populated.
        """
        chunks: list[RagChunk] = []
        title_path: str = ""
        title_level: int = 0

        # Current text buffer: list of _TextSpan from accumulated TEXT blocks
        text_spans: list[_TextSpan] = []
        offset = 0

        for block in blocks:
            if block.kind is BlockKind.DISCARDED:
                if not self.include_discarded:
                    logger.trace("RecursiveChunker: skipping DISCARDED block {}", block.block_index)
                    continue

            if block.kind is BlockKind.TITLE:
                # Flush current text buffer, update title context
                self._flush_buffer(text_spans, source, title_path, title_level, chunks)
                text_spans = []
                offset = 0
                title_path = block.text.strip()
                title_level = block.title_level or 1
                # Emit the title block itself as a standalone chunk
                chunks.append(RagChunk(
                    page_content=block.text,
                    source=source,
                    kind=BlockKind.TITLE,
                    title_path=title_path,
                    title_level=title_level,
                    position_int=[[block.page_idx, *block.bbox]],
                    block_indices=[block.block_index],
                    reading_order=block.reading_order,
                    chunk_index=len(chunks),
                    extras={},
                ))
                continue

            if block.kind in _STANDALONE:
                # Flush text buffer, then emit standalone block
                self._flush_buffer(text_spans, source, title_path, title_level, chunks)
                text_spans = []
                offset = 0
                extras: dict = {}
                if block.html:
                    extras["html"] = block.html
                if block.img_path:
                    extras["img_path"] = block.img_path
                if block.captions:
                    extras["captions"] = list(block.captions)
                if block.footnotes:
                    extras["footnotes"] = list(block.footnotes)
                chunks.append(RagChunk(
                    page_content=_page_content_for(block),
                    source=source,
                    kind=block.kind,
                    title_path=title_path,
                    title_level=title_level,
                    position_int=[[block.page_idx, *block.bbox]],
                    block_indices=[block.block_index],
                    reading_order=block.reading_order,
                    chunk_index=len(chunks),
                    extras=extras,
                ))
                continue

            # TEXT block — accumulate into buffer
            text = block.text
            if not text.strip():
                continue
            span = _TextSpan(text=text, start=offset, end=offset + len(text), block=block)
            text_spans.append(span)
            offset += len(text)

        # Flush remaining buffer
        self._flush_buffer(text_spans, source, title_path, title_level, chunks)

        # Wire prev/next links
        for i, c in enumerate(chunks):
            object.__setattr__(c, "prev_chunk_index", i - 1 if i > 0 else None)
            object.__setattr__(c, "next_chunk_index", i + 1 if i < len(chunks) - 1 else None)

        logger.debug("RecursiveChunker: {} blocks → {} chunks", len(blocks), len(chunks))
        return chunks

    def __repr__(self) -> str:
        return (
            f"RecursiveChunker(chunk_size={self.chunk_size}, "
            f"levels={len(self.rules)}, "
            f"min_characters_per_chunk={self.min_characters_per_chunk})"
        )
