"""Sentence-boundary-aware chunker with token-based size limits.

Design inspired by ``chonkie.SentenceChunker`` -- adapted for the openingestion
CHOMP pipeline (``ContentBlock`` -> ``RagChunk`` with spatial metadata).

Strategy
--------
* **TEXT** blocks are split into individual sentences.  Sentences are
  accumulated into a window until ``chunk_size`` tokens would be exceeded.
  When the limit is reached the window is emitted and the most-recent
  sentences whose total token count fits within ``chunk_overlap`` are carried
  over to seed the next window.
* A sentence that alone exceeds ``chunk_size`` tokens is passed whole (never
  split mid-sentence).
* **TITLE** blocks create a **hard boundary**: the current sentence buffer is
  flushed as-is (no overlap carried over), then discarded entirely before the
  new section starts.  This is intentional — a title marks a semantic break
  strong enough that overlap context from the previous section would be
  misleading for retrieval.  The title block itself is emitted as a standalone
  chunk, and the sentence buffer starts empty under the new ``title_path``.
* **TABLE / IMAGE / EQUATION** flush the buffer, are emitted standalone, reset
  the buffer.
* **DISCARDED** are silently dropped unless ``include_discarded=True``.

Sentence splitting
------------------
Sentences are split on configurable *delimiters* (``delim``).  The
``include_delim`` option controls where the delimiter character(s) land:

* ``"prev"``  (default) -- delimiter is appended to the *preceding* sentence,
  e.g. ``"Hello. World"`` -> ``["Hello. ", "World"]``
* ``"next"``  -- delimiter is prepended to the *next* sentence,
  e.g. ``"Hello. World"`` -> ``["Hello", ". World"]``
* ``None``    -- delimiter is discarded entirely,
  e.g. ``"Hello. World"`` -> ``["Hello", "World"]``

Fragments shorter than ``min_characters_per_sentence`` characters are filtered
out.

Injectable tokenizer
--------------------
Token counting uses the same :class:`~openingestion.refinery.protocols.Tokenizer`
protocol as the rest of the pipeline.  The built-in default is ``len(text)//4``.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Literal, Optional, Union

from loguru import logger

from openingestion.chunker.base import BaseChunker, _advance_title_context, _page_content_for
from openingestion.document import BlockKind, ContentBlock, RagChunk
from openingestion.utils.tokenizer import AutoTokenizer, BaseTokenizer

# Block kinds always emitted individually (never merged into windows)
_STANDALONE = frozenset({BlockKind.TABLE, BlockKind.IMAGE, BlockKind.EQUATION})


# ------------------------------------------------------------------------------
# Internal sentence representation
# ------------------------------------------------------------------------------

@dataclass
class _Sentence:
    """Lightweight sentence container used during chunking."""

    text: str
    token_count: int
    block: ContentBlock
    blocks: list[ContentBlock] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.blocks:
            self.blocks = [self.block]


@dataclass
class _TextSpan:
    """Character span of one source block inside a merged text run."""

    start: int
    end: int
    block: ContentBlock


# ------------------------------------------------------------------------------
# Sentence splitter helpers
# ------------------------------------------------------------------------------

def _split_text(
    text: str,
    delim: list[str],
    include_delim: Optional[Literal["prev", "next"]],
    min_characters_per_sentence: int,
) -> list[str]:
    """Split *text* into sentences.

    Attempts to use the ultra-fast Rust-based `chonkie_core` if installed.
    Falls back to regex-based splitting if not available.
    
    Args:
        text:                       Input text.
        delim:                      List of delimiter strings.
        include_delim:              ``"prev"``, ``"next"``, or ``None``.
        min_characters_per_sentence: Minimum character length to keep a sentence.

    Returns:
        Non-empty list of sentence strings.
    """
    try:
        import chonkie_core

        text_bytes = text.encode("utf-8")
        include_mode = include_delim or "prev"

        has_multibyte = any(len(d) > 1 for d in delim)

        if has_multibyte:
            patterns = [d.encode("utf-8") for d in delim]
            offsets = chonkie_core.split_pattern_offsets(
                text_bytes,
                patterns=patterns,
                include_delim=include_mode,
                min_chars=min_characters_per_sentence,
            )
        else:
            delim_bytes = "".join(delim).encode("utf-8")
            offsets = chonkie_core.split_offsets(
                text_bytes,
                delimiters=delim_bytes,
                include_delim=include_mode,
                min_chars=min_characters_per_sentence,
            )

        splits = [text_bytes[start:end].decode("utf-8") for start, end in offsets]
        return [s for s in splits if s]

    except ImportError:
        # Fallback to python standard library regex
        sorted_delim = sorted(delim, key=len, reverse=True)
        pattern_str = "|".join(re.escape(d) for d in sorted_delim)
        pattern = re.compile(f"({pattern_str})")

        parts = pattern.split(text)
        sentences: list[str] = []
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
            sentence = chunk.strip()
            if len(sentence) >= min_characters_per_sentence:
                sentences.append(sentence)

        return sentences or [text.strip()]


# ------------------------------------------------------------------------------
# Standalone chunk helper
# ------------------------------------------------------------------------------

def _make_standalone_chunk(
    block: ContentBlock,
    source: str,
    title_path: str,
    title_level: int,
    idx: int,
) -> RagChunk:
    """Build a RagChunk from a single standalone ContentBlock."""
    extras: dict = {}
    if block.html:
        extras["html"] = block.html
    if block.img_path:
        extras["img_path"] = block.img_path
    if block.captions:
        extras["captions"] = list(block.captions)
    if block.footnotes:
        extras["footnotes"] = list(block.footnotes)
    return RagChunk(
        page_content=_page_content_for(block),
        source=source,
        kind=block.kind,
        title_path=title_path,
        title_level=title_level,
        position_int=[[block.page_idx, *block.bbox]],
        block_indices=[block.block_index],
        reading_order=block.reading_order,
        chunk_index=idx,
        extras=extras,
    )


# ------------------------------------------------------------------------------
# SentenceChunker
# ------------------------------------------------------------------------------

class SentenceChunker(BaseChunker):
    """Split text ContentBlocks into chunks while preserving sentence boundaries.

    Each chunk contains complete sentences whose cumulative token count stays
    within ``chunk_size``.  The last few sentences fitting within
    ``chunk_overlap`` tokens are carried over to the next chunk to provide
    local context continuity.

    When sentences originate from multiple ContentBlocks the chunk's spatial
    metadata (``position_int``, ``block_indices``) reflects all contributing
    blocks.

    Args:
        chunk_size:                  Maximum tokens per chunk (default 512).
        chunk_overlap:               Tokens of overlap between consecutive
                                     chunks (default 128). Must be < ``chunk_size``.
        min_sentences_per_chunk:     Minimum number of sentences before
                                     emitting a chunk (default 1).
        min_characters_per_sentence: Sentences shorter than this (chars) are
                                     discarded during splitting (default 12).
        delim:                       Delimiters for sentence splitting.
                                     Defaults to ``[". ", "! ", "? ", "\\n"]``.
        include_delim:               Where the delimiter lands after splitting:
                                     ``"prev"`` (default) appends to preceding
                                     sentence; ``"next"`` prepends to following
                                     sentence; ``None`` discards it.
        tokenizer:                   Object implementing the
                                     :class:`~openingestion.refinery.protocols.Tokenizer`
                                     protocol. Defaults to the ``len(text)//4``
                                     heuristic.
        include_discarded:           Whether to keep ``DISCARDED`` blocks
                                     (default ``False``).

    Example:

        from openingestion.chunker.by_sentence import SentenceChunker

        chunker = SentenceChunker(chunk_size=256, chunk_overlap=32)
        chunks = chunker(blocks, source="/path/to/doc.pdf")

    Custom tokenizer (tiktoken)::

        import tiktoken
        enc = tiktoken.get_encoding("cl100k_base")

        class TiktokenWrapper:
            def count_tokens(self, text: str) -> int:
                return len(enc.encode(text))

        chunker = SentenceChunker(tokenizer=TiktokenWrapper())

    Custom delimiters (Chinese)::

        chunker = SentenceChunker(
            delim=["。", "！", "？", "\\n"],
            include_delim="prev",
        )
    """

    def __init__(
        self,
        chunk_size: int = 512,
        chunk_overlap: int = 128,
        min_sentences_per_chunk: int = 1,
        min_characters_per_sentence: int = 12,
        delim: Union[str, list[str]] = None,
        include_delim: Optional[Literal["prev", "next"]] = "prev",
        tokenizer: str | BaseTokenizer | None = None,
        include_discarded: bool = False,
    ) -> None:
        if chunk_size <= 0:
            raise ValueError("chunk_size must be > 0")
        if chunk_overlap < 0:
            raise ValueError("chunk_overlap must be >= 0")
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be strictly < chunk_size")
        if min_sentences_per_chunk < 1:
            raise ValueError("min_sentences_per_chunk must be >= 1")
        if min_characters_per_sentence < 1:
            raise ValueError("min_characters_per_sentence must be >= 1")
        if include_delim not in ("prev", "next", None):
            raise ValueError("include_delim must be 'prev', 'next' or None")

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.min_sentences_per_chunk = min_sentences_per_chunk
        self.min_characters_per_sentence = min_characters_per_sentence

        if delim is None:
            delim = [". ", "! ", "? ", "\n"]
        self.delim: list[str] = [delim] if isinstance(delim, str) else list(delim)
        self.include_delim = include_delim

        self._tok: BaseTokenizer = AutoTokenizer(tokenizer if tokenizer is not None else "heuristic")
        self.include_discarded = include_discarded

    # --------------------------------------------------------------------------

    def _prepare_sentences(
        self, text: str, block: ContentBlock
    ) -> list[_Sentence]:
        """Split *text* into ``_Sentence`` objects with token counts."""
        raw = _split_text(
            text,
            self.delim,
            self.include_delim,
            self.min_characters_per_sentence,
        )
        return [
            _Sentence(
                text=s,
                token_count=self._tok.count_tokens(s),
                block=block,
            )
            for s in raw
        ]

    def _prepare_sentences_from_run(
        self,
        blocks: list[ContentBlock],
    ) -> list[_Sentence]:
        """Split one merged text run and preserve multi-block provenance."""
        parts: list[str] = []
        spans: list[_TextSpan] = []
        cursor = 0

        for i, block in enumerate(blocks):
            text = (block.text or "").strip()
            if not text:
                continue
            if parts:
                parts.append("\n")
                cursor += 1
            start = cursor
            parts.append(text)
            cursor += len(text)
            spans.append(_TextSpan(start=start, end=cursor, block=block))

        merged_text = "".join(parts)
        if not merged_text.strip():
            return []

        raw_sentences = _split_text(
            merged_text,
            self.delim,
            self.include_delim,
            self.min_characters_per_sentence,
        )

        if not raw_sentences:
            return []

        sentences: list[_Sentence] = []
        search_start = 0

        for text in raw_sentences:
            pos = merged_text.find(text, search_start)
            if pos == -1:
                pos = merged_text.find(text)
            if pos == -1:
                pos = search_start
            sent_start = pos
            sent_end = pos + len(text)
            search_start = sent_end

            contributing: list[ContentBlock] = []
            seen: set[int] = set()
            for span in spans:
                if span.end > sent_start and span.start < sent_end:
                    bid = id(span.block)
                    if bid not in seen:
                        seen.add(bid)
                        contributing.append(span.block)

            if not contributing and spans:
                contributing = [spans[0].block]

            sentences.append(
                _Sentence(
                    text=text,
                    token_count=self._tok.count_tokens(text),
                    block=contributing[0],
                    blocks=contributing,
                )
            )

        return sentences

    # --------------------------------------------------------------------------

    def chunk(self, blocks: list[ContentBlock], source: str) -> list[RagChunk]:  # noqa: C901
        """Group ContentBlocks into sentence-boundary RagChunks.

        Args:
            blocks: ContentBlocks in reading order (Chef output).
            source: Absolute path to the source document.

        Returns:
            List of RagChunks with ``prev_chunk_index`` / ``next_chunk_index``
            already populated.
        """
        chunks: list[RagChunk] = []
        title_stack: list[str] = []
        title_path: str = ""
        title_level: int = 0

        buf: list[_Sentence] = []
        buf_tokens: int = 0

        # ---- helpers ---------------------------------------------------------

        def _emit() -> None:
            """Emit the buffer as a chunk, then trim to overlap tail."""
            nonlocal buf, buf_tokens
            if not buf:
                return
            if len(buf) < self.min_sentences_per_chunk:
                return   # keep accumulating

            text = "".join(s.text for s in buf)

            seen: dict[int, ContentBlock] = {}
            for s in buf:
                for block in s.blocks:
                    bid = id(block)
                    if bid not in seen:
                        seen[bid] = block
            ordered = list(seen.values())

            chunks.append(RagChunk(
                page_content=text,
                source=source,
                kind=BlockKind.TEXT,
                title_path=title_path,
                title_level=title_level,
                position_int=[[b.page_idx, *b.bbox] for b in ordered],
                block_indices=[b.block_index for b in ordered],
                reading_order=buf[0].block.reading_order,
                chunk_index=len(chunks),
                extras={},
            ))

            # Rebuild overlap: most-recent sentences within chunk_overlap tokens
            new_buf: list[_Sentence] = []
            new_tokens = 0
            for sent in reversed(buf):
                if new_tokens + sent.token_count <= self.chunk_overlap:
                    new_buf.insert(0, sent)
                    new_tokens += sent.token_count
                else:
                    break
            buf = new_buf
            buf_tokens = new_tokens

        def _force_emit() -> None:
            """Emit regardless of min_sentences_per_chunk (boundary flush)."""
            nonlocal buf, buf_tokens
            if not buf:
                return
            text = "".join(s.text for s in buf)
            seen: dict[int, ContentBlock] = {}
            for s in buf:
                for block in s.blocks:
                    bid = id(block)
                    if bid not in seen:
                        seen[bid] = block
            ordered = list(seen.values())
            chunks.append(RagChunk(
                page_content=text,
                source=source,
                kind=BlockKind.TEXT,
                title_path=title_path,
                title_level=title_level,
                position_int=[[b.page_idx, *b.bbox] for b in ordered],
                block_indices=[b.block_index for b in ordered],
                reading_order=buf[0].block.reading_order,
                chunk_index=len(chunks),
                extras={},
            ))
            buf = []
            buf_tokens = 0

        def _reset() -> None:
            nonlocal buf, buf_tokens
            buf = []
            buf_tokens = 0

        def _append_sentence_chunk(sent: _Sentence) -> None:
            """Append a single oversized sentence as its own standalone chunk.

            Used when a sentence alone exceeds ``chunk_size``.  The buffer is
            guaranteed to be empty before this is called.
            """
            chunks.append(RagChunk(
                page_content=sent.text,
                source=source,
                kind=sent.block.kind,
                title_path=title_path,
                title_level=title_level,
                position_int=[[b.page_idx, *b.bbox] for b in sent.blocks],
                block_indices=[b.block_index for b in sent.blocks],
                reading_order=min(b.reading_order for b in sent.blocks),
                chunk_index=len(chunks),
                extras={},
            ))

        def _flush_text_run(text_run: list[ContentBlock]) -> None:
            """Split one same-section text run into sentence units."""
            nonlocal buf, buf_tokens
            if not text_run:
                return

            sentences = self._prepare_sentences_from_run(text_run)

            for sent in sentences:
                if sent.token_count > self.chunk_size:
                    _emit()
                    _force_emit()
                    _append_sentence_chunk(sent)
                    continue

                if buf_tokens + sent.token_count > self.chunk_size:
                    _emit()
                    if buf_tokens + sent.token_count > self.chunk_size:
                        _force_emit()

                buf.append(sent)
                buf_tokens += sent.token_count

        # ---- main loop -------------------------------------------------------

        text_run: list[ContentBlock] = []

        for block in blocks:
            if block.kind == BlockKind.DISCARDED:
                if not self.include_discarded:
                    continue

            if block.kind == BlockKind.TITLE:
                _flush_text_run(text_run)
                text_run = []
                # Hard boundary: flush whatever was accumulating in the buffer
                # (even if it doesn't satisfy min_sentences_per_chunk yet),
                # then fully reset — no overlap is carried over to the new
                # section.  Overlap across a title boundary would mix context
                # from two distinct sections and hurt retrieval precision.
                _force_emit()
                _reset()  # <-- intentional: no overlap past a section boundary
                title_stack, title_path, title_level = _advance_title_context(
                    title_stack, block
                )
                chunks.append(
                    _make_standalone_chunk(
                        block, source, title_path, title_level, len(chunks)
                    )
                )
                continue

            if block.kind in _STANDALONE:
                _flush_text_run(text_run)
                text_run = []
                _force_emit()
                _reset()
                chunks.append(
                    _make_standalone_chunk(
                        block, source, title_path, title_level, len(chunks)
                    )
                )
                continue

            raw_text = (block.text or "").strip()
            if not raw_text:
                continue

            text_run.append(block)

        _flush_text_run(text_run)
        _force_emit()

        # ---- final pass: chunk_index + prev/next links ----------------------
        for i, ch in enumerate(chunks):
            ch.chunk_index = i
            ch.prev_chunk_index = i - 1 if i > 0 else None
            ch.next_chunk_index = i + 1 if i < len(chunks) - 1 else None

        logger.info(
            "SentenceChunker(chunk_size={}, overlap={}) -> {} chunks from {}",
            self.chunk_size, self.chunk_overlap, len(chunks), source,
        )
        return chunks

    # --------------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"SentenceChunker(chunk_size={self.chunk_size}, "
            f"chunk_overlap={self.chunk_overlap}, "
            f"min_sentences_per_chunk={self.min_sentences_per_chunk}, "
            f"min_characters_per_sentence={self.min_characters_per_sentence}, "
            f"delim={self.delim!r}, "
            f"include_delim={self.include_delim!r})"
        )
