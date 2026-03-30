"""Token-aware chunker — fixed-size windows with configurable overlap.

Strategy
--------
* **TEXT** blocks accumulate in a sliding window until ``max_tokens`` is
  reached, then the window is emitted and the tail (at most ``overlap_tokens``
  worth of recent blocks) is carried over to seed the next window.
* A single TEXT block that alone exceeds ``max_tokens`` is passed through
  as-is (no mid-block split).
* **TITLE** blocks flush the current window, update the ``title_path``
  context, and are emitted as standalone chunks. The window is reset after
  a TITLE (new section = fresh context).
* **TABLE / IMAGE / EQUATION** blocks flush the current window, are emitted
  as standalone chunks, then reset the window.
* **DISCARDED** blocks are silently dropped unless ``include_discarded=True``.
"""
from __future__ import annotations

from loguru import logger

from openingestion.chunker.base import BaseChunker, _advance_title_context, _page_content_for
from openingestion.document import BlockKind, ContentBlock, RagChunk
from openingestion.utils.tokenizer import AutoTokenizer, BaseTokenizer

# Block kinds that are always emitted individually (never merged into windows)
_STANDALONE = frozenset({BlockKind.TABLE, BlockKind.IMAGE, BlockKind.EQUATION})


# ──────────────────────────────────────────────────────────────────────────────
# Internal helper
# ──────────────────────────────────────────────────────────────────────────────

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


# ──────────────────────────────────────────────────────────────────────────────
# TokenChunker
# ──────────────────────────────────────────────────────────────────────────────

class TokenChunker(BaseChunker):
    """Split text ContentBlocks into fixed-size token windows with overlap.

    Args:
        max_tokens:        Maximum tokens per chunk (default 512).
        overlap_tokens:    Tokens carried over from the previous chunk to the
                           next one (default 64). Must be < ``max_tokens``.
        tokenizer:         String alias (``"heuristic"``, ``"word"``, ``"cl100k_base"``…)
                           or :class:`~openingestion.utils.tokenizer.BaseTokenizer` instance.
                           Resolved via :func:`~openingestion.utils.tokenizer.AutoTokenizer`.
                           Defaults to ``"heuristic"`` (``len(text)//4``).
        include_discarded: Whether to keep ``DISCARDED`` blocks (default ``False``).

    Example::

        from openingestion.chunker.by_token import TokenChunker

        chunker = TokenChunker(max_tokens=256, overlap_tokens=32)
        chunks = chunker(blocks, source="/path/to/doc.pdf")
    """

    def __init__(
        self,
        max_tokens: int = 512,
        overlap_tokens: int = 64,
        tokenizer: str | BaseTokenizer | None = None,
        include_discarded: bool = False,
    ) -> None:
        if max_tokens <= 0:
            raise ValueError("max_tokens must be > 0")
        if overlap_tokens < 0:
            raise ValueError("overlap_tokens must be >= 0")
        if overlap_tokens >= max_tokens:
            raise ValueError("overlap_tokens must be strictly < max_tokens")
        self.max_tokens = max_tokens
        self.overlap_tokens = overlap_tokens
        self._tok: BaseTokenizer = AutoTokenizer(tokenizer if tokenizer is not None else "heuristic")
        self.include_discarded = include_discarded

    # ──────────────────────────────────────────────────────────────────────────

    def chunk(self, blocks: list[ContentBlock], source: str) -> list[RagChunk]:  # noqa: C901
        """Group ContentBlocks into token-window RagChunks.

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

        # Sliding window: list of (block, token_count) pairs
        buf: list[tuple[ContentBlock, int]] = []
        buf_tokens: int = 0

        # ── inner helpers ─────────────────────────────────────────────────────

        def _flush() -> None:
            """Emit the current window as a RagChunk, then trim to overlap."""
            nonlocal buf, buf_tokens
            if not buf:
                return

            text = " ".join(b.text for b, _ in buf if b.text)
            positions = [[b.page_idx, *b.bbox] for b, _ in buf]
            indices = [b.block_index for b, _ in buf]
            ro = buf[0][0].reading_order

            chunks.append(RagChunk(
                page_content=text,
                source=source,
                kind=BlockKind.TEXT,
                title_path=title_path,
                title_level=title_level,
                position_int=positions,
                block_indices=indices,
                reading_order=ro,
                chunk_index=len(chunks),
                extras={},
            ))

            # Rebuild overlap: keep the most-recent blocks that fit within
            # overlap_tokens.  Blocks are considered from the end (most-recent
            # first) so that the freshest context is preserved.
            new_buf: list[tuple[ContentBlock, int]] = []
            new_tokens = 0
            for item in reversed(buf):
                _, t = item
                if new_tokens + t <= self.overlap_tokens:
                    new_buf.insert(0, item)
                    new_tokens += t
                else:
                    break  # remaining older blocks exceed the budget

            buf = new_buf
            buf_tokens = new_tokens

        def _reset() -> None:
            nonlocal buf, buf_tokens
            buf = []
            buf_tokens = 0

        # ── main loop ─────────────────────────────────────────────────────────

        for block in blocks:
            # ── DISCARDED ──────────────────────────────────────────────
            if block.kind == BlockKind.DISCARDED:
                if not self.include_discarded:
                    continue

            # ── TITLE ──────────────────────────────────────────────────
            if block.kind == BlockKind.TITLE:
                _flush()
                _reset()                    # titles open a new section context
                title_stack, title_path, title_level = _advance_title_context(
                    title_stack, block
                )
                chunks.append(
                    _make_standalone_chunk(
                        block, source, title_path, title_level, len(chunks)
                    )
                )
                continue

            # ── TABLE / IMAGE / EQUATION ───────────────────────────────
            if block.kind in _STANDALONE:
                _flush()
                _reset()
                chunks.append(
                    _make_standalone_chunk(
                        block, source, title_path, title_level, len(chunks)
                    )
                )
                continue

            # ── TEXT (or included DISCARDED) ───────────────────────────
            btoks = self._tok.count_tokens(block.text or "")

            # Single block that alone exceeds max_tokens → pass whole,
            # never split mid-block.
            if btoks > self.max_tokens:
                _flush()
                _reset()
                chunks.append(RagChunk(
                    page_content=block.text,
                    source=source,
                    kind=block.kind,
                    title_path=title_path,
                    title_level=title_level,
                    position_int=[[block.page_idx, *block.bbox]],
                    block_indices=[block.block_index],
                    reading_order=block.reading_order,
                    chunk_index=len(chunks),
                    extras={},
                ))
                continue

            # Would overflow the window → flush first, then start fresh
            if buf_tokens + btoks > self.max_tokens:
                _flush()

            buf.append((block, btoks))
            buf_tokens += btoks

        # Flush whatever remains in the buffer
        _flush()

        # ── final pass: chunk_index + prev / next links ────────────────────
        for i, ch in enumerate(chunks):
            ch.chunk_index = i
            ch.prev_chunk_index = i - 1 if i > 0 else None
            ch.next_chunk_index = i + 1 if i < len(chunks) - 1 else None

        logger.info(
            "TokenChunker(max={}, overlap={}) → {} chunks from {}",
            self.max_tokens, self.overlap_tokens, len(chunks), source,
        )
        return chunks

    # ──────────────────────────────────────────────────────────────────────────

    def __repr__(self) -> str:
        return (
            f"TokenChunker(max_tokens={self.max_tokens}, "
            f"overlap_tokens={self.overlap_tokens})"
        )
