"""Block-level chunker — one ContentBlock → one RagChunk.

This is the most faithful representation of the document's original
structure: every block becomes its own retrieval unit. Useful when
downstream systems (re-rankers, cross-encoders) can handle short,
heterogeneous passages AND when you want maximum positional fidelity.

Strategy
--------
* Each non-discarded ContentBlock is emitted directly as a RagChunk.
* TITLE blocks are emitted **and** update the running ``title_path``
  context so that all subsequent chunks carry the correct heading.
* DISCARDED blocks are silently dropped unless ``include_discarded=True``.
* ``prev_chunk_index`` / ``next_chunk_index`` links are wired at the end.
"""
from __future__ import annotations

from loguru import logger

from openingestion.chunker.base import BaseChunker
from openingestion.document import BlockKind, ContentBlock, RagChunk


class BlockChunker(BaseChunker):
    """Emit one RagChunk per ContentBlock.

    Args:
        include_discarded: Keep ``DISCARDED`` blocks (default ``False``).

    Example::

        from openingestion.chunker.by_block import BlockChunker

        chunker = BlockChunker()
        chunks = chunker(blocks, source="/path/to/doc.pdf")
    """

    def __init__(self, *, include_discarded: bool = False) -> None:
        self.include_discarded = include_discarded

    # ------------------------------------------------------------------

    def chunk(self, blocks: list[ContentBlock], source: str) -> list[RagChunk]:
        """Convert each ContentBlock into an individual RagChunk.

        Args:
            blocks: ContentBlocks in reading order (Chef output).
            source: Absolute path to the source document.

        Returns:
            List of RagChunks, one per non-discarded block.
        """
        chunks: list[RagChunk] = []
        title_path: str = ""
        title_level: int = 0

        for block in blocks:
            if block.kind is BlockKind.DISCARDED:
                if not self.include_discarded:
                    logger.trace("BlockChunker: skipping DISCARDED block {}", block.block_index)
                    continue

            # TITLE blocks update running context before being emitted.
            if block.kind is BlockKind.TITLE:
                title_path = block.text.strip()
                title_level = block.level or 1

            extras: dict = {}
            if block.html:
                extras["html"] = block.html
            if block.img_path:
                extras["img_path"] = block.img_path
            if block.captions:
                extras["captions"] = list(block.captions)
            if block.footnotes:
                extras["footnotes"] = list(block.footnotes)

            chunk = RagChunk(
                page_content=block.text,
                source=source,
                kind=block.kind,
                title_path=title_path,
                title_level=title_level,
                position_int=[[block.page_idx, *block.bbox]],
                block_indices=[block.block_index],
                reading_order=block.reading_order,
                chunk_index=len(chunks),
                extras=extras,
            )
            chunks.append(chunk)

        # Wire prev/next links
        for i, c in enumerate(chunks):
            object.__setattr__(c, "prev_chunk_index", i - 1 if i > 0 else None)
            object.__setattr__(c, "next_chunk_index", i + 1 if i < len(chunks) - 1 else None)

        logger.debug("BlockChunker: {} blocks → {} chunks", len(blocks), len(chunks))
        return chunks

    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return f"BlockChunker(include_discarded={self.include_discarded})"
