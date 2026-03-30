"""Base class for openingestion Chunkers (CHOMP step 2)."""
from __future__ import annotations

from abc import ABC, abstractmethod

from loguru import logger

from openingestion.document import BlockKind, ContentBlock, RagChunk


# ──────────────────────────────────────────────────────────────────────────────
# Shared helper — used by every chunker to determine page_content
# ──────────────────────────────────────────────────────────────────────────────

def _page_content_for(block: ContentBlock) -> str:
    """Return the most informative text representation of *block*.

    MinerU stores table content as HTML (``block.html``), not as plain text,
    so ``block.text`` is typically empty for TABLE blocks.  Using an empty
    string as ``page_content`` would make the chunk invisible to LLMs.

    Some tables are captured by MinerU **as images only** (scanned tables,
    complex layouts) — ``html`` is then also empty.  In that case the caption
    is the best available description.

    Fallback chain (first non-empty wins):

    1. ``block.text``                               — always preferred.
    2. ``block.html``          (TABLE only)         — structured HTML content.
    3. ``" ".join(captions)``  (TABLE or IMAGE)     — caption when html absent.
    4. ``""``                                       — no content available.

    Args:
        block: A :class:`~openingestion.document.ContentBlock`.

    Returns:
        Non-empty string when content is available, ``""`` otherwise.
    """
    if block.text:
        return block.text
    if block.kind is BlockKind.TABLE:
        if block.html:
            return block.html
        if block.captions:
            return " ".join(block.captions)
    if block.kind is BlockKind.IMAGE and block.captions:
        return " ".join(block.captions)
    return ""


def _advance_title_context(
    title_stack: list[str],
    block: ContentBlock,
) -> tuple[list[str], str, int]:
    """Return the updated hierarchical title context after a TITLE block.

    ``title_path`` is represented as a breadcrumb of heading ancestors:
    ``"Chapter > Section > Subsection"``.
    """
    level = block.title_level or 1
    new_stack = title_stack[: max(level - 1, 0)]
    text = block.text.strip()
    if text:
        new_stack.append(text)
    return new_stack, " > ".join(new_stack), level


class BaseChunker(ABC):
    """Base class for all Chunkers.

    A Chunker takes a list of ContentBlocks (from the Chef)
    and groups them into RagChunks according to a specific strategy.
    """

    @abstractmethod
    def chunk(self, blocks: list[ContentBlock], source: str) -> list[RagChunk]:
        """Group ContentBlocks into RagChunks.

        Args:
            blocks: ContentBlocks produced by the Chef, in reading order.
            source: Absolute path to the original document (set on each RagChunk).

        Returns:
            List of RagChunks with title_path, position_int,
            block_indices, chunk_index, and prev/next links.

        """
        raise NotImplementedError("Subclasses must implement chunk()")

    def chunk_batch(
        self,
        blocks_list: list[list[ContentBlock]],
        sources: list[str],
    ) -> list[list[RagChunk]]:
        """Chunk multiple documents in a batch.

        Args:
            blocks_list: List of ContentBlock lists, one per document.
            sources: Corresponding absolute source paths.

        Returns:
            List of RagChunk lists, one per document.

        """
        if len(blocks_list) != len(sources):
            raise ValueError("blocks_list and sources must have the same length")
        logger.info("Chunker batch: chunking {} documents", len(blocks_list))
        results = [self.chunk(blocks, src) for blocks, src in zip(blocks_list, sources)]
        logger.info("Chunker batch: completed {} documents", len(blocks_list))
        return results

    def __call__(self, blocks: list[ContentBlock], source: str) -> list[RagChunk]:
        """Shortcut: chunker(blocks, source) == chunker.chunk(blocks, source)."""
        logger.debug(
            "{} chunking {} blocks from {}",
            self.__class__.__name__, len(blocks), source
        )
        return self.chunk(blocks, source)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"
