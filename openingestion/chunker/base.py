"""Base class for openingestion Chunkers (CHOMP step 2)."""
from __future__ import annotations

from abc import ABC, abstractmethod

from loguru import logger

from openingestion.document import ContentBlock, RagChunk


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
