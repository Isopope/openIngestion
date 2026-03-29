"""Base class for openingestion Refineries (CHOMP step 3)."""
from __future__ import annotations

from abc import ABC, abstractmethod

from loguru import logger

from openingestion.document import RagChunk


class BaseRefinery(ABC):
    """Base class for all Refineries.

    A Refinery takes a list of raw RagChunks (from the CHUNKer)
    and enriches them with additional metadata.
    """

    @abstractmethod
    def enrich(self, chunks: list[RagChunk]) -> list[RagChunk]:
        """Enrich a list of RagChunks with metadata.

        Args:
            chunks: Raw RagChunks produced by the CHUNKer.

        Returns:
            Enriched RagChunks (same list, mutated in-place).

        """
        raise NotImplementedError("Subclasses must implement enrich()")

    def enrich_batch(
        self, chunks_list: list[list[RagChunk]]
    ) -> list[list[RagChunk]]:
        """Enrich multiple chunk lists in a batch.

        Args:
            chunks_list: List of RagChunk lists, one per document.

        Returns:
            List of enriched RagChunk lists.

        """
        logger.info("Refinery batch: enriching {} documents", len(chunks_list))
        results = [self.enrich(chunks) for chunks in chunks_list]
        logger.info("Refinery batch: completed {} documents", len(chunks_list))
        return results

    def __call__(self, chunks: list[RagChunk]) -> list[RagChunk]:
        """Shortcut: refinery(chunks) == refinery.enrich(chunks)."""
        logger.debug("{} enriching {} chunks", self.__class__.__name__, len(chunks))
        return self.enrich(chunks)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"
