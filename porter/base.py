"""Base class for openingestion Porters (CHOMP step 4)."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from loguru import logger

from openingestion.document import RagChunk


class BasePorter(ABC):
    """Base class for all Porters (and Handshakes).

    A Porter takes a list of enriched RagChunks and exports them
    to a target format (dicts, LangChain Documents, LlamaIndex Nodes, etc.).
    """

    @abstractmethod
    def export(self, chunks: list[RagChunk]) -> Any:
        """Export a list of RagChunks to the target format.

        Args:
            chunks: Enriched RagChunks to export.

        Returns:
            Data in the target format (type depends on implementation).

        """
        raise NotImplementedError("Subclasses must implement export()")

    def export_batch(self, chunks_list: list[list[RagChunk]]) -> list[Any]:
        """Export multiple chunk lists in a batch.

        Args:
            chunks_list: List of RagChunk lists, one per document.

        Returns:
            List of exported results.

        """
        logger.info("Porter batch: exporting {} documents", len(chunks_list))
        results = [self.export(chunks) for chunks in chunks_list]
        logger.info("Porter batch: completed {} documents", len(chunks_list))
        return results

    def __call__(self, chunks: list[RagChunk]) -> Any:
        """Shortcut: porter(chunks) == porter.export(chunks)."""
        logger.debug("{} exporting {} chunks", self.__class__.__name__, len(chunks))
        return self.export(chunks)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"
