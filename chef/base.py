"""Base class for openingestion Chefs (CHOMP step 1)."""
from __future__ import annotations

import os
from abc import ABC, abstractmethod

from loguru import logger

from openingestion.document import ContentBlock


class BaseChef(ABC):
    """Base class for all Chefs.

    A Chef takes a raw document (file path or output directory)
    and converts it into a list of ContentBlocks.
    """

    @abstractmethod
    def process(self, path: str | os.PathLike) -> list[ContentBlock]:
        """Process a file or directory and return ContentBlocks.

        Args:
            path: Path to the file or MinerU output directory.

        Returns:
            List of ContentBlocks extracted from the document.

        """
        raise NotImplementedError("Subclasses must implement process()")

    @abstractmethod
    def map_to_blocks(self, raw_items: list[dict]) -> list[ContentBlock]:
        """Map raw content_list items to ContentBlocks.

        Args:
            raw_items: List of raw dicts from *_content_list.json.

        Returns:
            List of typed ContentBlocks.

        """
        raise NotImplementedError("Subclasses must implement map_to_blocks()")

    def process_batch(
        self, paths: list[str | os.PathLike]
    ) -> list[list[ContentBlock]]:
        """Process multiple files or directories in a batch.

        Args:
            paths: List of file/directory paths to process.

        Returns:
            List of ContentBlock lists, one per input path.

        """
        logger.info("Chef batch: processing {} items", len(paths))
        results = [self.process(path) for path in paths]
        logger.info("Chef batch: completed {} items", len(paths))
        return results

    def __call__(self, path: str | os.PathLike) -> list[ContentBlock]:
        """Shortcut: chef(path) == chef.process(path)."""
        logger.debug("{} processing: {}", self.__class__.__name__, path)
        return self.process(path)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"
