"""Base class for openingestion Fetchers (pipeline entry point).

A Fetcher is the very first step of the pipeline:

    **Fetcher → Chef → Chunker → Refinery → Porter**

Its sole responsibility is to **locate and enumerate documents** from a
source (local disk, cloud storage, database, web …) and wrap each one in
a :class:`~openingestion.document.FetchedDocument`.  The Fetcher does *not*
parse document content — that is the Chef's job.
"""
from __future__ import annotations

import os
from abc import ABC, abstractmethod

from loguru import logger

from openingestion.document import FetchedDocument


class BaseFetcher(ABC):
    """Abstract base class for all Fetchers.

    Subclasses must implement :meth:`fetch`.
    :meth:`fetch_all` and :meth:`__call__` are provided for free.
    """

    @abstractmethod
    def fetch(
        self,
        path: str | os.PathLike | None = None,
        dir: str | os.PathLike | None = None,
        **kwargs,
    ) -> list[FetchedDocument]:
        """Enumerate documents from the source.

        Subclasses may add their own keyword arguments on top of the
        common ``path`` / ``dir`` pair.

        Args:
            path: Path to a single file or document.
            dir:  Path to a directory of documents.
            **kwargs: Implementation-specific arguments.

        Returns:
            Ordered list of :class:`~openingestion.document.FetchedDocument`.
        """
        raise NotImplementedError("Subclasses must implement fetch()")

    def fetch_file(
        self,
        dir: str | os.PathLike,
        name: str,
    ) -> FetchedDocument:
        """Return a single named document inside *dir*.

        Useful when you already know the exact filename, without scanning
        the whole directory.

        Args:
            dir:  Directory to search.
            name: Exact filename (not a glob).

        Returns:
            The matching :class:`~openingestion.document.FetchedDocument`.

        Raises:
            FileNotFoundError: If no file named *name* exists in *dir*.
        """
        raise NotImplementedError(
            "fetch_file() is not implemented for this fetcher."
        )

    def __call__(
        self,
        path: str | os.PathLike | None = None,
        dir: str | os.PathLike | None = None,
        **kwargs,
    ) -> list[FetchedDocument]:
        """Shortcut: ``fetcher(…)`` == ``fetcher.fetch(…)``."""
        logger.debug(
            "{} fetching — path={!r}  dir={!r}",
            self.__class__.__name__, path, dir,
        )
        return self.fetch(path=path, dir=dir, **kwargs)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"
