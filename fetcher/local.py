"""Local filesystem fetcher for the openingestion pipeline.

:class:`LocalFileFetcher` walks a directory (or accepts a single path) and
returns a list of :class:`~openingestion.document.FetchedDocument` ready to
be processed by a Chef.

Supported source types
----------------------
* **Single file** — ``fetcher(path="report.pdf")``
* **Directory** — ``fetcher(dir="./inputs/", ext=[".pdf"])``

Extension filtering
-------------------
Pass ``ext`` as a list of extensions **with** leading dot:
``ext=[".pdf", ".png", ".jpg"]``.
When ``ext`` is ``None`` (default), all files are returned.

Pre-configured vs ad-hoc mode
------------------------------
You can pre-configure defaults at construction time::

    fetcher = LocalFileFetcher(ext=[".pdf"], extra_metadata={"batch": "2026-03"})
    docs = fetcher(dir="./run_42/")      # uses pre-configured ext

Or override at call time::

    docs = fetcher(dir="./run_43/", ext=[".pdf", ".png"])

MIME detection
--------------
``mime_type`` is detected automatically via the stdlib :mod:`mimetypes`
module.  Falls back to ``"application/octet-stream"`` for unknown types.
"""
from __future__ import annotations

import mimetypes
import os
from pathlib import Path
from typing import Optional

from loguru import logger

from openingestion.document import FetchedDocument
from openingestion.fetcher.base import BaseFetcher


def _detect_mime(path: Path) -> str:
    """Return the MIME type for *path*, falling back to octet-stream."""
    mime, _ = mimetypes.guess_type(str(path))
    return mime or "application/octet-stream"


class LocalFileFetcher(BaseFetcher):
    """Fetch files from the local filesystem.

    Args:
        ext:            Default extension filter (list of strings with leading
                        dot, e.g. ``[".pdf"]``).  ``None`` accepts all files.
        recursive:      When ``True`` (default), ``os.walk`` descends into
                        sub-directories.  When ``False``, only the top-level
                        directory is scanned.
        extra_metadata: Key-value pairs injected into every
                        :class:`~openingestion.document.FetchedDocument`
                        produced by this fetcher.  Useful for batch tags,
                        run IDs, etc.

    Example::

        from openingestion.fetcher.local import LocalFileFetcher

        fetcher = LocalFileFetcher(ext=[".pdf"])
        docs = fetcher(dir="./data/papers/")
        for doc in docs:
            blocks = chef.process(doc)
    """

    def __init__(
        self,
        ext: Optional[list[str]] = None,
        recursive: bool = True,
        extra_metadata: Optional[dict] = None,
    ) -> None:
        self.ext = ext
        self.recursive = recursive
        self.extra_metadata: dict = extra_metadata or {}

    # ──────────────────────────────────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────────────────────────────────

    def fetch(
        self,
        path: str | os.PathLike | None = None,
        dir: str | os.PathLike | None = None,
        ext: Optional[list[str]] = None,
        recursive: Optional[bool] = None,
        extra_metadata: Optional[dict] = None,
    ) -> list[FetchedDocument]:
        """Enumerate local files as :class:`~openingestion.document.FetchedDocument`.

        Args:
            path:           Path to a single file.  Mutually exclusive with
                            *dir*.
            dir:            Path to a directory.  Mutually exclusive with
                            *path*.
            ext:            Override ``self.ext`` for this call only.
            recursive:      Override ``self.recursive`` for this call only.
            extra_metadata: Override / extend ``self.extra_metadata``.

        Returns:
            Ordered list of :class:`~openingestion.document.FetchedDocument`.

        Raises:
            ValueError:        If neither or both *path* and *dir* are given.
            FileNotFoundError: If *path* or *dir* does not exist.
        """
        if path is not None and dir is not None:
            raise ValueError("Provide either 'path' or 'dir', not both.")
        if path is None and dir is None:
            raise ValueError("Must provide either 'path' or 'dir'.")

        # Resolve call-time overrides
        effective_ext = ext if ext is not None else self.ext
        effective_recursive = recursive if recursive is not None else self.recursive
        effective_meta = {**self.extra_metadata, **(extra_metadata or {})}

        if path is not None:
            return self._fetch_single(Path(path), effective_meta)
        else:
            return self._fetch_dir(
                Path(dir),  # type: ignore[arg-type]
                effective_ext,
                effective_recursive,
                effective_meta,
            )

    def fetch_file(
        self,
        dir: str | os.PathLike,
        name: str,
    ) -> FetchedDocument:
        """Return the first file named *name* inside *dir*.

        Args:
            dir:  Directory to search (non-recursive).
            name: Exact filename.

        Returns:
            A single :class:`~openingestion.document.FetchedDocument`.

        Raises:
            FileNotFoundError: If *name* is not found in *dir*.
        """
        dir_path = Path(dir)
        if not dir_path.is_dir():
            raise FileNotFoundError(f"Directory not found: {dir}")
        for p in dir_path.iterdir():
            if p.is_file() and p.name == name:
                return self._make_doc(p, {})
        raise FileNotFoundError(f"File '{name}' not found in directory: {dir}")

    # ──────────────────────────────────────────────────────────────────────────
    # Internals
    # ──────────────────────────────────────────────────────────────────────────

    def _fetch_single(
        self,
        file_path: Path,
        metadata: dict,
    ) -> list[FetchedDocument]:
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        if not file_path.is_file():
            raise ValueError(f"Path is not a file: {file_path}")
        logger.info("LocalFileFetcher: 1 file — {}", file_path)
        return [self._make_doc(file_path, metadata)]

    def _fetch_dir(
        self,
        dir_path: Path,
        ext: Optional[list[str]],
        recursive: bool,
        metadata: dict,
    ) -> list[FetchedDocument]:
        if not dir_path.exists():
            raise FileNotFoundError(f"Directory not found: {dir_path}")
        if not dir_path.is_dir():
            raise ValueError(f"Path is not a directory: {dir_path}")

        docs: list[FetchedDocument] = []

        if recursive:
            walker = os.walk(dir_path, followlinks=False)
        else:
            # Simulate a single-level walk
            entries = [e for e in os.scandir(dir_path) if e.is_file(follow_symlinks=False)]
            walker = iter([
                (str(dir_path), [], [e.name for e in entries])
            ])

        for root, _dirs, filenames in walker:
            for filename in sorted(filenames):
                if ext is not None:
                    file_ext = os.path.splitext(filename)[1].lower()
                    if file_ext not in [e.lower() for e in ext]:
                        continue
                full_path = Path(root) / filename
                docs.append(self._make_doc(full_path, metadata))

        logger.info(
            "LocalFileFetcher: {} file(s) found in {} (ext={}, recursive={})",
            len(docs), dir_path, ext, recursive,
        )
        return docs

    @staticmethod
    def _make_doc(path: Path, metadata: dict) -> FetchedDocument:
        return FetchedDocument(
            source=str(path.resolve()),
            path=path.resolve(),
            mime_type=_detect_mime(path),
            metadata=dict(metadata),
        )

    # ──────────────────────────────────────────────────────────────────────────
    # Repr
    # ──────────────────────────────────────────────────────────────────────────

    def __repr__(self) -> str:
        return (
            f"LocalFileFetcher("
            f"ext={self.ext!r}, "
            f"recursive={self.recursive})"
        )
