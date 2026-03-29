"""JSONPorter — export RagChunks to JSON or JSONL files.

Two modes
---------
``lines=True`` (default)
    Newline-delimited JSON (JSONL / JSON Lines).  One chunk per line.
    Ideal for streaming ingestion into vector stores or batch pipelines.
    Default filename: ``chunks.jsonl``

``lines=False``
    Pretty-printed JSON array.
    Default filename: ``chunks.json``

Usage::

    from openingestion.porter.json_porter import JSONPorter

    porter = JSONPorter(lines=True)
    porter.export(enriched_chunks, file="out/doc1.jsonl")

    # or with __call__
    porter(enriched_chunks, file="out/doc1.jsonl")

Serialisation notes
-------------------
* ``RagChunk`` is a plain dataclass → serialised via ``dataclasses.asdict()``.
* ``BlockKind`` inherits from ``str`` so JSON handles it without a custom encoder.
* ``extras`` values are passed through as-is; ensure any binary data has already
  been converted to a string (e.g. ``image_b64`` data URI) by RagRefinery.
"""
from __future__ import annotations

import dataclasses
import json
import os
from pathlib import Path
from typing import Any

from loguru import logger

from openingestion.document import RagChunk
from openingestion.porter.base import BasePorter


def _chunk_to_dict(chunk: RagChunk) -> dict[str, Any]:
    """Serialise a RagChunk to a plain dict.

    ``dataclasses.asdict`` recurses into nested dataclasses/lists/dicts.
    ``BlockKind`` is a ``str`` subclass so it serialises to its string value
    without a custom encoder.
    """
    d = dataclasses.asdict(chunk)
    # Normalise kind to plain str (defensive — already str but avoids subclass
    # surprises with some JSON libraries).
    d["kind"] = str(chunk.kind.value)
    return d


class JSONPorter(BasePorter):
    """Porter that writes a list of RagChunks to a JSON or JSONL file.

    Args:
        lines:
            When ``True`` (default), writes JSONL (one JSON object per line).
            When ``False``, writes a pretty-printed JSON array.
        indent:
            JSON indentation when ``lines=False``.  Defaults to 4.
        encoding:
            File encoding.  Defaults to ``"utf-8"``.
        ensure_ascii:
            Passed to :func:`json.dumps` / :func:`json.dump`.
            Set to ``False`` (default) to preserve Unicode characters.
    """

    def __init__(
        self,
        lines: bool = True,
        indent: int = 4,
        encoding: str = "utf-8",
        ensure_ascii: bool = False,
    ) -> None:
        self.lines = lines
        self.indent = indent
        self.encoding = encoding
        self.ensure_ascii = ensure_ascii

    # ──────────────────────────────────────────────────────────────────────────
    # Internal writers
    # ──────────────────────────────────────────────────────────────────────────

    def _export_jsonl(
        self,
        chunks: list[RagChunk],
        file: str | os.PathLike,
    ) -> None:
        """Write JSONL — one JSON object per line."""
        logger.debug("JSONPorter: writing {} chunks → JSONL {}", len(chunks), file)
        Path(file).parent.mkdir(parents=True, exist_ok=True)
        with open(file, "w", encoding=self.encoding) as fh:
            for chunk in chunks:
                fh.write(
                    json.dumps(_chunk_to_dict(chunk), ensure_ascii=self.ensure_ascii)
                    + "\n"
                )
        logger.info("JSONPorter: exported {} chunks → {}", len(chunks), file)

    def _export_json(
        self,
        chunks: list[RagChunk],
        file: str | os.PathLike,
    ) -> None:
        """Write a pretty-printed JSON array."""
        logger.debug("JSONPorter: writing {} chunks → JSON {}", len(chunks), file)
        Path(file).parent.mkdir(parents=True, exist_ok=True)
        with open(file, "w", encoding=self.encoding) as fh:
            json.dump(
                [_chunk_to_dict(c) for c in chunks],
                fh,
                indent=self.indent,
                ensure_ascii=self.ensure_ascii,
            )
        logger.info("JSONPorter: exported {} chunks → {}", len(chunks), file)

    # ──────────────────────────────────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────────────────────────────────

    def export(
        self,
        chunks: list[RagChunk],
        file: str | os.PathLike = "chunks.jsonl",
    ) -> None:
        """Export *chunks* to *file*.

        Args:
            chunks: Enriched RagChunks to serialise.
            file:   Destination path.  Parent directories are created
                    automatically.  Defaults to ``"chunks.jsonl"``.
        """
        if self.lines:
            self._export_jsonl(chunks, file)
        else:
            self._export_json(chunks, file)

    def __call__(  # type: ignore[override]
        self,
        chunks: list[RagChunk],
        file: str | os.PathLike = "chunks.jsonl",
    ) -> None:
        """Shortcut: ``porter(chunks, file=…)``."""
        self.export(chunks, file)

    def __repr__(self) -> str:
        return (
            f"JSONPorter(lines={self.lines}, indent={self.indent}, "
            f"encoding={self.encoding!r})"
        )
