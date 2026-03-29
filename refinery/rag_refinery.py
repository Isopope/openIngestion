"""RagRefinery — CHOMP step 3: enrichment of raw RagChunks.

Responsibilities (per specv3)
-----------------------------
1. ``token_count``      ← ``len(page_content) // 4`` (default) or injectable tokenizer
2. ``content_hash``     ← ``sha256(page_content)[:16]`` (default) or injectable hasher
3. ``img_path`` absolu  ← relative MinerU path resolved to absolute in ``extras``
4. ``image_mode``       ← controls how image binary is exposed in ``extras``
5. ``infer_captions``   ← generates ``extras["inferred_caption"]`` when captions empty

Image modes
-----------
``"path"`` (default)
    ``extras["img_path"]`` is set to the absolute local path.
``"base64"``
    ``extras["image_b64"]`` is set to a ``data:image/…;base64,…`` URI.
    Requires the image file to be present on disk.
``"skip"``
    The chunk is kept (with captions / inferred caption) but no binary data
    is written to ``extras``.
``"ignore"``
    IMAGE chunks are removed entirely from the output list.

Injectable protocols
--------------------
Both ``tokenizer`` and ``hasher`` are structurally typed (duck-typed):

* ``tokenizer`` — any object with ``count_tokens(text: str) -> int``,
  or any :class:`~openingestion.utils.tokenizer.BaseTokenizer` instance,
  or any plain ``callable(str) -> int``.
* ``hasher`` — any object with ``hash(text: str) -> str``.
  Defaults to ``sha256[:16]``.

Usage::

    from openingestion.refinery.rag_refinery import RagRefinery

    refinery = RagRefinery(
        output_dir="./mineru_out/doc1/auto/",   # base dir for img resolution
        image_mode="path",
        infer_captions=True,
    )
    enriched = refinery.enrich(chunks)
"""
from __future__ import annotations

import base64
import hashlib
import mimetypes
import os
from pathlib import Path
from typing import Any, Callable, Literal, Optional

from loguru import logger

from openingestion.document import BlockKind, RagChunk
from openingestion.refinery.base import BaseRefinery
from openingestion.refinery.protocols import Hasher, Tokenizer


# ──────────────────────────────────────────────────────────────────────────────
# Default implementations (zero extra dependencies)
# ──────────────────────────────────────────────────────────────────────────────

class _DefaultTokenizer:
    """Heuristic tokenizer: len(text) // 4."""

    def count_tokens(self, text: str) -> int:  # noqa: D102
        return len(text) // 4


class _DefaultHasher:
    """SHA-256 hasher, first 16 hex chars."""

    def hash(self, text: str) -> str:  # noqa: D102
        return hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()[:16]


class _CallableTokenizer:
    """Wrap a plain ``callable(str) -> int`` as a tokenizer."""

    def __init__(self, fn: Callable[[str], int]) -> None:
        self._fn = fn

    def count_tokens(self, text: str) -> int:  # noqa: D102
        return int(self._fn(text))


def _resolve_tokenizer(tok: Any) -> Any:
    """Normalise any tokenizer-like object to one with ``count_tokens``."""
    if tok is None:
        return _DefaultTokenizer()
    if hasattr(tok, "count_tokens"):
        return tok
    if callable(tok):
        return _CallableTokenizer(tok)
    raise TypeError(
        f"tokenizer must have count_tokens(str)->int or be callable, got {type(tok)}"
    )


def _resolve_hasher(h: Any) -> Any:
    """Normalise any hasher-like object to one with ``hash``."""
    if h is None:
        return _DefaultHasher()
    if hasattr(h, "hash"):
        return h
    raise TypeError(
        f"hasher must have hash(str)->str, got {type(h)}"
    )


# ──────────────────────────────────────────────────────────────────────────────
# Image path resolution helpers
# ──────────────────────────────────────────────────────────────────────────────

def _resolve_img_path(img_path: str, base_dir: Optional[Path]) -> Optional[Path]:
    """Return an absolute Path for *img_path*.

    Args:
        img_path: Relative path from MinerU (e.g. ``"images/abc.jpg"``).
        base_dir: MinerU output directory.  If ``None``, resolution is skipped.

    Returns:
        Absolute ``Path`` if the file exists, otherwise ``None``.
    """
    if not img_path or base_dir is None:
        return None
    candidate = (base_dir / img_path).resolve()
    if candidate.exists():
        return candidate
    logger.warning("RagRefinery: image file not found — {}", candidate)
    return None


def _to_base64_uri(path: Path) -> Optional[str]:
    """Read *path* and return a ``data:image/…;base64,…`` URI."""
    mime, _ = mimetypes.guess_type(str(path))
    mime = mime or "application/octet-stream"
    try:
        data = path.read_bytes()
        b64 = base64.b64encode(data).decode("ascii")
        return f"data:{mime};base64,{b64}"
    except OSError as exc:
        logger.warning("RagRefinery: cannot read image for base64 — {}: {}", path, exc)
        return None


# ──────────────────────────────────────────────────────────────────────────────
# RagRefinery
# ──────────────────────────────────────────────────────────────────────────────

class RagRefinery(BaseRefinery):
    """Enrich a list of raw :class:`~openingestion.document.RagChunk`.

    Args:
        output_dir:
            Base directory of the MinerU output (the directory that contains
            ``images/``).  Used to resolve relative ``img_path`` values.
            Can also be derived automatically from ``chunk.source`` when
            ``None`` (tries the parent directory of the source path).
        image_mode:
            How to handle image binary data in ``extras``:

            * ``"path"``   — ``extras["img_path"]`` = absolute local path
            * ``"base64"`` — ``extras["image_b64"]`` = data URI
            * ``"skip"``   — no binary, but chunk is kept (with captions)
            * ``"ignore"`` — IMAGE chunks removed from output

        infer_captions:
            When ``True``, a human-readable ``extras["inferred_caption"]``
            is generated for IMAGE / TABLE / EQUATION chunks whose
            ``captions`` list is empty.  Format: ``"<Kind> — p.<page+1>"``.
        tokenizer:
            Object with ``count_tokens(str) -> int``, or a plain callable,
            or ``None`` (default: ``len(text) // 4``).
        hasher:
            Object with ``hash(str) -> str``, or ``None``
            (default: ``sha256[:16]``).

    Example::

        from openingestion.refinery.rag_refinery import RagRefinery

        refinery = RagRefinery(output_dir="./out/doc/auto/", infer_captions=True)
        enriched = refinery.enrich(raw_chunks)
    """

    def __init__(
        self,
        output_dir: Optional[str | os.PathLike] = None,
        image_mode: Literal["path", "base64", "skip", "ignore"] = "path",
        infer_captions: bool = True,
        tokenizer: Any = None,
        hasher: Any = None,
    ) -> None:
        if image_mode not in ("path", "base64", "skip", "ignore"):
            raise ValueError(
                f"image_mode must be 'path', 'base64', 'skip' or 'ignore', "
                f"got {image_mode!r}"
            )
        self._output_dir: Optional[Path] = Path(output_dir) if output_dir else None
        self.image_mode = image_mode
        self.infer_captions = infer_captions
        self._tokenizer = _resolve_tokenizer(tokenizer)
        self._hasher = _resolve_hasher(hasher)

    # ──────────────────────────────────────────────────────────────────────────
    # Public entry-point
    # ──────────────────────────────────────────────────────────────────────────

    def enrich(self, chunks: list[RagChunk]) -> list[RagChunk]:
        """Enrich chunks in-place and return the (possibly filtered) list.

        Operations applied per chunk:

        1. Compute ``token_count``
        2. Compute ``content_hash``
        3. Resolve image path and apply ``image_mode``
        4. Append ``inferred_caption`` if ``infer_captions=True``

        IMAGE chunks are removed when ``image_mode="ignore"``.

        Args:
            chunks: Raw RagChunks produced by a Chunker.

        Returns:
            Enriched list (may be shorter than input if ``image_mode="ignore"``).
        """
        result: list[RagChunk] = []
        for chunk in chunks:
            if chunk.kind == BlockKind.IMAGE and self.image_mode == "ignore":
                continue
            self._enrich_one(chunk)
            result.append(chunk)

        logger.info(
            "RagRefinery: enriched {} chunk(s) (image_mode={!r}, {} dropped)",
            len(result),
            self.image_mode,
            len(chunks) - len(result),
        )
        return result

    # ──────────────────────────────────────────────────────────────────────────
    # Per-chunk enrichment
    # ──────────────────────────────────────────────────────────────────────────

    def _enrich_one(self, chunk: RagChunk) -> None:
        # 1. token_count
        chunk.token_count = self._tokenizer.count_tokens(chunk.page_content)

        # 2. content_hash — composite key: source + chunk_index + content.
        #    Hashing page_content alone would produce the same hash for any two
        #    chunks sharing identical text (e.g. repeated "Introduction" titles
        #    across documents), which breaks UNIQUE constraints in vector stores.
        #    Using source + chunk_index makes the hash globally unique while
        #    remaining deterministic across re-ingestion runs.
        chunk.content_hash = self._hasher.hash(
            f"{chunk.source}\x00{chunk.chunk_index}\x00{chunk.page_content}"
        )

        # 3. Image handling
        if chunk.kind == BlockKind.IMAGE:
            self._handle_image(chunk)

        # 4. Inferred caption
        if self.infer_captions:
            self._maybe_infer_caption(chunk)

    def _handle_image(self, chunk: RagChunk) -> None:
        """Resolve img_path and apply image_mode to extras."""
        raw_img_path: str = chunk.extras.get("img_path", "")

        # Determine base_dir: constructor value, or parent of source
        base_dir = self._output_dir
        if base_dir is None and chunk.source:
            base_dir = Path(chunk.source).parent

        abs_path = _resolve_img_path(raw_img_path, base_dir)

        if self.image_mode == "path":
            if abs_path:
                chunk.extras["img_path"] = str(abs_path)
        elif self.image_mode == "base64":
            if abs_path:
                uri = _to_base64_uri(abs_path)
                if uri:
                    chunk.extras["image_b64"] = uri
            # keep original img_path for audit
        elif self.image_mode == "skip":
            # Remove binary data but keep the chunk
            chunk.extras.pop("img_path", None)
            chunk.extras.pop("image_b64", None)

    def _maybe_infer_caption(self, chunk: RagChunk) -> None:
        """Add inferred_caption to extras when captions are empty."""
        captions = chunk.extras.get("captions", [])
        if captions:
            return  # already has real captions

        kind_label = {
            BlockKind.IMAGE: "Image",
            BlockKind.TABLE: "Table",
            BlockKind.EQUATION: "Équation",
        }.get(chunk.kind)

        if kind_label is None:
            return  # only for visual / structural kinds

        # Derive page number from position_int if available
        if chunk.position_int:
            page_display = chunk.position_int[0][0] + 1  # 1-based
        else:
            page_display = "?"

        section = f" — {chunk.title_path}" if chunk.title_path else ""
        chunk.extras["inferred_caption"] = (
            f"{kind_label}{section} — p.{page_display}"
        )

    # ──────────────────────────────────────────────────────────────────────────
    # Repr
    # ──────────────────────────────────────────────────────────────────────────

    def __repr__(self) -> str:
        return (
            f"RagRefinery("
            f"image_mode={self.image_mode!r}, "
            f"infer_captions={self.infer_captions}, "
            f"tokenizer={self._tokenizer!r}, "
            f"hasher={self._hasher!r})"
        )
