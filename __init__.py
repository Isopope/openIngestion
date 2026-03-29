"""openingestion — RAG ingestion pipeline on top of MinerU.

Architecture CHOMP
------------------
Chef → cHunker → rOefinery → porter → (M)odel

Public API
----------
:func:`ingest`
    One-liner end-to-end: PDF/image/MinerU-dir → list[RagChunk].

:func:`ingest_from_output`
    Skip MinerU parsing — read an existing MinerU output directory.

:func:`ingest_from_json`
    Skip MinerU parsing — read a ``*_content_list.json`` directly.

Chunking strategies (``strategy`` parameter)
--------------------------------------------
``"by_block"``      1 ContentBlock = 1 RagChunk (maximum fidelity)
``"by_token"``      Fixed token-window + overlap  *(default)*
``"by_sentence"``   Sentence-boundary-aware windows
``"by_semantic"``   Embedding-based similarity breakpoints

Output formats (``output_format`` parameter)
--------------------------------------------
``"chunks"``        list[RagChunk]  *(default)*
``"dicts"``         list[dict]      — plain Python dicts (JSON-serialisable)
``"langchain"``     list[Document]  — requires ``langchain-core``
``"llamaindex"``    list[TextNode]  — requires ``llama-index-core``

Example::

    from openingestion import ingest

    # Simplest
    chunks = ingest("rapport.pdf")

    # Full options
    chunks = ingest(
        "rapport.pdf",
        strategy="by_token",
        max_tokens=512,
        overlap_tokens=64,
        image_mode="path",
        infer_captions=True,
        output_format="chunks",
    )

    # Re-ingestion without re-parsing
    chunks = ingest_from_output("./output/rapport/auto/")
    chunks = ingest_from_json("./output/rapport/auto/rapport_content_list.json")
"""
from __future__ import annotations

import dataclasses
import os
from pathlib import Path
from typing import Any, Literal

from loguru import logger

from openingestion.document import RagChunk
from openingestion.chef.mineru_chef import MinerUChef
from openingestion.chunker.by_block import BlockChunker
from openingestion.chunker.by_token import TokenChunker
from openingestion.chunker.by_sentence import SentenceChunker
from openingestion.refinery.rag_refinery import RagRefinery

__all__ = ["ingest", "ingest_from_output", "ingest_from_json"]

# ──────────────────────────────────────────────────────────────────────────────
# Strategy registry
# ──────────────────────────────────────────────────────────────────────────────

_STRATEGIES = frozenset(["by_block", "by_token", "by_sentence", "by_semantic"])
_OUTPUT_FORMATS = frozenset(["chunks", "dicts", "langchain", "llamaindex"])


def _build_chunker(
    strategy: str,
    max_tokens: int,
    overlap_tokens: int,
    include_discarded: bool,
    tokenizer: Any,
) -> Any:
    if strategy == "by_block":
        return BlockChunker(include_discarded=include_discarded)
    if strategy == "by_token":
        return TokenChunker(
            max_tokens=max_tokens,
            overlap_tokens=overlap_tokens,
            tokenizer=tokenizer,
            include_discarded=include_discarded,
        )
    if strategy == "by_sentence":
        return SentenceChunker(
            chunk_size=max_tokens,
            chunk_overlap=overlap_tokens,
            include_discarded=include_discarded,
        )
    if strategy == "by_semantic":
        from openingestion.chunker.by_semantic import SemanticChunker
        return SemanticChunker(
            include_discarded=include_discarded,
        )
    if strategy in ("by_slumber", "slumber"):
        from openingestion.chunker.by_slumber import SlumberChunker
        return SlumberChunker(include_discarded=include_discarded)
    raise ValueError(
        f"Unknown strategy {strategy!r}. Choose from: {sorted(_STRATEGIES)}"
    )


def _apply_output_format(chunks: list[RagChunk], output_format: str) -> list[Any]:
    if output_format == "chunks":
        return chunks

    if output_format == "dicts":
        result = []
        for c in chunks:
            d = dataclasses.asdict(c)
            d["kind"] = str(c.kind.value)
            result.append(d)
        return result

    if output_format == "langchain":
        try:
            from langchain_core.documents import Document  # type: ignore[import]
        except ImportError as exc:
            raise ImportError(
                "output_format='langchain' requires langchain-core: "
                "pip install langchain-core"
            ) from exc
        return [
            Document(
                page_content=c.page_content,
                metadata={
                    "source": c.source,
                    "kind": c.kind.value,
                    "title_path": c.title_path,
                    "title_level": c.title_level,
                    "chunk_index": c.chunk_index,
                    "token_count": c.token_count,
                    "content_hash": c.content_hash,
                    "position_int": c.position_int,
                    **c.extras,
                },
            )
            for c in chunks
        ]

    if output_format == "llamaindex":
        try:
            from llama_index.core.schema import TextNode  # type: ignore[import]
        except ImportError as exc:
            raise ImportError(
                "output_format='llamaindex' requires llama-index-core: "
                "pip install llama-index-core"
            ) from exc
        return [
            TextNode(
                text=c.page_content,
                metadata={
                    "source": c.source,
                    "kind": c.kind.value,
                    "title_path": c.title_path,
                    "title_level": c.title_level,
                    "chunk_index": c.chunk_index,
                    "token_count": c.token_count,
                    "content_hash": c.content_hash,
                    **c.extras,
                },
            )
            for c in chunks
        ]

    raise ValueError(
        f"Unknown output_format {output_format!r}. "
        f"Choose from: {sorted(_OUTPUT_FORMATS)}"
    )


# ──────────────────────────────────────────────────────────────────────────────
# Core pipeline runner (shared by all ingest* functions)
# ──────────────────────────────────────────────────────────────────────────────

def _run_pipeline(
    blocks,
    source: str,
    output_dir_for_images: str | os.PathLike | None,
    strategy: str,
    max_tokens: int,
    overlap_tokens: int,
    include_tables: bool,
    include_images: bool,
    include_equations: bool,
    include_discarded: bool,
    image_mode: str,
    infer_captions: bool,
    tokenizer: Any,
    hasher: Any,
    output_format: str,
) -> list[Any]:
    from openingestion.document import BlockKind

    # ---- filter optional block kinds ----
    if not include_tables:
        blocks = [b for b in blocks if b.kind != BlockKind.TABLE]
    if not include_images:
        blocks = [b for b in blocks if b.kind != BlockKind.IMAGE]
    if not include_equations:
        blocks = [b for b in blocks if b.kind != BlockKind.EQUATION]

    logger.info(
        "ingest: {} blocks after filter (source={})", len(blocks), source
    )

    # ---- chunker ----
    chunker = _build_chunker(strategy, max_tokens, overlap_tokens, include_discarded, tokenizer)
    chunks = chunker(blocks, source=source)
    logger.info("ingest: {} chunks from strategy={}", len(chunks), strategy)

    # ---- refinery ----
    refinery = RagRefinery(
        output_dir=output_dir_for_images,
        image_mode=image_mode,
        infer_captions=infer_captions,
        tokenizer=tokenizer,
        hasher=hasher,
    )
    chunks = refinery.enrich(chunks)

    # ---- output format ----
    return _apply_output_format(chunks, output_format)


# ──────────────────────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────────────────────

def ingest(
    source: str | os.PathLike,
    *,
    # --- pipeline ---
    parser: Literal["mineru", "docling"] = "mineru",
    strategy: Literal["by_block", "by_token", "by_sentence", "by_semantic"] = "by_token",
    # --- chunking ---
    max_tokens: int = 512,
    overlap_tokens: int = 64,
    # --- MinerU ---
    backend: str = "pipeline",
    mineru_output_dir: str | os.PathLike = "./output",
    # --- content filters ---
    include_tables: bool = True,
    include_images: bool = True,
    include_equations: bool = True,
    include_discarded: bool = False,
    # --- image handling ---
    image_mode: Literal["path", "base64", "skip", "ignore"] = "path",
    infer_captions: bool = True,
    # --- protocols ---
    tokenizer: Any = None,
    hasher: Any = None,
    # --- output ---
    output_format: Literal["chunks", "dicts", "langchain", "llamaindex"] = "chunks",
) -> list[Any]:
    """End-to-end ingestion: source → enriched RagChunks (or target format).

    Accepts any of:

    * A **PDF or image file** — MinerU is called to parse it first.
    * A **MinerU output directory** (contains ``*_content_list.json``) —
      skips parsing, reads the JSON directly.
    * A **``*_content_list.json`` file** — parsed directly.

    Args:
        source:            Path to PDF, image, MinerU output dir, or JSON.
        strategy:          Chunking strategy (``"by_token"`` default).
        max_tokens:        Max tokens per chunk (``by_token`` / ``by_sentence``).
        overlap_tokens:    Token overlap between chunks.
        backend:           MinerU backend (``"pipeline"`` default).
        mineru_output_dir: Where MinerU writes output for raw PDFs/images.
        include_tables:    Keep TABLE blocks.
        include_images:    Keep IMAGE blocks.
        include_equations: Keep EQUATION blocks.
        include_discarded: Keep DISCARDED blocks (headers/footers).
        image_mode:        How to handle images in extras.
        infer_captions:    Generate ``extras["inferred_caption"]`` when empty.
        tokenizer:         Injectable ``count_tokens(str)->int`` protocol.
        hasher:            Injectable ``hash(str)->str`` protocol.
        output_format:     ``"chunks"`` | ``"dicts"`` | ``"langchain"`` | ``"llamaindex"``.

    Returns:
        List of RagChunks (or converted objects if ``output_format`` != ``"chunks"``).

    Example::

        chunks = ingest("rapport.pdf", strategy="by_token", max_tokens=512)
        chunks = ingest("rapport.pdf", parser="docling", strategy="by_token")
    """
    source = Path(source)
    if parser == "docling":
        from openingestion.chef.docling_chef import DoclingChef
        chef = DoclingChef()
    else:
        chef = MinerUChef(output_dir=mineru_output_dir, mineru_backend=backend)
    blocks = chef.process(source)

    # Determine base dir for image resolution.
    # MinerUChef sets last_output_dir after process() (JSON parent, output dir,
    # or the dir MinerU actually wrote to for raw PDF/image inputs).
    # DoclingChef does not write a MinerU-style image folder → falls back to None.
    img_base = getattr(chef, "last_output_dir", None)

    return _run_pipeline(
        blocks=blocks,
        source=str(source.resolve()),
        output_dir_for_images=img_base,
        strategy=strategy,
        max_tokens=max_tokens,
        overlap_tokens=overlap_tokens,
        include_tables=include_tables,
        include_images=include_images,
        include_equations=include_equations,
        include_discarded=include_discarded,
        image_mode=image_mode,
        infer_captions=infer_captions,
        tokenizer=tokenizer,
        hasher=hasher,
        output_format=output_format,
    )


def ingest_from_output(
    output_dir: str | os.PathLike,
    *,
    strategy: Literal["by_block", "by_token", "by_sentence", "by_semantic"] = "by_token",
    max_tokens: int = 512,
    overlap_tokens: int = 64,
    include_tables: bool = True,
    include_images: bool = True,
    include_equations: bool = True,
    include_discarded: bool = False,
    image_mode: Literal["path", "base64", "skip", "ignore"] = "path",
    infer_captions: bool = True,
    tokenizer: Any = None,
    hasher: Any = None,
    output_format: Literal["chunks", "dicts", "langchain", "llamaindex"] = "chunks",
) -> list[Any]:
    """Ingest from an existing MinerU output directory (no re-parsing).

    Args:
        output_dir: Path to the MinerU output directory (contains
                    ``*_content_list.json`` and ``images/``).

    Returns:
        Same as :func:`ingest`.

    Example::

        chunks = ingest_from_output("./output/rapport/auto/")
    """
    output_dir = Path(output_dir)
    chef = MinerUChef()
    blocks = chef.process(output_dir)

    return _run_pipeline(
        blocks=blocks,
        source=str(output_dir.resolve()),
        output_dir_for_images=output_dir,
        strategy=strategy,
        max_tokens=max_tokens,
        overlap_tokens=overlap_tokens,
        include_tables=include_tables,
        include_images=include_images,
        include_equations=include_equations,
        include_discarded=include_discarded,
        image_mode=image_mode,
        infer_captions=infer_captions,
        tokenizer=tokenizer,
        hasher=hasher,
        output_format=output_format,
    )


def ingest_from_json(
    json_path: str | os.PathLike,
    *,
    strategy: Literal["by_block", "by_token", "by_sentence", "by_semantic"] = "by_token",
    max_tokens: int = 512,
    overlap_tokens: int = 64,
    include_tables: bool = True,
    include_images: bool = True,
    include_equations: bool = True,
    include_discarded: bool = False,
    image_mode: Literal["path", "base64", "skip", "ignore"] = "path",
    infer_captions: bool = True,
    tokenizer: Any = None,
    hasher: Any = None,
    output_format: Literal["chunks", "dicts", "langchain", "llamaindex"] = "chunks",
) -> list[Any]:
    """Ingest from a ``*_content_list.json`` file (no re-parsing).

    Args:
        json_path: Path to the MinerU content-list JSON file.

    Returns:
        Same as :func:`ingest`.

    Example::

        chunks = ingest_from_json("./output/rapport/auto/rapport_content_list.json")
    """
    json_path = Path(json_path)
    chef = MinerUChef()
    blocks = chef.process(json_path)

    return _run_pipeline(
        blocks=blocks,
        source=str(json_path.resolve()),
        output_dir_for_images=json_path.parent,
        strategy=strategy,
        max_tokens=max_tokens,
        overlap_tokens=overlap_tokens,
        include_tables=include_tables,
        include_images=include_images,
        include_equations=include_equations,
        include_discarded=include_discarded,
        image_mode=image_mode,
        infer_captions=infer_captions,
        tokenizer=tokenizer,
        hasher=hasher,
        output_format=output_format,
    )
