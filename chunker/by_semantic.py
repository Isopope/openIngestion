"""Semantic similarity-based chunker with Savitzky-Golay filtering.

Design
------
Sentences within TEXT blocks are embedded, then consecutive sentence-pair
similarities form a 1-D signal.  A **Savitzky-Golay** filter (or a simple
moving average when scipy is unavailable) smooths the curve; peaks in the
*inverse* signal (= valleys in similarity) become **breakpoints** that mark
sentence-group boundaries.  Optionally, a **skip-window** pass merges
semantically close non-adjacent groups back together.

Breakpoint detection modes
--------------------------
``"percentile"`` (default)
    A boundary is placed wherever the smoothed similarity drops below
    the Nth percentile of all similarities (e.g. 25th percentile).
    ``breakpoint_threshold_value`` is the percentile in ``[0, 100]``.
    Lower value → fewer, larger chunks.

``"threshold"``
    A boundary is placed wherever the smoothed similarity drops below
    the fixed value ``breakpoint_threshold_value`` (``[0, 1]``).

Special block handling
----------------------
* **TITLE** → hard boundary flush (no cross-section overlap), standalone chunk,
  new ``title_path``.
* **TABLE / IMAGE / EQUATION** → standalone chunk, buffer flushed first.
* **DISCARDED** → silently dropped unless ``include_discarded=True``.

Injectable embedder
-------------------
The ``model`` parameter accepts:

* ``str``                       — ``sentence-transformers`` model name
* :class:`~openingestion.utils.embedder.BaseEmbedder` — used as-is
* ``callable``                  — wrapped in ``_CallableEmbedder``
* any object with ``embed_batch`` — duck-typed

Required extras
---------------
``pip install sentence-transformers``  (when using string model names)
``pip install scipy``                  (for Savitzky-Golay, strongly recommended)

Both are optional for *import*; errors surface at call time only.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Literal, Optional

import numpy as np
from loguru import logger

from openingestion.chunker.base import BaseChunker
from openingestion.chunker.by_sentence import (
    _Sentence,
    _make_standalone_chunk,
    _split_text,
)
from openingestion.document import BlockKind, ContentBlock, RagChunk
from openingestion.utils.embedder import AutoEmbedder, BaseEmbedder

# Block kinds always emitted individually (never merged into windows)
_STANDALONE = frozenset({BlockKind.TABLE, BlockKind.IMAGE, BlockKind.EQUATION})

# ──────────────────────────────────────────────────────────────────────────────
# scipy guard (lazy)
# ──────────────────────────────────────────────────────────────────────────────

try:
    from scipy.signal import savgol_filter as _savgol_filter  # type: ignore

    _SCIPY_AVAILABLE = True
except ImportError:  # pragma: no cover
    _SCIPY_AVAILABLE = False


def _smooth(sims: np.ndarray, window_length: int, polyorder: int) -> np.ndarray:
    """Smooth a similarity array.

    Uses Savitzky-Golay if scipy is available, otherwise falls back to a
    simple uniform moving-average window.

    Args:
        sims:           1-D array of float similarities.
        window_length:  Smoothing window size (must be ≥ 1, odd preferred).
        polyorder:      Polynomial order for SG filter (ignored in fallback).

    Returns:
        Smoothed array of the same length.
    """
    n = len(sims)
    if window_length <= 1 or n < 2:
        return sims.copy()

    # scipy.signal.savgol_filter requires window_length <= data length
    wl = min(window_length, n)
    # Must be odd
    if wl % 2 == 0:
        wl -= 1
    if wl < 1:
        return sims.copy()
    # polyorder must be < window_length
    po = min(polyorder, wl - 1)

    if _SCIPY_AVAILABLE:
        return _savgol_filter(sims, window_length=wl, polyorder=po)
    else:
        logger.warning(
            "scipy not installed — falling back to moving-average smoothing. "
            "Run `pip install scipy` for better quality Savitzky-Golay filtering."
        )
        kernel = np.ones(wl, dtype=np.float64) / wl
        return np.convolve(sims, kernel, mode="same")


# ──────────────────────────────────────────────────────────────────────────────
# Internal group representation
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class _SentenceGroup:
    """A contiguous group of sentences destined to become one or more chunks."""

    sentences: list[_Sentence] = field(default_factory=list)

    @property
    def text(self) -> str:
        return " ".join(s.text for s in self.sentences)

    @property
    def token_count(self) -> int:
        return sum(s.token_count for s in self.sentences)

    @property
    def blocks(self) -> list[ContentBlock]:
        seen: set[int] = set()
        out: list[ContentBlock] = []
        for s in self.sentences:
            bid = id(s.block)
            if bid not in seen:
                seen.add(bid)
                out.append(s.block)
        return out


def _group_text(group: _SentenceGroup) -> str:
    """Join sentence texts, preserving trailing delimiter punctuation."""
    parts = [s.text for s in group.sentences]
    return " ".join(parts)


# ──────────────────────────────────────────────────────────────────────────────
# Similarity helpers
# ──────────────────────────────────────────────────────────────────────────────

def _cosine_matrix(vecs: np.ndarray) -> np.ndarray:
    """Row-wise cosine similarity between consecutive vectors.

    Args:
        vecs: (N, D) float array.

    Returns:
        (N-1,) similarity array.
    """
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    normed = vecs / norms
    return np.einsum("nd,nd->n", normed[:-1], normed[1:])


def _skip_window_sims(vecs: np.ndarray, skip_window: int) -> np.ndarray:
    """Compute skip-window similarities.

    For each position i, compute max(cosine(vecs[i], vecs[j]))
    for j in [i+1, i+skip_window].  This replaces the raw consecutive
    similarity with a broader look-ahead.

    Args:
        vecs:        (N, D) float array.
        skip_window: Number of positions to look ahead (>= 1).

    Returns:
        (N-1,) maximum similarity array.
    """
    n = len(vecs)
    sims = np.zeros(n - 1, dtype=np.float64)
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    normed = vecs / norms

    for i in range(n - 1):
        end = min(i + skip_window + 1, n)
        dot = normed[i] @ normed[i + 1: end].T  # (end-i-1,)
        sims[i] = float(np.max(dot))

    return sims


# ──────────────────────────────────────────────────────────────────────────────
# RagChunk builder
# ──────────────────────────────────────────────────────────────────────────────

def _build_chunk_from_group(
    group: _SentenceGroup,
    source: str,
    title_path: str,
    title_level: int,
    idx: int,
) -> RagChunk:
    """Convert a :class:`_SentenceGroup` into a :class:`RagChunk`."""
    blocks = group.blocks
    position_int = [[b.page_idx, *b.bbox] for b in blocks]
    block_indices = [b.block_index for b in blocks]
    reading_orders = [b.reading_order for b in blocks if b.reading_order is not None]
    reading_order = min(reading_orders) if reading_orders else None

    return RagChunk(
        page_content=_group_text(group),
        source=source,
        kind=BlockKind.TEXT,
        title_path=title_path,
        title_level=title_level,
        position_int=position_int,
        block_indices=block_indices,
        reading_order=reading_order,
        chunk_index=idx,
    )


# ──────────────────────────────────────────────────────────────────────────────
# SemanticChunker
# ──────────────────────────────────────────────────────────────────────────────

class SemanticChunker(BaseChunker):
    """Chunk TEXT blocks by semantic similarity between consecutive sentences.

    Pipeline (inside ``_process_text_run``)
    ----------------------------------------
    1. Split TEXT blocks into sentences (reuses ``_split_text`` from
       :mod:`~openingestion.chunker.by_sentence`).
    2. Embed all sentences in a single ``embed_batch`` call.
    3. Compute per-pair similarities (with optional ``skip_window`` look-ahead).
    4. Smooth the similarity curve with Savitzky-Golay (or moving average).
    5. Detect breakpoints: positions where similarity drops below a threshold.
    6. Segment sentences into initial groups at breakpoints.
    7. *(Optional)* Skip-and-merge: re-merge adjacent groups whose centroid
       similarity exceeds the breakpoint threshold.
    8. Emit one :class:`RagChunk` per group.

    TITLE blocks flush-and-reset: they are emitted as standalone chunks, and
    no overlap is carried across the section boundary.

    Args:
        model:
            String model name (``sentence-transformers``), a
            :class:`~openingestion.utils.embedder.BaseEmbedder` instance,
            a callable, or any duck-typed object with ``embed_batch``.
        breakpoint_threshold_type:
            ``"percentile"`` or ``"threshold"``.
        breakpoint_threshold_value:
            For ``"percentile"``: the Nth percentile (0–100) below which a
            gap is considered a breakpoint.  Lower → fewer, larger chunks.
            For ``"threshold"``: the fixed cosine similarity value (0–1).
        skip_window:
            When > 0, look-ahead ``skip_window`` positions when computing
            similarities, exposing longer-range connections.  Set to ``0`` to
            use only consecutive pairs.
        sg_window_length:
            Savitzky-Golay (or moving-average) window size.  Set to ``0`` or
            ``1`` to disable smoothing entirely.
        sg_polyorder:
            Polynomial order for SG filter.  Ignored when scipy is unavailable.
        delim:
            Sentence delimiter strings.
        include_delim:
            Where to attach the delimiter (``"prev"``, ``"next"``, or ``None``).
        min_characters_per_sentence:
            Fragments shorter than this are discarded.
        include_discarded:
            Emit ``DISCARDED`` blocks as standalone chunks when ``True``.

    Example::

        from openingestion.chunker.by_semantic import SemanticChunker

        chunker = SemanticChunker(
            model="all-MiniLM-L6-v2",
            breakpoint_threshold_type="percentile",
            breakpoint_threshold_value=25.0,
            skip_window=1,
        )
        chunks = chunker(blocks, source="my_doc.pdf")
    """

    def __init__(
        self,
        model: str | BaseEmbedder | Any = "sentence-transformers/all-MiniLM-L6-v2",
        breakpoint_threshold_type: Literal["percentile", "threshold"] = "percentile",
        breakpoint_threshold_value: float = 25.0,
        skip_window: int = 0,
        sg_window_length: int = 3,
        sg_polyorder: int = 2,
        delim: list[str] | None = None,
        include_delim: Optional[Literal["prev", "next"]] | None = "prev",
        min_characters_per_sentence: int = 12,
        include_discarded: bool = False,
    ) -> None:
        if breakpoint_threshold_type not in ("percentile", "threshold"):
            raise ValueError(
                f"breakpoint_threshold_type must be 'percentile' or 'threshold', "
                f"got {breakpoint_threshold_type!r}"
            )
        if not (0.0 <= breakpoint_threshold_value <= 100.0):
            raise ValueError(
                "breakpoint_threshold_value must be in [0, 100] "
                f"(got {breakpoint_threshold_value})"
            )

        self._embedder = AutoEmbedder(model)
        self.breakpoint_threshold_type = breakpoint_threshold_type
        self.breakpoint_threshold_value = breakpoint_threshold_value
        self.skip_window = skip_window
        self.sg_window_length = sg_window_length
        self.sg_polyorder = sg_polyorder
        self.delim = delim if delim is not None else [". ", "! ", "? ", "\n"]
        self.include_delim: Optional[Literal["prev", "next"]] = include_delim
        self.min_characters_per_sentence = min_characters_per_sentence
        self.include_discarded = include_discarded

    # ──────────────────────────────────────────────────────────────────────────
    # Public entry-point
    # ──────────────────────────────────────────────────────────────────────────

    def chunk(
        self,
        blocks: list[ContentBlock],
        source: str,
    ) -> list[RagChunk]:
        """Convert a list of :class:`ContentBlock` into semantic :class:`RagChunk`.

        Args:
            blocks: Ordered document blocks (output of a Chef).
            source: Document source identifier (filename, URL, …).

        Returns:
            Ordered list of :class:`RagChunk`.
        """
        results: list[RagChunk] = []
        title_path: str = ""
        title_level: int = 0

        # Accumulate consecutive TEXT blocks into runs; flush on non-TEXT blocks.
        text_run: list[ContentBlock] = []

        def flush_run() -> None:
            """Process the accumulated TEXT run and append chunks to results."""
            if not text_run:
                return
            run_chunks = self._process_text_run(
                text_run,
                source=source,
                title_path=title_path,
                title_level=title_level,
                start_idx=len(results),
            )
            results.extend(run_chunks)
            text_run.clear()

        for block in blocks:
            kind = block.kind

            if kind == BlockKind.DISCARDED:
                if self.include_discarded:
                    flush_run()
                    results.append(
                        _make_standalone_chunk(
                            block, source, title_path, title_level, len(results)
                        )
                    )
                continue

            if kind == BlockKind.TITLE:
                flush_run()
                title_level = block.title_level or 1
                sep = " > " if title_path else ""
                title_path = f"{title_path}{sep}{block.text}"
                results.append(
                    _make_standalone_chunk(
                        block, source, title_path, title_level, len(results)
                    )
                )
                continue

            if kind in _STANDALONE:
                flush_run()
                results.append(
                    _make_standalone_chunk(
                        block, source, title_path, title_level, len(results)
                    )
                )
                continue

            # TEXT block → accumulate
            text_run.append(block)

        flush_run()

        # Wire prev/next links
        for i, rc in enumerate(results):
            if i > 0:
                rc.prev_chunk_index = results[i - 1].chunk_index
            if i < len(results) - 1:
                rc.next_chunk_index = results[i + 1].chunk_index

        return results

    # ──────────────────────────────────────────────────────────────────────────
    # Core algorithm
    # ──────────────────────────────────────────────────────────────────────────

    def _process_text_run(
        self,
        blocks: list[ContentBlock],
        source: str,
        title_path: str,
        title_level: int,
        start_idx: int,
    ) -> list[RagChunk]:
        """Apply the full semantic chunking pipeline to a TEXT block run.

        Returns a list of :class:`RagChunk` (may be empty if no text).
        """
        # 1. Collect all sentences from all TEXT blocks in the run
        sentences: list[_Sentence] = []
        for block in blocks:
            texts = _split_text(
                block.text,
                self.delim,
                self.include_delim,
                self.min_characters_per_sentence,
            )
            for t in texts:
                sentences.append(
                    _Sentence(
                        text=t,
                        token_count=len(t) // 4,  # heuristic, consistent with HeuristicTokenizer
                        block=block,
                    )
                )

        if not sentences:
            return []

        if len(sentences) == 1:
            # Trivial case: single sentence → one chunk
            group = _SentenceGroup(sentences=sentences)
            rc = _build_chunk_from_group(
                group, source, title_path, title_level, start_idx
            )
            return [rc]

        # 2. Embed all sentences in a single batch call
        texts_to_embed = [s.text for s in sentences]
        logger.debug("Embedding {} sentences for semantic chunking", len(sentences))
        vecs_raw = self._embedder.embed_batch(texts_to_embed)
        vecs = np.array(vecs_raw, dtype=np.float64)  # (N, D)

        # 3. Compute per-pair similarities
        if self.skip_window > 0:
            sims = _skip_window_sims(vecs, self.skip_window)
        else:
            sims = _cosine_matrix(vecs)  # (N-1,)

        # 4 & 5. Smooth and detect breakpoints
        try:
            import chonkie_core
            # We still need a threshold depending on the mode
            if self.breakpoint_threshold_type == "percentile":
                # We calculate percentile on the smoothed signal for parity
                if self.sg_window_length > 1 and len(sims) >= 2:
                    smoothed_for_thresh = _smooth(sims, self.sg_window_length, self.sg_polyorder)
                else:
                    smoothed_for_thresh = sims.copy()
                threshold = float(np.percentile(smoothed_for_thresh, self.breakpoint_threshold_value))
            else:
                threshold = self.breakpoint_threshold_value

            if self.sg_window_length > 1 and len(sims) >= self.sg_window_length:
                sims_f64 = np.asarray(sims, dtype=np.float64)
                minima_indices, minima_values = chonkie_core.find_local_minima_interpolated(
                    sims_f64,
                    window_size=self.sg_window_length,
                    poly_order=self.sg_polyorder,
                    tolerance=0.1,  # Default heuristic for peak detection
                )
                
                if len(minima_indices) > 0:
                    filtered_indices, _ = chonkie_core.filter_split_indices(
                        minima_indices,
                        minima_values,
                        threshold,
                        1,  # min_sentences_per_chunk equivalent
                    )
                    breakpoints = {int(i) for i in filtered_indices}
                else:
                    breakpoints = set()
            else:
                breakpoints = {int(i) for i in np.where(sims < threshold)[0]}

        except ImportError:
            # 4. Smooth with Savitzky-Golay (or moving average) fallback
            if self.sg_window_length > 1 and len(sims) >= 2:
                smoothed = _smooth(sims, self.sg_window_length, self.sg_polyorder)
            else:
                smoothed = sims.copy()
            # 5. Detect breakpoints fallback
            breakpoints = self._get_breakpoints(smoothed)

        # 6. Initial segmentation
        groups = self._segment(sentences, breakpoints)

        # 7. Skip-and-merge (optional)
        if self.skip_window > 0 and len(groups) > 1:
            groups = self._skip_and_merge(groups, vecs, sentences)

        # 8. Build RagChunks
        chunks: list[RagChunk] = []
        for group in groups:
            rc = _build_chunk_from_group(
                group,
                source,
                title_path,
                title_level,
                start_idx + len(chunks),
            )
            chunks.append(rc)

        return chunks

    # ──────────────────────────────────────────────────────────────────────────
    # Breakpoint detection
    # ──────────────────────────────────────────────────────────────────────────

    def _get_breakpoints(self, smoothed: np.ndarray) -> set[int]:
        """Return the set of inter-sentence indices that are breakpoints.

        Index *i* in the returned set means "split between sentence i and i+1".

        Args:
            smoothed: (N-1,) smoothed similarity array.

        Returns:
            Set of break indices (0-based, in ``[0, N-2]``).
        """
        if self.breakpoint_threshold_type == "percentile":
            threshold = float(np.percentile(smoothed, self.breakpoint_threshold_value))
        else:
            threshold = self.breakpoint_threshold_value

        return {int(i) for i in np.where(smoothed < threshold)[0]}

    # ──────────────────────────────────────────────────────────────────────────
    # Segmentation
    # ──────────────────────────────────────────────────────────────────────────

    @staticmethod
    def _segment(
        sentences: list[_Sentence],
        breakpoints: set[int],
    ) -> list[_SentenceGroup]:
        """Partition sentences into groups at each breakpoint.

        Args:
            sentences:   Full ordered list of sentences.
            breakpoints: Set of indices *i* where a split occurs between
                         sentence[i] and sentence[i+1].

        Returns:
            Ordered list of non-empty :class:`_SentenceGroup`.
        """
        groups: list[_SentenceGroup] = []
        current: list[_Sentence] = []
        for i, sent in enumerate(sentences):
            current.append(sent)
            if i in breakpoints and i < len(sentences) - 1:
                groups.append(_SentenceGroup(sentences=current))
                current = []
        if current:
            groups.append(_SentenceGroup(sentences=current))
        return groups

    # ──────────────────────────────────────────────────────────────────────────
    # Skip-and-merge
    # ──────────────────────────────────────────────────────────────────────────

    def _skip_and_merge(
        self,
        groups: list[_SentenceGroup],
        vecs: np.ndarray,
        sentences: list[_Sentence],
    ) -> list[_SentenceGroup]:
        """Re-merge semantically close non-adjacent groups.

        For each pair of adjacent groups (A, B), compute the cosine similarity
        between their centroid embeddings.  If the similarity exceeds the
        (possibly smoothed) breakpoint threshold, merge them.  The process is
        repeated until no merges occur.

        Args:
            groups:    Initial groups after segmentation.
            vecs:      (N, D) sentence embedding matrix (same order as sentences).
            sentences: Full ordered sentence list (used to map groups → vecs).

        Returns:
            Possibly-merged groups.
        """
        # Build a mapping sentence → index in sentences list
        sent_to_idx: dict[int, int] = {id(s): i for i, s in enumerate(sentences)}

        def centroid(group: _SentenceGroup) -> np.ndarray:
            indices = [sent_to_idx[id(s)] for s in group.sentences]
            return vecs[indices].mean(axis=0)

        if self.breakpoint_threshold_type == "percentile":
            # We don't have the full smoothed array here; use a reasonable heuristic:
            # fall back to a fixed threshold = (1 - breakpoint_threshold_value/100)
            # so that 25th percentile → threshold ~0.75 for merging
            merge_threshold = 1.0 - self.breakpoint_threshold_value / 100.0
        else:
            merge_threshold = self.breakpoint_threshold_value

        changed = True
        while changed:
            changed = False
            merged: list[_SentenceGroup] = []
            i = 0
            while i < len(groups):
                if i + 1 < len(groups):
                    ca = centroid(groups[i])
                    cb = centroid(groups[i + 1])
                    na = np.linalg.norm(ca)
                    nb = np.linalg.norm(cb)
                    if na == 0 or nb == 0:
                        sim = 0.0
                    else:
                        sim = float(np.dot(ca / na, cb / nb))
                    if sim >= merge_threshold:
                        merged.append(
                            _SentenceGroup(
                                sentences=groups[i].sentences + groups[i + 1].sentences
                            )
                        )
                        i += 2
                        changed = True
                        continue
                merged.append(groups[i])
                i += 1
            groups = merged

        return groups

    # ──────────────────────────────────────────────────────────────────────────
    # Repr
    # ──────────────────────────────────────────────────────────────────────────

    def __repr__(self) -> str:
        return (
            f"SemanticChunker("
            f"model={self._embedder!r}, "
            f"breakpoint_threshold_type={self.breakpoint_threshold_type!r}, "
            f"breakpoint_threshold_value={self.breakpoint_threshold_value}, "
            f"skip_window={self.skip_window}, "
            f"sg_window_length={self.sg_window_length})"
        )
