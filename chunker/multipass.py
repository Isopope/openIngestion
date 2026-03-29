"""Multipass chunker — indexes the same document at N granularities simultaneously.

Strategy
--------
Each *pass* applies an independent :class:`~openingestion.chunker.base.BaseChunker`
to the same ``ContentBlock`` list.  The resulting chunks are tagged with a
``granularity`` label (e.g. ``"mini"``, ``"normal"``, ``"large"``) and
concatenated into a single flat list with globally-unique ``chunk_index`` values.

Default preset
--------------
Three ``TokenChunker`` passes are created automatically when no *passes*
argument is supplied:

============  ===========  ==============
 Granularity  max_tokens    overlap_tokens
============  ===========  ==============
mini          150           16
normal        512           64
large         2048          128
============  ===========  ==============

Parent links (small-to-big retrieval)
--------------------------------------
When ``include_parent_links=True`` (default), each chunk receives
``extras["parent_chunk_index"]`` pointing to the smallest chunk of the
immediately-larger granularity pass whose block set *covers* it.

This enables the **parent-document-retriever** pattern: retrieve at fine
granularity, then expand context by fetching the parent chunk.

Usage::

    from openingestion.chunker.multipass import MultipassChunker

    # Default three-pass preset
    chunker = MultipassChunker()
    chunks = chunker(blocks, source="/path/to/doc.pdf")

    mini_chunks   = [c for c in chunks if c.granularity == "mini"]
    normal_chunks = [c for c in chunks if c.granularity == "normal"]
    large_chunks  = [c for c in chunks if c.granularity == "large"]

    # Custom passes (e.g. sentence + token)
    from openingestion.chunker.by_sentence import SentenceChunker
    from openingestion.chunker.by_token import TokenChunker

    chunker = MultipassChunker(passes=[
        ("sentence", SentenceChunker(max_tokens=200)),
        ("large",    TokenChunker(max_tokens=1024)),
    ])
"""
from __future__ import annotations

from loguru import logger

from openingestion.chunker.base import BaseChunker
from openingestion.chunker.by_token import TokenChunker
from openingestion.document import ContentBlock, RagChunk


# ── Default preset ─────────────────────────────────────────────────────────────
# (name, max_tokens, overlap_tokens)
_DEFAULT_PASSES: list[tuple[str, int, int]] = [
    ("mini",   150,  16),
    ("normal", 512,  64),
    ("large",  2048, 128),
]


# ─────────────────────────────────────────────────────────────────────────────
# MultipassChunker
# ─────────────────────────────────────────────────────────────────────────────

class MultipassChunker(BaseChunker):
    """Run N independent chunking passes and return a single merged list.

    Args:
        passes:
            List of ``(name, chunker)`` pairs executed in order from finest
            to coarsest granularity.  When omitted, the three default
            ``TokenChunker`` presets (mini / normal / large) are used.
        include_parent_links:
            When ``True`` (default), each chunk receives
            ``extras["parent_chunk_index"]`` pointing to the immediately-larger
            enclosing chunk.  Uses block-index overlap as the matching
            criterion; majority vote resolves ambiguities caused by overlap
            windows.
    """

    def __init__(
        self,
        passes: list[tuple[str, BaseChunker]] | None = None,
        *,
        include_parent_links: bool = True,
    ) -> None:
        if passes is None:
            passes = [
                (name, TokenChunker(max_tokens=mt, overlap_tokens=ot))
                for name, mt, ot in _DEFAULT_PASSES
            ]
        if len(passes) < 2:
            raise ValueError(
                "MultipassChunker needs at least 2 passes; "
                "use a regular chunker for single-pass."
            )
        self.passes = passes
        self.include_parent_links = include_parent_links

    # ── BaseChunker contract ──────────────────────────────────────────────────

    def chunk(self, blocks: list[ContentBlock], source: str) -> list[RagChunk]:  # noqa: C901
        """Execute all passes and return the merged, tagged chunk list.

        ``chunk_index`` values are offset per pass so they are unique across
        the whole output.  ``prev/next_chunk_index`` links are adjusted to
        match the offsets and remain valid within each granularity.

        Args:
            blocks: ContentBlocks in reading order (Chef output).
            source: Absolute path to the source document.

        Returns:
            Flat list of RagChunks, ordered by granularity pass (fine → coarse).
        """
        pass_results: list[tuple[str, list[RagChunk]]] = []
        offset = 0

        for pass_name, chunker in self.passes:
            raw = chunker.chunk(blocks, source)

            for c in raw:
                # Tag the granularity
                c.granularity = pass_name
                # Shift indices to make them globally unique
                c.chunk_index += offset
                if c.prev_chunk_index is not None:
                    c.prev_chunk_index += offset
                if c.next_chunk_index is not None:
                    c.next_chunk_index += offset

            pass_results.append((pass_name, raw))
            logger.debug(
                "MultipassChunker: pass '{}' → {} chunks (offset {})",
                pass_name, len(raw), offset,
            )
            offset += len(raw)

        if self.include_parent_links:
            self._attach_parent_links(pass_results)

        all_chunks: list[RagChunk] = []
        for _, chunks in pass_results:
            all_chunks.extend(chunks)

        logger.info(
            "MultipassChunker: {} pass(es), {} total chunks from '{}'",
            len(self.passes), len(all_chunks), source,
        )
        return all_chunks

    # ── Parent-link computation ───────────────────────────────────────────────

    @staticmethod
    def _attach_parent_links(
        pass_results: list[tuple[str, list[RagChunk]]],
    ) -> None:
        """Populate ``extras["parent_chunk_index"]`` on each chunk.

        For every pass *i*, each chunk is linked to the smallest enclosing
        chunk in pass *i+1* based on ``block_indices`` overlap (majority
        vote so that overlap windows don't cause spurious links to distant
        large chunks).
        """
        for i in range(len(pass_results) - 1):
            _, fine_chunks = pass_results[i]
            _, coarse_chunks = pass_results[i + 1]

            # block_index → chunk_index of the coarse chunk that owns it
            # (last writer wins for blocks shared by overlap windows)
            block_to_coarse: dict[int, int] = {}
            for cc in coarse_chunks:
                for bi in cc.block_indices:
                    block_to_coarse[bi] = cc.chunk_index

            for fc in fine_chunks:
                if not fc.block_indices:
                    continue
                # Majority vote: which coarse chunk covers the most of my blocks?
                votes: dict[int, int] = {}
                for bi in fc.block_indices:
                    ci = block_to_coarse.get(bi)
                    if ci is not None:
                        votes[ci] = votes.get(ci, 0) + 1
                if votes:
                    best = max(votes, key=lambda k: votes[k])
                    fc.extras["parent_chunk_index"] = best

    # ── Helpers ───────────────────────────────────────────────────────────────

    def __repr__(self) -> str:
        names = ", ".join(f"'{n}'" for n, _ in self.passes)
        return (
            f"MultipassChunker("
            f"passes=[{names}], "
            f"include_parent_links={self.include_parent_links})"
        )
