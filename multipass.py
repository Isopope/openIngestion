"""Multipass chunker — indexes the same document at N granularities via bottom-up aggregation.

Strategy
--------
A base chunker (usually fine-grained, e.g., `SentenceChunker` at 150 tokens) generates the "mini" chunks.
Subsequent granularities ("normal", "large") are built by concatenating the chunks of the previous level
until the target token limit is reached.

This "Bottom-Up Aggregation" strategy guarantees that chunks align perfectly on boundaries, 
eliminating the need for majority voting on block overlap, and significantly speeds up chunking 
since the source text is parsed and tokenized only once.

Standalone chunks (Images, Tables) are preserved perfectly across passes without being 
merged into text.

Usage::

    from openingestion.chunker.multipass import MultipassChunker

    # Default preset (mini=150, normal=512, large=2048)
    chunker = MultipassChunker()
    chunks = chunker(blocks, source="/path/to/doc.pdf")

    # Custom base chunker and levels
    from openingestion.chunker.by_sentence import SentenceChunker

    chunker = MultipassChunker(
        base_chunker=SentenceChunker(chunk_size=100, chunk_overlap=0),
        base_name="micro",
        pass_levels=[("medium", 400), ("massive", 3000)]
    )
"""
from __future__ import annotations

from loguru import logger
from typing import Optional

from openingestion.chunker.base import BaseChunker
from openingestion.chunker.by_sentence import SentenceChunker
from openingestion.document import ContentBlock, RagChunk, BlockKind
from openingestion.utils.tokenizer import AutoTokenizer, BaseTokenizer


# ── Default preset ─────────────────────────────────────────────────────────────
_DEFAULT_BASE_NAME = "mini"
_DEFAULT_BASE_SIZE = 150
_DEFAULT_PASS_LEVELS: list[tuple[str, int]] = [
    ("normal", 512),
    ("large", 2048),
]

_STANDALONE = frozenset({BlockKind.TABLE, BlockKind.IMAGE, BlockKind.EQUATION, BlockKind.TITLE})


# ─────────────────────────────────────────────────────────────────────────────
# MultipassChunker
# ─────────────────────────────────────────────────────────────────────────────

class MultipassChunker(BaseChunker):
    """Run a base chunking pass, then aggregate bottom-up for coarser granularities.

    Args:
        base_chunker:
            The chunker used for the finest granularity. When omitted, a 
            ``SentenceChunker`` with ``chunk_size=150`` and ``chunk_overlap=0`` is used.
        base_name:
            The granularity name assigned to the output of ``base_chunker``. Default "mini".
        pass_levels:
            List of ``(name, max_tokens)`` pairs for the aggregated passes.
            Executed in order.
        tokenizer:
            Tokenizer to measure token counts for aggregation. Defaults to "heuristic".
        include_parent_links:
            When ``True`` (default), each chunk receives
            ``extras["parent_chunk_index"]`` pointing to the immediately-larger
            enclosing chunk. Because of bottom-up aggregation, this is deterministic.
    """

    def __init__(
        self,
        base_chunker: BaseChunker | None = None,
        base_name: str = _DEFAULT_BASE_NAME,
        pass_levels: list[tuple[str, int]] | None = None,
        tokenizer: str | BaseTokenizer | None = None,
        *,
        include_parent_links: bool = True,
    ) -> None:
        if base_chunker is None:
            self.base_chunker = SentenceChunker(chunk_size=_DEFAULT_BASE_SIZE, chunk_overlap=0)
        else:
            self.base_chunker = base_chunker
            
        self.base_name = base_name
        self.pass_levels = pass_levels if pass_levels is not None else _DEFAULT_PASS_LEVELS
        self.include_parent_links = include_parent_links
        self._tok: BaseTokenizer = AutoTokenizer(tokenizer if tokenizer is not None else "heuristic")

    # ── BaseChunker contract ──────────────────────────────────────────────────

    def chunk(self, blocks: list[ContentBlock], source: str) -> list[RagChunk]:
        """Execute the base pass, then build subsequent passes via bottom-up aggregation.

        Args:
            blocks: ContentBlocks in reading order (Chef output).
            source: Absolute path to the source document.

        Returns:
            Flat list of RagChunks, ordered by granularity pass (fine → coarse).
        """
        all_chunks: list[RagChunk] = []
        pass_results: list[tuple[str, list[RagChunk]]] = []
        offset = 0

        # --- 1. Base pass ---
        raw_base = self.base_chunker.chunk(blocks, source)
        for c in raw_base:
            c.granularity = self.base_name
            c.chunk_index += offset
            if c.prev_chunk_index is not None:
                c.prev_chunk_index += offset
            if c.next_chunk_index is not None:
                c.next_chunk_index += offset
            # Calculate token count if it is 0 (SentenceChunker doesn't set it)
            if c.token_count == 0 and c.page_content:
                c.token_count = self._tok.count_tokens(c.page_content)

        pass_results.append((self.base_name, raw_base))
        all_chunks.extend(raw_base)
        
        logger.debug(
            "MultipassChunker: pass '{}' → {} chunks (offset {})",
            self.base_name, len(raw_base), offset,
        )
        offset += len(raw_base)

        # --- 2. Aggregation passes ---
        current_level_chunks = raw_base

        for level_name, max_tokens in self.pass_levels:
            new_level_chunks = self._aggregate_level(
                current_level_chunks, 
                level_name, 
                max_tokens, 
                offset
            )
            pass_results.append((level_name, new_level_chunks))
            all_chunks.extend(new_level_chunks)
            
            logger.debug(
                "MultipassChunker: pass '{}' → {} chunks (offset {})",
                level_name, len(new_level_chunks), offset,
            )
            offset += len(new_level_chunks)
            current_level_chunks = new_level_chunks

        logger.info(
            "MultipassChunker: {} pass(es), {} total chunks from '{}'",
            1 + len(self.pass_levels), len(all_chunks), source,
        )
        return all_chunks

    def _aggregate_level(
        self, 
        children: list[RagChunk], 
        level_name: str, 
        max_tokens: int,
        offset: int
    ) -> list[RagChunk]:
        """Group children RagChunks to form the next granularity level."""
        new_chunks: list[RagChunk] = []
        
        buf: list[RagChunk] = []
        buf_tokens = 0
        
        def _emit() -> None:
            nonlocal buf, buf_tokens
            if not buf:
                return
                
            new_idx = offset + len(new_chunks)
            
            # Combine content
            content = "\n\n".join(c.page_content for c in buf)
            
            # Combine positions and blocks
            pos_int = []
            block_indices = []
            for c in buf:
                pos_int.extend(c.position_int)
                block_indices.extend(c.block_indices)
                
            # Deduplicate block_indices while preserving order
            seen = set()
            dedup_block_indices = []
            for bi in block_indices:
                if bi not in seen:
                    seen.add(bi)
                    dedup_block_indices.append(bi)
                    
            # Combine extras
            merged_extras = {}
            captions = []
            footnotes = []
            for c in buf:
                if "html" in c.extras and "html" not in merged_extras:
                    merged_extras["html"] = c.extras["html"]
                if "img_path" in c.extras and "img_path" not in merged_extras:
                    merged_extras["img_path"] = c.extras["img_path"]
                if "captions" in c.extras:
                    captions.extend(c.extras["captions"])
                if "footnotes" in c.extras:
                    footnotes.extend(c.extras["footnotes"])
            
            if captions:
                merged_extras["captions"] = list(dict.fromkeys(captions))
            if footnotes:
                merged_extras["footnotes"] = list(dict.fromkeys(footnotes))
                
            # Create parent chunk
            parent_chunk = RagChunk(
                page_content=content,
                source=buf[0].source,
                kind=buf[0].kind if len(buf) == 1 else BlockKind.TEXT,
                title_path=buf[0].title_path,
                title_level=buf[0].title_level,
                position_int=pos_int,
                extras=merged_extras,
                chunk_index=new_idx,
                block_indices=dedup_block_indices,
                reading_order=buf[0].reading_order,
                prev_chunk_index=new_idx - 1 if new_chunks else None,
                next_chunk_index=None,
                token_count=self._tok.count_tokens(content),
                granularity=level_name
            )
            
            # Apply parent links to children deterministically
            if self.include_parent_links:
                for child in buf:
                    child.extras["parent_chunk_index"] = new_idx
            
            if new_chunks:
                new_chunks[-1].next_chunk_index = new_idx
                
            new_chunks.append(parent_chunk)
            buf = []
            buf_tokens = 0

        for child in children:
            if child.kind in _STANDALONE:
                _emit()
                # Standalone block creates its own chunk without being merged
                buf.append(child)
                buf_tokens += child.token_count
                _emit()
                continue
                
            # If adding this child exceeds max_tokens AND we already have something in the buffer
            if buf_tokens + child.token_count > max_tokens and buf:
                _emit()
                
            buf.append(child)
            buf_tokens += child.token_count
            
        _emit()
        return new_chunks

    # ── Helpers ───────────────────────────────────────────────────────────────

    def __repr__(self) -> str:
        names = [self.base_name] + [n for n, _ in self.pass_levels]
        return (
            f"MultipassChunker("
            f"levels={names}, "
            f"include_parent_links={self.include_parent_links})"
        )
