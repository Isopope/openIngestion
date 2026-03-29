"""Slumber Chunker — LLM-guided agentic chunking for openingestion.

Design inspired by ``chonkie.SlumberChunker`` (itself inspired by LumberChunker),
adapted for the openingestion CHOMP pipeline (``ContentBlock`` -> ``RagChunk``).

Strategy
--------
For each accumulated text buffer (between TITLE / standalone blocks):

1. **Pre-split** the text into small *candidate* pieces using the same
   recursive hierarchy as :class:`~openingestion.chunker.by_recursive.RecursiveChunker`.
   Each candidate is at most ``candidate_size`` tokens.

2. **Ask the Genie** (LLM) to identify the first candidate ID where the topic
   clearly shifts from the preceding candidates.  A sliding context window of
   ``chunk_size`` tokens is sent per call.

3. **Group** the candidates between two consecutive split points into a single
   :class:`~openingestion.document.RagChunk` with spatial metadata from all
   contributing ContentBlocks.

TITLE, TABLE, IMAGE and EQUATION blocks always flush the current buffer and
are emitted individually as hard boundaries.
"""
from __future__ import annotations

import re
from bisect import bisect_left
from dataclasses import dataclass
from itertools import accumulate
from typing import Literal, Optional, Union

from loguru import logger

from openingestion.chunker.base import BaseChunker, _page_content_for
from openingestion.chunker.by_recursive import (
    RecursiveLevel,
    RecursiveRules,
    _split_by_delimiters,
    _split_by_whitespace,
)
from openingestion.document import BlockKind, ContentBlock, RagChunk
from openingestion.genie.base import BaseGenie
from openingestion.utils.tokenizer import AutoTokenizer, BaseTokenizer

try:
    from tqdm import tqdm
    _TQDM_AVAILABLE = True
except ImportError:
    _TQDM_AVAILABLE = False

_STANDALONE = frozenset({BlockKind.TABLE, BlockKind.IMAGE, BlockKind.EQUATION})

# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------

_JSON_PROMPT = """\
<task>
You are given a set of text passages between <passages> and </passages>.
Each passage is labeled ID N. Find the FIRST passage where the topic / semantics
clearly shifts from the previous passages.
</task>

<rules>
- Return a JSON object with a single key "split_index" whose value is the ID of
  the first passage that starts a new topic.
- If no clear shift exists, return N+1 where N is the last passage ID.
- Aim for balanced group sizes; avoid very long single groups.
</rules>

<passages>
{passages}
</passages>
"""

_TEXT_PROMPT = """\
<context>
You are chunking text for a RAG system. Good chunks should be topically coherent.
</context>

<task>
Find the first passage ID where the topic shifts enough to start a new chunk.
</task>

<passages>
{passages}
</passages>

<format>
Return ONLY the integer ID number, or N+1 (where N is the last passage ID) if all
passages belong together. No explanation.
</format>

<answer>"""


# ---------------------------------------------------------------------------
# Internal types
# ---------------------------------------------------------------------------

@dataclass
class _Span:
    """Intermediate candidate piece with character offsets in the aggregate text."""
    text: str
    start: int
    end: int
    token_count: int


@dataclass
class _TextSpanRef:
    """Text span with a reference back to the originating ContentBlock."""
    text: str
    start: int
    end: int
    block: ContentBlock


# ---------------------------------------------------------------------------
# SlumberChunker
# ---------------------------------------------------------------------------

class SlumberChunker(BaseChunker):
    """Chunk ContentBlocks using an LLM to detect semantic topic boundaries.

    Args:
        genie: A :class:`~openingestion.genie.BaseGenie` instance.
               Defaults to :class:`~openingestion.genie.OpenAIGenie` if ``None``.
        chunk_size: Maximum tokens per final chunk and LLM context window per call.
                    Default 512.
        rules: Recursive splitting rules for the pre-split step.
        candidate_size: Token budget for each candidate piece (default 128).
        min_characters_per_chunk: Pieces shorter than this are discarded (default 24).
        tokenizer: Tokenizer name or instance. Defaults to ``"heuristic"``.
        extract_mode: ``"json"`` uses structured output; ``"text"`` parses a plain
                      integer; ``"auto"`` (default) detects from the genie.
        max_retries: Maximum LLM call retries on parse failure (default 3).
        verbose: Show a tqdm progress bar during chunking (default ``True``).
        include_discarded: Keep ``DISCARDED`` blocks (default ``False``).

    Example::

        from openingestion.chunker.by_slumber import SlumberChunker
        from openingestion.genie import OpenAIGenie

        genie = OpenAIGenie(model="gpt-4o-mini")
        chunker = SlumberChunker(genie=genie, chunk_size=512)
        chunks = chunker(blocks, source="/path/to/doc.pdf")
    """

    def __init__(
        self,
        genie: Optional[BaseGenie] = None,
        chunk_size: int = 512,
        rules: Optional[RecursiveRules] = None,
        candidate_size: int = 128,
        min_characters_per_chunk: int = 24,
        tokenizer: Union[str, BaseTokenizer, None] = None,
        extract_mode: Literal["text", "json", "auto"] = "auto",
        max_retries: int = 3,
        verbose: bool = True,
        include_discarded: bool = False,
    ) -> None:
        if chunk_size <= 0:
            raise ValueError("chunk_size must be > 0")
        if candidate_size <= 0:
            raise ValueError("candidate_size must be > 0")
        if candidate_size > chunk_size:
            raise ValueError("candidate_size must be <= chunk_size")

        if genie is None:
            from openingestion.genie import OpenAIGenie
            genie = OpenAIGenie()

        self.genie = genie
        self.chunk_size = chunk_size
        self.candidate_size = candidate_size
        self.rules: RecursiveRules = rules if rules is not None else RecursiveRules()
        self.min_characters_per_chunk = min_characters_per_chunk
        self._tok: BaseTokenizer = AutoTokenizer(tokenizer if tokenizer is not None else "heuristic")
        self.max_retries = max_retries
        self.verbose = verbose
        self.include_discarded = include_discarded

        self.extract_mode: Literal["text", "json"] = self._determine_extract_mode(extract_mode)

        self._Split = None
        if self.extract_mode == "json":
            try:
                from pydantic import BaseModel

                class _SplitSchema(BaseModel):
                    split_index: int

                self._Split = _SplitSchema
            except ImportError as exc:
                raise ImportError(
                    "extract_mode='json' requires pydantic. Run: pip install pydantic"
                ) from exc

        self._template = _JSON_PROMPT if self.extract_mode == "json" else _TEXT_PROMPT

    # -----------------------------------------------------------------------
    # Extract-mode resolution
    # -----------------------------------------------------------------------

    def _determine_extract_mode(
        self, mode: Literal["text", "json", "auto"]
    ) -> Literal["text", "json"]:
        if mode == "json":
            return "json"
        if mode == "text":
            return "text"
        try:
            if type(self.genie).generate_json is not BaseGenie.generate_json:
                return "json"
        except Exception:
            pass
        return "text"

    # -----------------------------------------------------------------------
    # Index extraction
    # -----------------------------------------------------------------------

    def _extract_index_from_text(self, response: str) -> int:
        cleaned = response.strip()
        try:
            return int(cleaned)
        except ValueError:
            pass
        match = re.search(r"(\d+)", cleaned)
        if match:
            return int(match.group(1))
        raise ValueError(f"Could not extract integer from LLM response: {response!r}")

    def _get_split_index(self, prompt: str, group_end_index: int) -> int:
        last_exc: Exception = Exception("no attempts")
        for attempt in range(self.max_retries):
            try:
                if self.extract_mode == "json":
                    raw = self.genie.generate_json(prompt, self._Split)
                    idx = int(raw["split_index"])
                else:
                    raw = self.genie.generate(prompt)
                    idx = self._extract_index_from_text(raw)
                if idx > group_end_index:
                    raise ValueError(
                        f"split_index={idx} exceeds group_end={group_end_index}"
                    )
                return idx
            except (KeyboardInterrupt, SystemExit):
                raise
            except Exception as exc:
                last_exc = exc
                logger.debug("SlumberChunker: attempt {}/{} failed — {}", attempt + 1, self.max_retries, exc)

        logger.warning("SlumberChunker: all retries failed ({}), keeping group together.", last_exc)
        return group_end_index

    # -----------------------------------------------------------------------
    # Recursive pre-split  (produces _Span candidates)
    # -----------------------------------------------------------------------

    def _count(self, text: str) -> int:
        return self._tok.count_tokens(text)

    def _split_at_level(self, text: str, level: RecursiveLevel) -> list[str]:
        if level.delimiters is not None:
            return _split_by_delimiters(
                text, level.delimiters, level.include_delim, self.min_characters_per_chunk
            )
        return _split_by_whitespace(text, level.include_delim, self.min_characters_per_chunk)

    def _recursive_split(self, text: str, depth: int = 0, offset: int = 0) -> list[_Span]:
        if not text.strip():
            return []
        tc = self._count(text)
        if tc <= self.candidate_size:
            return [_Span(text=text, start=offset, end=offset + len(text), token_count=tc)]
        if depth >= len(self.rules):
            return [_Span(text=text, start=offset, end=offset + len(text), token_count=tc)]

        pieces = self._split_at_level(text, self.rules[depth])
        if len(pieces) <= 1 and (not pieces or pieces[0] == text.strip()):
            return self._recursive_split(text, depth + 1, offset)

        spans: list[_Span] = []
        current_offset = offset
        for piece in pieces:
            if self._count(piece) > self.candidate_size:
                spans.extend(self._recursive_split(piece, depth + 1, current_offset))
            else:
                spans.append(_Span(
                    text=piece,
                    start=current_offset,
                    end=current_offset + len(piece),
                    token_count=self._count(piece),
                ))
            current_offset += len(piece)
        return spans

    # -----------------------------------------------------------------------
    # Genie-driven grouping
    # -----------------------------------------------------------------------

    def _genie_chunk(self, spans: list[_Span], agg_text: str) -> list[str]:
        if not spans:
            return []

        prepared = [f"ID {i}: " + s.text.replace("\n", " ").strip() for i, s in enumerate(spans)]
        cumulative = list(accumulate([0] + [s.token_count for s in spans]))

        pbar = None
        if self.verbose and _TQDM_AVAILABLE:
            pbar = tqdm(total=len(spans), desc="SlumberChunker", unit="span", leave=False)

        result: list[str] = []
        current_pos = 0
        current_token_acc = 0

        while current_pos < len(spans):
            group_end = min(
                bisect_left(cumulative, current_token_acc + self.chunk_size) - 1,
                len(spans),
            )
            if group_end <= current_pos:
                group_end = current_pos + 1

            prompt = self._template.format(
                passages="\n".join(prepared[current_pos:group_end]),
            )
            split_at = self._get_split_index(prompt, group_end)
            if split_at <= current_pos:
                split_at = current_pos + 1

            result.append(agg_text[spans[current_pos].start:spans[split_at - 1].end])
            current_token_acc = cumulative[split_at]
            current_pos = split_at

            if pbar is not None:
                pbar.update(current_pos - pbar.n)

        if pbar is not None:
            pbar.close()

        return result

    # -----------------------------------------------------------------------
    # Spatial metadata helper
    # -----------------------------------------------------------------------

    def _blocks_for_chunk(
        self,
        chunk_text: str,
        block_spans: list[_TextSpanRef],
        agg_text: str,
    ) -> list[ContentBlock]:
        pos = agg_text.find(chunk_text)
        if pos == -1:
            return [s.block for s in block_spans]
        chunk_end = pos + len(chunk_text)
        contributing: list[ContentBlock] = []
        seen: set[int] = set()
        for span in block_spans:
            if span.end > pos and span.start < chunk_end:
                bid = id(span.block)
                if bid not in seen:
                    contributing.append(span.block)
                    seen.add(bid)
        return contributing or [block_spans[0].block]

    # -----------------------------------------------------------------------
    # Buffer flush
    # -----------------------------------------------------------------------

    def _flush_buffer(
        self,
        block_spans: list[_TextSpanRef],
        source: str,
        title_path: str,
        title_level: int,
        chunks: list[RagChunk],
    ) -> None:
        if not block_spans:
            return
        agg_text = "".join(s.text for s in block_spans)
        if not agg_text.strip():
            return

        candidates = self._recursive_split(agg_text, depth=0, offset=0)
        if not candidates:
            return

        logger.debug(
            "SlumberChunker: {} blocks → {} candidates → calling Genie",
            len(block_spans), len(candidates),
        )

        text_pieces = self._genie_chunk(candidates, agg_text)

        for piece in text_pieces:
            if not piece.strip():
                continue
            contributing = self._blocks_for_chunk(piece, block_spans, agg_text)
            chunks.append(RagChunk(
                page_content=piece,
                source=source,
                kind=BlockKind.TEXT,
                title_path=title_path,
                title_level=title_level,
                position_int=[[b.page_idx, *b.bbox] for b in contributing],
                block_indices=[b.block_index for b in contributing],
                reading_order=contributing[0].reading_order,
                chunk_index=len(chunks),
                extras={},
            ))

    # -----------------------------------------------------------------------
    # Public API
    # -----------------------------------------------------------------------

    def chunk(self, blocks: list[ContentBlock], source: str) -> list[RagChunk]:
        """Chunk ContentBlocks using LLM-guided semantic boundary detection.

        Args:
            blocks: ContentBlocks in reading order (Chef output).
            source: Absolute path to the source document.

        Returns:
            List of RagChunks with prev/next links populated.
        """
        chunks: list[RagChunk] = []
        title_path = ""
        title_level = 0
        buf: list[_TextSpanRef] = []
        offset = 0

        for block in blocks:
            if block.kind is BlockKind.DISCARDED:
                if not self.include_discarded:
                    continue

            if block.kind is BlockKind.TITLE:
                self._flush_buffer(buf, source, title_path, title_level, chunks)
                buf = []
                offset = 0
                title_path = block.text.strip()
                title_level = block.title_level or 1
                chunks.append(RagChunk(
                    page_content=block.text,
                    source=source,
                    kind=BlockKind.TITLE,
                    title_path=title_path,
                    title_level=title_level,
                    position_int=[[block.page_idx, *block.bbox]],
                    block_indices=[block.block_index],
                    reading_order=block.reading_order,
                    chunk_index=len(chunks),
                    extras={},
                ))
                continue

            if block.kind in _STANDALONE:
                self._flush_buffer(buf, source, title_path, title_level, chunks)
                buf = []
                offset = 0
                extras: dict = {}
                if block.html:
                    extras["html"] = block.html
                if block.img_path:
                    extras["img_path"] = block.img_path
                if block.captions:
                    extras["captions"] = list(block.captions)
                if block.footnotes:
                    extras["footnotes"] = list(block.footnotes)
                chunks.append(RagChunk(
                    page_content=_page_content_for(block),
                    source=source,
                    kind=block.kind,
                    title_path=title_path,
                    title_level=title_level,
                    position_int=[[block.page_idx, *block.bbox]],
                    block_indices=[block.block_index],
                    reading_order=block.reading_order,
                    chunk_index=len(chunks),
                    extras=extras,
                ))
                continue

            text = block.text
            if not text.strip():
                continue
            buf.append(_TextSpanRef(text=text, start=offset, end=offset + len(text), block=block))
            offset += len(text)

        self._flush_buffer(buf, source, title_path, title_level, chunks)

        for i, c in enumerate(chunks):
            object.__setattr__(c, "prev_chunk_index", i - 1 if i > 0 else None)
            object.__setattr__(c, "next_chunk_index", i + 1 if i < len(chunks) - 1 else None)

        logger.debug("SlumberChunker: {} blocks -> {} chunks", len(blocks), len(chunks))
        return chunks

    def __repr__(self) -> str:
        return (
            f"SlumberChunker(genie={self.genie!r}, "
            f"chunk_size={self.chunk_size}, "
            f"candidate_size={self.candidate_size}, "
            f"extract_mode={self.extract_mode!r}, "
            f"max_retries={self.max_retries})"
        )
