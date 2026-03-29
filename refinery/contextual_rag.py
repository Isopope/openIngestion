"""ContextualRagRefinery — enriches RagChunks with LLM-generated context.

Implements the *Contextual RAG* technique described by Anthropic:

* ``doc_summary``   : ~100-token LLM summary of the whole document, shared by
                      every chunk of the same source document.
* ``chunk_context`` : ~50-token LLM description situating a specific chunk
                      within the document, unique per chunk.

Prompt caching
--------------
For each document all chunk-context calls share the same *prefix* (the document
text).  Providers that support prompt caching reuse that prefix at a reduced
token cost:

* **Anthropic** — override :meth:`~openingestion.genie.base.BaseGenie.generate_contextual`
  to send the document prefixed with ``cache_control: {"type": "ephemeral"}``.
* **OpenAI** — prefix caching is automatic; no override required.
* **Gemini** — use the ``cachedContent`` API in your :class:`BaseGenie` subclass.

The default :meth:`~openingestion.genie.base.BaseGenie.generate_contextual`
simply concatenates context + prompt, so every provider works out of the box.

Parallelism
-----------
Chunk-context calls are dispatched to a :class:`~concurrent.futures.ThreadPoolExecutor`
so N chunks of a document are processed concurrently.  Control the degree of
parallelism with ``max_workers`` (default: 4).

Error handling
--------------
By default (``on_error="skip"``) a failed LLM call leaves the field empty and
logs a warning — the pipeline continues.  Set ``on_error="raise"`` to propagate
the exception instead.

Usage::

    from openingestion.genie.openai_genie import OpenAIGenie
    from openingestion.refinery.contextual_rag import ContextualRagRefinery

    genie = OpenAIGenie(api_key="sk-n...", model="gpt-4o-mini")
    refinery = ContextualRagRefinery(genie=genie, max_workers=8)
    enriched = refinery.enrich(chunks)
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from typing import Callable, Literal

from loguru import logger

from openingestion.document import RagChunk
from openingestion.refinery.base import BaseRefinery


# ── Utility ───────────────────────────────────────────────────────────────────

def _trim_middle(text: str, max_chars: int, head_ratio: float = 0.4) -> str:
    """Trim *text* to *max_chars* keeping the head and tail.

    Unlike a simple ``text[:max_chars]`` tail-cut this preserves both the
    beginning (rich in title / introduction context) and the end (conclusion,
    bibliography) of the document.

    Args:
        text:       Full document text.
        max_chars:  Maximum length of the returned string.
        head_ratio: Fraction of *max_chars* kept from the head.  The tail
                    receives ``1 - head_ratio``.  Default: ``0.4`` (40 / 60).

    Returns:
        Original text when shorter than *max_chars*, otherwise trimmed text
        with a ``[…]`` marker at the cut point.
    """
    if len(text) <= max_chars:
        return text
    head = int(max_chars * head_ratio)
    tail = max_chars - head - 5  # 5 chars for the " […] " marker
    trimmed = text[:head] + " […] " + text[len(text) - tail :]
    logger.debug(
        "ContextualRagRefinery: doc text trimmed head={} tail={} (original={} chars)",
        head, tail, len(text),
    )
    return trimmed


# ── Prompt templates ──────────────────────────────────────────────────────────

_DOC_SUMMARY_PROMPT = """\
Here is a document:
<document>
{doc_text}
</document>

Please give a short succinct summary of the entire document in 2-3 sentences. \
Answer ONLY with the summary, no preamble or conclusion.\
"""

# The document prefix is sent as the cacheable `context` arg of generate_contextual.
_CHUNK_CONTEXT_PREFIX = """\
Here is a document:
<document>
{doc_text}
</document>

"""

# The chunk-specific suffix is sent as the `prompt` arg.
_CHUNK_CONTEXT_SUFFIX = """\
Here is a chunk from that document:
<chunk>
{chunk_text}
</chunk>

Please give a short succinct context (1-2 sentences) explaining what this chunk \
is about and how it fits in the overall document, for the purpose of improving \
search retrieval. Answer ONLY with the context, no preamble.\
"""


# ── ContextualRagRefinery ──────────────────────────────────────────────────────

class ContextualRagRefinery(BaseRefinery):
    """Enrich RagChunks with LLM-generated ``doc_summary`` and ``chunk_context``.

    Args:
        genie:
            Any :class:`~openingestion.genie.base.BaseGenie` instance *or* a
            plain ``callable(str) -> str`` for ad-hoc usage.
        max_doc_tokens:
            Document text is trimmed (middle-cut) to approximately
            *max_doc_tokens* tokens (``× 4`` chars) before LLM calls.
            Default: ``4000``.
        generate_doc_summary:
            If ``True`` (default), generate a ``doc_summary`` shared by all
            chunks of the same document.
        generate_chunk_context:
            If ``True`` (default), generate a ``chunk_context`` per chunk.
        full_doc_token_limit:
            When the document text fits within this many tokens it is sent
            verbatim as context for chunk-context calls; otherwise the
            pre-computed ``doc_summary`` is used instead (mirrors Onyx
            ``MAX_TOKENS_FOR_FULL_INCLUSION``).  Default: ``2000``.
        summary_max_tokens:
            ``max_tokens`` cap passed to the LLM for doc-summary calls.
            Default: ``150``.
        context_max_tokens:
            ``max_tokens`` cap passed to the LLM for chunk-context calls.
            Default: ``100``.
        max_workers:
            Parallel threads for chunk-context generation.  Default: ``4``.
        on_error:
            ``"skip"`` (default) — log a warning and leave the field empty.
            ``"raise"``          — propagate the exception.
    """

    def __init__(
        self,
        genie,  # BaseGenie | Callable[[str], str]
        *,
        max_doc_tokens: int = 4000,
        generate_doc_summary: bool = True,
        generate_chunk_context: bool = True,
        full_doc_token_limit: int = 2000,
        summary_max_tokens: int = 150,
        context_max_tokens: int = 100,
        max_workers: int = 4,
        on_error: Literal["skip", "raise"] = "skip",
    ) -> None:
        self._genie = genie
        self._max_chars = max_doc_tokens * 4
        self._full_doc_chars = full_doc_token_limit * 4
        self.generate_doc_summary = generate_doc_summary
        self.generate_chunk_context = generate_chunk_context
        self.summary_max_tokens = summary_max_tokens
        self.context_max_tokens = context_max_tokens
        self.max_workers = max_workers
        self.on_error = on_error

    # ── Public entry-point ────────────────────────────────────────────────────

    def enrich(self, chunks: list[RagChunk]) -> list[RagChunk]:
        """Enrich *chunks* in-place with ``doc_summary`` and ``chunk_context``.

        Chunks are grouped by ``source``; each group triggers at most one
        doc-summary LLM call and one chunk-context call per chunk.

        Args:
            chunks: RagChunks produced by a Chunker (or a prior Refinery stage).

        Returns:
            The same list, mutated in-place.
        """
        if not chunks:
            return chunks

        # Group by source document
        by_source: dict[str, list[RagChunk]] = {}
        for chunk in chunks:
            by_source.setdefault(chunk.source, []).append(chunk)

        for source, doc_chunks in by_source.items():
            logger.info(
                "ContextualRagRefinery: enriching {} chunk(s) from '{}'",
                len(doc_chunks),
                source,
            )
            self._enrich_document(doc_chunks)

        return chunks

    # ── Per-document enrichment ───────────────────────────────────────────────

    def _enrich_document(self, chunks: list[RagChunk]) -> None:
        """Run doc_summary + chunk_context for one document."""
        full_text = self._build_doc_text(chunks)
        # Trim for LLM context window (middle-cut preserves head + tail)
        doc_text = _trim_middle(full_text, self._max_chars)

        doc_summary: str = ""

        # 1. doc_summary — one LLM call, result injected into every chunk
        if self.generate_doc_summary:
            doc_summary = self._safe_call(
                lambda: self._call_doc_summary(doc_text),
                context=f"doc_summary for '{chunks[0].source}'",
            )
            if doc_summary:
                for chunk in chunks:
                    chunk.doc_summary = doc_summary
                logger.debug(
                    "ContextualRagRefinery: doc_summary set on {} chunk(s)",
                    len(chunks),
                )

        # 2. chunk_context — one call per chunk, parallelized.
        #    doc_info selection (mirrors Onyx MAX_TOKENS_FOR_FULL_INCLUSION):
        #      • short doc  → send the full text verbatim (best quality)
        #      • long doc   → reuse doc_summary if available
        #      • long + no summary → compute a fallback summary on-the-fly
        if self.generate_chunk_context:
            if len(full_text) <= self._full_doc_chars:
                doc_info = full_text
            elif doc_summary:
                doc_info = doc_summary
            else:
                # Document too long and doc_summary disabled/failed → fallback
                logger.debug(
                    "ContextualRagRefinery: doc too long and no summary — "
                    "computing fallback summary"
                )
                doc_info = self._safe_call(
                    lambda: self._call_doc_summary(doc_text),
                    context=f"fallback doc_info for '{chunks[0].source}'",
                ) or doc_text  # last resort: truncated raw text

            doc_prefix = _CHUNK_CONTEXT_PREFIX.format(doc_text=doc_info)
            self._fill_chunk_contexts(chunks, doc_prefix)

    def _fill_chunk_contexts(
        self, chunks: list[RagChunk], doc_prefix: str
    ) -> None:
        """Dispatch chunk-context calls concurrently."""
        non_empty = [c for c in chunks if c.page_content.strip()]
        if not non_empty:
            return

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_chunk = {
                executor.submit(
                    self._safe_call,
                    # default arg binding avoids late-binding closure bug
                    lambda c=chunk: self._call_chunk_context(doc_prefix, c),
                    context=f"chunk_context #{chunk.chunk_index}",
                ): chunk
                for chunk in non_empty
            }
            for future, chunk in future_to_chunk.items():
                result = future.result()
                if result:
                    chunk.chunk_context = result

        logger.debug(
            "ContextualRagRefinery: chunk_context set on {} / {} chunk(s)",
            sum(1 for c in non_empty if c.chunk_context),
            len(non_empty),
        )

    # ── LLM call helpers ──────────────────────────────────────────────────────

    def _call_doc_summary(self, doc_text: str) -> str:
        prompt = _DOC_SUMMARY_PROMPT.format(doc_text=doc_text)
        return self._generate(prompt, max_tokens=self.summary_max_tokens)

    def _call_chunk_context(self, doc_prefix: str, chunk: RagChunk) -> str:
        """Prefer generate_contextual (enables provider-side prompt caching)."""
        suffix = _CHUNK_CONTEXT_SUFFIX.format(chunk_text=chunk.page_content)
        if hasattr(self._genie, "generate_contextual"):
            import inspect
            sig = inspect.signature(self._genie.generate_contextual)
            accepts_max_tokens = "max_tokens" in sig.parameters
            accepts_kwargs = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values())
            
            if accepts_max_tokens or accepts_kwargs:
                return self._genie.generate_contextual(
                    doc_prefix, suffix, max_tokens=self.context_max_tokens
                )
            return self._genie.generate_contextual(doc_prefix, suffix)
        return self._generate(doc_prefix + suffix, max_tokens=self.context_max_tokens)

    def _generate(self, prompt: str, *, max_tokens: int | None = None) -> str:
        """Dispatch to genie.generate() or a plain callable.

        Passes *max_tokens* as a keyword argument when the genie's
        ``generate`` method accepts it (duck-typed check).
        """
        if hasattr(self._genie, "generate"):
            import inspect
            sig = inspect.signature(self._genie.generate)
            if "max_tokens" in sig.parameters and max_tokens is not None:
                return self._genie.generate(prompt, max_tokens=max_tokens)
            return self._genie.generate(prompt)
        if callable(self._genie):
            return self._genie(prompt)
        raise TypeError(
            f"genie must have a generate(str)->str method or be callable, "
            f"got {type(self._genie)}"
        )

    def _safe_call(
        self, fn: Callable[[], str], *, context: str = ""
    ) -> str:
        try:
            return fn()
        except Exception as exc:
            if self.on_error == "raise":
                raise
            logger.warning(
                "ContextualRagRefinery: LLM call failed ({}) — {}: {}",
                context,
                type(exc).__name__,
                exc,
            )
            return ""

    # ── Document text builder ─────────────────────────────────────────────────

    def _build_doc_text(self, chunks: list[RagChunk]) -> str:
        """Concatenate page_content of all chunks (no trimming — caller trims)."""
        parts = [c.page_content for c in chunks if c.page_content.strip()]
        return "\n\n".join(parts)

    def __repr__(self) -> str:
        return (
            f"ContextualRagRefinery("
            f"doc_summary={self.generate_doc_summary}, "
            f"chunk_context={self.generate_chunk_context}, "
            f"summary_max_tokens={self.summary_max_tokens}, "
            f"context_max_tokens={self.context_max_tokens}, "
            f"max_workers={self.max_workers})"
        )
