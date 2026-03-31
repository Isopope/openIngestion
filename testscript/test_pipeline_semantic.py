"""Test pipeline using SemanticChunker and ContextualRagRefinery on main-4.pdf."""
from __future__ import annotations

import argparse
import sys
import time
from collections import Counter
from pathlib import Path

from loguru import logger
from openingestion.genie import OpenAIGenie

# -- config -------------------------------------------------------------------
PDF_PATH   = Path(__file__).parent / "main-4LE.pdf"
OUTPUT_DIR = Path(__file__).parent / "output"
IMAGE_MODE = "path"
# -----------------------------------------------------------------------------

def _find_mineru_output(pdf: Path, output_root: Path):
    for candidate in (
        output_root / pdf.name / "auto",
        output_root / pdf.stem / "auto",
    ):
        if candidate.is_dir() and list(candidate.glob("*_content_list.json")):
            return candidate
    return None

def _section(title: str) -> None:
    print(f"\n{'─' * 60}\n  {title}\n{'─' * 60}")

def run(force_reparse: bool = False, skip_contextual: bool = False) -> None:
    from openingestion.chef.mineru_chef import MinerUChef
    from openingestion.chunker import SemanticChunker
    from openingestion.refinery import ContextualRagRefinery, RagRefinery
    from openingestion.porter import JSONPorter

    if not PDF_PATH.exists():
        logger.error("PDF introuvable : {}", PDF_PATH)
        sys.exit(1)

    # -- Etape 1 : Chef (MinerU) -----------------------------------------------
    _section("ETAPE 1 — MinerU Chef")
    existing = _find_mineru_output(PDF_PATH, OUTPUT_DIR)

    t_chef = time.perf_counter()
    chef = MinerUChef(output_dir=OUTPUT_DIR)

    if existing and not force_reparse:
        logger.info("Output MinerU existant — re-lecture : {}", existing)
        blocks = chef.process(existing)
    else:
        logger.info("Parsing MinerU complet de {}", PDF_PATH)
        blocks = chef.process(PDF_PATH)

    elapsed_chef = time.perf_counter() - t_chef
    print(f"  Blocs extraits : {len(blocks)}  ({elapsed_chef:.1f}s)")
    for k, n in sorted(Counter(b.kind.value for b in blocks).items()):
        print(f"    {k:<12} {n:>4}")

    # -- Etape 2 : SemanticChunker ---------------------------------------------
    _section("ETAPE 2 — SemanticChunker")
    t_chunk = time.perf_counter()

    # Uses the default 'sentence-transformers/all-MiniLM-L6-v2'
    chunker = SemanticChunker()
    chunks = chunker(blocks, source=str(PDF_PATH))

    elapsed_chunk = time.perf_counter() - t_chunk
    print(f"  Total chunks : {len(chunks)}  ({elapsed_chunk:.2f}s)")
    
    indices = [c.chunk_index for c in chunks]
    assert len(indices) == len(set(indices)), "ERREUR : chunk_index non uniques !"
    print("  chunk_index uniques : OK")

    # -- Etape 3 : RagRefinery -------------------------------------------------
    _section("ETAPE 3 — RagRefinery (token_count + hash + images)")
    t_ref = time.perf_counter()

    rag_img_dir = existing or (OUTPUT_DIR / PDF_PATH.name / "auto")
    rag_refinery = RagRefinery(
        output_dir=rag_img_dir,
        image_mode=IMAGE_MODE,
        infer_captions=True,
    )
    chunks = rag_refinery.enrich(chunks)

    elapsed_ref = time.perf_counter() - t_ref
    tok_normal = [c.token_count for c in chunks]
    print(f"  Enrichissement termine ({elapsed_ref:.2f}s)")
    if tok_normal:
        print(f"  Token count min={min(tok_normal)} max={max(tok_normal)} moy={sum(tok_normal)/len(tok_normal):.0f}")
    assert all(c.content_hash for c in chunks if c.page_content.strip()), "ERREUR : content_hash vide sur chunk non-vide"
    print("  Assertions RagRefinery : OK")

    # -- Etape 4 : ContextualRagRefinery ---------------------------------------
    if not skip_contextual:
        _section("ETAPE 4 — ContextualRagRefinery (doc_summary + chunk_context)")
        print("  Genie : OpenAIGenie")
        t_ctx = time.perf_counter()

        genie = OpenAIGenie(model="gpt-4o-mini")

        ctx_refinery = ContextualRagRefinery(
            genie=genie,
            generate_doc_summary=True,
            generate_chunk_context=True,
            max_workers=4,
            summary_max_tokens=150,
            context_max_tokens=100,
        )
        ctx_refinery.enrich(chunks)

        elapsed_ctx = time.perf_counter() - t_ctx
        with_summary = sum(1 for c in chunks if c.doc_summary)
        with_context = sum(1 for c in chunks if c.chunk_context)
        print(f"  Appels OpenAIGenie : OK  ({elapsed_ctx:.2f}s)")
        print(f"  doc_summary renseigne   : {with_summary}/{len(chunks)} chunks")
        print(f"  chunk_context renseigne : {with_context}/{len(chunks)} chunks")
        assert with_summary > 0, "ERREUR : aucun doc_summary genere"
        assert with_context > 0, "ERREUR : aucun chunk_context genere"
        print("  Assertions ContextualRagRefinery : OK")

    # -- Etape 5 : Export JSONL ------------------------------------------------
    _section("ETAPE 5 — Export JSONL")
    out_jsonl = Path(__file__).parent / "output_chunks_semantic.jsonl"
    JSONPorter(lines=True)(chunks, file=out_jsonl)
    print(f"  Exporte ({len(chunks)} chunks) -> {out_jsonl}")

    # -- Resume ----------------------------------------------------------------
    total = time.perf_counter() - t_chef
    _section("RESUME")
    print(f"  Blocs MinerU    : {len(blocks)}")
    print(f"  Chunks total    : {len(chunks)}")
    print(f"  Temps total     : {total:.1f}s")
    print(f"\n  PIPELINE OK")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Test pipeline avec Semantic Chunking & Contextual RAG")
    ap.add_argument("--reparse", action="store_true", help="Force re-run MinerU meme si l'output existe")
    ap.add_argument("--no-contextual", action="store_true", help="Sauter l'etape ContextualRagRefinery")
    args = ap.parse_args()
    
    logger.remove()
    logger.add(sys.stderr, level="INFO")
    run(force_reparse=args.reparse, skip_contextual=args.no_contextual)
