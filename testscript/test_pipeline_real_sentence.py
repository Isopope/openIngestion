"""Test pipeline using SentenceChunker and ContextualRagRefinery on main-4.pdf.

This version uses the real SentenceChunker and the real OpenAIGenie for context generation!
"""
from __future__ import annotations

import argparse
import sys
import time
from collections import Counter
from pathlib import Path

from loguru import logger

# ==============================================================================
# CONFIGURATION
# ==============================================================================
# Ajoutez votre cle API OpenAI ci-dessous pour activer le Contextual RAG
OPENAI_API_KEY = "VOTRE_CLE_API_ICI"  # <-- REMPLISSEZ ICI AVEC VOTRE CLE API OPENAI

PDF_PATH   = Path(__file__).parent / "main-4.pdf"
OUTPUT_DIR = Path(__file__).parent / "output"
IMAGE_MODE = "path"
# ==============================================================================

def _find_mineru_output(pdf: Path, output_root: Path):
    """Finds the MinerU extraction output."""
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
    from openingestion.chunker import SentenceChunker
    from openingestion.refinery import ContextualRagRefinery, RagRefinery, VisionRefinery
    from openingestion.porter import JSONPorter
    from openingestion.genie import OpenAIGenie

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

    # -- Etape 2 : SentenceChunker ---------------------------------------------
    _section("ETAPE 2 — SentenceChunker")
    t_chunk = time.perf_counter()

    chunker = SentenceChunker(
        chunk_size=512,
        chunk_overlap=128
    )
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
    tok_count = [c.token_count for c in chunks]
    print(f"  Enrichissement termine ({elapsed_ref:.2f}s)")
    if tok_count:
        print(f"  Token count min={min(tok_count)} max={max(tok_count)} moy={sum(tok_count)/len(tok_count):.0f}")
    assert all(c.content_hash for c in chunks if c.page_content.strip()), "ERREUR : content_hash vide sur chunk non-vide"
    print("  Assertions RagRefinery : OK")

    # -- Etape 3.5 : VisionRefinery --------------------------------------------
    _section("ETAPE 3.5 — VisionRefinery (Extraction tableaux scannés)")
    t_vis = time.perf_counter()
    
    vision_genie = OpenAIGenie(
        model="gpt-4o-mini",
        api_key=OPENAI_API_KEY
    )
    from openingestion.document import BlockKind
    vision_refinery = VisionRefinery(
        genie=vision_genie,
        kinds={BlockKind.TABLE, BlockKind.IMAGE},  # Tableaux mal parsés ET images
        only_if_empty=True,
        image_detail="low"        # Economique
    )
    chunks = vision_refinery.enrich(chunks)
    
    elapsed_vis = time.perf_counter() - t_vis
    print(f"  Vision extraction terminée ({elapsed_vis:.2f}s)")

    # -- Etape 4 : ContextualRagRefinery ---------------------------------------
    if not skip_contextual:
        _section("ETAPE 4 — ContextualRagRefinery (OpenAIGenie)")
        
        if OPENAI_API_KEY == "VOTRE_CLE_API_ICI" or not OPENAI_API_KEY:
            logger.error("Veuillez remplir le champ OPENAI_API_KEY avec votre cle API dans le fichier test_pipeline_real_sentence.py")
            sys.exit(1)

        t_ctx = time.perf_counter()

        genie = OpenAIGenie(
            model="gpt-4o-mini",
            api_key=OPENAI_API_KEY
        )

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
        print(f"  Requetes d'API OpenAI effectuee! ({elapsed_ctx:.2f}s)")
        print(f"  doc_summary renseigne   : {with_summary}/{len(chunks)} chunks")
        print(f"  chunk_context renseigne : {with_context}/{len(chunks)} chunks")
        assert with_summary > 0, "ERREUR : aucun doc_summary genere"
        assert with_context > 0, "ERREUR : aucun chunk_context genere"
        print("  Assertions ContextualRagRefinery : OK")

    # -- Etape 5 : Export JSONL ------------------------------------------------
    _section("ETAPE 5 — Export JSONL")
    out_jsonl = Path(__file__).parent / "output_chunks_real_sentence.jsonl"
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
    ap = argparse.ArgumentParser(description="Test pipeline avec real SentenceChunker & ContextualRAG via OpenAPI")
    ap.add_argument("--reparse", action="store_true", help="Force re-run MinerU meme si l'output existe")
    ap.add_argument("--no-contextual", action="store_true", help="Sauter l'etape ContextualRagRefinery")
    args = ap.parse_args()
    
    logger.remove()
    logger.add(sys.stderr, level="INFO")
    run(force_reparse=args.reparse, skip_contextual=args.no_contextual)
