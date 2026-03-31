"""Test end-to-end de la pipeline complete openingestion sur main-4.pdf.

Etapes couvertes
----------------
1. Chef     -- MinerU parse le PDF (reutilise l'output existant si present)
2. Chunker  -- MultipassChunker (mini 150 / normal 512 / large 2048 tokens)
3. Refinery -- RagRefinery (token_count, content_hash, img_path absolu)
4. Refinery -- ContextualRagRefinery (doc_summary + chunk_context via mock genie)
5. Porter   -- JSONPorter -> output_chunks_multipass.jsonl

Usage
-----
    python test_pipeline.py
    python test_pipeline.py --reparse       # force re-run MinerU meme si output existe
    python test_pipeline.py --no-contextual # saute ContextualRagRefinery
"""
from __future__ import annotations

import argparse
import sys
import time
from collections import Counter
from pathlib import Path

from loguru import logger

# -- config -------------------------------------------------------------------
PDF_PATH   = Path(__file__).parent / "main-4LE.pdf"
OUTPUT_DIR = Path(__file__).parent / "output"
IMAGE_MODE = "path"
# -----------------------------------------------------------------------------


def _find_mineru_output(pdf: Path, output_root: Path):
    """Return the MinerU output sub-dir for *pdf* if it already exists."""
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
    from openingestion.chunker import MultipassChunker
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

    # -- Etape 2 : MultipassChunker --------------------------------------------
    _section("ETAPE 2 — MultipassChunker (mini / normal / large)")
    t_chunk = time.perf_counter()

    chunker = MultipassChunker()
    chunks = chunker(blocks, source=str(PDF_PATH))

    elapsed_chunk = time.perf_counter() - t_chunk
    counts_gran = Counter(c.granularity for c in chunks)
    print(f"  Total chunks : {len(chunks)}  ({elapsed_chunk:.2f}s)")
    for gran, n in sorted(counts_gran.items()):
        print(f"    {gran:<8} {n:>4} chunks")

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
    normal_chunks = [c for c in chunks if c.granularity == "normal"]
    tok_normal = [c.token_count for c in normal_chunks]
    print(f"  Enrichissement termine ({elapsed_ref:.2f}s)")
    if tok_normal:
        print(f"  Token count [normal]  min={min(tok_normal)}  "
              f"max={max(tok_normal)}  "
              f"moy={sum(tok_normal)/len(tok_normal):.0f}")
    assert all(c.content_hash for c in chunks if c.page_content.strip()), \
        "ERREUR : content_hash vide sur chunk non-vide"
    assert all(c.token_count >= 0 for c in chunks), "ERREUR : token_count negatif"
    print("  Assertions RagRefinery : OK")

    # -- Etape 4 : ContextualRagRefinery ---------------------------------------
    if not skip_contextual:
        _section("ETAPE 4 — ContextualRagRefinery (doc_summary + chunk_context)")
        print("  Genie : mock (lambda) — remplacer par OpenAIGenie / GeminiGenie en prod")
        t_ctx = time.perf_counter()

        call_count = {"n": 0}

        def mock_genie(prompt: str) -> str:
            call_count["n"] += 1
            snippet = prompt[-60:].replace("\n", " ").strip()
            return f"[MOCK ctx #{call_count['n']}] ...{snippet}"

        ctx_refinery = ContextualRagRefinery(
            genie=mock_genie,
            generate_doc_summary=True,
            generate_chunk_context=True,
            max_workers=4,
            summary_max_tokens=150,
            context_max_tokens=100,
        )
        # On applique uniquement sur les chunks "normal" pour garder le test rapide
        ctx_refinery.enrich(normal_chunks)

        elapsed_ctx = time.perf_counter() - t_ctx
        with_summary = sum(1 for c in normal_chunks if c.doc_summary)
        with_context = sum(1 for c in normal_chunks if c.chunk_context)
        print(f"  Appels mock genie : {call_count['n']}  ({elapsed_ctx:.2f}s)")
        print(f"  doc_summary renseigne   : {with_summary}/{len(normal_chunks)} chunks [normal]")
        print(f"  chunk_context renseigne : {with_context}/{len(normal_chunks)} chunks [normal]")
        assert with_summary > 0, "ERREUR : aucun doc_summary genere"
        assert with_context > 0, "ERREUR : aucun chunk_context genere"
        print("  Assertions ContextualRagRefinery : OK")
    else:
        _section("ETAPE 4 — ContextualRagRefinery (ignoree via --no-contextual)")

    # -- Etape 5 : parent links ------------------------------------------------
    _section("ETAPE 5 — Verification des liens parent (small-to-big)")
    mini_chunks = [c for c in chunks if c.granularity == "mini"]
    linked = sum(1 for c in mini_chunks if "parent_chunk_index" in c.extras)
    print(f"  Chunks mini avec parent_chunk_index : {linked}/{len(mini_chunks)}")

    # -- Etape 6 : Apercu ------------------------------------------------------
    _section("ETAPE 6 — Apercu des 3 premiers chunks [normal]")
    chunk_map = {c.chunk_index: c for c in chunks}
    for c in normal_chunks[:3]:
        title = f"[{c.title_path}]" if c.title_path else "[sans titre]"
        preview = c.page_content[:100].replace("\n", " ")
        print(f"\n  chunk_index={c.chunk_index}  kind={c.kind.value}  tok={c.token_count}  {title}")
        print(f"  contenu  : {preview!r}")
        if c.doc_summary:
            print(f"  summary  : {c.doc_summary[:80]!r}")
        if c.chunk_context:
            print(f"  context  : {c.chunk_context[:80]!r}")
        parent_idx = c.extras.get("parent_chunk_index")
        if parent_idx is not None:
            parent = chunk_map.get(parent_idx)
            if parent:
                print(f"  parent   : chunk_index={parent_idx}  gran={parent.granularity}  tok={parent.token_count}")

    # -- Etape 7 : Export JSONL ------------------------------------------------
    _section("ETAPE 7 — Export JSONL")
    out_jsonl = Path(__file__).parent / "output_chunks_multipass.jsonl"
    JSONPorter(lines=True)(chunks, file=out_jsonl)
    print(f"  Exporte ({len(chunks)} chunks) -> {out_jsonl}")
    out_json = Path(__file__).parent / "output_chunks_multipass.json"
    JSONPorter(lines=False, indent=2)(chunks, file=out_json)
    print(f"  Exporte ({len(chunks)} chunks) -> {out_json}")

    # -- Resume ----------------------------------------------------------------
    total = time.perf_counter() - t_chef
    _section("RESUME")
    print(f"  Blocs MinerU    : {len(blocks)}")
    print(f"  Chunks total    : {len(chunks)}")
    for gran, n in sorted(counts_gran.items()):
        print(f"    {gran:<8}      {n:>4}")
    print(f"  Temps total     : {total:.1f}s")
    print(f"\n  PIPELINE OK")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Test pipeline complet openingestion")
    ap.add_argument("--reparse", action="store_true",
                    help="Force re-run MinerU meme si l'output existe")
    ap.add_argument("--no-contextual", action="store_true",
                    help="Sauter l'etape ContextualRagRefinery")
    args = ap.parse_args()
    run(force_reparse=args.reparse, skip_contextual=args.no_contextual)
