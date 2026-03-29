"""Test end-to-end de la pipeline openingestion avec Docling.

Stratégie : by_semantic (SemanticChunker)
Utilise DoclingChef à la place de MinerU — pas besoin de GPU ni d'output dir.

Usage
-----
    python test_pipeline_docling.py
    python test_pipeline_docling.py --strategy by_token
    python test_pipeline_docling.py --strategy by_block
    python test_pipeline_docling.py --pdf autre_fichier.pdf
"""
from __future__ import annotations

import argparse
import sys
import time
from collections import Counter
from pathlib import Path

from loguru import logger

# ── config ────────────────────────────────────────────────────────────────────
PDF_PATH  = Path(__file__).parent / "main-4.pdf"
STRATEGY  = "by_slumber"
IMAGE_MODE = "path"
# ─────────────────────────────────────────────────────────────────────────────


def run(pdf: Path, strategy: str) -> None:
    from openingestion import ingest
    from openingestion.porter import JSONPorter

    if not pdf.exists():
        logger.error("PDF introuvable : {}", pdf)
        sys.exit(1)

    logger.info("DoclingChef — parsing de {} (stratégie={})", pdf.name, strategy)

    t0 = time.perf_counter()

    chunks = ingest(
        pdf,
        parser="docling",
        strategy=strategy,
        image_mode=IMAGE_MODE,
        infer_captions=True,
    )

    elapsed = time.perf_counter() - t0

    # ── résultats ─────────────────────────────────────────────────────────────
    print()
    print("=" * 60)
    print(f"  Pipeline Docling terminée en {elapsed:.1f}s")
    print(f"  PDF        : {pdf.name}")
    print(f"  Stratégie  : {strategy}")
    print(f"  Chunks     : {len(chunks)}")
    print("=" * 60)

    # Distribution des kinds
    kinds = Counter(c.kind.value for c in chunks)
    print("\nDistribution des kinds :")
    for kind, n in sorted(kinds.items()):
        print(f"  {kind:<12} {n:>4}")

    # Stats token_count
    token_counts = [c.token_count for c in chunks]
    if token_counts:
        print(f"\nToken count  min={min(token_counts)}  "
              f"max={max(token_counts)}  "
              f"moy={sum(token_counts)/len(token_counts):.0f}")

    # Stats spatiales — vérifier que les bboxes sont bien remplies
    with_bbox = [c for c in chunks if c.position_int and c.position_int[0][1:] != [0, 0, 0, 0]]
    print(f"\nChunks avec bbox spatiale : {len(with_bbox)}/{len(chunks)}")

    # Aperçu des 5 premiers chunks
    print("\n-- Aperçu des 5 premiers chunks ---------------------")
    for i, c in enumerate(chunks[:5]):
        title = f"[{c.title_path}]" if c.title_path else ""
        preview = c.page_content[:120].replace("\n", " ")
        page_no = c.position_int[0][0] + 1 if c.position_int else "?"
        bbox = c.position_int[0][1:] if c.position_int else "?"
        print(f"  [{i}] kind={c.kind.value:<8} p={page_no}  bbox={bbox}")
        print(f"       tok={c.token_count:>4}  hash={c.content_hash}  {title}")
        print(f"       {preview!r}")
        print()

    # Export JSON
    out_json = Path(__file__).parent / "output_chunks_docling.json"
    JSONPorter(lines=False, indent=2)(chunks, file=out_json)
    print(f"Chunks exportés -> {out_json}")

    # Export JSONL
    out_jsonl = Path(__file__).parent / "output_chunks_docling.jsonl"
    JSONPorter(lines=True)(chunks, file=out_jsonl)
    print(f"Chunks exportés (JSONL) -> {out_jsonl}")

    # Santé basique
    assert all(c.content_hash for c in chunks), "content_hash vide détecté"
    assert all(c.token_count >= 0 for c in chunks), "token_count négatif détecté"
    print("\nAssertions de santé : OK")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Test pipeline openingestion + Docling")
    p.add_argument("--pdf", type=Path, default=PDF_PATH,
                   help="Chemin vers le PDF à ingérer")
    p.add_argument("--strategy",
                   choices=["by_block", "by_token", "by_sentence", "by_semantic", "by_slumber"],
                   default=STRATEGY,
                   help="Stratégie de chunking (défaut: by_semantic)")
    args = p.parse_args()
    run(pdf=args.pdf, strategy=args.strategy)
