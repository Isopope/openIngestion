"""Test end-to-end de la pipeline openingestion sur main-4.pdf.

Stratégie : by_semantic (SemanticChunker)
Lance MinerU si l'output n'existe pas encore, sinon re-lit directement.

Usage
-----
    python test_pipeline.py
    python test_pipeline.py --reparse   # force re-run MinerU même si output existe
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from loguru import logger

# ── config ────────────────────────────────────────────────────────────────────
PDF_PATH     = Path(__file__).parent / "main-4.pdf"
OUTPUT_DIR   = Path(__file__).parent / "output"
STRATEGY     = "by_semantic"
IMAGE_MODE   = "path"
# ─────────────────────────────────────────────────────────────────────────────


def _find_mineru_output(pdf: Path, output_root: Path) -> Path | None:
    """Return the MinerU output sub-dir for *pdf* if it already exists.

    MinerU uses the full filename (including extension) as sub-dir name,
    e.g. ``output/main-4.pdf/auto/``.
    """
    # MinerU names the sub-dir after the full file name (with extension)
    candidate = output_root / pdf.name / "auto"
    if candidate.is_dir() and list(candidate.glob("*_content_list.json")):
        return candidate
    # Fallback: stem-only (older behaviour)
    candidate_stem = output_root / pdf.stem / "auto"
    if candidate_stem.is_dir() and list(candidate_stem.glob("*_content_list.json")):
        return candidate_stem
    return None


def run(force_reparse: bool = False) -> None:
    from openingestion import ingest, ingest_from_output
    from openingestion.porter import JSONPorter

    if not PDF_PATH.exists():
        logger.error("PDF introuvable : {}", PDF_PATH)
        sys.exit(1)

    # Decide route: fast (existing output) or full MinerU parse
    existing = _find_mineru_output(PDF_PATH, OUTPUT_DIR)

    t0 = time.perf_counter()

    if existing and not force_reparse:
        logger.info("Répertoire MinerU détecté — re-lecture directe : {}", existing)
        chunks = ingest_from_output(
            existing,
            strategy=STRATEGY,
            image_mode=IMAGE_MODE,
            infer_captions=True,
        )
    else:
        if force_reparse:
            logger.info("--reparse demandé — parsing MinerU complet")
        else:
            logger.info("Aucun output MinerU trouvé — parsing complet de {}", PDF_PATH)
        chunks = ingest(
            PDF_PATH,
            strategy=STRATEGY,
            image_mode=IMAGE_MODE,
            infer_captions=True,
            mineru_output_dir=OUTPUT_DIR,
        )

    elapsed = time.perf_counter() - t0

    # ── résultats ─────────────────────────────────────────────────────────────
    print()
    print("=" * 60)
    print(f"  Pipeline terminee en {elapsed:.1f}s")
    print(f"  Strategie  : {STRATEGY}")
    print(f"  Chunks     : {len(chunks)}")
    print("=" * 60)

    # Distribution des kinds
    from collections import Counter
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

    # Aperçu des 5 premiers chunks
    print("\n-- Apercu des 5 premiers chunks ---------------------")
    for i, c in enumerate(chunks[:5]):
        title = f"[{c.title_path}]" if c.title_path else ""
        preview = c.page_content[:120].replace("\n", " ")
        print(f"  [{i}] kind={c.kind.value:<8} hash={c.content_hash}  "
              f"tok={c.token_count:>4}  p={c.position_int[0][0]+1 if c.position_int else '?'}"
              f"  {title}")
        print(f"       {preview!r}")
        print()

    # Export JSON array (inspectable dans VS Code / tout viewer JSON)
    out_json = Path(__file__).parent / "output_chunks.json"
    JSONPorter(lines=False, indent=2)(chunks, file=out_json)
    print(f"Chunks exportes -> {out_json}")

    # Export JSONL (format production - 1 objet par ligne)
    out_jsonl = Path(__file__).parent / "output_chunks.jsonl"
    JSONPorter(lines=True)(chunks, file=out_jsonl)
    print(f"Chunks exportes (JSONL) -> {out_jsonl}")

    # Sante basique
    assert all(c.content_hash for c in chunks), "content_hash vide detecte"
    assert all(c.token_count >= 0 for c in chunks), "token_count negatif detecte"
    print("\nAssertions de sante : OK")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test pipeline openingestion")
    parser.add_argument("--reparse", action="store_true",
                        help="Force re-run MinerU même si l'output existe")
    args = parser.parse_args()
    run(force_reparse=args.reparse)
