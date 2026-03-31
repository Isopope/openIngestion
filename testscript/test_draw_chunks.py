"""Genere un PDF annote avec les chunks RAG superposes sur le document original.

Fonctionne avec les deux parsers (MinerU et Docling).
Le systeme de coordonnees est detecte automatiquement.

Usage
-----
    # Depuis un JSON multipass produit par test_pipeline.py (recommande)
    python test_draw_chunks.py --from-json output_chunks_multipass.json
    python test_draw_chunks.py --from-json output_chunks_multipass.json --granularity normal
    python test_draw_chunks.py --from-json output_chunks_multipass.json --granularity mini

    # Parse a la volee (MinerU, strategie simple)
    python test_draw_chunks.py --parser mineru --strategy by_sentence
    python test_draw_chunks.py --parser mineru --strategy by_token

    # Parse a la volee (MinerU, multipass via SentenceChunker)
    python test_draw_chunks.py --parser mineru --strategy multipass
    python test_draw_chunks.py --parser mineru --strategy multipass --granularity mini
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

from loguru import logger

PDF_PATH = Path(__file__).parent / "main-4LE.pdf"

# Couleurs par granularite (pour le multipass)
_GRAN_LEGEND = {
    "mini":   "vert",
    "normal": "bleu",
    "large":  "orange",
    "":       "bleu",   # pipeline single-pass
}


def _load_from_json(json_path: Path) -> list:
    """Reload RagChunks from a serialised JSON or JSONL file."""
    from openingestion.document import RagChunk, BlockKind

    text = json_path.read_text(encoding="utf-8").strip()
    if json_path.suffix == ".jsonl" or text.startswith("{"):
        raw = [json.loads(line) for line in text.splitlines() if line.strip()]
    else:
        raw = json.loads(text)

    chunks = []
    for d in raw:
        d2 = dict(d)
        d2["kind"] = BlockKind(d2["kind"])
        # granularity may be absent in old exports
        d2.setdefault("granularity", "")
        chunks.append(RagChunk(**d2))
    return chunks


def _build_multipass_chunks(pdf: Path) -> list:
    """Run the full multipass pipeline (SentenceChunker + TokenChunker)."""
    from openingestion.chef.mineru_chef import MinerUChef
    from openingestion.chunker import MultipassChunker, SentenceChunker, TokenChunker
    from openingestion.refinery import RagRefinery

    output_dir = Path(__file__).parent / "output"
    existing = None
    for candidate in (output_dir / pdf.name / "auto", output_dir / pdf.stem / "auto"):
        if candidate.is_dir() and list(candidate.glob("*_content_list.json")):
            existing = candidate
            break

    chef = MinerUChef(output_dir=output_dir)
    if existing:
        logger.info("Output MinerU existant — re-lecture : {}", existing)
        blocks = chef.process(existing)
    else:
        logger.info("Parsing MinerU complet de {}", pdf)
        blocks = chef.process(pdf)

    chunker = MultipassChunker(passes=[
        ("mini",   SentenceChunker(chunk_size=150,  chunk_overlap=16)),
        ("normal", SentenceChunker(chunk_size=512,  chunk_overlap=64)),
        ("large",  TokenChunker(max_tokens=2048, overlap_tokens=128)),
    ])
    chunks = chunker(blocks, source=str(pdf))

    refinery = RagRefinery(
        output_dir=existing or (output_dir / pdf.name / "auto"),
        image_mode="path",
        infer_captions=True,
    )
    return refinery.enrich(chunks)


def run(
    pdf: Path,
    parser: str,
    strategy: str,
    from_json: Path | None,
    granularity: str,
) -> None:
    from openingestion.utils.draw_chunks import draw_chunks_on_pdf

    if not pdf.exists():
        logger.error("PDF introuvable : {}", pdf)
        sys.exit(1)

    # ── Chargement des chunks ─────────────────────────────────────────────────
    if from_json is not None:
        if not from_json.exists():
            logger.error("JSON introuvable : {}", from_json)
            sys.exit(1)
        logger.info("Chargement des chunks depuis {}", from_json.name)
        t0 = time.perf_counter()
        chunks = _load_from_json(from_json)
        t_parse = time.perf_counter() - t0
        logger.info("{} chunks charges en {:.2f}s", len(chunks), t_parse)
        if "docling" in from_json.name:
            parser = "docling"
        elif "mineru" in from_json.name or "multipass" in from_json.name:
            parser = "mineru"

    elif strategy == "multipass":
        t0 = time.perf_counter()
        chunks = _build_multipass_chunks(pdf)
        t_parse = time.perf_counter() - t0
        logger.info("{} chunks produits en {:.1f}s", len(chunks), t_parse)

    else:
        from openingestion import ingest

        logger.info("Parsing {} avec parser={} strategy={}", pdf.name, parser, strategy)
        t0 = time.perf_counter()
        chunks = ingest(pdf, parser=parser, strategy=strategy,
                        image_mode="path", infer_captions=True)
        t_parse = time.perf_counter() - t0
        logger.info("{} chunks produits en {:.1f}s", len(chunks), t_parse)

    all_chunks = chunks
    total_all = len(all_chunks)

    # ── Filtrage par granularite quand --granularity specifie ─────────────────
    grans_present = sorted(set(c.granularity for c in chunks))
    is_multipass = any(g for g in grans_present)

    if granularity and is_multipass:
        chunks = [c for c in chunks if c.granularity == granularity]
        if not chunks:
            logger.error(
                "Aucun chunk avec granularity={!r}. Disponibles : {}",
                granularity, grans_present,
            )
            sys.exit(1)
        logger.info(
            "Filtre granularite={!r} : {} / {} chunks retenus",
            granularity, len(chunks), total_all,
        )
        gran_label = granularity
    elif is_multipass and not granularity:
        # Pas de filtre : on annote tout, normal par defaut pour la lisibilite
        gran_label = "all"
    else:
        gran_label = ""

    # ── Nom du fichier de sortie ──────────────────────────────────────────────
    suffix_parts = [parser]
    if is_multipass:
        suffix_parts.append("multipass")
        if gran_label and gran_label != "all":
            suffix_parts.append(gran_label)
    else:
        suffix_parts.append(strategy)
    out_pdf = pdf.with_name(f"{pdf.stem}_chunks_{'_'.join(suffix_parts)}.pdf")

    # ── Annotation PDF ────────────────────────────────────────────────────────
    t1 = time.perf_counter()
    result = draw_chunks_on_pdf(chunks, pdf, output_path=out_pdf)
    t_draw = time.perf_counter() - t1

    # ── Rapport ───────────────────────────────────────────────────────────────
    print()
    print("=" * 60)
    print(f"  PDF annote genere en {t_draw:.1f}s")
    print(f"  Parser     : {parser}")
    print(f"  Strategie  : {strategy if not is_multipass else 'multipass'}")
    if grans_present and any(grans_present):
        print(f"  Granularites disponibles : {', '.join(g for g in grans_present if g)}")
        print(f"  Granularite annotee      : {gran_label or 'toutes'}")
    print(f"  Chunks annotes : {len(chunks)}  (total pipeline : {total_all})")
    print(f"  Sortie     : {result}")
    print("=" * 60)

    kinds = Counter(c.kind.value for c in chunks)
    print("\nDistribution des kinds (= couleurs dans le PDF) :")
    legend_kind = {
        "text":      "bleu",
        "title":     "orange",
        "table":     "jaune",
        "image":     "vert",
        "equation":  "cyan",
        "discarded": "gris",
    }
    for kind, n in sorted(kinds.items()):
        colour = legend_kind.get(kind, "violet")
        print(f"  {kind:<12} {n:>4}   -> {colour}")

    if is_multipass:
        print("\nDistribution par granularite :")
        for g, n in sorted(Counter(c.granularity for c in all_chunks).items()):
            selected = " [annotes]" if (not granularity or g == granularity) else ""
            print(f"  {g:<8}  {n:>4} chunks{selected}")

    with_bbox = [c for c in chunks if c.position_int and any(
        e[1:5] != [0, 0, 0, 0] for e in c.position_int
    )]
    print(f"\nChunks avec bbox non-nulle : {len(with_bbox)}/{len(chunks)}")
    print(f"\nOuvrez le fichier annote :\n  {result}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Annote un PDF avec les bboxes des chunks RAG")
    p.add_argument("--pdf", type=Path, default=PDF_PATH)
    p.add_argument("--parser", choices=["docling", "mineru"], default="mineru")
    p.add_argument(
        "--strategy",
        choices=["by_block", "by_token", "by_sentence", "by_semantic", "multipass"],
        default="multipass",
    )
    p.add_argument(
        "--granularity",
        choices=["mini", "normal", "large", ""],
        default="normal",
        help="Filtrer les chunks par granularite (multipass seulement). "
             "Vide = tout annoter.",
    )
    p.add_argument(
        "--from-json",
        type=Path,
        default=None,
        metavar="PATH",
        help="Charger les chunks depuis un JSON/JSONL existant "
             "(ex: output_chunks_multipass.json)",
    )
    args = p.parse_args()
    run(
        pdf=args.pdf,
        parser=args.parser,
        strategy=args.strategy,
        from_json=args.from_json,
        granularity=args.granularity,
    )
