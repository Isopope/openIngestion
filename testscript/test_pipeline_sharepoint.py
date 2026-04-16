"""Pipeline d'ingestion RAG depuis SharePoint.

Orchestration complète :
    SharepointFetcher → MinerUChef → SentenceChunker → RagRefinery → JSONPorter

Usage
-----
    python testscript/test_pipeline_sharepoint.py
    python testscript/test_pipeline_sharepoint.py --no-contextual
    python testscript/test_pipeline_sharepoint.py --reparse
    python testscript/test_pipeline_sharepoint.py --folder "Documents/RH"
"""
from __future__ import annotations

import argparse
import sys
import time
from collections import Counter
from pathlib import Path

from loguru import logger

# ==============================================================================
# CONFIGURATION — credentials Azure AD & SharePoint
# ==============================================================================
AZURE_CLIENT_ID     = "VOTRE_CLIENT_ID"
AZURE_CLIENT_SECRET = "VOTRE_CLIENT_SECRET"
AZURE_TENANT_ID     = "VOTRE_TENANT_ID"

# Identifier le site SharePoint : fournir l'un ou l'autre
SHAREPOINT_SITE_URL  = "https://VOTRE_TENANT.sharepoint.com/sites/VOTRE_SITE"
SHAREPOINT_SITE_NAME = None           # alternatif si pas de site_url exact

# Sous-dossier à parcourir (None = racine "Documents Partagés")
SHAREPOINT_FOLDER    = None           # ex. "Documents/RH/Politiques"

# Répertoires locaux
OUTPUT_DIR      = Path(__file__).parent / "output_sharepoint"
DOWNLOAD_DIR    = Path(__file__).parent / "downloads_sharepoint"
IMAGE_MODE      = "path"

# Clé API OpenAI (pour ContextualRagRefinery et VisionRefinery)
OPENAI_API_KEY  = "VOTRE_CLE_OPENAI"
# ==============================================================================


def _section(title: str) -> None:
    print(f"\n{'─' * 60}\n  {title}\n{'─' * 60}")


def _find_mineru_output(pdf: Path, output_root: Path) -> Path | None:
    """Cherche un répertoire de sortie MinerU existant pour éviter le re-parsing."""
    for candidate in (
        output_root / pdf.name / "auto",
        output_root / pdf.stem / "auto",
    ):
        if candidate.is_dir() and list(candidate.glob("*_content_list.json")):
            return candidate
    return None


def run(
    folder: str | None = None,
    force_reparse: bool = False,
    skip_contextual: bool = False,
) -> None:
    from openingestion.fetcher.sharepoint import SharepointFetcher
    from openingestion.chef.mineru_chef import MinerUChef
    from openingestion.chunker import SentenceChunker
    from openingestion.refinery import ContextualRagRefinery, RagRefinery, VisionRefinery
    from openingestion.porter import JSONPorter
    from openingestion.genie import OpenAIGenie
    from openingestion.document import BlockKind

    # Valider la configuration minimale
    if "VOTRE_CLIENT_ID" in AZURE_CLIENT_ID:
        logger.error(
            "Veuillez renseigner AZURE_CLIENT_ID, AZURE_CLIENT_SECRET et "
            "AZURE_TENANT_ID dans le fichier test_pipeline_sharepoint.py"
        )
        sys.exit(1)

    if "VOTRE_TENANT" in SHAREPOINT_SITE_URL and SHAREPOINT_SITE_NAME is None:
        logger.error(
            "Veuillez renseigner SHAREPOINT_SITE_URL ou SHAREPOINT_SITE_NAME "
            "dans le fichier test_pipeline_sharepoint.py"
        )
        sys.exit(1)

    folder_target = folder or SHAREPOINT_FOLDER
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)

    t_total = time.perf_counter()

    # ── Étape 0 : SharepointFetcher ───────────────────────────────────────────
    _section("ÉTAPE 0 — SharepointFetcher (téléchargement depuis SharePoint)")
    t_fetch = time.perf_counter()

    fetcher = SharepointFetcher(
        client_id=AZURE_CLIENT_ID,
        client_secret=AZURE_CLIENT_SECRET,
        tenant_id=AZURE_TENANT_ID,
        output_dir=DOWNLOAD_DIR,
    )

    fetched_docs = fetcher.fetch(
        site_url=SHAREPOINT_SITE_URL if "VOTRE_TENANT" not in SHAREPOINT_SITE_URL else None,
        site_name=SHAREPOINT_SITE_NAME,
        folder_path=folder_target,
    )

    elapsed_fetch = time.perf_counter() - t_fetch
    print(f"  Documents récupérés : {len(fetched_docs)}  ({elapsed_fetch:.1f}s)")

    if not fetched_docs:
        logger.warning("Aucun document trouvé sur SharePoint. Vérifiez vos paramètres.")
        sys.exit(0)

    for doc in fetched_docs:
        print(f"    • [{doc.mime_type or '?'}]  {doc.path.name}")

    # Filtrer uniquement les PDFs (le Chef MinerU ne supporte que PDF/images)
    supported_ext = {".pdf", ".png", ".jpg", ".jpeg", ".tiff", ".bmp", ".webp"}
    supported_docs = [d for d in fetched_docs if d.path.suffix.lower() in supported_ext]
    skipped = len(fetched_docs) - len(supported_docs)
    if skipped:
        logger.warning("{} fichier(s) ignoré(s) (format non supporté par MinerU).", skipped)
    if not supported_docs:
        logger.error("Aucun fichier PDF/image parmi les documents téléchargés.")
        sys.exit(1)

    print(f"  Fichiers traités par le pipeline : {len(supported_docs)}")

    # ── Étape 1 : MinerUChef ──────────────────────────────────────────────────
    _section("ÉTAPE 1 — MinerUChef (parsing layout)")
    t_chef = time.perf_counter()

    chef = MinerUChef(output_dir=OUTPUT_DIR)

    all_blocks_per_doc: list[tuple[Path, list]] = []
    for doc in supported_docs:
        pdf_path = doc.path
        existing = _find_mineru_output(pdf_path, OUTPUT_DIR)

        if existing and not force_reparse:
            logger.info("Output MinerU existant — re-lecture : {}", existing)
            blocks = chef.process(existing)
        else:
            logger.info("Parsing MinerU de {}", pdf_path.name)
            blocks = chef.process(pdf_path)

        print(f"  [{pdf_path.name}]  {len(blocks)} blocs extraits")
        for k, n in sorted(Counter(b.kind.value for b in blocks).items()):
            print(f"      {k:<12} {n:>4}")

        all_blocks_per_doc.append((pdf_path, blocks))

    elapsed_chef = time.perf_counter() - t_chef
    total_blocks = sum(len(b) for _, b in all_blocks_per_doc)
    print(f"\n  Total blocs extraits : {total_blocks}  ({elapsed_chef:.1f}s)")

    # ── Traitement par document ───────────────────────────────────────────────
    all_chunks = []

    for pdf_path, blocks in all_blocks_per_doc:
        doc_name = pdf_path.stem

        # ── Étape 2 : SentenceChunker ─────────────────────────────────────────
        _section(f"ÉTAPE 2 — SentenceChunker  [{doc_name}]")
        t_chunk = time.perf_counter()

        chunker = SentenceChunker(chunk_size=512, chunk_overlap=128)
        chunks = chunker(blocks, source=str(pdf_path))

        elapsed_chunk = time.perf_counter() - t_chunk
        print(f"  Chunks générés : {len(chunks)}  ({elapsed_chunk:.2f}s)")

        indices = [c.chunk_index for c in chunks]
        assert len(indices) == len(set(indices)), "ERREUR : chunk_index non uniques !"
        print("  chunk_index uniques : OK")

        # ── Étape 3 : RagRefinery ──────────────────────────────────────────────
        _section(f"ÉTAPE 3 — RagRefinery  [{doc_name}]")
        t_ref = time.perf_counter()

        existing = _find_mineru_output(pdf_path, OUTPUT_DIR)
        rag_img_dir = existing or (OUTPUT_DIR / pdf_path.name / "auto")

        rag_refinery = RagRefinery(
            output_dir=rag_img_dir,
            image_mode=IMAGE_MODE,
            infer_captions=True,
        )
        chunks = rag_refinery.enrich(chunks)

        elapsed_ref = time.perf_counter() - t_ref
        tok_count = [c.token_count for c in chunks]
        print(f"  Enrichissement terminé ({elapsed_ref:.2f}s)")
        if tok_count:
            print(
                f"  Token count  min={min(tok_count)}  "
                f"max={max(tok_count)}  "
                f"moy={sum(tok_count)/len(tok_count):.0f}"
            )
        assert all(
            c.content_hash for c in chunks if c.page_content.strip()
        ), "ERREUR : content_hash vide sur chunk non-vide"
        print("  Assertions RagRefinery : OK")

        # ── Étape 3.5 : VisionRefinery ────────────────────────────────────────
        if OPENAI_API_KEY and "VOTRE_CLE" not in OPENAI_API_KEY:
            _section(f"ÉTAPE 3.5 — VisionRefinery  [{doc_name}]")
            t_vis = time.perf_counter()

            vision_genie = OpenAIGenie(model="gpt-4o-mini", api_key=OPENAI_API_KEY)
            vision_refinery = VisionRefinery(
                genie=vision_genie,
                kinds={BlockKind.TABLE, BlockKind.IMAGE},
                only_if_empty=True,
                image_detail="low",
            )
            chunks = vision_refinery.enrich(chunks)

            elapsed_vis = time.perf_counter() - t_vis
            print(f"  Vision extraction terminée ({elapsed_vis:.2f}s)")

        # ── Étape 4 : ContextualRagRefinery ───────────────────────────────────
        if not skip_contextual and OPENAI_API_KEY and "VOTRE_CLE" not in OPENAI_API_KEY:
            _section(f"ÉTAPE 4 — ContextualRagRefinery  [{doc_name}]")
            t_ctx = time.perf_counter()

            genie = OpenAIGenie(model="gpt-4o-mini", api_key=OPENAI_API_KEY)
            ctx_refinery = ContextualRagRefinery(
                genie=genie,
                generate_doc_summary=True,
                generate_chunk_context=True,
                max_workers=4,
                summary_max_tokens=150,
                context_max_tokens=100,
            )
           # chunks = ctx_refinery.enrich(chunks)

            elapsed_ctx = time.perf_counter() - t_ctx
            with_summary = sum(1 for c in chunks if c.doc_summary)
            with_context = sum(1 for c in chunks if c.chunk_context)
            print(f"  Contextual RAG terminé ({elapsed_ctx:.2f}s)")
            print(f"  doc_summary renseigné   : {with_summary}/{len(chunks)}")
            print(f"  chunk_context renseigné : {with_context}/{len(chunks)}")

        all_chunks.extend(chunks)

    # ── Étape 5 : Export JSONL ────────────────────────────────────────────────
    _section("ÉTAPE 5 — Export JSONL")
    out_jsonl = Path(__file__).parent / "output_chunks_sharepoint.jsonl"
    JSONPorter(lines=True)(all_chunks, file=out_jsonl)
    print(f"  Exporté ({len(all_chunks)} chunks) → {out_jsonl}")

    # ── Résumé ────────────────────────────────────────────────────────────────
    total_elapsed = time.perf_counter() - t_total
    _section("RÉSUMÉ")
    print(f"  Documents SharePoint  : {len(supported_docs)}")
    print(f"  Blocs MinerU totaux   : {total_blocks}")
    print(f"  Chunks totaux         : {len(all_chunks)}")
    print(f"  Temps total           : {total_elapsed:.1f}s")
    print(f"  Sortie JSONL          : {out_jsonl}")
    print(f"\n  PIPELINE OK")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="Pipeline RAG SharePoint → MinerU → SentenceChunker → RagRefinery → JSONL"
    )
    ap.add_argument(
        "--folder",
        default=None,
        metavar="PATH",
        help="Sous-dossier SharePoint à parcourir (ex: 'Documents/RH'). "
             "Remplace SHAREPOINT_FOLDER du script.",
    )
    ap.add_argument(
        "--reparse",
        action="store_true",
        help="Force le re-parsing MinerU même si un output existant est trouvé.",
    )
    ap.add_argument(
        "--no-contextual",
        action="store_true",
        help="Sauter l'étape ContextualRagRefinery (plus rapide, pas de résumé LLM).",
    )
    args = ap.parse_args()

    logger.remove()
    logger.add(sys.stderr, level="INFO")

    run(
        folder=args.folder,
        force_reparse=args.reparse,
        skip_contextual=args.no_contextual,
    )
