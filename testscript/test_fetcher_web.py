"""Test simple du WebFetcher.

Prérequis
---------
    pip install playwright
    playwright install chromium

Usage
-----
    python testscript/test_fetcher_web.py
    python testscript/test_fetcher_web.py https://example.com
    python testscript/test_fetcher_web.py https://example.com https://github.com --mode html
    python testscript/test_fetcher_web.py --headed --wait-for load
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from loguru import logger

DEFAULT_URLS = [
    "https://example.com",
]
DEFAULT_OUTPUT_DIR = Path(__file__).parent / "output_web_fetcher"


def _section(title: str) -> None:
    print(f"\n{'─' * 60}\n  {title}\n{'─' * 60}")


def run(
    urls: list[str],
    output_dir: Path,
    mode: str,
    headless: bool,
    wait_for: str,
) -> None:
    from openingestion.fetcher.web import WebFetcher

    output_dir.mkdir(parents=True, exist_ok=True)

    _section("ETAPE 1 — WebFetcher")
    print(f"  URLs à traiter : {len(urls)}")
    print(f"  Mode           : {mode}")
    print(f"  Headless       : {headless}")
    print(f"  wait_for       : {wait_for}")
    print(f"  Output dir     : {output_dir}")

    t0 = time.perf_counter()
    fetcher = WebFetcher(
        output_dir=output_dir,
        mode=mode,
        headless=headless,
    )
    docs = fetcher.fetch(urls=urls, wait_for=wait_for)
    elapsed = time.perf_counter() - t0

    _section("ETAPE 2 — Résultats")
    print(f"  Documents récupérés : {len(docs)}  ({elapsed:.2f}s)")

    if not docs:
        raise AssertionError("Aucun document n'a été récupéré.")

    for i, doc in enumerate(docs, start=1):
        exists = doc.path is not None and doc.path.exists()
        size = doc.path.stat().st_size if exists else 0
        print(f"  [{i}] source    : {doc.source}")
        print(f"      path      : {doc.path}")
        print(f"      mime_type : {doc.mime_type}")
        print(f"      exists    : {exists}")
        print(f"      size      : {size} bytes")

    _section("ETAPE 3 — Vérifications")
    assert all(doc.path is not None for doc in docs), "Un document n'a pas de chemin local"
    assert all(doc.path.exists() for doc in docs if doc.path is not None), "Un fichier téléchargé est introuvable"

    expected_mime = "application/pdf" if mode == "pdf" else "text/html"
    assert all(doc.mime_type == expected_mime for doc in docs), "mime_type incohérent avec le mode demandé"

    print("  Assertions : OK")

    _section("RESUME")
    print(f"  URLs testées      : {len(urls)}")
    print(f"  Documents créés   : {len(docs)}")
    print(f"  Répertoire sortie : {output_dir}")
    print(f"  Temps total       : {elapsed:.2f}s")
    print("\n  WEB FETCHER OK")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test du WebFetcher openingestion")
    parser.add_argument(
        "urls",
        nargs="*",
        default=DEFAULT_URLS,
        help="Une ou plusieurs URLs publiques à récupérer",
    )
    parser.add_argument(
        "--mode",
        choices=["pdf", "html"],
        default="pdf",
        help="Format de sortie du WebFetcher",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Répertoire de sortie local",
    )
    parser.add_argument(
        "--wait-for",
        choices=["networkidle", "load", "domcontentloaded", "commit"],
        default="networkidle",
        help="Stratégie d'attente avant sauvegarde de la page",
    )
    parser.add_argument(
        "--headed",
        action="store_true",
        help="Lance Chromium avec interface visible pour debug",
    )
    args = parser.parse_args()

    logger.remove()
    logger.add(sys.stderr, level="INFO")

    run(
        urls=args.urls,
        output_dir=args.output_dir,
        mode=args.mode,
        headless=not args.headed,
        wait_for=args.wait_for,
    )
