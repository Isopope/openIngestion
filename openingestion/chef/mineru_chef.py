"""MinerU Chef — CHOMP step 1 implementation."""
from __future__ import annotations

import json
import mimetypes
import os
from pathlib import Path

from loguru import logger

from openingestion.document import BlockKind, ContentBlock, FetchedDocument
from openingestion.chef.base import BaseChef

try:
    from mineru.cli.common import do_parse
except ImportError:
    do_parse = None

# MIME types that MinerU can ingest directly
_MINERU_MIME = {
    "application/pdf",
    "image/png",
    "image/jpeg",
    "image/jpg",
    "image/tiff",
    "image/bmp",
    "image/webp",
}


def _is_mineru_output_dir(path: Path) -> bool:
    """Return True if *path* is a directory containing a *_content_list.json."""
    return path.is_dir() and bool(list(path.glob("*_content_list.json")))


def _mime(path: Path) -> str:
    mime, _ = mimetypes.guess_type(str(path))
    return mime or "application/octet-stream"


def _is_parseable(path: Path) -> bool:
    """Return True if *path* is a file MinerU can parse (PDF or image)."""
    return path.is_file() and _mime(path) in _MINERU_MIME


class MinerUChef(BaseChef):
    """Chef that ingests MinerU output **or** raw PDF/image files.

    :meth:`process` is smart: it detects what it receives and routes
    accordingly:

    * **MinerU output directory** (contains ``*_content_list.json``) →
      reads the JSON directly (fast, no GPU required).
    * **Raw PDF or image file** → runs MinerU, then reads the JSON output.
    * **``*_content_list.json`` file** → parsed directly.
    * :class:`~openingestion.document.FetchedDocument` → unwrapped and
      routed via the rules above.

    Args:
        output_dir: Where MinerU writes its output when parsing raw files.
                    Defaults to ``"./output"``.
        mineru_backend: MinerU backend to use (``"pipeline"`` by default).
    """

    def __init__(
        self,
        output_dir: str | os.PathLike = "./output",
        mineru_backend: str = "pipeline",
    ) -> None:
        self.output_dir = Path(output_dir)
        self.mineru_backend = mineru_backend

    def process(
        self,
        source: str | os.PathLike | FetchedDocument,
    ) -> list[ContentBlock]:
        """Smart entry point — accepts a path, directory, or FetchedDocument.

        Routing logic::

            FetchedDocument      → unwrap .path, then route as Path
            *_content_list.json  → parse JSON directly
            dir with JSON        → _read_ocr_dir()
            PDF / image file     → _run_mineru() → _read_ocr_dir()

        Args:
            source: A :class:`~openingestion.document.FetchedDocument`,
                    an absolute path to a PDF/image/JSON file, or a
                    MinerU output directory.

        Returns:
            List of :class:`~openingestion.document.ContentBlock` in
            reading order.

        Raises:
            ValueError: If *source* cannot be routed.
            FileNotFoundError: If the path does not exist.
        """
        # Unwrap FetchedDocument
        if isinstance(source, FetchedDocument):
            if source.path is None:
                raise ValueError(
                    f"FetchedDocument '{source.source}' has no local path. "
                    "Cloud/stream sources must be materialised before processing."
                )
            target = source.path
        else:
            target = Path(source)

        if not target.exists():
            raise FileNotFoundError(f"Source not found: {target}")

        # Route
        if target.is_file() and target.suffix == ".json":
            logger.info("MinerUChef: JSON file → {}", target)
            return self.parse_json(target)

        if _is_mineru_output_dir(target):
            logger.info("MinerUChef: MinerU output dir → {}", target)
            return self._read_ocr_dir(target)

        if _is_parseable(target):
            logger.info("MinerUChef: raw file ({}) → launching MinerU → {}", _mime(target), target)
            ocr_dir = self._run_mineru(target)
            return self._read_ocr_dir(ocr_dir)

        raise ValueError(
            f"Cannot process source: {target}\n"
            "Expected: MinerU output dir, *_content_list.json, PDF, or image file."
        )

    def process_batch(
        self,
        sources: list[str | os.PathLike | FetchedDocument],
    ) -> list[list[ContentBlock]]:
        """Process multiple sources in sequence.

        Args:
            sources: List of paths or :class:`~openingestion.document.FetchedDocument`.

        Returns:
            List of ContentBlock lists, one per source.
        """
        logger.info("MinerUChef batch: {} source(s)", len(sources))
        return [self.process(s) for s in sources]

    # ──────────────────────────────────────────────────────────────────────────
    # Low-level helpers
    # ──────────────────────────────────────────────────────────────────────────

    def _read_ocr_dir(self, ocr_path: Path) -> list[ContentBlock]:
        content_list_files = list(ocr_path.glob("*_content_list.json"))

        if not content_list_files:
            raise FileNotFoundError(
                f"No *_content_list.json found in: {ocr_path}"
            )
        if len(content_list_files) > 1:
            logger.warning(
                "Multiple _content_list.json found — using: {}",
                content_list_files[0],
            )

        return self.parse_json(content_list_files[0])

    def parse_json(self, json_path: str | Path) -> list[ContentBlock]:
        """Read a ``*_content_list.json`` file directly.

        Args:
            json_path: Path to the JSON file.

        Returns:
            List of :class:`~openingestion.document.ContentBlock`.
        """
        with open(json_path, encoding="utf-8") as f:
            raw_data = json.load(f)
        logger.info("MinerUChef: {} blocks from {}", len(raw_data), json_path)
        return self.map_to_blocks(raw_data)

    def _run_mineru(self, file_path: Path) -> Path:
        """Run MinerU on *file_path* and return the output directory.

        Args:
            file_path: Path to the PDF or image file.

        Returns:
            Path to the MinerU output directory containing
            ``*_content_list.json``.

        Raises:
            ImportError: If the ``mineru`` package is not installed.
        """
        if do_parse is None:
            raise ImportError(
                "The 'mineru' package is not installed or could not be imported. "
                "Either install it with `pip install mineru` or pre-process your "
                "files with MinerU and pass the output directory to process()."
            )

        self.output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(
            "MinerUChef: running MinerU (backend={}) on {} → {}",
            self.mineru_backend, file_path, self.output_dir,
        )

        do_parse(
            str(file_path),
            output_dir=str(self.output_dir),
            backend=self.mineru_backend,
        )

        # MinerU writes to output_dir/<stem>/auto/
        candidate = self.output_dir / file_path.stem / "auto"
        if not candidate.is_dir():
            # Fallback: search for any subdir with a content_list.json
            for sub in self.output_dir.rglob("*_content_list.json"):
                return sub.parent
            raise FileNotFoundError(
                f"MinerU ran but no *_content_list.json found under {self.output_dir}"
            )
        return candidate

    def map_to_blocks(self, raw_items: list[dict]) -> list[ContentBlock]:
        """Map raw content_list items to typed ContentBlocks.

        Args:
            raw_items: List of raw dicts from ``*_content_list.json``.

        Returns:
            List of ContentBlocks in reading order.
        """
        blocks: list[ContentBlock] = []
        page_counters: dict[int, int] = {}

        for idx, item in enumerate(raw_items):
            m_type = item.get("type", "text")

            kind = BlockKind.TEXT
            if m_type == "image":
                kind = BlockKind.IMAGE
            elif m_type == "table":
                kind = BlockKind.TABLE
            elif m_type == "discarded":
                kind = BlockKind.DISCARDED

            title_level = item.get("text_level") or 0
            if title_level > 0:
                kind = BlockKind.TITLE

            page = item.get("page_idx", 0)
            block_index = page_counters.get(page, 0)
            page_counters[page] = block_index + 1

            captions = (
                item.get("image_caption")
                or item.get("table_caption")
                or []
            )
            footnotes = (
                item.get("image_footnote")
                or item.get("table_footnote")
                or []
            )

            blocks.append(ContentBlock(
                kind=kind,
                text=item.get("text", ""),
                page_idx=page,
                bbox=item.get("bbox", [0, 0, 0, 0]),
                title_level=title_level,
                html=item.get("html", ""),
                img_path=item.get("img_path", ""),
                captions=captions,
                footnotes=footnotes,
                block_index=block_index,
                reading_order=idx,
                raw=item,
            ))

        logger.debug("map_to_blocks → {} ContentBlocks", len(blocks))
        return blocks


        # Si c'est un fichier JSON direct
        if ocr_path.is_file() and ocr_path.suffix == ".json":
            return self.parse_json(str(ocr_path))

        # Sinon on cherche *_content_list.json dans le dossier
        return self._read_ocr_dir(ocr_path)

    def _read_ocr_dir(self, ocr_path: Path) -> list[ContentBlock]:
        content_list_files = list(ocr_path.glob("*_content_list.json"))

        if not content_list_files:
            raise FileNotFoundError(
                f"Aucun fichier *_content_list.json trouvé dans : {ocr_path}"
            )
        if len(content_list_files) > 1:
            logger.warning(
                "Plusieurs _content_list.json trouvés — utilisation de : {}",
                content_list_files[0],
            )

        with open(content_list_files[0], encoding="utf-8") as f:
            raw_data = json.load(f)

        logger.info(
            "MinerUChef : {} blocs lus depuis {}", len(raw_data), content_list_files[0]
        )
        return self.map_to_blocks(raw_data)

    def parse_json(self, json_path: str) -> list[ContentBlock]:
        """Read a *_content_list.json file directly.

        Args:
            json_path: Absolute path to the JSON file.

        Returns:
            List of ContentBlocks.

        """
        with open(json_path, encoding="utf-8") as f:
            raw_data = json.load(f)
        logger.info("MinerUChef : {} blocs lus depuis {}", len(raw_data), json_path)
        return self.map_to_blocks(raw_data)

    def parse_file(self, pdf_path: str, output_dir: str = "output") -> list[ContentBlock]:
        """Run MinerU on a raw PDF file.

        Requires MinerU to be installed (mineru package).

        Args:
            pdf_path: Path to the PDF file.
            output_dir: Directory where MinerU will write its output.

        Returns:
            List of ContentBlocks.

        """
        if do_parse is None:
            raise ImportError(
                "MinerU n'est pas installé. "
                "Utilisez process() sur un dossier ocr existant."
            )
        raise NotImplementedError(
            "parse_file() sera complété lors de l'intégration MinerU CLI."
        )

    def map_to_blocks(self, raw_items: list[dict]) -> list[ContentBlock]:
        """Map raw content_list items to typed ContentBlocks.

        Args:
            raw_items: List of raw dicts from *_content_list.json.

        Returns:
            List of ContentBlocks in reading order.

        """
        blocks: list[ContentBlock] = []
        page_counters: dict[int, int] = {}

        for idx, item in enumerate(raw_items):
            m_type = item.get("type", "text")

            kind = BlockKind.TEXT
            if m_type == "image":
                kind = BlockKind.IMAGE
            elif m_type == "table":
                kind = BlockKind.TABLE
            elif m_type == "discarded":
                kind = BlockKind.DISCARDED

            title_level = item.get("text_level") or 0
            if title_level > 0:
                kind = BlockKind.TITLE

            page = item.get("page_idx", 0)
            block_index = page_counters.get(page, 0)
            page_counters[page] = block_index + 1

            captions = (
                item.get("image_caption")
                or item.get("table_caption")
                or []
            )
            footnotes = (
                item.get("image_footnote")
                or item.get("table_footnote")
                or []
            )

            blocks.append(ContentBlock(
                kind=kind,
                text=item.get("text", ""),
                page_idx=page,
                bbox=item.get("bbox", [0, 0, 0, 0]),
                title_level=title_level,
                html=item.get("html", ""),
                img_path=item.get("img_path", ""),
                captions=captions,
                footnotes=footnotes,
                block_index=block_index,
                reading_order=idx,
                raw=item,
            ))

        logger.debug("map_to_blocks → {} ContentBlocks", len(blocks))
        return blocks
