"""MinerU Chef — CHOMP step 1 implementation."""
from __future__ import annotations

import json
import os
from pathlib import Path

from loguru import logger

from openingestion.document import BlockKind, ContentBlock
from openingestion.chef.base import BaseChef

try:
    from mineru.cli.common import do_parse
except ImportError:
    do_parse = None


class MinerUChef(BaseChef):
    """Chef that reads MinerU output directories and maps them to ContentBlocks.

    Supports two entry points:
        - process(ocr_dir)   : reads an existing *_content_list.json
        - parse_json(path)   : reads a specific JSON file directly
        - parse_file(pdf)    : runs MinerU on a raw PDF (requires MinerU installed)
    """

    def process(self, path: str | os.PathLike) -> list[ContentBlock]:
        """Process a MinerU output directory.

        Args:
            path: Path to the 'ocr' directory produced by MinerU.

        Returns:
            List of ContentBlocks.

        """
        ocr_path = Path(path)

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
