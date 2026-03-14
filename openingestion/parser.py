from __future__ import annotations
import json
import logging
from pathlib import Path
from typing import List, Optional

try:
    from mineru.cli.common import do_parse
except ImportError:
    do_parse = None

from .document import ContentBlock, BlockKind

logger = logging.getLogger(__name__)


class MinerUChef:
    """
    CHOMP étape 1 — Chef.
    Transforme un document brut ou un dossier d'output MinerU en ContentBlocks.
    """

    def parse_file(self, pdf_path: str, output_dir: str = "output") -> List[ContentBlock]:
        """
        Exécute MinerU sur un fichier PDF et retourne les blocs.
        Requiert MinerU installé.
        """
        if do_parse is None:
            raise ImportError(
                "MinerU n'est pas installé. "
                "Utilisez parse_output_dir() si les fichiers JSON sont déjà générés."
            )
        # Intégration do_parse à affiner selon la signature exacte de la version MinerU utilisée
        raise NotImplementedError("parse_file() sera complétée dans le prochain sprint.")

    def parse_output_dir(self, ocr_dir: str) -> List[ContentBlock]:
        """
        Lit un répertoire 'ocr' déjà généré par MinerU.
        Cherche le fichier *_content_list.json.
        """
        ocr_path = Path(ocr_dir)
        content_list_files = list(ocr_path.glob("*_content_list.json"))

        if not content_list_files:
            raise FileNotFoundError(
                f"Aucun fichier *_content_list.json trouvé dans : {ocr_dir}"
            )

        if len(content_list_files) > 1:
            logger.warning(
                "%d fichiers _content_list.json trouvés — on utilise : %s",
                len(content_list_files), content_list_files[0]
            )

        with open(content_list_files[0], encoding="utf-8") as f:
            raw_data = json.load(f)

        logger.info("Chef : %d blocs lus depuis %s", len(raw_data), content_list_files[0])
        return self.map_to_blocks(raw_data)

    def parse_json(self, json_path: str) -> List[ContentBlock]:
        """Lit directement un fichier *_content_list.json."""
        with open(json_path, encoding="utf-8") as f:
            raw_data = json.load(f)
        return self.map_to_blocks(raw_data)

    def map_to_blocks(self, raw_items: List[dict]) -> List[ContentBlock]:
        """
        Mappe les dicts bruts de MinerU en objets ContentBlock typés.
        Une règle simple : si text_level > 0 → TITLE.
        """
        blocks: List[ContentBlock] = []
        page_counters: dict[int, int] = {}   # page_idx → compteur de blocks dans la page

        for idx, item in enumerate(raw_items):
            m_type = item.get("type", "text")

            kind = BlockKind.TEXT
            if m_type == "image":
                kind = BlockKind.IMAGE
            elif m_type == "table":
                kind = BlockKind.TABLE
            elif m_type == "discarded":
                kind = BlockKind.DISCARDED

            title_level = item.get("text_level", 0) or 0
            if title_level > 0:
                kind = BlockKind.TITLE

            page = item.get("page_idx", 0)
            block_index = page_counters.get(page, 0)
            page_counters[page] = block_index + 1

            # Captions : images et tables ont des champs différents
            captions = (
                item.get("image_caption") or
                item.get("table_caption") or
                []
            )
            footnotes = (
                item.get("image_footnote") or
                item.get("table_footnote") or
                []
            )

            block = ContentBlock(
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
            )
            blocks.append(block)

        logger.debug("Chef : map_to_blocks → %d ContentBlocks", len(blocks))
        return blocks
