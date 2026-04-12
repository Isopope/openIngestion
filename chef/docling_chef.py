"""Docling Chef — spatial-aware PDF parser alternative to MinerUChef.

Produces identical :class:`~openingestion.document.ContentBlock` objects
(with normalized ``[x0, y0, x1, y1]`` bboxes in [0, 1000]) so all
downstream chunkers, refineries and porters work unchanged.

Coordinate normalization
------------------------
Docling uses a **bottom-left** origin (PDF convention).  This chef flips
the y-axis so that ``y0`` is the **top** of the box — matching MinerU's
convention — and normalizes all coordinates to [0, 1000] relative to page
dimensions.

Usage::

    from openingestion.chef.docling_chef import DoclingChef

    chef = DoclingChef()
    blocks = chef.process("rapport.pdf")

Or via the top-level :func:`~openingestion.ingest` helper::

    from openingestion import ingest

    chunks = ingest("rapport.pdf", parser="docling", strategy="by_token")
"""
from __future__ import annotations

import mimetypes
import os
from pathlib import Path
from typing import Any

from loguru import logger

from openingestion.document import BlockKind, ContentBlock, FetchedDocument
from openingestion.chef.base import BaseChef

# ---------------------------------------------------------------------------
# Optional Docling import — kept lazy to avoid loading heavy ML deps at import time
# ---------------------------------------------------------------------------
try:
    import importlib.util as _util
    _DOCLING_AVAILABLE = _util.find_spec("docling") is not None
    DocumentConverter = None  # loaded on first use inside _get_converter()
except Exception:
    _DOCLING_AVAILABLE = False
    DocumentConverter = None  # type: ignore[assignment,misc]

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_SUPPORTED_MIME = {
    "application/pdf",
    "image/png",
    "image/jpeg",
    "image/jpg",
    "image/tiff",
    "image/bmp",
    "image/webp",
}

# Docling DocItemLabel value → BlockKind
_LABEL_TO_KIND: dict[str, BlockKind] = {
    "title":             BlockKind.TITLE,
    "section_header":    BlockKind.TITLE,
    "text":              BlockKind.TEXT,
    "paragraph":         BlockKind.TEXT,
    "list_item":         BlockKind.TEXT,
    "code":              BlockKind.TEXT,
    "caption":           BlockKind.TEXT,
    "footnote":          BlockKind.TEXT,
    "form":              BlockKind.TEXT,
    "key_value_region":  BlockKind.TEXT,
    "table":             BlockKind.TABLE,
    "picture":           BlockKind.IMAGE,
    "figure":            BlockKind.IMAGE,
    "formula":           BlockKind.EQUATION,
    "equation":          BlockKind.EQUATION,
    "page_header":       BlockKind.DISCARDED,
    "page_footer":       BlockKind.DISCARDED,
    "watermark":         BlockKind.DISCARDED,
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _label_str(label) -> str:
    """Normalise a DocItemLabel (enum or str) to a lowercase plain string."""
    if hasattr(label, "value"):
        return str(label.value).lower()
    raw = str(label).lower()
    # Strip possible "docitemlabel." prefix
    for prefix in ("docitemlabel.", "labeltype."):
        if raw.startswith(prefix):
            return raw[len(prefix):]
    return raw


def _normalize_bbox(
    bbox,
    page_w: float,
    page_h: float,
) -> list[int]:
    """Convert a Docling BoundingBox → MinerU-style [x0, y0, x1, y1] in [0, 1000].

    Docling PDFs use **bottom-left** origin (``CoordOrigin.BOTTOMLEFT``).
    We flip the y-axis so that y0 is the visual *top* of the box, consistent
    with MinerU's convention.

    Args:
        bbox:   Docling ``BoundingBox`` (attributes ``l``, ``t``, ``r``, ``b``
                and ``coord_origin``).
        page_w: Page width in pts (from ``doc.pages[n].size.width``).
        page_h: Page height in pts.

    Returns:
        ``[x0, y0, x1, y1]`` each in [0, 1000].
    """
    w = max(page_w, 1.0)
    h = max(page_h, 1.0)

    l, t, r, b = bbox.l, bbox.t, bbox.r, bbox.b

    # Detect coordinate origin
    try:
        origin_str = str(bbox.coord_origin).upper()
        is_bottom_left = "BOTTOMLEFT" in origin_str
    except Exception:
        is_bottom_left = True  # PDF default

    if is_bottom_left:
        # t > b in bottom-left coords (t is closer to the top of the page)
        # After flip: visual y0 (top of box) = h - t,  y1 (bottom) = h - b
        # But if l/b/r/t are in PDF pts, t may be the TOP y in PDF space
        # Docling: l=left, t=top, r=right, b=bottom, all measured from
        # BOTTOM-LEFT corner → t > b always.
        # Visual top  = page_h - t  (small number → near top of rendered page)
        # Visual bottom = page_h - b
        y0 = (h - t) / h * 1000
        y1 = (h - b) / h * 1000
    else:
        y0 = t / h * 1000
        y1 = b / h * 1000

    x0 = l / w * 1000
    x1 = r / w * 1000

    return [int(x0), int(y0), int(x1), int(y1)]


# ---------------------------------------------------------------------------
# DoclingChef
# ---------------------------------------------------------------------------

class DoclingChef(BaseChef):
    """Chef that uses **Docling** (IBM) to parse PDFs and images.

    Unlike :class:`MinerUChef`, this chef does **not** require MinerU or
    any heavy GPU models by default.  Docling's layout analysis is
    CPU-friendly while still providing:

    * Accurate bounding boxes normalised to ``[0, 1000]`` (same as MinerU)
    * Table detection with HTML export
    * Section hierarchy and heading levels
    * Reading-order preservation

    The output ``ContentBlock`` objects are **drop-in compatible** with all
    downstream chunkers, refineries, and porters.

    Args:
        pipeline_options: Optional ``PdfPipelineOptions`` instance for
                          fine-grained Docling pipeline configuration
                          (e.g. enabling OCR, table structure model, etc.).

    Example::

        chef = DoclingChef()
        blocks = chef.process("rapport.pdf")

        # With custom options
        from docling.datamodel.pipeline_options import PdfPipelineOptions
        opts = PdfPipelineOptions()
        opts.do_ocr = True
        chef = DoclingChef(pipeline_options=opts)
    """

    def __init__(self, pipeline_options=None) -> None:
        if not _DOCLING_AVAILABLE:
            raise ImportError(
                "docling is not installed. "
                "Install it with:  pip install docling"
            )
        self._pipeline_options = pipeline_options
        self._converter: Any = None  # lazy init

    # ------------------------------------------------------------------
    # BaseChef interface
    # ------------------------------------------------------------------

    def process(
        self,
        source: str | os.PathLike | FetchedDocument,
    ) -> list[ContentBlock]:
        """Parse a PDF or image with Docling and return ContentBlocks.

        Args:
            source: Path to a PDF/image file or a
                    :class:`~openingestion.document.FetchedDocument`.

        Returns:
            List of :class:`~openingestion.document.ContentBlock` in
            reading order, each carrying normalised spatial coordinates.

        Raises:
            ImportError: If *docling* is not installed.
            FileNotFoundError: If the file does not exist.
            ValueError: If the file MIME type is not supported.
        """
        # --- unwrap FetchedDocument ---
        if isinstance(source, FetchedDocument):
            if source.path is None:
                raise ValueError(
                    f"FetchedDocument '{source.source}' has no local path. "
                    "Materialise cloud sources before calling DoclingChef."
                )
            target = source.path
        else:
            target = Path(source)

        if not target.exists():
            raise FileNotFoundError(f"Source not found: {target}")

        mime, _ = mimetypes.guess_type(str(target))
        if mime not in _SUPPORTED_MIME:
            raise ValueError(
                f"DoclingChef: unsupported file type '{mime}' for {target}.\n"
                f"Supported MIME types: {sorted(_SUPPORTED_MIME)}"
            )

        logger.info("DoclingChef: parsing {} ({})", target.name, mime)
        converter = self._get_converter()
        result = converter.convert(str(target))
        doc = result.document

        blocks = self._doc_to_blocks(doc)
        logger.info(
            "DoclingChef: {} blocks extracted from {}", len(blocks), target.name
        )
        return blocks

    def map_to_blocks(self, raw_items: list[dict]) -> list[ContentBlock]:
        """Map pre-exported Docling-style dicts to ContentBlocks.

        Useful for re-ingesting a previously serialised Docling output
        without re-running the converter.

        Args:
            raw_items: List of dicts with keys ``label``, ``text``,
                       ``page_idx``, ``bbox``, ``title_level``, ``html``,
                       ``captions``.

        Returns:
            List of :class:`~openingestion.document.ContentBlock`.
        """
        blocks: list[ContentBlock] = []
        for idx, item in enumerate(raw_items):
            label = item.get("label", "text")
            kind = _LABEL_TO_KIND.get(label, BlockKind.TEXT)
            blocks.append(ContentBlock(
                kind=kind,
                text=item.get("text", ""),
                page_idx=item.get("page_idx", 0),
                bbox=item.get("bbox", [0, 0, 0, 0]),
                title_level=item.get("title_level", 0),
                html=item.get("html", ""),
                img_path=item.get("img_path", ""),
                captions=item.get("captions", []),
                footnotes=item.get("footnotes", []),
                block_index=idx,
                reading_order=idx,
                raw=item,
            ))
        return blocks

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_converter(self) -> Any:
        """Lazily instantiate the Docling DocumentConverter."""
        if self._converter is None:
            from docling.document_converter import DocumentConverter as _DC  # type: ignore[import]
            if self._pipeline_options is not None:
                from docling.document_converter import PdfFormatOption  # type: ignore[import]
                from docling.datamodel.base_models import InputFormat  # type: ignore[import]
                self._converter = _DC(
                    format_options={
                        InputFormat.PDF: PdfFormatOption(
                            pipeline_options=self._pipeline_options
                        )
                    }
                )
            else:
                self._converter = _DC()
        return self._converter

    def _doc_to_blocks(self, doc) -> list[ContentBlock]:
        """Convert a ``DoclingDocument`` to a flat list of ``ContentBlock``.

        Iterates over the document in reading order.  For items that span
        multiple pages, one ``ContentBlock`` is emitted per provenance
        entry so that each block carries a single, accurate bbox.
        """
        blocks: list[ContentBlock] = []
        reading_order = 0

        for item, level in doc.iterate_items():
            prov_list = getattr(item, "prov", []) or []
            label_str = _label_str(getattr(item, "label", "text"))
            kind = _LABEL_TO_KIND.get(label_str, BlockKind.TEXT)

            # ── title level ──────────────────────────────────────────────
            title_level = 0
            if label_str == "title":
                title_level = 1
            elif label_str == "section_header":
                # Docling `level` is the 0-based depth in the doc tree
                title_level = min((level or 0) + 2, 6)

            # ── text ─────────────────────────────────────────────────────
            text = getattr(item, "text", "") or ""

            # ── HTML (tables) ────────────────────────────────────────────
            html = ""
            if kind == BlockKind.TABLE:
                try:
                    html = item.export_to_html(doc)
                except Exception:
                    try:
                        html = item.export_to_markdown()
                    except Exception:
                        pass

            # ── captions ─────────────────────────────────────────────────
            captions: list[str] = []
            for cap in (getattr(item, "captions", None) or []):
                cap_text = getattr(cap, "text", "") or str(cap)
                if cap_text:
                    captions.append(cap_text)

            # ── footnotes ────────────────────────────────────────────────
            footnotes: list[str] = []
            for fn in (getattr(item, "footnotes", None) or []):
                fn_text = getattr(fn, "text", "") or str(fn)
                if fn_text:
                    footnotes.append(fn_text)

            # ── image path (pictures saved by Docling pipeline) ───────────
            img_path = ""
            if kind == BlockKind.IMAGE:
                try:
                    uri = getattr(getattr(item, "image", None), "uri", None)
                    if uri:
                        img_path = str(uri)
                except Exception:
                    pass

            # ── raw dict for round-tripping ───────────────────────────────
            base_raw: dict = {"label": label_str, "text": text}

            # ── emit one block per provenance (= per page span) ───────────
            if not prov_list:
                blocks.append(ContentBlock(
                    kind=kind,
                    text=text,
                    page_idx=0,
                    bbox=[0, 0, 0, 0],
                    title_level=title_level,
                    html=html,
                    img_path=img_path,
                    captions=captions,
                    footnotes=footnotes,
                    block_index=len(blocks),
                    reading_order=reading_order,
                    raw=base_raw,
                ))
                reading_order += 1
                continue

            for prov in prov_list:
                page_no = prov.page_no   # 1-based in Docling
                page_idx = page_no - 1   # 0-based for ContentBlock

                # Page dimensions for normalisation
                page_size = None
                if hasattr(doc, "pages") and page_no in doc.pages:
                    page_size = doc.pages[page_no].size

                if page_size and getattr(page_size, "width", 0) and getattr(page_size, "height", 0):
                    bbox = _normalize_bbox(prov.bbox, page_size.width, page_size.height)
                else:
                    bbox = [0, 0, 0, 0]

                raw = {
                    **base_raw,
                    "page_no": page_no,
                    "bbox_raw": {
                        "l": prov.bbox.l,
                        "t": prov.bbox.t,
                        "r": prov.bbox.r,
                        "b": prov.bbox.b,
                    },
                }

                blocks.append(ContentBlock(
                    kind=kind,
                    text=text,
                    page_idx=page_idx,
                    bbox=bbox,
                    title_level=title_level,
                    html=html,
                    img_path=img_path,
                    captions=captions,
                    footnotes=footnotes,
                    block_index=len(blocks),
                    reading_order=reading_order,
                    raw=raw,
                ))

            reading_order += 1

        return blocks
