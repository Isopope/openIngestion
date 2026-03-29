"""draw_chunks — annotate a PDF with RAG chunk bounding boxes.

Produces a ``*_chunks.pdf`` where every :class:`~openingestion.document.RagChunk`
is drawn on its source page with a colour-coded filled rectangle and its
``chunk_index`` number, exactly like MinerU's ``*_layout.pdf`` but at the
**chunk level** rather than the raw-block level.

Coordinate systems
------------------
Two coordinate systems are handled transparently:

* **MinerU** — ``ContentBlock.bbox`` is in **PDF points** (pts), top-left origin,
  i.e. values typically in ``[0, page_height]``.  Auto-detected when any
  coordinate exceeds ``1001``.
* **Docling / normalised** — ``ContentBlock.bbox`` is in **[0, 1000]** with a
  top-left origin (as produced by :class:`~openingestion.chef.DoclingChef`).

The function accepts an explicit ``normalized`` flag or auto-detects.

Usage::

    from openingestion import ingest
    from openingestion.utils.draw_chunks import draw_chunks_on_pdf

    chunks = ingest("rapport.pdf", parser="docling", strategy="by_token")
    out = draw_chunks_on_pdf(chunks, "rapport.pdf")
    print("Annotated PDF →", out)   # rapport_chunks.pdf
"""
from __future__ import annotations

import os
from io import BytesIO
from pathlib import Path

from loguru import logger

from openingestion.document import BlockKind, RagChunk

# ── colour palette per BlockKind (R, G, B) ──────────────────────────────────
_KIND_RGB: dict[BlockKind, tuple[int, int, int]] = {
    BlockKind.TEXT:      (60,  120, 220),   # blue
    BlockKind.TITLE:     (220, 100,  20),   # orange
    BlockKind.TABLE:     (180, 180,   0),   # yellow
    BlockKind.IMAGE:     ( 60, 180,  60),   # green
    BlockKind.EQUATION:  (  0, 180, 180),   # cyan
    BlockKind.DISCARDED: (140, 140, 140),   # grey
}
_DEFAULT_RGB = (180, 60, 180)   # purple fallback


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def draw_chunks_on_pdf(
    chunks: list[RagChunk],
    pdf_path: str | os.PathLike,
    output_path: str | os.PathLike | None = None,
    *,
    normalized: bool | None = None,
) -> Path:
    """Draw chunk bounding boxes on the original PDF.

    Args:
        chunks:      List of :class:`~openingestion.document.RagChunk` as
                     returned by :func:`~openingestion.ingest`.
        pdf_path:    Path to the **original** PDF file.
        output_path: Where to write the annotated PDF.  Defaults to
                     ``<stem>_chunks.pdf`` next to the original.
        normalized:  ``True``  → coordinates are in [0, 1000] (Docling).
                     ``False`` → coordinates are in PDF pts (MinerU).
                     ``None``  → auto-detect from the data.

    Returns:
        Path to the annotated PDF file.

    Raises:
        ImportError: If ``pypdf`` or ``reportlab`` are not installed.
        FileNotFoundError: If ``pdf_path`` does not exist.
    """
    try:
        from pypdf import PdfReader, PdfWriter, PageObject
        from reportlab.pdfgen import canvas as rl_canvas
    except ImportError as exc:
        raise ImportError(
            "draw_chunks_on_pdf requires pypdf and reportlab:\n"
            "  pip install pypdf reportlab"
        ) from exc

    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    if output_path is None:
        output_path = pdf_path.with_name(pdf_path.stem + "_chunks.pdf")
    output_path = Path(output_path)

    # --- detect coordinate system -------------------------------------------
    if normalized is None:
        normalized = _auto_detect_normalized(chunks)
    logger.info(
        "draw_chunks_on_pdf: {} chunks, coord={}, pdf={}",
        len(chunks), "normalized[0-1000]" if normalized else "pts",
        pdf_path.name,
    )

    # --- build per-page index: page_idx → list[(chunk_index, kind, bbox)] ----
    # bbox here is always in the source coord system
    per_page: dict[int, list[tuple[int, BlockKind, list[int]]]] = {}
    for chunk in chunks:
        for entry in chunk.position_int:
            if len(entry) < 5:
                continue
            page_idx, x0, y0, x1, y1 = entry[0], entry[1], entry[2], entry[3], entry[4]
            per_page.setdefault(page_idx, []).append(
                (chunk.chunk_index, chunk.kind, [x0, y0, x1, y1])
            )

    # --- open source PDF ----------------------------------------------------
    reader = PdfReader(str(pdf_path))
    writer = PdfWriter()

    for page_idx, page in enumerate(reader.pages):
        page_width  = float(page.cropbox[2])
        page_height = float(page.cropbox[3])

        rotation_obj = page.get("/Rotate", 0)
        try:
            rotation = int(rotation_obj) % 360
        except (ValueError, TypeError):
            rotation = 0

        # swap dims for rotated pages
        if rotation in (90, 270):
            page_width, page_height = page_height, page_width

        packet = BytesIO()
        c = rl_canvas.Canvas(packet, pagesize=(page_width, page_height))

        entries = per_page.get(page_idx, [])
        for chunk_idx, kind, bbox in entries:
            rgb = _KIND_RGB.get(kind, _DEFAULT_RGB)
            r, g, b = [v / 255.0 for v in rgb]

            rl_rect = _to_reportlab_rect(
                bbox, page_width, page_height, rotation, normalized
            )
            if rl_rect is None:
                continue
            rx, ry, rw, rh = rl_rect

            # filled semi-transparent rectangle
            c.setFillColorRGB(r, g, b, 0.25)
            c.rect(rx, ry, rw, rh, stroke=0, fill=1)

            # coloured border
            c.setStrokeColorRGB(r, g, b, 0.9)
            c.setLineWidth(1.0)
            c.rect(rx, ry, rw, rh, stroke=1, fill=0)

            # chunk index label (top-right corner of the box)
            c.setFillColorRGB(r, g, b, 1.0)
            c.setFontSize(8)
            label = str(chunk_idx)
            c.saveState()
            if rotation == 0:
                c.translate(rx + rw + 1, ry + rh - 8)
            elif rotation == 90:
                c.translate(rx + 8, ry + rh + 1)
            elif rotation == 180:
                c.translate(rx - 1, ry + 8)
            elif rotation == 270:
                c.translate(rx + rw - 8, ry - 1)
            c.rotate(rotation)
            c.drawString(0, 0, label)
            c.restoreState()

        c.save()
        packet.seek(0)

        from pypdf import PdfReader as _PR
        overlay_reader = _PR(packet)
        if overlay_reader.pages:
            new_page = PageObject(pdf=None)
            new_page.update(page)
            new_page.merge_page(overlay_reader.pages[0])
            writer.add_page(new_page)
        else:
            writer.add_page(page)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "wb") as f:
        writer.write(f)

    logger.info("Annotated PDF written → {}", output_path)
    return output_path


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

def _auto_detect_normalized(chunks: list[RagChunk]) -> bool:
    """Return True if all bbox coordinates appear to be in [0, 1000].

    Heuristic: if any coordinate value exceeds 1001 we assume PDF-pts.
    """
    for chunk in chunks:
        for entry in chunk.position_int:
            if len(entry) >= 5:
                if any(v > 1001 for v in entry[1:5]):
                    return False
    return True


def _to_reportlab_rect(
    bbox: list[int],
    page_width: float,
    page_height: float,
    rotation: int,
    normalized: bool,
) -> tuple[float, float, float, float] | None:
    """Convert a [x0, y0, x1, y1] bbox to reportlab (x, y, w, h).

    Reportlab uses bottom-left origin. Both MinerU and Docling use
    top-left origin — the only difference is the unit (pts vs [0, 1000]).

    Args:
        bbox:        [x0, y0, x1, y1] in source coordinate system.
        page_width:  Page width in pts (after rotation swap if any).
        page_height: Page height in pts.
        rotation:    Page /Rotate value (0, 90, 180, 270).
        normalized:  True → [0, 1000]; False → pts.

    Returns:
        ``(x, y_bottom, width, height)`` for reportlab, or ``None`` if
        the bbox is degenerate.
    """
    x0, y0, x1, y1 = bbox

    if normalized:
        # Scale [0, 1000] → pts (top-left origin → bottom-left for reportlab)
        x0_pts = x0 / 1000.0 * page_width
        x1_pts = x1 / 1000.0 * page_width
        # y0 is visual top, y1 is visual bottom (in [0,1000], larger = lower)
        # reportlab y0 (bottom of box) = page_height - (y1/1000 * page_height)
        rl_y0  = page_height - (y1 / 1000.0 * page_height)
        rl_y1  = page_height - (y0 / 1000.0 * page_height)
    else:
        # MinerU pts — top-left origin, y0 = visual top in pts
        x0_pts = float(x0)
        x1_pts = float(x1)
        rl_y0  = page_height - float(y1)   # reportlab bottom
        rl_y1  = page_height - float(y0)   # reportlab top

    rw = abs(x1_pts - x0_pts)
    rh = abs(rl_y1 - rl_y0)

    if rw < 1 or rh < 1:
        return None

    # Handle rotated pages
    if rotation == 90:
        rl_y0, x0_pts = x0_pts, rl_y0
        rw, rh = rh, rw
    elif rotation == 180:
        x0_pts = page_width - x1_pts
    elif rotation == 270:
        rw, rh = rh, rw
        x0_pts = page_height - rl_y1
        rl_y0  = page_width  - x1_pts

    return (x0_pts, rl_y0, rw, rh)
