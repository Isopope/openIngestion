"""Vision Refinery \u2014 uses Vision LLMs to extract text/markdown from images/tables.

This refinery identifies chunk with missing or impoverished content (e.g. 
image-only tables from MinerU where HTML was lost, or actual document images
containing charts/diagrams), loads the associated image, and calls a multimodal
LLM (like GPT-4o-mini) to extract its content or describe it.

It should be placed **after** RagRefinery in the pipeline (so that ``img_path``
and ``image_b64`` are properly resolved/populated) and **before** ContextualRagRefinery.
"""
from __future__ import annotations

import base64
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from loguru import logger

from openingestion.document import BlockKind, RagChunk
from openingestion.refinery.base import BaseRefinery


class VisionRefinery(BaseRefinery):
    """Enriches visually-dependent chunks (tables, images) using a Vision LLM.
    
    Args:
        genie: An instantiated Genie that supports ``generate_vision`` 
               (e.g., :class:`~openingestion.genie.openai_genie.OpenAIGenie`).
        kinds: Set of :class:`BlockKind` to target. Defaults to ``{TABLE, IMAGE}``.
        only_if_empty: If ``True`` (default), skips tables that already have HTML
                       and only targets image-only tables (or decorative captions).
        image_detail: OpenAI detail mode for images (``"low"``, ``"high"``, or ``"auto"``).
                      Defaults to ``"low"`` to optimize costs (85 tokens/image).
        max_workers: Number of parallel LLM requests to make.
    """

    def __init__(
        self,
        genie,
        kinds: set[BlockKind] | None = None,
        only_if_empty: bool = True,
        image_detail: str = "low",
        max_workers: int = 5,
    ) -> None:
        self.genie = genie
        self.kinds = kinds or {BlockKind.TABLE, BlockKind.IMAGE}
        self.only_if_empty = only_if_empty
        self.image_detail = image_detail
        self.max_workers = max_workers

        # Target prompts based on the block type.
        self.prompts = {
            BlockKind.TABLE: (
                "Extract the data from this table and format it as a valid Markdown table. "
                "Do not add any surrounding text or explanations, just the Markdown table itself."
            ),
            BlockKind.IMAGE: (
                "Describe this image in detail. Extract any visible text, translate flowcharts "
                "into textual steps, and describe diagrams thoroughly."
            ),
        }

    def enrich(self, chunks: list[RagChunk]) -> list[RagChunk]:
        """Apply the Vision LLM to target chunks modifying them in place."""
        targets = [c for c in chunks if self._should_process(c)]
        
        if not targets:
            return chunks

        logger.info(f"VisionRefinery: analyzing {len(targets)} visual chunk(s) with Vision LLM...")

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # We map list to exhaust the generator and force execution.
            list(executor.map(self._process_chunk, targets))

        return chunks

    def _should_process(self, chunk: RagChunk) -> bool:
        """Decide if a chunk requires Vision LLM intervention."""
        if chunk.kind not in self.kinds:
            return False
            
        if self.only_if_empty:
            # If the page_content is just the inferred caption from RagRefinery,
            # or completely empty, we consider it "missing actual content".
            is_generic_caption = chunk.page_content == chunk.extras.get("inferred_caption", "")
            has_real_content = bool(chunk.page_content) and not is_generic_caption
            # Exception: if kind is TABLE but html is already rich, skip.
            # (Though if chunk.page_content is full, it already hits has_real_content)
            if has_real_content:
                return False
                
        # We can only process it if it has an image attached
        return "img_path" in chunk.extras or "image_b64" in chunk.extras

    def _process_chunk(self, chunk: RagChunk) -> None:
        """Execute the API call for a single chunk."""
        try:
            # 1. Obtain Base64 representation (either from extras or by reading the file)
            img_b64 = chunk.extras.get("image_b64")
            if not img_b64:
                img_path = chunk.extras.get("img_path")
                if img_path:
                    path_obj = Path(img_path)
                    if path_obj.exists():
                        mime = "image/png" if path_obj.suffix.lower() == ".png" else "image/jpeg"
                        with open(path_obj, "rb") as f:
                            encoded = base64.b64encode(f.read()).decode("utf-8")
                            img_b64 = f"data:{mime};base64,{encoded}"
                    else:
                        logger.warning(f"VisionRefinery: image path not found on disk: {img_path}")
            
            if not img_b64:
                return

            # 2. Query the LLM
            prompt = self.prompts.get(chunk.kind, self.prompts[BlockKind.IMAGE])
            
            if hasattr(self.genie, "generate_vision"):
                extracted_text = self.genie.generate_vision(
                    prompt=prompt, 
                    image_b64=img_b64, 
                    detail=self.image_detail,
                    system="You are an expert data extractor assistant."
                )
            else:
                logger.error("VisionRefinery: Provider Genie does not support generate_vision()")
                return
            
            # 3. Write back and clean up
            if extracted_text and extracted_text.strip():
                # On met à jour le contenu
                chunk.page_content = extracted_text.strip()
                
                # Optionnel: on pourrait recalculer le token_count / content_hash ici, 
                # mais ce sera fait proprement si on place VisionRefinery AVANT ContextualRagRefinery
                
                logger.debug(f"VisionRefinery: Successfully extracted content for chunk {chunk.chunk_index} ({chunk.kind.value})")

        except Exception as e:
            logger.error(f"VisionRefinery failed on chunk {chunk.chunk_index}: {e}")
