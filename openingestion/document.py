from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional


class BlockKind(str, Enum):
    """Types de blocs supportés par openingestion."""
    TEXT = "text"
    TITLE = "title"
    IMAGE = "image"
    TABLE = "table"
    EQUATION = "equation"
    DISCARDED = "discarded"


@dataclass(frozen=True)
class ContentBlock:
    """
    Unité atomique extraite par MinerU (CHOMP : étape Chef).
    Immuable — représentation fidèle d'un item content_list.json.
    """
    kind: BlockKind
    text: str
    page_idx: int           # 0-based
    bbox: list[int]         # [x0, y0, x1, y1] normalisé [0-1000]

    # Hiérarchie
    title_level: int = 0    # 0 = pas un titre, 1-4 = niveau heading

    # Contenu riche (présent selon kind)
    html: str = ""          # Tables
    img_path: str = ""      # Chemin relatif issu de MinerU (images/hash.jpg)

    # Annexes
    captions: list[str] = field(default_factory=list)
    footnotes: list[str] = field(default_factory=list)

    # Traçabilité
    block_index: int = 0    # Index dans la page
    reading_order: int = 0  # Index global dans content_list.json — assigné par le Chef
    raw: dict[str, Any] = field(default_factory=dict)  # Item brut original


@dataclass
class RagChunk:
    """
    Unité RAG prête pour l'embedding / retrieval (CHOMP : sortie REfinery).
    Mutable pour permettre l'enrichissement progressif.
    """
    page_content: str
    source: str             # Chemin absolu du document source (résolu à l'entrée)

    # Classification
    kind: BlockKind
    title_path: str = ""    # "H1 > H2 > ..." contexte hiérarchique
    title_level: int = 0    # Niveau du titre parent (0 = aucun)

    # Spatial awareness — [[page_idx, x0, y0, x1, y1], ...]
    # JSON-safe : listes de listes (pas de tuples)
    position_int: list[list[int]] = field(default_factory=list)

    # Métadonnées riches (vide {} pour chunks texte bruts)
    # kind=table  → {"html": "<table>..."}
    # kind=image  → {"img_path": "/abs/...", "captions": [...]}
    # kind=*      → {"inferred_caption": "..."} si captions vides
    extras: dict[str, Any] = field(default_factory=dict)

    # Navigation
    chunk_index: int = 0    # Index 0-based dans le document
    block_indices: list[int] = field(default_factory=list)
    reading_order: int = 0  # reading_order du premier bloc fusionné

    # Liens (calculés en passe finale du CHUNKer)
    prev_chunk_index: Optional[int] = None
    next_chunk_index: Optional[int] = None

    # Enrichissements (calculés par la RagRefinery)
    token_count: int = 0
    content_hash: str = ""
