# openingestion — Spécifications Techniques v3

*Statut: En cours d'implémentation — flux CHOMP*
*Basé sur: analyse empirique des outputs MinerU (document Alternance_260314_120625)*

---

## 1. Vision

Créer une **bibliothèque Python** au-dessus de MinerU qui :

- Expose les sorties MinerU comme **chunks RAG exploitables**
- **Préserve la richesse géométrique et sémantique** de MinerU au lieu de l'abstraire
- Offre une **API simple** pour le cas courant et un **contrôle fin** pour les besoins avancés
- Reste **agnostique du framework** (pas de dépendance dure sur LangChain ou LlamaIndex)

> **Principe fondateur** : MinerU ne produit pas juste du texte — il produit de la connaissance
> spatiale et structurelle sur chaque fragment du document. C'est un asset stratégique pour
> la qualité RAG. Notre lib doit le préserver, pas le jeter.

---

## 2. Fondements Empiriques

*Issus de l'analyse de `output/Alternance_260314_120625/ocr/`*

### 2.1 Les fichiers produits par MinerU

| Fichier | Contenu | Usage dans notre lib |
|---|---|---|
| `*_content_list.json` | Array de blocs typés (texte, image, table) | **SEUL FICHIER LU** |
| `*_middle.json` | Structure fine : pages → para_blocks → lines → spans | **Non utilisé** |
| `*.md` | Rendu markdown flat | Référence / debug |
| `*_model.json` | Sorties brutes modèles de détection | Non utilisé |
| `images/` | Images extraites (hash.jpg) | Référencées via img_path |

### 2.2 Structure d'un item content_list

```json
{
  "type": "text",                  // "text" | "image" | "table" | "discarded"
  "text": "Amélioration et...",    // Contenu textuel
  "text_level": 1,                 // Présent si titre (1, 2, 3...)
  "bbox": [100, 243, 965, 379],    // [x0, y0, x1, y1] normalisé [0-1000]
  "page_idx": 0,                   // 0-based

  // Pour images:
  "img_path": "images/hash.jpg",
  "image_caption": [],
  "image_footnote": [],

  // Pour tables:
  "html": "<table>...</table>",    // HTML du tableau (si reconnu)
  "table_caption": [],
  "table_footnote": []
}
```

### 2.3 Observations clés sur le document réel

- ~2776 items content_list pour ~432 lignes markdown
- Beaucoup de **fragments courts** ("Vespa", "FastAPI") → chunking critique
- **Type "discarded"** = headers/footers → à filtrer ou garder séparément
- **`page_size` et `middle.json` non utilisés** : coordonnées normalisées [0-1000] suffisent
- `text_level` observé : 1 (pas encore vu >1 mais structure supportée)

---

## 3. Architecture — Flux CHOMP

L'architecture suit le flux **CHOMP** (inspiré de Chonkie) : chaque étape est indépendante, testable et composable.

```
 Document PDF/Image
       │
       ▼
┌─────────────────┐
│  1. MinerUChef  │  parser.py       → ContentBlock[]
│  (CHef)         │  Pilote MinerU, isole tables/images
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  2. CHUNKer     │  chunker.py      → RagChunk[] bruts
│                 │  by_section, by_page, by_block, fixed_size
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  3. REfinery    │  enricher.py     → RagChunk[] enrichis
│                 │  tokens, hash, img absolu, captions
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  4. Porter /    │  adapters.py     → format cible
│     Handshake   │  to_dicts(), to_langchain(), to_llamaindex()
└─────────────────┘

+ ingest() = shortcut complet Chef → CHUNKer → REfinery → Porter
```

**Responsabilités par étape :**

| Étape CHOMP | Composant | Produit | Champs calculés |
|---|---|---|---|
| 1. Chef | `parser.py` — `MinerUChef` | `ContentBlock[]` | `reading_order` (index dans content_list) |
| 2. CHUNKer | `chunker.py` | `RagChunk[]` bruts | `title_path`, `block_indices`, `chunk_index`, `position_int`, `prev/next` |
| 3. REfinery | `enricher.py` — `RagRefinery` | `RagChunk[]` enrichis | `token_count`, `content_hash`, `img_path` absolu |
| 4. Porter | `adapters.py` | format cible | `to_dicts()`, `to_langchain()`, `to_llamaindex()` |

**Principes** :
- Chaque étape est indépendante et testable unitairement
- Composable : court-circuiter n'importe quelle étape
- Extensible : enrichissement custom entre REfinery et Porter
- `ingest()` = raccourci clé-en-main, sans sur-ingénierie (pas de DAG/engine)

---

## 4. Data Model

### 4.1 ContentBlock — Représentation fidèle d'un item MinerU

```python
@dataclass
class ContentBlock:
    """
    Représentation directe d'un item content_list MinerU.
    Toutes les données originales sont préservées.
    """
    kind: BlockKind            # Enum: TEXT, TITLE, IMAGE, TABLE, EQUATION,
                               #       HEADER, FOOTER, FOOTNOTE, DISCARDED
    text: str                  # Contenu textuel (vide si image pure)
    page_idx: int              # Page 0-based
    bbox: list[int]            # [x0, y0, x1, y1] normalisé [0-1000]

    # Hiérarchie
    title_level: int           # 0 = pas titre, 1-4 = niveau heading

    # Contenu riche
    html: str                  # HTML pour les tables
    img_path: str              # Chemin relatif issu de MinerU (ex: images/hash.jpg)
                               # → résolu en absolu par la REfinery dans extras
    captions: list[str]        # Captions (images et tables)
    footnotes: list[str]       # Footnotes (images et tables)

    # Traçabilité
    block_index: int           # Index du bloc dans la page (0-based)
    reading_order: int         # ← PARSER : index dans content_list.json
                               # = position absolue dans le flux de lecture MinerU
                               # utile pour reconstruire l'ordre après filtrage
    raw: dict                  # Item original complet depuis content_list.json
```

### 4.2 RagChunk — Unité RAG, riche en métadonnées

```python
@dataclass
class RagChunk:
    """
    Une unité de texte prête pour l'embedding / retrieval,
    avec toutes les métadonnées de traçabilité et spatial awareness.
    """
    page_content: str          # Texte à embedder

    # --- Provenance ---
    source: str                # Chemin fichier source

    # --- Sémantique ---
    kind: BlockKind            # str-enum : chunk.kind == "image" fonctionne toujours
    title_path: str            # "Section > Sous-section > ..."
    title_level: int           # Niveau du titre parent (0 si aucun)

    # --- Spatial awareness ---
    position_int: list[list[int]]  # JSON-safe : jamais de tuples
                               # Chaque sous-liste : [page_idx, x0, y0, x1, y1]
                               # page_idx : 0-based, x0/y0/x1/y1 normalisés [0-1000]
                               # Ex mono-page  : [[2, 50, 200, 750, 450]]
                               # Ex multi-page : [[2, 50, 200, 750, 1000],
                               #                  [3, 50, 0,   750, 150]]

    # --- Contenu enrichi (présent selon kind, vide sinon) ---
    extras: dict               # Données spécifiques au kind :
                               # kind=table  → {"html": "<table>..."}
                               # kind=image  → {"img_path": "/abs/path/abc.jpg"}
                               #               {"image_b64": "data:image/..."} si image_mode=base64
                               # kind=*      → {"captions": [...], "footnotes": [...]}
                               #               {"inferred_caption": "..."} si captions vide
                               # {} pour les chunks texte (pas de pollution)

    # --- Indexation & traçabilité ---
    chunk_index: int           # ← CHUNKER : index séquentiel dans la liste produite
    block_indices: list[int]   # ← CHUNKER : indices des ContentBlock fusionnés
    reading_order: int         # ← CHUNKER : reading_order du premier bloc fusionné
                               #   = position dans le flux de lecture original MinerU
    prev_chunk_index: int | None  # ← CHUNKER (passe finale) : chunk précédent
    next_chunk_index: int | None  # ← CHUNKER (passe finale) : chunk suivant

    # --- Enrichissements ---
    token_count: int           # ← ENRICHER : len(page_content) // 4 par défaut
                               #   ou tokenizer.count_tokens() si injecté
    content_hash: str          # ← ENRICHER : sha256(page_content)[:16] par défaut
                               #   ou hasher.hash() si injecté
```

### 4.3 Décision Data Model

- **Source unique** : seul `content_list.json` est lu — `middle.json` n'est jamais chargé
- **Pas de scores OCR** : `ocr_confidence` et `ocr_scores` supprimés, pas nécessaires
- **Pas de `page_size`** : les coordonnées normalisées [0-1000] de `position_int` suffisent pour tout raisonnement spatial
- **`reading_order`** dans `ContentBlock` ET `RagChunk` : le parser l'assigné (index brut), le chunker le propage (premier bloc fusionné) — survit au filtrage
- **`prev/next`** : calculés en passe finale dans le chunker, après que la liste complète des chunks est connue
- **`token_count` et `content_hash`** : calculés dans l'enricher, jamais avant (le texte final du chunk n'est connu qu'après fusion des blocs)
- **Protocoles injectables** : `tokenizer` et `hasher` sont des interfaces minimales — la lib n'impose aucune dépendance
- **`extras` dict** : vide `{}` pour les chunks texte, peuplé selon `kind` et `image_mode`

---

## 5. API Publique

### 5.1 Fonction principale (one-liner)

```python
from openingestion import ingest

# Cas le plus simple
chunks = ingest("rapport.pdf")

# Avec options
chunks = ingest(
    "rapport.pdf",
    strategy="by_section",     # by_block | by_section | by_page | fixed_size
    lang="fr",                  # Hint OCR MinerU
    backend="pipeline",         # pipeline | vlm-auto-engine | hybrid-auto-engine
    start_page=0,
    end_page=None,
    max_chars=1500,             # Pour fixed_size
    overlap_chars=200,          # Pour fixed_size
    include_tables=True,
    include_images=True,
    image_mode="path",          # "path"   → extras["img_path"] absolu (défaut)
                                # "base64" → extras["image_b64"] = "data:image/..."
                                # "skip"   → chunk image sans binaire (captions only)
                                # "ignore" → équivalent include_images=False
    infer_captions=True,        # extras["inferred_caption"] si captions vide
    include_equations=True,
    include_discarded=False,    # Headers/footers → False par défaut
    output_format="chunks",     # "chunks" | "langchain" | "llamaindex" | "dicts"
    # --- Protocoles injectables ---
    tokenizer=None,             # Protocol: count_tokens(str) -> int
                                # Défaut : len(text) // 4
    hasher=None,                # Protocol: hash(str) -> str
                                # Défaut : sha256(text)[:16]
)

# Re-ingestion sans re-parse (sorties MinerU déjà sur disque)
chunks = ingest_from_output("output/mon_doc/ocr/")          # dossier MinerU
# ou
chunks = ingest_from_json("output/mon_doc/ocr/_content_list.json")
```

### 5.2 API modulaire (contrôle fin)

```python
from openingestion.parser import MinerUChef
from openingestion.chunker import chunk
from openingestion.enricher import RagRefinery
from openingestion.adapters import to_langchain

# Étape 1 — Chef
chef = MinerUChef()
blocks = chef.parse_output_dir("output/rapport/ocr/")
# → list[ContentBlock]

# Étape 2 — CHUNKer
chunks = chunk(blocks, strategy="by_section", source="rapport.pdf")
# → list[RagChunk] avec title_path, position_int, block_indices, prev/next

# Étape 3 — REfinery
refinery = RagRefinery(
    image_mode="path",
    infer_captions=True,
    tokenizer=my_tokenizer,  # optionnel
    hasher=my_hasher,        # optionnel
)
chunks = refinery.enrich(chunks)
# → token_count, content_hash, img_path absolu

# Étape 4 (optionnelle) — Enrichissement custom applicatif
for c in chunks:
    c.extras["custom"] = my_enrichment(c)

# Étape 5 — Porter
docs = to_langchain(chunks)
```

### 5.3 Accès aux blocs bruts

```python
# Besoins avancés : accès au ContentBlock original derrière un chunk
block = blocks[chunk.block_indices[0]]
print(block.raw)           # item content_list.json complet
print(block.bbox)          # [x0, y0, x1, y1] normalisé [0-1000]
print(block.img_path)      # chemin image local original
print(block.reading_order) # position absolue dans le flux MinerU
```

### 5.4 Protocoles injectables (RagRefinery)

```python
from typing import Protocol

class Tokenizer(Protocol):
    def count_tokens(self, text: str) -> int: ...

class Hasher(Protocol):
    def hash(self, text: str) -> str: ...

# Exemple tiktoken
import tiktoken
enc = tiktoken.get_encoding("cl100k_base")

class TiktokenWrapper:
    def count_tokens(self, text: str) -> int:
        return len(enc.encode(text))

# Exemple xxhash
import xxhash

class XxhashWrapper:
    def hash(self, text: str) -> str:
        return xxhash.xxh64(text.encode()).hexdigest()

# Injection
chunks = ingest(
    "doc.pdf",
    tokenizer=TiktokenWrapper(),
    hasher=XxhashWrapper(),
)
```

**Défauts** (zéro dépendance) :
- `token_count` : `len(text) // 4`
- `content_hash` : `hashlib.sha256(text.encode()).hexdigest()[:16]`

---

## 6. Stratégies de Chunking

### `by_block`
> **1 ContentBlock = 1 RagChunk**

- Granularité maximale, aucune perte d'information
- Fragments très courts possibles ("Vespa", "FastAPI")
- **Cas d'usage** : retrieval ultra-précis, documents courts

### `by_section` *(défaut recommandé)*
> **Agrège les blocs sous le même titre jusqu'au prochain titre ou bloc standalone**

- Tables et images = toujours chunks isolés
- Textes = fusionnés sous le même titre parent
- title_path tracé automatiquement
- **Cas d'usage** : rapports, articles, papers académiques

### `by_page`
> **Tous les blocs d'une page = 1 RagChunk**

- bbox englobant calculé sur tous les blocs
- Préserve logique de lecture mais peut être volumineux
- **Cas d'usage** : docs scannés bruités, slides, présentations

### `fixed_size`
> **Accumule jusqu'à max_chars, puis sliding window avec overlap**

- Blocs standalone (table, image, équation) passent toujours seuls
- Overlap préserve contexte aux frontières
- **Cas d'usage** : très longs documents, modèles avec token limit strict

### Spatial chunking *(Phase 2)*
> **Agrège des blocs proches spatialement sur la même page**

- Exploite bbox pour détecter colonnes, légendes, encadrés
- **Cas d'usage** : documents multi-colonnes, présentations, formulaires

---

## 7. Spatial Awareness — Usages concrets

La conservation de `position_int` (coordonnées normalisées [0-1000]) dans chaque `RagChunk` déverrouille :

```python
# Helpers de base — position_int est list[list[int]]
def pages(chunk):       return [row[0] for row in chunk.position_int]
def first_bbox(chunk):  return chunk.position_int[0][1:]   # [x0, y0, x1, y1]

# 1. Trouver voisins spatiaux sur la même page (context expansion)
def neighbors(chunk: RagChunk, all_chunks: list[RagChunk], gap_max: int = 50):
    page_idx, x0, y0, x1, y1 = chunk.position_int[0]
    return [
        c for c in all_chunks
        if c is not chunk
        and c.position_int[0][0] == page_idx
        and (
            abs(c.position_int[0][2] - y1) <= gap_max
            or abs(y0 - c.position_int[0][4]) <= gap_max
        )
    ]

# 2. Filtrer par zone (haut / corps / bas)
header_zone = [c for c in chunks if c.position_int[0][2] < 100]
body_zone   = [c for c in chunks if 100 <= c.position_int[0][2] < 900]
footer_zone = [c for c in chunks if c.position_int[0][2] >= 900]

# 3. Visualisation : highlight dans le PDF (page_size fournie par l'appelant)
def position_to_pixels(chunk: RagChunk, page_w: int, page_h: int) -> list:
    return [
        [p, x0*page_w//1000, y0*page_h//1000, x1*page_w//1000, y1*page_h//1000]
        for p, x0, y0, x1, y1 in chunk.position_int
    ]

# 4. Injection dans vectorstore avec filtres géométriques (ex: Qdrant)
vectorstore.add(
    embedding=embed(chunk.page_content),
    payload={**chunk_to_dict(chunk)}   # position_int inclus → filtre possible
)
results = vectorstore.search(q, filter={"position_int[0][0]": {"gte": 2, "lte": 5}})
```

---

## 8. Gestion des Assets (images)

> **Hors lib** — l'upload MinIO/S3 est une responsabilité applicative. La lib fournit uniquement `extras["img_path"]` en chemin absolu local résolu par la REfinery.

### 8.1 Pattern d'upload (code applicatif)

```python
# app/ingestion.py — exemple avec MinIO
from openingestion import ingest_from_output
from openingestion.document import BlockKind
from pathlib import Path

chunks = ingest_from_output("output/rapport/ocr/")

for chunk in chunks:
    if chunk.kind == BlockKind.IMAGE and chunk.extras.get("img_path"):
        local_path = chunk.extras["img_path"]   # absolu, résolu par la REfinery
        filename   = Path(local_path).name
        minio_client.put_object("rag-assets", f"{doc_id}/{filename}", ...)
        chunk.extras["asset_id"] = f"rag-assets/{doc_id}/{filename}"
        # img_path local conservé pour audit / re-ingestion idempotente

vectorstore.add(chunks)
```

### 8.2 Récupération d'un asset (côté application)

```python
# asset_id format : "bucket/path/file.jpg" (séparateur /)
bkt, name = asset_id.split("/", 1)
raw_bytes = minio_client.get_object(bkt, name).read()

# PIL jamais importé dans la lib :
from io import BytesIO
from PIL import Image
img = Image.open(BytesIO(raw_bytes))
```

### 8.3 Décisions

- **Upload hors lib** : openingestion ne dépend ni de minio ni de boto3
- **`img_path` absolu** dans `extras` : résolu par la REfinery, utilisable localement sans upload
- **Séparateur `/`** dans `asset_id` : convention naturelle des object stores
- **Nommage** : `{doc_id}/{hash_fichier}.{ext}` → idempotent, pas de doublon si re-ingestion

---

## 9. Adaptateurs

```python
# LangChain (opt-in, pip install langchain-core)
from openingestion.adapters import to_langchain
docs = to_langchain(chunks)
# → list[langchain_core.documents.Document]
# metadata: source, page_idx, kind, title_path, chunk_index, position_int

# LlamaIndex (opt-in, pip install llama-index-core)
from openingestion.adapters import to_llamaindex
nodes = to_llamaindex(chunks)
# → list[llama_index.core.schema.TextNode]
# Avec prev/next relationships chaînés automatiquement

# Dicts (aucune dépendance)
from openingestion.adapters import to_dicts
records = to_dicts(chunks)
# → list[dict] — JSON-serializable, compatible tout vectorstore
```

---

## 10. Gestion d'Erreurs

| Cas | Comportement |
|---|---|
| PDF vide ou invalide | `logger.warning` + retourne `[]` |
| Format non supporté | `raise ValueError` explicite avec formats supportés |
| Modèles MinerU manquants | `raise RuntimeError` avec commande de download |
| Chunk vide après filtrage | Silencieusement ignoré |
| Encodage unicode | Normalisé silencieusement |
| OOM pendant parsing | Propagé tel quel (gestion caller) |

---

## 11. Dépendances

```
# Obligatoire
mineru>=2.7.6          # Core — déjà installé

# Optionnelles (opt-in via try/except dans les adaptateurs)
langchain-core         # to_langchain()
llama-index-core       # to_llamaindex()
```

**Contraintes** :
- ✅ Dataclasses (stdlib Python 3.10+)
- ✅ loguru (déjà présent dans MinerU)
- ❌ Pydantic (non justifié pour des dataclasses simples)
- ❌ Async obligatoire (sync suffit, async = phase 2)
- ❌ minio / boto3 (responsabilité applicative, hors lib)

---

## 12. Structure du Repo

```
openingestion/
├── __init__.py          # Public API : ingest(), ingest_from_output(), ingest_from_json()
├── document.py          # BlockKind, ContentBlock, RagChunk   ✅ FAIT
├── parser.py            # MinerUChef (CHOMP : Chef)            ✅ FAIT
├── chunker.py           # CHUNKer — by_section, by_block, by_page, fixed_size
├── enricher.py          # RagRefinery (CHOMP : REfinery)
└── adapters.py          # Porter / Handshake (CHOMP : Porter)

examples/
├── demo_simple.py       # ingest("doc.pdf") → chunks
├── demo_advanced.py     # Pipeline CHOMP modulaire + enrichissement custom
├── demo_spatial.py      # Exploitation position_int, neighbors
└── demo_reingestion.py  # ingest_from_output() sur sorties MinerU existantes
```

---

## 13. Phases

### Phase 1 — MVP *(en cours)*
- [x] Spécifications validées (v3, flux CHOMP)
- [x] `document.py` : `BlockKind`, `ContentBlock`, `RagChunk`
- [x] `parser.py` : `MinerUChef` — `parse_output_dir()`, `map_to_blocks()`
- [ ] `chunker.py` : `CHUNKer` — `by_block`, `by_section`, `by_page`, `fixed_size`
- [ ] `enricher.py` : `RagRefinery` — tokens, hash, img_path absolu, captions
- [ ] `adapters.py` : `to_dicts()`, `to_langchain()`
- [ ] `__init__.py` : `ingest()`, `ingest_from_output()`, `ingest_from_json()`
- [ ] Test sur `Alternance_260314_120625.pdf`

### Phase 2 — Enrichissements
- [ ] Spatial chunking (multi-colonnes)
- [ ] to_llamaindex() adapter
- [ ] Description VLM : extras["description"] via modèle vision
- [ ] Batch async (ingest_many)
- [ ] Cache / fingerprinting

### Phase 3 — Production
- [ ] Tests unitaires
- [ ] Benchmarks (vitesse, qualité chunking)
- [ ] Packaging pyproject.toml standalone


---

---

## ANNEXE — Intégration dans le RAG Agentique Personnel

*Cette section ne fait pas partie de la lib publique. Elle documente l'usage de openingestion dans le contexte du RAG agentic à arbre de décision.*

### A.1 Contexte

- RAG **mono-tenant**, pas d'authentification sur les assets
- Décisions de retrieval pilotées par un **arbre de décision agentique**
- Images potentiellement passées à un **VLM** pour description textuelle (phase 2)
- Stack : MinIO pour le stockage des assets, Quart/FastAPI pour l'API

### A.2 Ce que la lib fournit (contrat)

Chaque chunk image sorti de `rag_ingestor` contient :

```python
RagChunk(
    kind="image",
    extras={
        "img_path": "/absolute/path/to/abc123.jpg",    # chemin absolu résolu
        "captions": [...],                              # depuis MinerU si dispo
        "inferred_caption": "Architecture — image p.4",# toujours présent
        "description": "Diagramme montrant...",         # si VLM activé (phase 2)
    },
    title_path="Section > Sous-section",               # contexte sémantique
    position_int=[[3, 92, 410, 908, 620]],             # localisation sur la page
)
```

### A.3 Responsabilité de l'application (hors lib)

#### Upload MinIO à l'ingestion

```python
# app/ingestion.py
from openingestion import ingest
from pathlib import Path

chunks = ingest("rapport.pdf", strategy="by_section")

for chunk in chunks:
    if chunk.kind == "image" and chunk.extras.get("img_path"):
        local_path = chunk.extras["img_path"]
        filename   = Path(local_path).name
        asset_id   = f"rag-assets/{doc_id}/{filename}"  # "bucket/path/file.jpg"

        with open(local_path, "rb") as f:
            minio_client.put_object("rag-assets", f"{doc_id}/{filename}", f, ...)

        chunk.extras["asset_id"] = asset_id  # enrichissement applicatif
        # chunk.extras["img_path"] conservé pour audit

# Stocker les chunks dans le vectorstore
vectorstore.add(chunks)
```

#### Endpoint de service des images

```python
# app/routes.py
@manager.route("/image/<path:asset_id>", methods=["GET"])
async def get_image(asset_id):
    """
    asset_id format : "bucket/path/to/file.jpg"
    Séparateur / — robuste même si bucket contient des tirets (ex: rag-assets)
    <path:asset_id> dans la route : les / passent sans être encodés en %2F
    """
    try:
        parts = asset_id.split("/", 1)
        if len(parts) != 2:
            return error("Image not found.")

        bkt, nm = parts
        data = await thread_pool_exec(settings.STORAGE_IMPL.get, bkt, nm)

        import mimetypes
        content_type, _ = mimetypes.guess_type(nm)

        response = await make_response(data)
        response.headers.set("Content-Type", content_type or "application/octet-stream")
        response.headers.set("Cache-Control", "max-age=3600, private")
        response.headers.set("ETag", nm)  # nm = hash → ETag gratuit
        return response

    except Exception as e:
        return error_response(e)
```

### A.4 Flux agentique complet

```
Document PDF
    │
    ▼
openingestion.ingest()
    │  → chunks texte  : page_content + title_path + position_int
    │  → chunks image  : extras["img_path"] absolu + inferred_caption
    │  → chunks table  : extras["html"] + captions
    │
    ▼
Upload MinIO (app)
    │  → chunk.extras["asset_id"] = "rag-assets/doc/abc123.jpg"
    │
    ▼
Vectorstore (texte embedé + payload complet du chunk)
    │
    ▼
Requête utilisateur
    │
    ▼
Arbre de décision agentique
    ├── évalue chunks texte  → contexte LLM
    └── évalue chunks image
            ├── via extras["inferred_caption"] + extras["description"] (VLM)
            ├── via title_path (contexte sémantique)
            └── décision : pertinent ?
                    ├── OUI → inclut asset_id dans la réponse structurée
                    │         → UI : GET /image/<asset_id> → affichage
                    └── NON → ignoré
```

### A.5 Points de vigilance

| Point | Note |
|---|---|
| `img_path` absolu | Résolu par la **REfinery** (`enricher.py`), pas par le Chef ni le CHUNKer |
| Nommage asset_id | `{doc_id}/{hash_fichier}.{ext}` → idempotent, pas de doublon si re-ingestion |
| Tables | Servies inline via `extras["html"]`, pas via cet endpoint |
| VLM description | Stocker dans `extras["description"]` avant envoi au vectorstore |
