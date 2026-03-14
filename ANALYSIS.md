# Analyse des Sorties MinerU — Findings Empiriques

## Source
- Document: `Alternance_260314_120625.pdf` (rapport d'alternance)
- Outputs MinerU (backend: `pipeline`, method: `ocr`)
- Fichiers clés:
  - `*_content_list.json` (2776 lignes) ← **CIBLE RAG**
  - `*_middle.json` (26789 lignes) ← structure détaillée
  - `*.md` (432 lignes) ← rendu markdown
  - `*_model.json` ← sorties brutes modèles
  - images/ ← images extraites

---

## 1. Structure content_list.json (le plus pertinent)

### Format: Array de items, chaque item repr. un "bloc"

```json
{
  "type": "text" | "image" | "discarded" | "table",
  "text": "...",
  "text_level": 0|1|2|...,    // Pour titles uniquement
  "bbox": [x0, y0, x1, y1],   // Coordinates normalized [0-1000]
  "page_idx": 0,               // 0-based page number
  
  // Pour images:
  "img_path": "images/hash.jpg",
  "image_caption": [],         // List de strings
  "image_footnote": [],        // List de strings
  
  // Pour tables (non observé ici, mais selon code MinerU):
  // "html": "...",           // HTML table markup
  // "table_caption": [],
  // "table_footnote": [],
}
```

### Types observés dans ce document

1. **"text"** : Texte normal
   ```json
   {
     "type": "text",
     "text": "Vespa",
     "bbox": [177, 213, 368, 333],
     "page_idx": 1
   }
   ```

2. **"text" avec "text_level"** : Titre/heading
   ```json
   {
     "type": "text",
     "text": "Amélioration et évaluation du système Retrieval-Augmented...",
     "text_level": 1,
     "bbox": [100, 243, 965, 379],
     "page_idx": 0
   }
   ```

3. **"image"** : Bloc image
   ```json
   {
     "type": "image",
     "img_path": "images/ed9abc33a8087055dece64e2c3dd16915abd63dd905d2b1bf156ad8ac1e8f07c.jpg",
     "image_caption": [],
     "image_footnote": [],
     "bbox": [59, 200, 163, 327],
     "page_idx": 1
   }
   ```

4. **"discarded"** : Headers/footers/metadata (keep separé)
   ```json
   {
     "type": "discarded",
     "text": "POLYTECH TOURS",
     "bbox": [57, 7, 277, 129],
     "page_idx": 0
   }
   ```

### Observations

- **Total items**: 2776 (dont beaucoup de "text" courts)
- **Pas de "table" observation**: Document sans tableau (logical: rapport alternance)
- **Images**: Plusieurs, avec empty captions/footnotes
- **text_level**: Observe 1, pas vu >1 (no nested hierarchy visible)
- **bbox**: Coordonnées normalisées [0-1000], pas pixels bruts
- **page_idx**: 0-based, correspond aux pages du PDF

---

## 2. Structure middle.json (enrichi)

Plus détaillé que content_list. Contient:

```json
{
  "pdf_info": [                    // Array, 1 item par page
    {
      "page_idx": 0,
      "page_size": [1440, 810],    // Dimensions page
      "preproc_blocks": [...],     // Blocs avant post-processing
      "discarded_blocks": [...],   // Headers/footer rejetés
      "para_blocks": [...]         // Blocs conservés
    }
  ]
}
```

### Granularité fine: lines + spans

Chaque bloc contient `lines[]`, chaque line contient `spans[]`:

```json
{
  "type": "title",
  "lines": [
    {
      "bbox": [...],
      "spans": [
        {
          "bbox": [...],
          "score": 0.998,           // OCR confidence
          "content": "text",
          "type": "text"            // vs "image", "equation_inline", etc.
        }
      ]
    }
  ]
}
```

**Avantage**: Permet de récupérer scores OCR, caractériser les spans par type (texte vs formula inline, etc.)

---

## 3. Markdown render (pour référence)

```markdown
# Amélioration et évaluation du système RAG...

# Contexte technologique de l'alternance

![](images/ed9abc33a8087055dece64e2c3dd16915abd63dd905d2b1bf156ad8ac1e8f07c.jpg)

Vespa

FastAPI
...
```

**Observation**: Images inline, titres avec `#`, pas de hiérarchie multi-level apparent.

---

## 4. Implications pour la Bibliothèque RAG

### 4a. Input idéal: content_list.json
- Taille raisonnable (2776 items pour 432 lignes MD)
- Structure plate et prévisible
- Enough metadata (bbox, page_idx, type)
- **Problem**: Comment obtenir content_list.json ?
  - Option 1: Parser lui-même (comme MinerU le fait)
  - Option 2: MinerU.parse() → retourner content_list nativement
  - Option 3: Passer middle.json et le convertir

### 4b. Atomicité: 1 item content_list ≠ 1 chunk RAG

Examples:
- Short texts ("Vespa", "FastAPI") → **multiple items, maybe merge**
- Title + following params → **natural grouping**
- Image ← **isolated item, could be standalone chunk**

### 4c. Métadonnées réelles disponibles

**Direct du content_list**:
- `type` (text, image, discarded)
- `text_level` (if title)
- `page_idx`
- `bbox`

**À dériver**:
- Section path (track titles as ancestors)
- OCR confidence (available in middle.json, not content_list)
- Whether "discarded" (filtered by type)

### 4d. Cas d'usage découvert

1. **Documents Français avec OCR** → OCR backend nécessaire
2. **Beaucoup de text fragments** → Chunking strategy crucial
3. **Peu de tables/equations** → Peut ignorer ces types complètement
4. **Images numerous mais vides de captions** → Images need caption inference

---

## 5. Questions Ouvertes

### Q1: Should we work with content_list.json or middle.json?

**content_list.json** :
- ✅ Clean, simple structure
- ✅ Already aggregated (lines → text)
- ❌ No OCR scores
- ❌ No fine spans

**middle.json** :
- ✅ Richer metadata (scores, lines/spans)
- ✅ Can extract fine details if needed
- ❌ Much larger (26k lines vs 2.7k)
- ❌ More complex to parse

**Recommendation**: Start with content_list, offer middle.json access if advanced.

### Q2: How to get content_list from MinerU API?

Currently MinerU doesn't expose an easy way to get content_list programmatically — you'd need to:

1. Run pipeline (`doc_analyze()`)
2. Get middle_json
3. Call `union_make()` with `MakeMode.CONTENT_LIST`

This should be **wrapped in our library**.

### Q3: Chunking strategy for this doc?

Test case: "Alternance_260314_120625"
- ~2776 items total
- Mostly short texts, some images, few titles with text_level
- **by_section**: Would group texts under titles → Natural sections
- **by_block**: 2776 chunks → Too granular? But very atomic
- **fixed_size**: 1500 chars → ~45 chunks → Reasonable

**To test**: Try all 3 strategies and evaluate output.

---

## 6. Dimensions à Mesurer

When we build library, test against this real doc:

- [ ] Parse time (cold start + subsequent)
- [ ] Chunking time (each strategy)
- [ ] Output size (avg chunk size, metadata overhead)
- [ ] Quality (is title_path correct? Do chunks make sense?)
- [ ] Metadata completeness (are all fields populated?)

---

## 7. Next Steps

1. **Look at middle.json structure** in detail (spans, lines structure)
2. **Look at other output folder** if available (Alternance_260314_115929)
3. **Prototype quick converter**: content_list.json → List[RagChunk]
4. **Test chunking strategies** on real data
5. **Decide on API surface** after seeing all this

---

**Files to keep open for reference**:
- `output/Alternance_260314_120625/ocr/Alternance_260314_120625_content_list.json`
- `output/Alternance_260314_120625/ocr/Alternance_260314_120625_middle.json`
- Source PDF: `Alternance_260314_120625.pdf` (if available)
