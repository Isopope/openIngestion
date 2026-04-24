# openingestion

> Modular RAG ingestion pipeline — from raw documents to retrieval-ready chunks.

```
Fetcher → Chef → Chunker → Refinery → Porter
```

**Version:** 0.1.2 · **Python:** 3.10 – 3.13 · **License:** MIT

---

## Overview

**openingestion** orchestrates the full journey from raw documents to enriched,
retrieval-ready chunks through five composable stages:

| Stage | Classes | Input → Output |
|---|---|---|
| **Fetcher** | `LocalFileFetcher`, `WebFetcher`, `SharepointFetcher` | Source → `FetchedDocument[]` |
| **Chef** | `MinerUChef`, `DoclingChef` | File/dir → `ContentBlock[]` |
| **Chunker** | `TokenChunker`, `SentenceChunker`, `SemanticChunker`, `SlumberChunker`, `BlockChunker`, `PageChunker`, `SectionChunker`, `RecursiveChunker` | `ContentBlock[]` → `RagChunk[]` |
| **Refinery** | `RagRefinery`, `ContextualRagRefinery`, `VisionRefinery` | `RagChunk[]` → enriched `RagChunk[]` |
| **Porter** | `JSONPorter`, `to_dicts`, `to_langchain`, `to_llamaindex` | `RagChunk[]` → target format |

---

## Installation

### From PyPI

```bash
pip install openingestion
```

> The base install (`loguru` + `chonkie-core`) is intentionally minimal — no heavy
> ML dependencies. Parsers, tokenizers, and advanced chunkers are opt-in extras.

### From source (editable)

```bash
git clone https://github.com/Isopope/openIngestion.git
cd openIngestion
pip install -e .
```

### Windows / PowerShell

openingestion requires **Python 3.10 – 3.13** (3.14+ not yet supported due to MinerU).

```powershell
py -3.13 -m venv .venv
. .\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e .
```

---

## Optional extras

| Extra | Installs | Use case |
|---|---|---|
| `mineru` | `mineru[pipeline]==3.0.4` | GPU-accelerated PDF parsing (full layout analysis) |
| `docling` | `docling` | CPU-only PDF parsing (IBM Docling, no GPU required) |
| `semantic` | `sentence-transformers`, `numpy`, `scipy` | `SemanticChunker` — embedding-based splitting |
| `slumber` | `openai`, `pydantic`, `tenacity`, `tqdm` | `SlumberChunker` + `OpenAIGenie` — LLM-guided chunking |
| `tiktoken` | `tiktoken` | Exact OpenAI tokenizer (`cl100k_base`, `o200k_base` …) |
| `hf-tokenizers` | `tokenizers` | Fast HuggingFace tokenizers (Rust, BPE/WordPiece) |
| `transformers` | `transformers` | HuggingFace `AutoTokenizer` (full model loading) |
| `langchain` | `langchain-core` | `output_format="langchain"` |
| `llamaindex` | `llama-index-core` | `output_format="llamaindex"` |
| `web` | `playwright` | `WebFetcher` — render websites to PDF/HTML |
| `sharepoint` | `msal`, `office365-rest-python-client` | `SharepointFetcher` — Microsoft 365 / SharePoint |

**Convenience bundles:**

```bash
# CPU pipeline (Docling + semantic + tiktoken)
pip install -e ".[cpu]"

# GPU pipeline (MinerU + semantic + tiktoken)
pip install -e ".[mineru,gpu]"

# Everything
pip install -e ".[all]"
```

**Individual extras:**

```bash
pip install -e ".[mineru]"          # MinerU parser (GPU recommended)
pip install -e ".[docling]"         # Docling parser (CPU)
pip install -e ".[semantic]"        # SemanticChunker
pip install -e ".[slumber]"         # SlumberChunker + OpenAI
pip install -e ".[web]"             # WebFetcher (then: playwright install chromium)
pip install -e ".[sharepoint]"      # SharepointFetcher
```

---

## Quick start

```python
from openingestion import ingest, ingest_from_output, ingest_from_json

# Parse a raw PDF with MinerU (requires [mineru] extra)
chunks = ingest("rapport.pdf")

# Skip re-parsing — reuse an existing MinerU output directory
chunks = ingest_from_output("./output/rapport/auto/")

# Load directly from a content_list.json
chunks = ingest_from_json("./output/rapport/auto/rapport_content_list.json")

# Use Docling instead of MinerU (CPU, no GPU needed)
chunks = ingest("rapport.pdf", parser="docling", strategy="by_sentence")

# Full control
chunks = ingest(
    "rapport.pdf",
    parser="mineru",                # or "docling"
    strategy="by_token",            # by_block | by_token | by_sentence | by_semantic | by_slumber
    max_tokens=512,
    overlap_tokens=64,
    image_mode="path",              # path | base64 | skip | ignore
    infer_captions=True,
    output_format="chunks",         # chunks | dicts | langchain | llamaindex
)

# Export to LangChain Documents
docs = ingest("rapport.pdf", output_format="langchain")

# Export to LlamaIndex TextNodes
nodes = ingest("rapport.pdf", output_format="llamaindex")
```

---

## Fetchers

```python
from openingestion.fetcher import LocalFileFetcher, WebFetcher, SharepointFetcher

# Local filesystem
fetcher = LocalFileFetcher(ext=[".pdf"])
docs = fetcher(dir="./inputs/")

# Website → PDF (requires [web] extra + playwright install chromium)
fetcher = WebFetcher(output_dir="./downloads/", mode="pdf")
docs = fetcher.fetch(urls=["https://example.com"])

# SharePoint / Microsoft 365 (requires [sharepoint] extra)
fetcher = SharepointFetcher(
    client_id="...", client_secret="...", tenant_id="...",
    output_dir="./downloads/",
)
docs = fetcher.fetch(site_url="https://tenant.sharepoint.com/sites/MySite")
```

---

## Refineries

```python
from openingestion.refinery import RagRefinery, ContextualRagRefinery, VisionRefinery
from openingestion.genie import OpenAIGenie

# Standard enrichment: token count, content hash, image paths
refinery = RagRefinery(output_dir="./output/doc/auto/", image_mode="path")
chunks = refinery.enrich(chunks)

# Contextual RAG: LLM-generated doc summary + per-chunk context
genie = OpenAIGenie(model="gpt-4o-mini", api_key="sk-...")
ctx_refinery = ContextualRagRefinery(genie=genie, generate_doc_summary=True)
chunks = ctx_refinery.enrich(chunks)

# Vision: extract text from scanned tables / images via GPT-4o
vision_refinery = VisionRefinery(genie=genie, only_if_empty=True)
chunks = vision_refinery.enrich(chunks)
```

---

## Export

```python
from openingestion.porter import JSONPorter

# JSONL (one chunk per line)
JSONPorter(lines=True)(chunks, file="output.jsonl")

# Pretty JSON array
JSONPorter(lines=False, indent=2)(chunks, file="output.json")
```

---

## Architecture

Each stage follows a uniform **Abstract Base Class** pattern:

- Abstract method: `process()` / `chunk()` / `enrich()` / `export()`
- Batch processing: `process_batch()` / `chunk_batch()` / `enrich_batch()` / `export_batch()`
- Callable shortcut: `instance(input)` == `instance.main_method(input)`
- Unified logging via `loguru`

The three core data models flow through the whole pipeline:

```
FetchedDocument  →  ContentBlock  →  RagChunk
   (Fetcher)          (Chef)       (Chunker + Refinery)
```

`BlockKind` (`TEXT`, `TITLE`, `TABLE`, `IMAGE`, `LIST`, `EQUATION`, `DISCARDED`)
is preserved from Chef through to the final export.

---

## License

MIT — see [LICENSE](LICENSE).

> **Note on optional dependencies:** the `[mineru]` extra installs MinerU which is
> licensed under **AGPL-3.0**. Its licence terms apply when that extra is installed.

See [specv3.md](specv3.md) for full technical specifications.
