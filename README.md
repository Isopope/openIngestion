# openingestion

Pipeline d'ingestion RAG au-dessus de MinerU / Docling.

```
Fetcher → Chef → Chunker → Refinery → Porter
```

## Installation

### 1. Cloner et installer en mode éditable

```bash
git clone <repo-url>
cd openingestion
pip install -e .
```

> L'installation éditable (`-e`) est **obligatoire** pour que les imports
> `from openingestion import …` se résolvent correctement depuis les scripts
> et notebooks, car la racine du dépôt *est* le package Python.

### 1bis. Setup Windows / PowerShell

Le projet demande `Python >= 3.10`. Sur Windows, un setup simple ressemble Ã  :

```powershell
py -3.14 -m venv .venv
. .\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e .
```

Pour un premier run CPU sans GPU, ajoutez Docling :

```powershell
python -m pip install -e ".[docling]"
```

### 2. Extras optionnels

```bash
# Parser MinerU (GPU recommandé)
pip install -e ".[mineru]"

# Parser Docling (CPU, pas de GPU nécessaire)
pip install -e ".[docling]"

# Chunking sémantique (sentence-transformers + scipy)
pip install -e ".[semantic]"

# Chunking LLM-guidé SlumberChunker + OpenAIGenie
pip install -e ".[slumber]"

# Tokenizer OpenAI exact (cl100k_base, o200k_base…)
pip install -e ".[tiktoken]"

# Tokenizer HuggingFace rapide (Rust, BPE/WordPiece…)
pip install -e ".[hf-tokenizers]"

# AutoTokenizer HuggingFace (transformers complet)
pip install -e ".[transformers]"

# Tout à la fois
pip install -e ".[mineru,docling,semantic,slumber,tiktoken]"
```

## Utilisation rapide

```python
from openingestion import ingest

# Depuis un PDF brut (MinerU tourne en arrière-plan)
chunks = ingest("rapport.pdf")

# Depuis un répertoire de sortie MinerU existant (pas de re-parsing)
chunks = ingest("./output/rapport/auto/")

# Avec Docling (CPU, pas de GPU)
chunks = ingest("rapport.pdf", parser="docling", strategy="by_token")

# Format LangChain
docs = ingest("rapport.pdf", output_format="langchain")
```

## Architecture

| Étape | Classe | Rôle |
|---|---|---|
| Chef | `MinerUChef`, `DoclingChef` | Parse le document → `ContentBlock[]` |
| Chunker | `TokenChunker`, `SentenceChunker`, `SemanticChunker`… | Groupe les blocs → `RagChunk[]` |
| Refinery | `RagRefinery`, `ContextualRagRefinery` | Enrichit les chunks (tokens, hash, images, contexte LLM) |
| Porter | `JSONPorter`, `to_langchain`, `to_llamaindex` | Exporte vers le format cible |

Voir [specv3.md](specv3.md) pour les spécifications techniques détaillées.
