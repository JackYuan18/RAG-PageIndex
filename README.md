# PageIndex

Reasoning-based, vectorless document indexing and RAG for structured PDFs (and Markdown), extended for local LLMs (Ollama), switchable rule/LLM structure extraction, and embedding-based summarization / keyword pipelines.

This repository is a modified fork of [VectifyAI/PageIndex](https://github.com/VectifyAI/PageIndex), adapted for NSTSCE-style document collections, DocIndex-based retrieval, and offline-capable workflows.

---

## What this project does

At a high level, the system:

1. **Indexes** PDFs into hierarchical trees (sections → subsections) with page spans and optional node text.
2. **Summarizes** leaves and parents (LLM or extractive MMR over embeddings).
3. **Builds a document abstract** and **keywords** for each file.
4. **Maintains a DocIndex** — a keyword → list of structure-JSON paths used as the first stage of retrieval.
5. **Answers queries** by matching keywords → loading trees → reasoning-based tree search → combining contexts into an answer.

Unlike classic vector RAG, the primary retrieval step navigates a **document outline** with LLM (or rule-based) reasoning rather than embedding similarity over chunks—though embeddings are now also used optionally for **summaries** and **keywords**.

```text
PDF / Markdown
    │
    ▼
pageindex (tree + summaries + abstract + keywords)
    │
    ▼
results/*_structure.json  +  results/DocIndex.json
    │
    ▼
RAG (keyword match → tree search → answer)
```

---

## Major entry points

| Script / module | Role |
|-----------------|------|
| [`run_pageindex.py`](run_pageindex.py) | Process **one** PDF: build structure, optional DocIndex update, keyword merge. |
| [`build_docindex.py`](build_docindex.py) | Batch-index **all** PDFs under `Database/` and rebuild/update DocIndex. |
| [`RAG/rag_query.py`](RAG/rag_query.py) | CLI RAG: query → DocIndex → tree search → answer. |
| [`benchmarks/run_benchmarks.py`](benchmarks/run_benchmarks.py) | Time indexing + RAG runs (supports `--pageindex-ai-mode`). |
| [`DocIndex-Interface/app.py`](DocIndex-Interface/app.py) | Flask UI to manage Database / DocIndex / re-index jobs. |
| [`RAG-Interface/app.py`](RAG-Interface/app.py) | Flask UI for querying the RAG pipeline. |
| [`eval/ragas_eval.py`](eval/ragas_eval.py) | Optional RAGAS-style evaluation of answers. |

Typical CLI flow:

```bash
# Index one document
python run_pageindex.py --pdf_path Database/example.pdf --pageindex-ai-mode rule

# Index a whole collection + DocIndex
python build_docindex.py --database-dir Database --pageindex-ai-mode rule

# Ask a question
python RAG/rag_query.py --query "What are the challenges of L3 systems?"
```

Important switches on indexing (also available in `pageindex/config.yaml`):

- `--pageindex-ai-mode {llm,rule,auto}` — TOC / tree construction via LLM vs rule-based helpers.
- `--node-summary-method` / `--parent-summary-method` — `llm` or `mmr`.
- `--abstract-method` — `llm` or `mmr`.
- `--keyword-method` — `llm`, `rich`, or `embed_mmr`.
- `--keyword-merge-method` — `llm` or `embed` (DocIndex keyword unification).

---

## Core package: `pageindex/`

### [`pageindex/page_index.py`](pageindex/page_index.py)

Main PDF indexing pipeline (`page_index_main`):

1. Parse PDF pages / tokens  
2. Extract title and authors  
3. Build hierarchical tree (`tree_parser`) — TOC detection, mapping, verification, optional fixes  
4. Assign node IDs and (optional) node text  
5. Generate leaf summaries, then parent summaries (bottom-up)  
6. Generate document abstract and keywords  

Includes switchable **LLM vs rule-based** implementations for TOC-related steps (`generate_toc`, detectors, page-number mapping, etc.), controlled by `PAGEINDEX_AI_MODE` / `--pageindex-ai-mode`.

### [`pageindex/utils.py`](pageindex/utils.py)

Shared utilities and the “content” backends used after the tree exists:

- **LLM clients** — `ChatGPT_API` / `ChatGPT_API_async` for Ollama / OpenAI / HuggingFace  
- **Embeddings** — `embed_texts_ollama` (cached under `cache/embeddings/`)  
- **MMR utilities** — chunking, `mmr_select`, near-duplicate merge  
- **Summaries** — leaf/parent summarization (`llm` or `mmr`), `summary_payload` for RAG  
- **Abstracts & keywords** — `generate_doc_abstract`, `generate_doc_keywords` (`llm` / `rich` / `embed_mmr`)  
- **Config** — `ConfigLoader` over [`pageindex/config.yaml`](pageindex/config.yaml)

### [`pageindex/page_index_md.py`](pageindex/page_index_md.py)

Markdown → tree path (`md_to_tree`), with thinning / summary options for MD documents.

### [`pageindex/transform_to_json.py`](pageindex/transform_to_json.py)

Rule-based text → structured TOC JSON used when AI mode is rule-based (`generate_toc_re`).

### [`pageindex/config.yaml`](pageindex/config.yaml)

Defaults for models, summarization / abstract / keyword / merge methods, MMR and embedding knobs.

---

## Indexing & DocIndex scripts

### [`run_pageindex.py`](run_pageindex.py)

- Wraps `page_index_main` for a single PDF  
- Writes `results/<name>_structure.json`  
- Optionally updates `results/DocIndex.json`  
- **Keyword merge**: `merge_keywords` with `llm` (pairwise LLM) or `embed` (LSH / ANN over keyword embeddings)

### [`build_docindex.py`](build_docindex.py)

Batch loop over `Database/*.pdf`, calling the same `process_document` helper used by `run_pageindex.py`.

---

## RAG: `RAG/`

### [`RAG/rag_query.py`](RAG/rag_query.py)

End-to-end query CLI: load DocIndex → match keywords → load structures → tree search → extract context → answer.

### [`RAG/utils.py`](RAG/utils.py)

Keyword matching, structure loading, reasoning-based tree search prompts, context extraction, and answer combination. Aware of `summary_payload` (`llm` vs `mmr`) for node summaries.

---

## Interfaces & evaluation

| Path | Purpose |
|------|---------|
| `DocIndex-Interface/` | Manage PDFs, trigger re-index, inspect DocIndex / structures |
| `RAG-Interface/` | Chat-style query UI over the RAG stack |
| `eval/` | Dataset + RAGAS evaluation helpers |
| `benchmarks/` | Indexing / RAG timing harness |
| `tutorials/`, `cookbook/` | Examples and notebooks |

---

## Data layout

```text
Database/                 # Source PDFs
results/
  *_structure.json        # Per-document tree (+ summaries, abstract, keywords)
  DocIndex.json           # keyword → [structure file paths]
cache/embeddings/         # Cached Ollama embedding vectors
```

A structure JSON typically includes: document metadata, `structure` tree (`title`, `node_id`, page indices, optional `text` / `summary` / `summary_payload`), `doc_abstract`, `doc_abstract_payload`, `keywords`, and related payloads.

---

## Configuration & environment

Create a `.env` in the project root (see also [`OLLAMA_SETUP.md`](OLLAMA_SETUP.md)):

```bash
API_PROVIDER=ollama
OLLAMA_BASE_URL=http://localhost:11434/v1
OLLAMA_MODEL=qwen2.5:14b
CHATGPT_API_KEY=...          # if using OpenAI
HF_TOKEN=...                 # if using HuggingFace
```

Install via Conda (`environment.yml`) or `pip install -r requirements.txt`. Pull an embedding model for MMR / `embed_mmr` / embed merge when needed, e.g. `ollama pull mxbai-embed-large`.

---

## License & credits

Non-commercial license for modifications in this repository — see [LICENSE](LICENSE).

Original framework: [PageIndex](https://github.com/VectifyAI/PageIndex) by [VectifyAI](https://github.com/VectifyAI) ([pageindex.ai](https://pageindex.ai)).
