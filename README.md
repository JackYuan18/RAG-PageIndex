# PageIndex — NSTSCE Document Indexing & RAG

Reasoning-based, vectorless document indexing and retrieval for structured PDFs. This repository is a modified fork of [VectifyAI/PageIndex](https://github.com/VectifyAI/PageIndex), adapted for NSTSCE/VTTI research document collections, local LLMs (Ollama), and a DocIndex-driven RAG chatbot.

**New to the project?** Read [What this system does](#what-this-system-does), then [New maintainer setup](#new-maintainer-setup) (minimum steps), then [Quick start](#quick-start) for detail.

### Handoff package

| Artifact | Purpose |
|----------|---------|
| [README.md](README.md) | Main setup, architecture, reference |
| [docs/PageIndex_Handoff.pptx](docs/PageIndex_Handoff.pptx) | Slide deck for handoff meeting |
| [RAG-Interface/README.md](RAG-Interface/README.md) | Chat UI (multi-turn, port 5001) |
| [DocIndex-Interface/README.md](DocIndex-Interface/README.md) | Document manager (upload/index/remove, port 5002) |
| [.env.example](.env.example) | Secrets template |
| [llm_models.yaml](llm_models.yaml) | Model aliases and defaults |
| `python scripts/generate_handoff_ppt.py` | Regenerate the PowerPoint deck |

---

## What this system does

The pipeline has two main phases: **indexing** and **querying**.

```text
┌─────────────────────────────────────────────────────────────────────────┐
│  INDEXING (offline, run when PDFs are added or changed)                 │
├─────────────────────────────────────────────────────────────────────────┤
│  Database/*.pdf                                                         │
│       │                                                                 │
│       ▼                                                                 │
│  pageindex — parse PDF → section tree → summaries → abstract/keywords   │
│       │                                                                 │
│       ▼                                                                 │
│  results/<name>_structure.json   (per-document tree + metadata)         │
│  results/DocIndex.json           (keyword → structure file paths)       │
└─────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────┐
│  QUERYING (online, when a user asks a question)                         │
├─────────────────────────────────────────────────────────────────────────┤
│  User question (+ optional prior chat turns in web UI)                  │
│       │                                                                 │
│       ▼  Step 1: embed query (with context) → match keywords → PDFs   │
│       ▼  Step 2: load structure JSON for matched documents              │
│       ▼  Step 3: LLM "tree search" — pick relevant sections           │
│       ▼  Step 4: extract text from selected sections                    │
│       ▼  Step 5–6: LLM answer per doc → merge with citations            │
│       │                                                                 │
│       ▼                                                                 │
│  Final answer (shown in RAG chat UI or CLI)                             │
└─────────────────────────────────────────────────────────────────────────┘
```

Unlike classic vector RAG (chunk → embed → similarity search), retrieval here navigates a **document outline** with LLM reasoning. Embeddings are still used for keyword matching, optional MMR summarization, and keyword merging—but not as the primary retrieval mechanism.

---

## Repository layout

```text
PageIndex/
├── Database/                  # Source PDFs (put documents here)
├── results/                   # Indexed outputs (structure JSON + DocIndex)
├── cache/                     # Cached embedding vectors (auto-created)
├── logs/                      # Per-run indexing logs
│
├── pageindex/                 # Core indexing Python package
├── build_docindex.py          # Batch-index all PDFs in Database/
├── run_pageindex.py           # Index a single PDF (+ optional DocIndex update)
│
├── RAG/                       # Query / retrieval pipeline
├── RAG-Interface/             # Web chat UI (port 5001)
├── DocIndex-Interface/        # Document management UI (port 5002)
│
├── benchmarks/                # Indexing + RAG timing harness
├── eval/                      # RAGAS answer-quality evaluation
├── cookbook/                  # Example Jupyter notebooks (upstream PageIndex)
├── tutorials/                 # Concept guides (doc search, tree search)
├── docs/                      # Internal performance / optimization notes
├── tests/                     # Small test PDFs and expected outputs
├── debug/                     # Scratch files from development (safe to ignore)
│
├── llm_models.yaml            # Model aliases + indexing vs RAG settings
├── .env.example               # Template for secrets (copy to .env)
├── pageindex/config.yaml      # Summarization / keyword pipeline defaults
├── requirements.txt
├── environment.yml
└── OLLAMA_SETUP.md            # Detailed Ollama install notes
```

### Folder reference

#### Data & generated artifacts

| Folder | Description |
|--------|-------------|
| **`Database/`** | **Input documents.** Place source PDFs here before indexing. Served by the RAG chat UI when users click source links. |
| **`results/`** | **Indexing output.** One `*_structure.json` per indexed PDF (tree, summaries, abstract, keywords, metadata) plus `DocIndex.json` (keyword → structure paths). Required for RAG to work. |
| **`cache/embeddings/`** | **Embedding cache.** SHA-keyed JSON files for Ollama embedding calls (keyword matching, MMR, keyword merge). Safe to delete; will be rebuilt on next run. |
| **`logs/`** | **Indexing run logs.** JSON logs per PDF run from `page_index_main` (via `JsonLogger`). Useful for debugging TOC/tree issues. |
| **`RAG-Interface/cache/`** | Separate embedding cache created when the chat UI warms DocIndex keywords on startup. Same format as `cache/embeddings/`. |

#### Core code

| Folder / file | Description |
|---------------|-------------|
| **`pageindex/`** | Core indexing library. See [pageindex package](#pageindex-package) below. |
| **`build_docindex.py`** | Batch driver: loops over `Database/*.pdf`, calls `process_document()`, updates `DocIndex.json`. |
| **`run_pageindex.py`** | Single-document driver: wraps `page_index_main`, saves structure JSON, optional DocIndex merge. |
| **`RAG/`** | RAG query stack. `rag_query.py` orchestrates the 6-step pipeline; `utils.py` has keyword match, tree search, context extraction, and answer prompts. Also contains `example.html` (flow diagram) and `Untitled1.ipynb` (experimental notebook). |
| **`RAG-Interface/`** | Flask app for the **VTTI AI Chatbot** UI. `app.py` exposes `/api/query` (SSE progress stream) and `/api/status`. `templates/chatbot.html` tracks in-browser conversation history and sends prior turns with each request. Default port **5001**. See [RAG-Interface/README.md](RAG-Interface/README.md). |
| **`DocIndex-Interface/`** | Flask app to **manage the document collection**: upload PDFs, index or re-index, remove documents (updates `Database/`, structure JSON, and `DocIndex.json`). Default port **5002**. See [DocIndex-Interface/README.md](DocIndex-Interface/README.md). |

#### Evaluation, benchmarks & docs

| Folder | Description |
|--------|-------------|
| **`benchmarks/`** | Performance harness. `run_benchmarks.py` runs indexing + RAG on fixed PDFs/queries and writes timing JSON under `benchmarks/runs/<timestamp>/`. Subfolders like `runs_rule/`, `runs_llm/` are **historical benchmark outputs** — not used at runtime. See also `docs/perf.md`. |
| **`eval/`** | **RAGAS evaluation** of answer quality. `ragas_eval.py` runs labeled Q&A against `eval/datasets/*.jsonl` and writes scores to `eval/runs/<timestamp>/`. See `eval/README.md`. |
| **`docs/`** | Internal documentation: `perf.md` (benchmarking notes), `PageIndex_Handoff.pptx` (handoff slide deck; regenerate with `python scripts/generate_handoff_ppt.py`). |
| **`scripts/`** | Utility scripts (e.g. `generate_handoff_ppt.py` to rebuild the handoff PowerPoint). |
| **`cookbook/`** | **Upstream PageIndex notebooks** (not required for NSTSCE deployment): vectorless RAG walkthrough, vision RAG, agentic retrieval, PageIndex Chat API quickstart. See `cookbook/README.md`. |
| **`tutorials/`** | **Upstream concept guides** from PageIndex: how tree search works (`tutorials/tree-search/`), multi-document search strategies (`tutorials/doc-search/` — metadata, semantics, description). Reference material, not executed by this repo's scripts. |

#### Development & testing

| Folder | Description |
|--------|-------------|
| **`tests/`** | Lightweight test fixtures: `tests/pdfs/` (sample PDFs) and `tests/results/` (expected structure outputs). Not a full automated test suite. |
| **`debug/`** | Ad-hoc debug dumps (e.g. `toc_with_page_number_debug.json`). Safe to ignore or delete. |

#### Root config files

| File | Description |
|------|-------------|
| **`llm_models.yaml`** | Model aliases and which models indexing vs RAG use. Loaded by `pageindex/model_registry.py`. |
| **`pageindex/config.yaml`** | Default indexing pipeline settings (summary methods, MMR, keyword merge, concurrency). |
| **`.env`** | Secrets and optional overrides. Copy from `.env.example`. **Do not commit.** |
| **`requirements.txt`** / **`environment.yml`** | Python dependencies (Conda or pip). |
| **`OLLAMA_SETUP.md`** | Step-by-step Ollama install and troubleshooting. |

### pageindex package

| File | Description |
|------|-------------|
| **`page_index.py`** | Main PDF indexing pipeline (`page_index_main`): parse PDF → title/authors → tree → summaries → abstract → keywords. |
| **`page_index_md.py`** | Markdown → tree path (`md_to_tree`) for `.md` documents. |
| **`utils.py`** | Shared backends: LLM clients (`ChatGPT_API` / `_async`), Ollama embeddings, MMR summarization, abstract/keyword generation, `ConfigLoader`. |
| **`transform_to_json.py`** | Rule-based text → TOC JSON (`generate_toc_re`) used when `--pageindex-ai-mode rule`. |
| **`model_registry.py`** | Resolves aliases in `llm_models.yaml` to actual model names; separates indexing vs RAG model getters. |
| **`config.yaml`** | Default options merged with CLI flags during indexing. |

---

## Prerequisites

| Requirement | Notes |
|-------------|-------|
| **Python 3.12** | Conda (`environment.yml`) or `pip install -r requirements.txt` |
| **Ollama** | **Required for embeddings** (`mxbai-embed-large`) used in indexing and RAG keyword matching, even when chat models use OpenAI |
| **Chat LLM** | Local Ollama models and/or **OpenAI API key** — depends on `llm_models.yaml` (see [Configuration](#configuration)) |
| **GPU** | Recommended for local Ollama (indexing chat + embeddings). RAG chat can use OpenAI, but **embeddings still run on local Ollama** in the default setup. |
| **NVIDIA drivers** | If using GPU: `nvidia-smi` must work (reboot after driver updates) |

**Run all commands from the project root** (`PageIndex/`). Paths like `Database/`, `results/`, and `cache/` are relative to the current working directory.

---

## New maintainer setup

Minimum steps to run on a fresh machine (current default: **hybrid** — Ollama for indexing + embeddings, OpenAI for RAG chat):

1. **Install:** `pip install -r requirements.txt` (or Conda env + pip for full deps)
2. **Secrets:** `cp .env.example .env` and set `CHATGPT_API_KEY` (required for `gpt51` in `llm_models.yaml`)
3. **Ollama:** `ollama serve`, then `ollama pull mxbai-embed-large` and `ollama pull qwen2.5:7b`
4. **Index:** put PDFs in `Database/`, then `python build_docindex.py` from project root
5. **Query:** `python RAG/rag_query.py --query "…"` or `python RAG-Interface/app.py`
6. **Manage docs (optional):** `python DocIndex-Interface/app.py` → http://localhost:5002

For a **fully local** setup (no OpenAI), change all models in `llm_models.yaml` to Ollama aliases (e.g. `qwen7`) — see [Quick start](#quick-start).

---

## Quick start

### 1. Clone and install

```bash
cd PageIndex   # project root — stay here for all commands below

# Option A: Conda
conda env create -f environment.yml
conda activate pageindex

# Option B: pip (includes ragas for eval/)
pip install -r requirements.txt
```

> `environment.yml` does not list every package in `requirements.txt` (e.g. `ragas`). Use `pip install -r requirements.txt` if you need evaluation tools.

### 2. Configure models and API keys

```bash
cp .env.example .env
# Edit .env — at minimum set CHATGPT_API_KEY if using OpenAI chat models
```

**How models are chosen:** `llm_models.yaml` sets aliases and defaults. The code routes each model to a provider automatically:

| Alias pattern | Provider | Requires |
|---------------|----------|----------|
| `qwen7`, `qwen`, `llama8`, `ollama`, … | **Ollama** (local) | `ollama serve` + `ollama pull <model>` |
| `gpt4mini`, `gpt4`, `gpt51`, `gpt4o`, … | **OpenAI API** | `CHATGPT_API_KEY` in `.env` |
| `huggingface` | **Hugging Face router** | `HF_TOKEN` in `.env` |
| `mxbai-embed-large` (embeddings) | **Ollama** always | `ollama pull mxbai-embed-large` |

> `API_PROVIDER` in older docs is **not used**. Provider is inferred from the model alias in `llm_models.yaml`.

Edit **`llm_models.yaml`** for your setup. Example **hybrid** (local indexing + OpenAI RAG, matches the current repo default):

```yaml
indexing:
  chat_model: qwen7              # Ollama — summaries/abstracts during indexing
  embed_model: mxbai-embed-large # Ollama — keyword embeddings

rag:
  chat_model: gpt51              # OpenAI — final answers (needs CHATGPT_API_KEY)
  tree_search_model: gpt51       # OpenAI — section selection
  keyword_embed_model: mxbai-embed-large  # Ollama — query/keyword match
```

Example **fully local** setup (no OpenAI key):

```yaml
indexing:
  chat_model: qwen7
  embed_model: mxbai-embed-large
rag:
  chat_model: qwen7
  tree_search_model: qwen7
  keyword_embed_model: mxbai-embed-large
```

### 3. Start Ollama and pull models

Ollama is needed for **embeddings** in all setups, and for **chat** in fully-local setups.

```bash
ollama serve                    # in a separate terminal (or use system service)
ollama pull mxbai-embed-large   # required for keyword matching + embed merge
ollama pull qwen2.5:7b          # if using qwen7 alias for indexing/chat
```

See [OLLAMA_SETUP.md](OLLAMA_SETUP.md) for install details.

### 4. Add PDFs and build the index

Place PDFs in `Database/` (create the folder if it does not exist), then:

```bash
python build_docindex.py --pageindex-ai-mode rule
```

`--database-dir` defaults to `./Database` under the project root; you only need to pass it if PDFs live elsewhere.

This creates `results/*_structure.json` and `results/DocIndex.json`. To index one PDF:

```bash
python run_pageindex.py --pdf_path Database/example.pdf --update_docindex --pageindex-ai-mode rule
```

**Expect several minutes per document** — with default flags, tree building uses fast `rule` heuristics, but summaries and abstract still call `indexing.chat_model` (steps 6–8). Keywords use Ollama embeddings (`embed_mmr`), not the chat LLM.

### 5. Ask questions

**CLI** (from project root):

```bash
python RAG/rag_query.py --query "What are the challenges of Level 3 automated driving systems?"
```

**Web UI** (from project root):

```bash
python RAG-Interface/app.py
# Open http://localhost:5001
```

The chat UI requires `results/DocIndex.json`. It streams progress for each RAG step and supports **multi-turn conversation** — follow-up questions use prior Q&A from the current browser session.

### Fresh-machine checklist

- [ ] Python 3.12 env created; `pip install -r requirements.txt`
- [ ] `.env` created from `.env.example`; `CHATGPT_API_KEY` set if using `gpt*` aliases
- [ ] `ollama serve` running; `mxbai-embed-large` pulled
- [ ] Chat model pulled in Ollama **or** OpenAI key configured per `llm_models.yaml`
- [ ] PDFs in `Database/`; `python build_docindex.py` completed successfully
- [ ] `results/DocIndex.json` exists
- [ ] `python RAG/rag_query.py --query "test"` returns an answer (or a clear error)
- [ ] RAG chat UI: ask a follow-up question that references the previous answer
- [ ] DocIndex manager: upload a PDF, index it, then remove it (smoke test)
- [ ] `docs/PageIndex_Handoff.pptx` reviewed (or regenerate with `python scripts/generate_handoff_ppt.py`)

---

## Day-to-day workflows

### Add a new PDF to the collection

1. **Upload** in the DocIndex manager (http://localhost:5002) **or** copy the PDF into `Database/`.
2. Index the document:
   - **Web UI:** click **Add Structure to DocIndex** on the row (PDF only)
   - **Batch CLI:** `python build_docindex.py` (re-indexes all PDFs)
   - **Single CLI:** `python run_pageindex.py --pdf_path Database/new.pdf --update_docindex`
3. Confirm `results/DocIndex.json` was updated.

### Remove a PDF from the collection

Use the DocIndex manager (**Remove** on a row) or manually delete the PDF, `results/<stem>_structure.json`, and prune keyword entries from `DocIndex.json`. The web UI handles all three steps automatically. See [DocIndex-Interface/README.md](DocIndex-Interface/README.md#remove-a-document).

### Re-index after changing indexing settings

If you change `pageindex/config.yaml` or summary/keyword methods, re-run indexing for affected PDFs. Structure files are not updated automatically.

### Change which LLM answers questions

Edit `rag.chat_model` and `rag.tree_search_model` in `llm_models.yaml`, then restart the RAG interface. No re-indexing required.

### Customize answer style or citations

Edit prompts in `RAG/utils.py`:

| Function | Purpose |
|----------|---------|
| `generate_answer_with_citations()` | Per-document answer + `[Source: file.pdf]` |
| `combine_answers()` | Merge answers when multiple PDFs match |
| `tree_search()` | Which sections to retrieve (not the final answer) |
| `format_conversation_for_prompt()` | Format prior turns for LLM prompts |
| `build_retrieval_query()` | Combine recent user turns for keyword matching |

---

## Indexing reference

### `build_docindex.py` — batch index

Processes every `*.pdf` in `Database/`.

```bash
python build_docindex.py \
  --pageindex-ai-mode rule \
  --keyword-method embed_mmr \
  --keyword-merge-method embed
```

| Flag | Default | Meaning |
|------|---------|---------|
| `--pageindex-ai-mode` | `rule` | `rule` = TOC/tree heuristics; `llm` = LLM-based TOC; `auto` = leave `PAGEINDEX_AI_MODE` env unchanged |
| `--summary-method` | `llm` | Leaf summaries: `llm` or `mmr` (embedding-based extractive) |
| `--parent-summary-method` | `llm` | Parent node summaries |
| `--abstract-method` | `llm` | Document-level abstract |
| `--keyword-method` | `embed_mmr` | Keywords: `llm`, `rich`, or `embed_mmr` |
| `--keyword-merge-method` | `embed` | Merge similar keywords in DocIndex: `embed` (fast) or `llm` |

If `rule` mode fails for a PDF, `build_docindex.py` automatically retries once with `llm` mode.

### `run_pageindex.py` — single document

Same indexing flags as `build_docindex.py`, plus `--pdf_path` or `--md_path` and `--update_docindex`:

```bash
python run_pageindex.py \
  --pdf_path Database/my_report.pdf \
  --update_docindex \
  --pageindex-ai-mode rule
```

Markdown files are supported via `--md_path` only through this script (not `build_docindex.py`, which is PDF-only).

### What each indexed document contains

`results/<name>_structure.json` includes:

- `doc_title`, `doc_authors`, `doc_path`
- `doc_abstract`, `keywords` / `keywords_list`
- `structure` — nested tree of sections with `node_id`, page ranges, `summary`, optional `text`

`results/DocIndex.json` maps **keyword strings → list of structure file paths** used in RAG Step 1.

### Indexing pipeline (`pageindex/page_index.py`)

Main stages inside `page_index_main` (terminal shows Steps 1–9):

1. Parse PDF pages  
2. Extract title and authors  
3. Build hierarchical tree (TOC detection, page mapping)  
4. Assign node IDs  
5. Attach node text (if enabled)  
6. Generate leaf summaries  
7. Generate parent summaries  
8. Generate document abstract  
9. Generate keywords  

`run_pageindex.py` / `build_docindex.py` add further steps: save structure JSON, load/prune DocIndex, merge keywords, write `DocIndex.json`.

With default `keyword_method: embed_mmr`, step 9 uses embeddings (Ollama), not the indexing chat model. Summaries/abstract (steps 6–8) still use `indexing.chat_model`.

Further defaults live in `pageindex/config.yaml` (concurrency, MMR parameters, etc.).

---

## RAG query reference

### Pipeline (`RAG/rag_query.py`)

| Step | What happens |
|------|----------------|
| **1** | Embed query (with recent user turns if history provided) → cosine similarity against DocIndex keywords → ranked PDF list |
| **2** | Load `*_structure.json` for matched documents |
| **3** | LLM reads section tree + prior conversation → selects relevant `node_id`s |
| **4** | Pull text from selected nodes into context |
| **5** | LLM generates per-document answer with citations (aware of prior turns) |
| **6** | LLM merges into one final answer (aware of prior turns) |

CLI options:

```bash
python RAG/rag_query.py \
  --query "Your question" \
  --keyword-match-method embed \
  --keyword-match-threshold 0.30 \
  --keyword-match-top-k 8
```

### Conversation history (web UI)

The chat UI keeps an in-memory history of user questions and assistant answers (up to **20 turns**). Each `POST /api/query` sends:

```json
{
  "query": "What sensors does it use?",
  "history": [
    {"role": "user", "content": "What is ADAS?"},
    {"role": "assistant", "content": "ADAS is …"}
  ]
}
```

History is used in:

| Stage | How |
|-------|-----|
| **Keyword match (Step 1)** | Recent user turns are combined with the current query for embedding (`build_retrieval_query`) |
| **Tree search (Step 3)** | Prior turns included in the LLM prompt so follow-ups like “tell me more” resolve correctly |
| **Answer generation (Steps 5–6)** | Prior turns included so answers stay coherent across the session |

**Limits:** History is **browser-session only** — refreshing the page clears it. There is no server-side persistence. The CLI remains single-turn unless you call `rag_query(..., conversation_history=[...])` programmatically.

---

## Web interfaces

| Interface | Command (from project root) | URL | Purpose |
|-----------|----------------------------|-----|---------|
| **RAG chat** | `python RAG-Interface/app.py` | http://localhost:5001 | Multi-turn Q&A over indexed docs (session history in browser) |
| **DocIndex manager** | `python DocIndex-Interface/app.py` | http://localhost:5002 | Upload, index, re-index, and remove documents |

Both apps can also be started from their own folders (`python app.py`). **Restart Flask apps after pulling code changes** so new API routes load.

| Doc | Description |
|-----|-------------|
| [RAG-Interface/README.md](RAG-Interface/README.md) | Chat UI setup, conversation history, API |
| [DocIndex-Interface/README.md](DocIndex-Interface/README.md) | Document manager: upload, index, remove |
| [docs/PageIndex_Handoff.pptx](docs/PageIndex_Handoff.pptx) | Handoff slide deck |

Environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `FLASK_PORT` | `5001` | RAG interface port |
| `DOCINDEX_FLASK_PORT` | `5002` | DocIndex interface port |
| `DOCINDEX_MAX_UPLOAD_MB` | `100` | Max upload size per file in DocIndex manager |
| `FLASK_HOST` | `0.0.0.0` | Bind address |
| `FLASK_OPEN_BROWSER` | off | Set to `1` to auto-open browser on start |

---

## Configuration

### `llm_models.yaml` (recommended)

Central place for model aliases and which model is used for indexing vs RAG. Loaded by `pageindex/model_registry.py`.

- **`indexing.chat_model`** — summaries and abstracts during indexing (steps 6–8); keywords too when `keyword_method` is `llm`
- **`indexing.embed_model`** — embeddings during indexing (MMR / `embed_mmr` keywords / embed merge)
- **`rag.chat_model`** — final answer generation
- **`rag.tree_search_model`** — section selection (often a faster/cheaper model)
- **`rag.keyword_embed_model`** — query ↔ keyword matching at query time
- **`rag.max_tree_search_docs`** — optional cap on documents searched (uncomment in yaml)

**Provider routing** is automatic: Ollama aliases/tags go to `OLLAMA_BASE_URL`; `gpt-*` aliases use `CHATGPT_API_KEY`. Embeddings always use Ollama.

Environment overrides: `PAGEINDEX_CHAT_MODEL`, `PAGEINDEX_EMBED_MODEL`, `RAG_CHAT_MODEL`, `RAG_TREE_SEARCH_MODEL`, `RAG_EMBED_MODEL`, `OLLAMA_MODEL`, `OLLAMA_BASE_URL`, `LLM_MODELS_PATH`.

### `pageindex/config.yaml`

Indexing pipeline defaults: summary methods, MMR knobs, keyword merge thresholds, concurrency limits. Merged with CLI flags in `build_docindex.py` / `run_pageindex.py`. CLI flags from `build_docindex.py` override some yaml defaults (e.g. `if_add_node_text` defaults to `yes` in the script vs `no` in yaml).

### `.env`

Copy `.env.example` → `.env`. Required keys depend on `llm_models.yaml`:

| If you use… | Set in `.env` |
|-------------|----------------|
| `gpt4mini`, `gpt51`, `gpt4`, etc. | `CHATGPT_API_KEY` |
| Local Ollama chat (`qwen7`, `llama8`, …) | `OLLAMA_BASE_URL` (default `http://localhost:11434/v1`) |
| `huggingface` alias | `HF_TOKEN` |
| Embeddings (`mxbai-embed-large`) | Ollama running (no separate key) |

Never commit `.env`.

---

## Evaluation & benchmarks

| Path | Purpose |
|------|---------|
| `benchmarks/run_benchmarks.py` | Run indexing + RAG on chosen PDFs/queries; save wall-clock and per-step timings to `benchmarks/runs/<timestamp>/` |
| `eval/ragas_eval.py` | Score RAG answers with RAGAS metrics against labeled datasets in `eval/datasets/` |
| `docs/perf.md` | How timing is instrumented and how to compare before/after performance |
| `cookbook/*.ipynb` | Standalone notebooks demonstrating upstream PageIndex patterns |
| `tutorials/` | Written guides on tree search and multi-document retrieval strategies |

Example benchmark run:

```bash
python benchmarks/run_benchmarks.py \
  --model qwen \
  --pdf "Database/NSTSCE_L3System_Final.pdf" \
  --query "What are the challenges of L3 automated driving systems?"
```

Example RAGAS eval (see `eval/README.md`; judge LLM uses `AsyncOpenAI()` → set **`OPENAI_API_KEY`** in the environment, or export `OPENAI_API_KEY=$CHATGPT_API_KEY` if you only have the latter):

```bash
python eval/ragas_eval.py \
  --dataset eval/datasets/example_node_ids.jsonl \
  --results-dir ./results \
  --model qwen
```

---

## Troubleshooting

| Problem | Likely cause | Fix |
|---------|--------------|-----|
| `No DocIndex found` | Indexing not run | Run `build_docindex.py` from project root |
| `Database directory not found` | Missing `Database/` or wrong cwd | `mkdir Database`; run commands from project root |
| `No documents found matching the query` | Keywords don't match question | Re-index; lower `--keyword-match-threshold`; check DocIndex keywords |
| Embedding / Ollama errors during RAG | Ollama not running or model missing | `ollama serve`; `ollama pull mxbai-embed-large` |
| OpenAI errors | Missing/invalid key | Set `CHATGPT_API_KEY` in `.env` when using `gpt*` aliases in `llm_models.yaml` |
| `Model config warning: … not in Ollama` | Model not pulled | `ollama pull <model>` or change alias in `llm_models.yaml` |
| `Failed to initialize NVML` / GPU errors | Driver mismatch after update | Reboot the machine |
| Slow indexing | Many LLM calls per PDF | Use `rule` mode for tree; consider `mmr` summaries; reduce concurrency in `config.yaml` |
| Empty `doc_authors` in structure JSON | Old index | Re-run indexing for that PDF |
| RAG chat shows progress but no answer | LLM error | Check terminal running `app.py`; verify keys/models for `rag.*` in yaml |
| Follow-up questions ignore prior context | Page refreshed or CLI used | History is web-UI only; refresh clears session; CLI is single-turn by default |
| DocIndex API returns HTML 404 | Flask server not restarted after code update | Stop and restart `python DocIndex-Interface/app.py` |
| Remove / upload buttons do nothing | Stale server process | Restart the DocIndex manager; hard-refresh browser |
| Wrong `results/` or `cache/` location | Started app from wrong directory | Always `cd` to project root before running scripts |

Indexing logs are written under `logs/` (JSON per run).

---

## Major entry points (quick reference)

| Script | Role |
|--------|------|
| `build_docindex.py` | Batch-index all PDFs in `Database/` |
| `run_pageindex.py` | Index one PDF or Markdown file; optional DocIndex update |
| `RAG/rag_query.py` | CLI question answering |
| `RAG-Interface/app.py` | Chat web UI |
| `DocIndex-Interface/app.py` | Document management web UI |

---

## License & credits

Non-commercial license for modifications in this repository — see [LICENSE](LICENSE).

Original framework: [PageIndex](https://github.com/VectifyAI/PageIndex) by [VectifyAI](https://github.com/VectifyAI) ([pageindex.ai](https://pageindex.ai)).
