# DocIndex Manager — Web UI

Flask web interface for managing the document collection, running indexing jobs, and inspecting `DocIndex.json`. Use this UI when you add or update PDFs in `Database/` and need to rebuild structure files or refresh the keyword index.

For querying indexed documents, use the [RAG chat UI](../RAG-Interface/) on port **5001**. See the [project README](../README.md) for the full pipeline overview.

---

## Features

- **DocIndex overview** — path, keyword count, and which document stems are indexed
- **Database listing** — all PDF and Markdown files in `Database/` with indexing status
- **File upload** — upload PDF or Markdown files directly into `Database/` (drag-and-drop or file picker)
- **Per-document indexing** — run `run_pageindex.py --update_docindex` for one PDF
- **Remove documents** — delete from `Database/`, remove structure JSON, and prune DocIndex keyword entries
- **Full reconstruction** — run `build_docindex.py` to re-index every PDF in `Database/`
- **Live job logs** — streaming terminal output while a job runs
- **Open documents** — click a row to open the PDF in a new tab

Upload **PDF** (`.pdf`) or **Markdown** (`.md`) files via the upload zone in the UI. Files are saved to `Database/` on the server. If a filename already exists, the server saves as `name_1.pdf`, `name_2.pdf`, etc.

After uploading, click **Add Structure to DocIndex** on the new row to index it (PDF only from the UI).

---

## Prerequisites

Same as indexing in the main project:

| Requirement | Notes |
|-------------|-------|
| **Python 3.12** + project deps | `pip install -r requirements.txt` from project root |
| **Ollama** | Required for embeddings (`mxbai-embed-large`) and local chat models |
| **`.env`** | Copy from `.env.example`; set keys per `llm_models.yaml` |
| **`Database/`** | Source PDFs (created automatically if missing) |

Indexing uses **`indexing.chat_model`** and **`indexing.embed_model`** from `llm_models.yaml` (default: `qwen7` + `mxbai-embed-large`).

---

## Quick start

From the **project root** (`PageIndex/`) or from `DocIndex-Interface/`:

```bash
python DocIndex-Interface/app.py
# or: cd DocIndex-Interface && python app.py

# Open http://localhost:5002
```

> **Restart the server** after pulling code changes (`Ctrl+C`, then re-run). New API routes (upload, remove) will not load until you restart.

On first use with an empty index, click **DocIndex Reconstruction** to batch-index all PDFs, or upload a PDF and click **Add Structure to DocIndex**.

---

## Usage

### Add a new document

1. **Upload** a PDF via the upload zone **or** copy it into `Database/` on the server.
2. Click **Add Structure to DocIndex** on that row in the table.
3. Wait for the job to finish (progress log appears under the row).
4. Confirm `results/<stem>_structure.json` exists and the row shows `structure · in DocIndex`.

### Update an existing document

After replacing a PDF or changing indexing settings in `pageindex/config.yaml`:

1. Click **Update Structure** on the document row (same command as add — rebuilds structure + merges into DocIndex).

### Remove a document

Click **Remove** on a row and confirm. This will:

1. Delete the file from `Database/` (if present)
2. Delete `results/<stem>_structure.json` (if present)
3. Remove all DocIndex keyword entries that reference that structure file (empty keywords are dropped)

You cannot remove a document while an indexing job is running for it.

### Rebuild the entire index

Click **DocIndex Reconstruction** in the header. This runs:

```bash
python build_docindex.py --database-dir Database/ --output-dir results/
```

Use this when you have many new PDFs, want a clean DocIndex, or after bulk changes.

### What each status means

| Status | Meaning |
|--------|---------|
| `structure` | `results/<stem>_structure.json` exists |
| `in DocIndex` | At least one keyword in `DocIndex.json` points to this document's structure file |
| `—` | Not indexed yet |

Markdown files (`.md`) appear in the list but **cannot be indexed from this UI** — only PDFs are supported. Use the CLI for Markdown:

```bash
python run_pageindex.py --md_path Database/notes.md --update_docindex
```

---

## Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `DOCINDEX_FLASK_PORT` | `5002` | HTTP port for this UI |
| `DOCINDEX_MAX_UPLOAD_MB` | `100` | Maximum upload size per file (MB) |
| `FLASK_HOST` | `0.0.0.0` | Bind address |
| `FLASK_OPEN_BROWSER` | off | Set to `1` to auto-open the browser on start |

Model selection is **not** exposed in the UI dropdown; jobs use `indexing.chat_model` from `llm_models.yaml` (via `pageindex/model_registry.py`). Change models there and restart the app.

Paths are fixed relative to the project root:

| Path | Role |
|------|------|
| `Database/` | Source PDFs |
| `results/` | `*_structure.json` and `DocIndex.json` |
| `logs/` | Per-run indexing logs (from underlying scripts) |

---

## API reference

All routes are served by `DocIndex-Interface/app.py`.

### `GET /`

Main DocIndex manager page (`templates/docindex.html`).

### `GET /api/overview`

Returns project paths, DocIndex stats, and the document table data.

**Response (abbreviated):**

```json
{
  "project_root": "/path/to/PageIndex",
  "database_dir": "/path/to/PageIndex/Database",
  "results_dir": "/path/to/PageIndex/results",
  "docindex": {
    "path": "/path/to/PageIndex/results/DocIndex.json",
    "exists": true,
    "keyword_count": 142,
    "indexed_document_stems": ["NSTSCE_L3System_Final"]
  },
  "documents": [
    {
      "filename": "report.pdf",
      "stem": "report",
      "kind": "pdf",
      "structure_filename": "report_structure.json",
      "structure_exists": true,
      "in_docindex": true,
      "open_url": "/database/report.pdf"
    }
  ]
}
```

### `POST /api/upload`

Upload a PDF or Markdown file into `Database/`.

**Request:** `multipart/form-data` with field `file`.

**Response:**

```json
{
  "success": true,
  "filename": "report.pdf",
  "original_filename": "report.pdf",
  "renamed": false,
  "message": "Uploaded report.pdf"
}
```

If `report.pdf` already exists, the file is saved as `report_1.pdf` and `renamed` is `true`.

### `POST /api/remove-document`

Remove a document and its index entries.

**Request body:**

```json
{
  "filename": "report.pdf",
  "delete_file": true,
  "delete_structure": true
}
```

Both `delete_file` and `delete_structure` default to `true`.

**Response:**

```json
{
  "success": true,
  "message": "Removed report.pdf",
  "removed": {
    "filename": "report.pdf",
    "stem": "report",
    "database_file_deleted": true,
    "structure_file_deleted": true,
    "docindex": {
      "keywords_removed": 12,
      "keyword_entries_updated": 12,
      "structure_refs_removed": 12
    }
  }
}
```

### `POST /api/reconstruct`

Starts a background job running `build_docindex.py` over all PDFs in `Database/`.

**Response:**

```json
{ "success": true, "job_id": "uuid" }
```

### `POST /api/process-document`

Indexes or re-indexes a single PDF.

**Request body:**

```json
{
  "filename": "report.pdf",
  "mode": "update_structure"
}
```

Optional `model` field overrides the default indexing chat model for that job.

Runs:

```bash
python run_pageindex.py --pdf_path Database/report.pdf --model <indexing.chat_model> --update_docindex
```

**Response:**

```json
{ "success": true, "job_id": "uuid" }
```

### `GET /api/job/<job_id>`

Poll job status and captured stdout/stderr.

**Response:**

```json
{
  "success": true,
  "job": {
    "id": "uuid",
    "label": "run_pageindex --update_docindex: report.pdf",
    "status": "running",
    "output": "…",
    "returncode": null,
    "error": null
  }
}
```

`status` is `running`, `done`, or `failed`. The UI polls every 400 ms until the job completes.

### `GET /database/<filename>`

Serves a file from `Database/` (path traversal protected). Used when clicking a document row.

---

## Troubleshooting

| Problem | Fix |
|---------|-----|
| Empty document list | Upload a PDF in the UI or add files to `Database/`; reload the page |
| **Add Structure** disabled | Only PDFs are supported in the UI; `.md` files need the CLI |
| Job fails immediately | Check terminal running `app.py`; verify Ollama is up and models are pulled |
| `DocIndex.json missing` | Run reconstruction or index at least one PDF |
| Slow indexing | Normal — several LLM/embedding calls per PDF; see [indexing reference](../README.md#indexing-reference) |
| Wrong paths | Always start the app from the project root: `python DocIndex-Interface/app.py` |
| Stale status after job | Page auto-refreshes when the job finishes; reload manually if needed |
| Remove returns 404 / JSON parse error | Server running old code | Restart `python app.py`; hard-refresh browser (`Ctrl+Shift+R`) |
| Upload fails (file too large) | Exceeds limit | Default max 100 MB; set `DOCINDEX_MAX_UPLOAD_MB` |

---

## Related tools

| Tool | When to use |
|------|-------------|
| `build_docindex.py` | Batch index all PDFs (same as **DocIndex Reconstruction**) |
| `run_pageindex.py` | Single PDF or Markdown; same as per-row **Update Structure** |
| `RAG-Interface/app.py` | Ask questions after indexing is complete |

---

## Customization

- **UI:** edit `templates/docindex.html`
- **Indexing defaults:** `pageindex/config.yaml` and CLI flags in `build_docindex.py` / `run_pageindex.py`
- **Models:** `llm_models.yaml` → `indexing.*` keys

---

## License

Same as the main PageIndex project — see [LICENSE](../LICENSE).
