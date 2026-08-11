# VTTI AI Chatbot — RAG Web UI

Flask web interface for querying indexed documents using the PageIndex RAG pipeline. Supports **multi-turn conversation** — follow-up questions use prior Q&A from the current browser session.

For managing documents (upload, index, remove), use the [DocIndex manager](../DocIndex-Interface/) on port **5002**. See the [project README](../README.md) for the full pipeline.

---

## Features

- Multi-turn chat with in-browser conversation history (up to 20 turns)
- Server-Sent Events (SSE) progress stream for each RAG step
- Clickable `[Source: file.pdf]` citations
- Per-step timing in the response footer
- DocIndex keyword warm-up on startup (faster first query)

---

## Prerequisites

| Requirement | Notes |
|-------------|-------|
| Indexed corpus | `results/DocIndex.json` must exist — run `build_docindex.py` or use the DocIndex manager |
| Models / keys | Configured via `llm_models.yaml` and `.env` (see main README) |
| Ollama | Required for keyword embeddings even when RAG chat uses OpenAI |

---

## Quick start

From the **project root**:

```bash
python RAG-Interface/app.py
# Open http://localhost:5001
```

Or from this directory:

```bash
cd RAG-Interface
python app.py
```

Models are read from `llm_models.yaml` (`rag.chat_model`, `rag.tree_search_model`) — not from a UI dropdown. Restart the app after changing `llm_models.yaml`.

---

## Conversation history

Each `POST /api/query` sends the current question plus prior turns:

```json
{
  "query": "What sensors does it use?",
  "history": [
    {"role": "user", "content": "What is ADAS?"},
    {"role": "assistant", "content": "ADAS is …"}
  ]
}
```

History is **browser-session only** — refreshing the page clears it. The CLI (`RAG/rag_query.py`) is single-turn unless you pass `conversation_history` programmatically.

---

## Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `FLASK_PORT` | `5001` | HTTP port |
| `FLASK_HOST` | `0.0.0.0` | Bind address |
| `FLASK_OPEN_BROWSER` | off | Set to `1` to auto-open browser on start |

RAG models: `llm_models.yaml` → `rag.*` keys. Secrets: `.env` (copy from `.env.example`).

---

## API reference

### `GET /`

Main chat page (`templates/chatbot.html`).

### `POST /api/query`

Submit a question. Returns an **SSE stream** (`text/event-stream`).

**Request body:**

```json
{
  "query": "Your question",
  "history": [
    {"role": "user", "content": "…"},
    {"role": "assistant", "content": "…"}
  ]
}
```

`history` is optional (omit for first turn).

**SSE event types:**

| Type | Payload |
|------|---------|
| `progress` | `{ "message": "Step 1: …", "clear_previous": true }` |
| `result` | `{ "data": { "answer", "sources", "matched_documents", "step_timings", … } }` |
| `error` | `{ "error": "…" }` |

### `GET /api/status`

```json
{
  "docindex_available": true,
  "docindex_path": "/path/to/results/DocIndex.json"
}
```

### `GET /docs/<filename>`

Serves PDFs from `Database/` for source link clicks.

---

## Troubleshooting

| Problem | Fix |
|---------|-----|
| DocIndex not found | Run `build_docindex.py` or index via DocIndex manager |
| No answer / LLM errors | Check terminal; verify `CHATGPT_API_KEY` for `gpt*` aliases or Ollama for local models |
| Follow-ups ignore context | Page was refreshed; history is session-only |
| Port in use | Set `FLASK_PORT` or stop the other process |
| Slow first query | Normal — keyword embeddings warm up on startup |

---

## Customization

- **UI:** `templates/chatbot.html`
- **Answer prompts:** `RAG/utils.py` (`generate_answer_with_citations`, `combine_answers`, `tree_search`)
- **Pipeline:** `RAG/rag_query.py`

---

## License

Same as the main PageIndex project — see [LICENSE](../LICENSE).
