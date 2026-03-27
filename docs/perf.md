## Performance optimization notes

This document explains the performance-related changes made to:
- `run_pageindex.py` / `pageindex/page_index.py` (indexing)
- `RAG/rag_query.py` + `RAG/utils.py` (query answering)

It also provides a reproducible benchmark method and where timing outputs are written.

### How to run benchmarks

Run indexing + RAG benchmarks and save machine-readable JSON outputs:

```bash
python3 benchmarks/run_benchmarks.py \
  --model qwen \
  --pdf "Database/NSTSCE_L3System_Final.pdf" \
  --pdf "Database/AVSC Best Practice for Developing ADS.pdf" \
  --query "In vtti, who works on the challenges of L3 autonomous systems?" \
  --query "What are the challenges of L3 autonomous driving systems?"
```

The script prints a run directory like `benchmarks/runs/YYYYMMDD_HHMMSS/` containing:
- `meta.json` (subprocess wall times, exit codes, stdout/stderr tails)
- `index_<doc>.json` (indexing timings from `run_pageindex.py --timings-out`)
- `rag_<n>.json` (RAG result dict from `RAG/rag_query.py --out`)

### Timing sources

- **Indexing**:
  - Steps 1–10 are timed inside `pageindex/page_index.py` (`page_index_main(..., step_timings=...)`).
  - Steps 11–14 are timed inside `run_pageindex.py` (save structure JSON, load+prune DocIndex, merge keywords, write DocIndex).
- **RAG**:
  - `RAG/rag_query.py` returns `step_timings` for keyword match, structure load, tree search, context extraction, per-doc answers, and combine.


### Before/after reporting

To document before/after numbers:
- Run `benchmarks/run_benchmarks.py` on the same inputs before and after changes.
- Compare:
  - `meta.json` `wall_seconds`
  - per-step totals in `index_*.json` and `rag_*.json` `step_timings`

