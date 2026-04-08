## RAGAS eval for PageIndex RAG

This folder contains a small evaluation harness for the PageIndex RAG pipeline (`RAG/rag_query.py`) using **RAGAS** metrics.

### Setup

- **Install**:

```bash
pip install -r requirements.txt
```

- **Env vars** (OpenAI judging LLM):
  - **`OPENAI_API_KEY`**: required by `openai` + RAGAS (via `openai.AsyncOpenAI()`).

### Dataset format (JSONL)

Each line is one JSON object:

- **`question`**: string
- **`references`**: list of:
  - **`doc`**: a `*_structure.json` filename (typically under `./results/`)
  - **`node_ids`**: list of PageIndex `node_id` strings you labeled as relevant
- **`meta`**: optional dict

See `eval/datasets/example_node_ids.jsonl`.

### Run

From repo root:

```bash
python3 eval/ragas_eval.py \
  --dataset eval/datasets/example_node_ids.jsonl \
  --results-dir ./results \
  --model qwen \
  --judge-model gpt-4o-mini
```

Artifacts are written to `eval/runs/<timestamp>/`:
- `items.jsonl`: inputs + raw outputs per question
- `scores.csv`: per-item metric scores
- `summary.json`: aggregate metrics

