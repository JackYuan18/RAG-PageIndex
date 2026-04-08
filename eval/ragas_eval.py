#!/usr/bin/env python3
import argparse
import asyncio
import csv
import json
import os
import time
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@dataclass(frozen=True)
class Reference:
    doc: str
    node_ids: Tuple[str, ...]


@dataclass(frozen=True)
class EvalItem:
    question: str
    references: Tuple[Reference, ...]
    meta: Dict[str, Any]


def _read_jsonl(path: str) -> Iterable[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            s = line.strip()
            if not s:
                continue
            try:
                obj = json.loads(s)
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON on line {line_no} of {path}: {e}") from e
            if not isinstance(obj, dict):
                raise ValueError(f"Line {line_no} of {path} must be a JSON object")
            yield obj


def load_eval_items(dataset_path: str) -> List[EvalItem]:
    items: List[EvalItem] = []
    for i, obj in enumerate(_read_jsonl(dataset_path), start=1):
        q = obj.get("question")
        if not isinstance(q, str) or not q.strip():
            raise ValueError(f"Item #{i}: missing/invalid 'question'")

        refs_raw = obj.get("references")
        if not isinstance(refs_raw, list) or not refs_raw:
            raise ValueError(f"Item #{i}: missing/invalid 'references' (must be non-empty list)")

        refs: List[Reference] = []
        for j, r in enumerate(refs_raw, start=1):
            if not isinstance(r, dict):
                raise ValueError(f"Item #{i} ref #{j}: must be an object")
            doc = r.get("doc")
            node_ids = r.get("node_ids")
            if not isinstance(doc, str) or not doc.strip():
                raise ValueError(f"Item #{i} ref #{j}: missing/invalid 'doc'")
            if not isinstance(node_ids, list) or not all(isinstance(x, str) and x.strip() for x in node_ids):
                raise ValueError(f"Item #{i} ref #{j}: missing/invalid 'node_ids' (list[str])")
            refs.append(Reference(doc=doc, node_ids=tuple(node_ids)))

        meta = obj.get("meta") or {}
        if not isinstance(meta, dict):
            raise ValueError(f"Item #{i}: 'meta' must be an object if present")

        items.append(EvalItem(question=q.strip(), references=tuple(refs), meta=meta))
    return items


def _ensure_abs(path: str) -> str:
    return path if os.path.isabs(path) else os.path.join(PROJECT_ROOT, path)


def resolve_structure_path(doc: str, results_dir: str) -> str:
    # If user provides a full/relative path, honor it.
    if os.path.isabs(doc):
        return doc

    # Most common: doc is a *_structure.json basename stored under results_dir.
    p1 = os.path.join(results_dir, os.path.basename(doc))
    if os.path.exists(p1):
        return p1

    # Also allow doc relative to project root.
    p2 = os.path.join(PROJECT_ROOT, doc.lstrip("./"))
    if os.path.exists(p2):
        return p2

    return p1  # default best guess; downstream will error with a clear message


async def build_reference_contexts(
    references: Tuple[Reference, ...],
    results_dir: str,
) -> Tuple[List[str], List[str]]:
    """
    Returns (reference_contexts, reference_context_ids)
    - reference_contexts: extracted text blobs for labeled nodes (one per reference doc entry)
    - reference_context_ids: stable IDs for ID-based recall (doc + node_id)
    """
    # Lazy import to keep CLI usable even if PageIndex deps are partially missing.
    from RAG.utils import create_node_mapping, extract_context, load_document_structure

    ref_contexts: List[str] = []
    ref_ids: List[str] = []

    for ref in references:
        structure_path = resolve_structure_path(ref.doc, results_dir=results_dir)
        structure = load_document_structure(structure_path, results_dir=results_dir, project_root=PROJECT_ROOT)
        if not structure:
            raise FileNotFoundError(
                f"Could not load document structure for ref doc '{ref.doc}'. "
                f"Tried '{structure_path}'. Ensure the structure JSON exists under results dir."
            )

        tree = structure.get("structure", structure)
        node_map = create_node_mapping(tree)

        # Extract text for labeled nodes.
        text = await extract_context(node_map, list(ref.node_ids), doc_path=structure_path, results_dir=results_dir, project_root=PROJECT_ROOT)
        ref_contexts.append(text)

        # Build IDs used by IDBasedContextRecall.
        doc_key = os.path.basename(structure_path)
        for nid in ref.node_ids:
            ref_ids.append(f"{doc_key}::{nid}")

    return ref_contexts, ref_ids


def build_retrieved_context_ids(retrieved: List[Dict[str, Any]]) -> List[str]:
    """
    Convert rag_query() retrieved contexts into IDs compatible with our reference IDs.
    We use the structure basename (doc) plus the retrieved node_ids if present.

    If node_ids are not present in the rag_query output, this returns an empty list and
    ID-based metrics are skipped.
    """
    out: List[str] = []
    for item in retrieved:
        if not isinstance(item, dict):
            continue
        doc_path = item.get("path") or item.get("doc_path")
        if not doc_path:
            continue
        doc_key = os.path.basename(str(doc_path))
        node_ids = item.get("node_ids") or item.get("retrieved_node_ids")
        if not isinstance(node_ids, list):
            continue
        for nid in node_ids:
            if isinstance(nid, str) and nid.strip():
                out.append(f"{doc_key}::{nid}")
    return out


async def run_rag(question: str, model: str, docindex_path: Optional[str]) -> Dict[str, Any]:
    from RAG.rag_query import rag_query

    return await rag_query(question, model=model, doc_index_path=docindex_path)


def _safe_list_str(x: Any) -> List[str]:
    if not isinstance(x, list):
        return []
    return [str(v) for v in x if isinstance(v, str) and v.strip()]


def _extract_contexts_from_rag_result(r: Dict[str, Any]) -> List[str]:
    retrieved = r.get("retrieved_contexts") or []
    if not isinstance(retrieved, list):
        return []
    contexts: List[str] = []
    for item in retrieved:
        if isinstance(item, dict):
            c = item.get("context")
            if isinstance(c, str) and c.strip():
                contexts.append(c)
        elif isinstance(item, str) and item.strip():
            contexts.append(item)
    return contexts


def _timestamp() -> str:
    return time.strftime("%Y%m%d_%H%M%S")


def _write_jsonl(path: str, rows: Iterable[Dict[str, Any]]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _write_scores_csv(path: str, rows: List[Dict[str, Any]]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if not rows:
        with open(path, "w", encoding="utf-8") as f:
            f.write("")
        return
    # Stable field order: core columns first, then metric columns alphabetically.
    core = ["question", "answer", "num_contexts", "num_reference_contexts"]
    metric_cols = sorted([k for k in rows[0].keys() if k not in core])
    fieldnames = core + metric_cols
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k) for k in fieldnames})


def _aggregate_summary(score_rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    metrics: Dict[str, List[float]] = {}
    for row in score_rows:
        for k, v in row.items():
            if k in {"question", "answer", "num_contexts", "num_reference_contexts"}:
                continue
            if isinstance(v, (int, float)):
                metrics.setdefault(k, []).append(float(v))
    summary: Dict[str, Any] = {"count": len(score_rows), "metrics": {}}
    for k, vals in metrics.items():
        if not vals:
            continue
        vals_sorted = sorted(vals)
        mean = sum(vals_sorted) / len(vals_sorted)
        median = vals_sorted[len(vals_sorted) // 2] if len(vals_sorted) % 2 else (vals_sorted[len(vals_sorted)//2 - 1] + vals_sorted[len(vals_sorted)//2]) / 2
        summary["metrics"][k] = {"mean": mean, "median": median}
    return summary


async def main_async() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, help="Path to JSONL dataset (question + references)")
    ap.add_argument("--results-dir", default=os.path.join(PROJECT_ROOT, "results"), help="Where *_structure.json files live")
    ap.add_argument("--docindex", default=None, help="Optional DocIndex.json path for rag_query")
    ap.add_argument("--model", default="qwen", help="LLM choice for the RAG pipeline (passed to rag_query)")
    ap.add_argument("--judge-model", default="gpt-4o-mini", help="OpenAI model name used by RAGAS judges")
    ap.add_argument("--out-dir", default=os.path.join(PROJECT_ROOT, "eval", "runs"), help="Base output directory")
    ap.add_argument("--max-items", type=int, default=0, help="If >0, evaluate only first N items")
    args = ap.parse_args()

    dataset_path = _ensure_abs(args.dataset)
    results_dir = _ensure_abs(args.results_dir)
    docindex_path = _ensure_abs(args.docindex) if args.docindex else None

    items = load_eval_items(dataset_path)
    if args.max_items and args.max_items > 0:
        items = items[: args.max_items]

    run_dir = os.path.join(_ensure_abs(args.out_dir), _timestamp())
    os.makedirs(run_dir, exist_ok=True)

    # RAGAS setup (OpenAI judging LLM)
    try:
        from openai import AsyncOpenAI
        from ragas import evaluate
        from ragas.llms import llm_factory
        from ragas.metrics.collections import AnswerRelevancy, ContextRecall, Faithfulness
        from datasets import Dataset
    except Exception as e:
        raise RuntimeError(
            "Missing required dependencies for RAGAS evaluation. "
            "Install with `pip install -r requirements.txt`."
        ) from e

    client = AsyncOpenAI()
    judge_llm = llm_factory(args.judge_model, client=client)

    metrics = [
        Faithfulness(llm=judge_llm),
        AnswerRelevancy(llm=judge_llm),
        ContextRecall(llm=judge_llm),
    ]

    raw_rows: List[Dict[str, Any]] = []
    ragas_rows: List[Dict[str, Any]] = []

    for idx, item in enumerate(items, start=1):
        rag_result = await run_rag(item.question, model=args.model, docindex_path=docindex_path)
        answer = rag_result.get("answer") if isinstance(rag_result, dict) else None
        if not isinstance(answer, str):
            answer = ""

        contexts = _extract_contexts_from_rag_result(rag_result if isinstance(rag_result, dict) else {})

        reference_contexts, reference_context_ids = await build_reference_contexts(
            item.references,
            results_dir=results_dir,
        )

        # For ContextRecall, RAGAS expects a "ground_truth" (reference answer). We don't have that.
        # We use the reference contexts text as a proxy reference by concatenating them.
        ground_truth = "\n\n---\n\n".join([c for c in reference_contexts if isinstance(c, str)])

        raw_rows.append(
            {
                "idx": idx,
                "question": item.question,
                "answer": answer,
                "contexts": contexts,
                "reference_contexts": reference_contexts,
                "references": [
                    {"doc": r.doc, "node_ids": list(r.node_ids)} for r in item.references
                ],
                "meta": item.meta,
                "rag_result": rag_result,
            }
        )

        ragas_rows.append(
            {
                "question": item.question,
                "answer": answer,
                "contexts": contexts,
                "ground_truth": ground_truth,
            }
        )

    _write_jsonl(os.path.join(run_dir, "items.jsonl"), raw_rows)

    ds = Dataset.from_list(ragas_rows)
    result = evaluate(ds, metrics=metrics, raise_exceptions=False, show_progress=True)
    df = result.to_pandas()

    score_rows: List[Dict[str, Any]] = []
    for i, row in enumerate(raw_rows):
        score_row: Dict[str, Any] = {
            "question": row["question"],
            "answer": row["answer"],
            "num_contexts": len(row.get("contexts") or []),
            "num_reference_contexts": len(row.get("reference_contexts") or []),
        }
        # Add metric columns from ragas dataframe
        for col in df.columns:
            v = df.iloc[i][col]
            if v is None:
                continue
            try:
                if isinstance(v, float) or isinstance(v, int):
                    score_row[str(col)] = float(v)
                else:
                    # pandas may store numpy types; try float cast
                    score_row[str(col)] = float(v)
            except Exception:
                continue
        score_rows.append(score_row)

    _write_scores_csv(os.path.join(run_dir, "scores.csv"), score_rows)
    summary = _aggregate_summary(score_rows)
    with open(os.path.join(run_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(run_dir)
    return 0


def main() -> int:
    return asyncio.run(main_async())


if __name__ == "__main__":
    raise SystemExit(main())

