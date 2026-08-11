#!/usr/bin/env python3
"""
RAG Query Script using PageIndex DocIndex

This script performs reasoning-based RAG by:
1. Matching queries to keywords in DocIndex
2. Loading corresponding document structures
3. Performing tree search to find relevant nodes
4. Extracting context and generating answers

Usage:
    # From project root:
    python3 RAG/rag_query.py --query "What are the conclusions in this document?"
    python3 RAG/rag_query.py --query "Explain the methodology" --model llama3.1:8b
    
    # From RAG directory:
    cd RAG
    python3 rag_query.py --query "What is discussed about X?"
    
    # With custom paths:
    python3 RAG/rag_query.py --query "..." --results-dir ./custom_results --docindex ./custom_results/DocIndex

Requirements:
    - DocIndex file must exist at ./results/DocIndex (relative to project root, or specified path)
    - Document structure JSON files must exist in results directory
    - Original PDF files should be available for text extraction (optional)
    - Script can be run from any directory; paths are resolved relative to project root
"""

import os
import sys
import json
import time
import argparse
import asyncio
import textwrap
from collections import defaultdict
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv

# Add parent directory to path to ensure imports work
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Import utils after path setup
from RAG.utils import *
from pageindex.utils import ChatGPT_API, ChatGPT_API_async, get_model_name, extract_json
from pageindex.model_registry import get_rag_chat_model, get_rag_embed_model, get_rag_tree_search_model, get_rag_max_tree_search_docs, validate_configured_models
import openai

load_dotenv()

# Configuration - paths relative to project root
PROJECT_ROOT = project_root
RESULTS_DIR = os.path.join(PROJECT_ROOT, 'results')
DOCINDEX_PATH = os.path.join(RESULTS_DIR, 'DocIndex.json')


def _finish_rag_step(
    step_timings: List[Dict[str, Any]],
    label: str,
    t0: float,
    log_fn: Optional[Any] = None,
) -> float:
    sec = round(time.perf_counter() - t0, 3)
    step_timings.append({"step": label, "seconds": sec})
    if log_fn:
        log_fn(f"  {label} — completed in {sec} s")
    return sec


async def rag_query(
    query: str,
    model: Optional[str] = None,
    doc_index_path: Optional[str] = None,
    progress_callback=None,
    keyword_match_method: str = "embed",
    keyword_match_top_k: int = 8,
    keyword_match_threshold: float = 0.30,
    embed_model: Optional[str] = None,
    max_tree_search_docs: Optional[int] = None,
    conversation_history: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """
    Main RAG query function.
    
    Args:
        query: User query/question
        model: LLM model to use (defaults to environment setting)
        doc_index_path: Optional path to DocIndex file
        progress_callback: Optional callback function(message: str) to receive progress updates
        conversation_history: Optional list of prior turns ``[{"role": "user"|"assistant", "content": "..."}]``
    
    Returns:
        Dictionary with query results, including ``step_timings`` (``step`` / ``seconds`` per stage).
    """
    def log(message):
        """Log message to progress callback (UI) and terminal."""
        if progress_callback:
            progress_callback(message)
        text = (message or "").rstrip()
        if text:
            print(text, flush=True)

    global DOCINDEX_PATH

    model = model or get_rag_chat_model()
    embed_model = embed_model or get_rag_embed_model()
    tree_search_model = get_rag_tree_search_model() or model
    from pageindex.model_registry import uses_ollama_chat_provider
    if tree_search_model != model and not uses_ollama_chat_provider(tree_search_model):
        log(f"Tree search model {tree_search_model} is not an Ollama alias; using chat model {model}")
        tree_search_model = model
    log(f"Using chat model: {model} (resolved: {get_model_name(model)})")
    if tree_search_model != model:
        log(f"Using tree search model: {tree_search_model} (resolved: {get_model_name(tree_search_model)})")
    log(f"Using keyword embed model: {embed_model}")
    for warning in validate_configured_models():
        log(f"Model config warning: {warning}")

    # Load DocIndex
    if doc_index_path:
        DOCINDEX_PATH = doc_index_path
    
    step_timings: List[Dict[str, Any]] = []

    log("Loading DocIndex...")
    doc_index = load_docindex(DOCINDEX_PATH, quiet=True)
    if doc_index:
        log(f"DocIndex loaded ({len(doc_index)} keywords)")
    
    if not doc_index:
        return {
            "error": "No DocIndex found. Please run run_pageindex.py first.",
            "query": query,
            "step_timings": step_timings,
        }
    
    history = normalize_conversation_history(conversation_history)
    retrieval_query = build_retrieval_query(query, history)
    if history and retrieval_query != query:
        log(f"Retrieval query (with conversation context): {retrieval_query}")

    log(f"Query: {query}\n")
    log("=" * 80)
    
    # Match query to keywords
    log("\nStep 1: Matching query to keywords in DocIndex...")
    log(f"  Using keyword match method: {keyword_match_method}")
    embed_model_resolved = embed_model or None
    if keyword_match_method == "embed":
        warm_docindex_keyword_embeddings(
            doc_index,
            embed_model=embed_model_resolved,
            progress_callback=log,
        )
    t0 = time.perf_counter()
    matched_docs = match_query_to_keywords(
        retrieval_query,
        doc_index,
        model=model,
        method=keyword_match_method,
        embed_model=embed_model,
        top_k=keyword_match_top_k,
        min_similarity=keyword_match_threshold,
        progress_callback=log,
    )
    _finish_rag_step(
        step_timings, "Step 1: Matching query to keywords in DocIndex", t0, log_fn=log
    )
    
    if not matched_docs:
        return {
            "error": "No documents found matching the query.",
            "query": query,
            "matched_documents": [],
            "step_timings": step_timings,
        }
    
    log(f"Found {len(matched_docs)} matching document(s):")
    for doc in matched_docs:
        log(f"  - {doc}")

    max_tree_search_docs = max_tree_search_docs if max_tree_search_docs is not None else get_rag_max_tree_search_docs()
    docs_for_retrieval = matched_docs
    if max_tree_search_docs and len(matched_docs) > max_tree_search_docs:
        log(
            f"\nLimiting tree search to top {max_tree_search_docs} document(s) "
            f"(of {len(matched_docs)} matched)."
        )
        docs_for_retrieval = matched_docs[:max_tree_search_docs]
        for doc in docs_for_retrieval:
            log(f"  - {doc}")
    
    # Load document structures
    log("\nStep 2: Loading document structures...")
    t0 = time.perf_counter()
    all_trees_node_maps = []
    for structure_path in docs_for_retrieval:
        structure = load_document_structure(structure_path, results_dir=RESULTS_DIR, project_root=PROJECT_ROOT)
        all_trees_node_maps = extract_tree_and_node_map(structure, structure_path, all_trees_node_maps)
    _finish_rag_step(step_timings, "Step 2: Loading document structures", t0, log_fn=log)
    
    if not all_trees_node_maps:
        return {
            "error": "Could not load any document structures.",
            "query": query,
            "matched_documents": matched_docs,
            "step_timings": step_timings,
        }
    
    # Perform tree search for each document (parallel)
    log("\nStep 3: Performing reasoning-based tree search...")
    log(f"  Using tree search model: {tree_search_model} (resolved: {get_model_name(tree_search_model)})")
    log(f"  Searching {len(all_trees_node_maps)} document(s)")
    t0 = time.perf_counter()

    async def _tree_search_one(doc_info: Dict[str, Any]) -> tuple[Dict[str, Any], Dict[str, Any]]:
        log(f"\nSearching in: {os.path.basename(doc_info['path'])}")
        search_result = await tree_search(
            query, doc_info, model=tree_search_model, conversation_history=history
        )
        return doc_info, search_result

    search_pairs = await asyncio.gather(
        *(_tree_search_one(doc_info) for doc_info in all_trees_node_maps),
        return_exceptions=True,
    )

    for item in search_pairs:
        if isinstance(item, Exception):
            log(f"Tree search error: {item}")
            continue
        doc_info, search_result = item

        thinking = search_result.get("thinking", "N/A")
        log(f"\nReasoning Process ({os.path.basename(doc_info['path'])}):")
        log(textwrap.fill(thinking, width=80))

        node_ids = search_result.get("node_list", [])
        retrieved_lines = ["\nRetrieved Nodes:"]
        for node_id in node_ids:
            if node_id in doc_info["node_map"]:
                node = doc_info["node_map"][node_id]
                retrieved_lines.append(
                    f"  Node ID: {node['node_id']}\t Page: "
                    f"{node.get('page_index', node.get('start_index', 'N/A'))}\t "
                    f"Title: {node.get('title', 'Unknown')}"
                )
        log("\n".join(retrieved_lines))

        doc_info["retrieved_node_ids"] = node_ids

    _finish_rag_step(
        step_timings, "Step 3: Performing reasoning-based tree search", t0, log_fn=log
    )
    
    # Extract context from retrieved nodes
    log("\nStep 4: Extracting context from retrieved nodes...")
    t0 = time.perf_counter()
    for doc_info in all_trees_node_maps:
        node_ids = doc_info['retrieved_node_ids']
        context = await extract_context(doc_info['node_map'], node_ids, doc_path=doc_info['path'], results_dir=RESULTS_DIR, project_root=PROJECT_ROOT)
        # if len(context)>0:
        doc_info['context'] = context
        
        log(f"  Extracted {len(context)} characters from {os.path.basename(doc_info['path'])}")
    _finish_rag_step(
        step_timings, "Step 4: Extracting context from retrieved nodes", t0, log_fn=log
    )
    # print(f"all_contexts length: {len(all_contexts)}")
    # Combine all contexts
    
    # Generate answer with inline citations
    log("\nStep 5: Generating answer...")
    t0 = time.perf_counter()
    all_trees_node_maps_with_answers = await generate_answer_for_each_context(
        query, all_trees_node_maps, model=model, conversation_history=history
    )
    _finish_rag_step(
        step_timings, "Step 5: Generating per-document answers", t0, log_fn=log
    )
    log("\nStep 6: Combining answers...")
    t0 = time.perf_counter()
    answer, citation_sources = await combine_answers(
        query, all_trees_node_maps_with_answers, model=model, conversation_history=history
    )
    _finish_rag_step(step_timings, "Step 6: Combining answers", t0, log_fn=log)
    
    log("\n" + "=" * 80)
    log("\nAnswer:")
    wrapped_answer = textwrap.fill(answer, width=80)
    log(wrapped_answer)
    
    retrieved_contexts = [
        {"path": d["path"], "context": d.get("context", ""), "doc_path": d.get("doc_path", d["path"])}
        for d in all_trees_node_maps
        if len(d.get("context", "")) > 0
    ]
    
    return {
        "query": query,
        "matched_documents": matched_docs,
        "retrieved_contexts": retrieved_contexts,
        "answer": answer,
        "sources": citation_sources,
        "step_timings": step_timings,
    }


def main():
    parser = argparse.ArgumentParser(description='RAG Query using PageIndex DocIndex')
    parser.add_argument('--query', type=str, required=True, help='Query/question to answer')
    parser.add_argument(
        '--model',
        type=str,
        default=None,
        help='Chat model alias or name (default: llm_models.yaml rag.chat_model)',
    )
    parser.add_argument('--docindex', type=str, default=None, help='Path to DocIndex file (default: ./results/DocIndex)')
    parser.add_argument('--results-dir', type=str, default='./results', help='Results directory (default: ./results)')
    parser.add_argument('--out', type=str, default=None, help='Optional path to write full result JSON')
    parser.add_argument(
        '--keyword-match-method',
        choices=['embed', 'llm'],
        default='embed',
        help='How to match query to DocIndex keywords (default: embed)',
    )
    parser.add_argument(
        '--keyword-match-top-k',
        type=int,
        default=8,
        help='Max keywords to select when using embed matching (default: 8)',
    )
    parser.add_argument(
        '--keyword-match-threshold',
        type=float,
        default=0.30,
        help='Min cosine similarity for embed keyword matching (default: 0.30)',
    )
    parser.add_argument(
        '--embed-model',
        type=str,
        default=None,
        help='Embedding model for keyword match (default: llm_models.yaml rag.keyword_embed_model)',
    )
    parser.add_argument(
        '--max-tree-search-docs',
        type=int,
        default=None,
        help='Max documents for tree search (default: llm_models.yaml rag.max_tree_search_docs)',
    )
    
    args = parser.parse_args()
    
    # Update global paths
    global RESULTS_DIR, DOCINDEX_PATH
    # Handle relative paths for results_dir
    if not os.path.isabs(args.results_dir):
        RESULTS_DIR = os.path.join(PROJECT_ROOT, args.results_dir.lstrip('./'))
    else:
        RESULTS_DIR = args.results_dir
    
    if args.docindex:
        if not os.path.isabs(args.docindex):
            DOCINDEX_PATH = os.path.join(PROJECT_ROOT, args.docindex.lstrip('./'))
        else:
            DOCINDEX_PATH = args.docindex
    else:
        DOCINDEX_PATH = os.path.join(RESULTS_DIR, 'DocIndex.json')
    
    # Get model from registry when CLI omits --model
    model = args.model or get_rag_chat_model()
    # if not model:
    #     api_provider = os.getenv("API_PROVIDER", "ollama").lower()
    #     if api_provider == "ollama":
    #         model = os.getenv("OLLAMA_MODEL", "llama3.1:8b")
    #     else:
    #         model = "gpt-4o-2024-11-20"
    
    # Run RAG query
    result = asyncio.run(
        rag_query(
            args.query,
            model=model,
            doc_index_path=DOCINDEX_PATH,
            keyword_match_method=args.keyword_match_method,
            keyword_match_top_k=args.keyword_match_top_k,
            keyword_match_threshold=args.keyword_match_threshold,
            embed_model=args.embed_model,
            max_tree_search_docs=args.max_tree_search_docs,
        )
    )
    if args.out:
        try:
            os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
            with open(args.out, "w", encoding="utf-8") as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
            print(f"Wrote result to: {args.out}")
        except Exception as e:
            print(f"Warning: could not write result JSON: {e}")
    
    # Print summary
    if 'error' in result:
        print(f"\nError: {result['error']}")
    else:
        print(f"\n\nSummary:")
        print(f"  Query: {result['query']}")
        print(f"  Matched Documents: {len(result.get('matched_documents', []))}")
        print(f"  Retrieved Contexts: {len(result.get('retrieved_contexts', []))}")
        for item in result.get("step_timings") or []:
            print(f"  {item.get('step')}: {item.get('seconds')} s")


if __name__ == "__main__":
    main()