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
import openai

load_dotenv()

# Configuration - paths relative to project root
PROJECT_ROOT = project_root
RESULTS_DIR = os.path.join(PROJECT_ROOT, 'results')
DOCINDEX_PATH = os.path.join(RESULTS_DIR, 'DocIndex')



async def rag_query(query: str, model: Optional[str] = None, doc_index_path: Optional[str] = None, progress_callback=None) -> Dict[str, Any]:
    """
    Main RAG query function.
    
    Args:
        query: User query/question
        model: LLM model to use (defaults to environment setting)
        doc_index_path: Optional path to DocIndex file
        progress_callback: Optional callback function(message: str) to receive progress updates
    
    Returns:
        Dictionary with query results including matched docs, retrieved nodes, and answer
    """
    def log(message):
        """Log message to both callback and print."""
        if progress_callback:
            progress_callback(message)
        print(message)
    
    # Load DocIndex
    global DOCINDEX_PATH
    if doc_index_path:
        DOCINDEX_PATH = doc_index_path
    
    doc_index = load_docindex(DOCINDEX_PATH)
    
    if not doc_index:
        return {
            "error": "No DocIndex found. Please run run_pageindex.py first.",
            "query": query
        }
    
    log(f"Query: {query}\n")
    log("=" * 80)
    
    # Match query to keywords
    log("\nStep 1: Matching query to keywords in DocIndex...")
    matched_docs = match_query_to_keywords(query, doc_index, model=model)
    
    if not matched_docs:
        return {
            "error": "No documents found matching the query.",
            "query": query,
            "matched_documents": []
        }
    
    log(f"Found {len(matched_docs)} matching document(s):")
    for doc in matched_docs:
        log(f"  - {doc}")
    
    # Load document structures
    log("\nStep 2: Loading document structures...")
    
    all_trees = []
    all_node_maps = []
    for doc_path in matched_docs:
        structure = load_document_structure(doc_path, results_dir=RESULTS_DIR, project_root=PROJECT_ROOT)
        all_trees, all_node_maps = extract_tree_and_node_map(structure, doc_path, all_trees, all_node_maps)
        
    
    if not all_trees:
        return {
            "error": "Could not load any document structures.",
            "query": query,
            "matched_documents": matched_docs
        }
    
    # Perform tree search for each document
    log("\nStep 3: Performing reasoning-based tree search...")
    all_retrieved_nodes = []
    
    for i, doc_info in enumerate(all_trees):
        log(f"\nSearching in: {os.path.basename(doc_info['path'])}")
        search_result = await tree_search(query, doc_info['tree'], model=model)
        
        thinking = search_result.get('thinking', 'N/A')
        log(f"\nReasoning Process:")
        # Use callback-aware print_wrapped
        wrapped_thinking = textwrap.fill(thinking, width=80)
        log(wrapped_thinking)
        
        node_ids = search_result.get('node_list', [])
        # Use callback-aware print_retrieved_nodes
        retrieved_lines = ["\nRetrieved Nodes:"]
        for node_id in node_ids:
            if node_id in all_node_maps[i]['node_map']:
                node = all_node_maps[i]['node_map'][node_id]
                retrieved_lines.append(f"  Node ID: {node['node_id']}\t Page: {node.get('page_index', node.get('start_index', 'N/A'))}\t Title: {node.get('title', 'Unknown')}")
        retrieved_info = "\n".join(retrieved_lines)
        log(retrieved_info)
        
        all_retrieved_nodes.append({
            'path': doc_info['path'],
            'node_ids': node_ids,
            'thinking': thinking
        })
    
    # Extract context from retrieved nodes
    log("\nStep 4: Extracting context from retrieved nodes...")
    all_contexts = []
    
    for i, doc_info in enumerate(all_node_maps):
        node_ids = all_retrieved_nodes[i]['node_ids']
        context = await extract_context(doc_info['node_map'], node_ids, doc_path=doc_info['path'], results_dir=RESULTS_DIR, project_root=PROJECT_ROOT)
        all_contexts.append({
            'path': doc_info['path'],
            'context': context
        })
        log(f"  Extracted {len(context)} characters from {os.path.basename(doc_info['path'])}")
    
    # Combine all contexts
    combined_context = "\n\n---\n\n".join([c['context'] for c in all_contexts])
    
    # Generate answer
    log("\nStep 5: Generating answer...")
    answer = await generate_answer(query, combined_context, model=model)
    
    log("\n" + "=" * 80)
    log("\nAnswer:")
    wrapped_answer = textwrap.fill(answer, width=80)
    log(wrapped_answer)
    
    return {
        "query": query,
        "matched_documents": matched_docs,
        "retrieved_nodes": all_retrieved_nodes,
        "context_length": len(combined_context),
        "answer": answer
    }


def main():
    parser = argparse.ArgumentParser(description='RAG Query using PageIndex DocIndex')
    parser.add_argument('--query', type=str, required=True, help='Query/question to answer')
    parser.add_argument('--model', type=str, default="ollama", help='LLM model to use (defaults to API_PROVIDER setting)')
    parser.add_argument('--docindex', type=str, default=None, help='Path to DocIndex file (default: ./results/DocIndex)')
    parser.add_argument('--results-dir', type=str, default='./results', help='Results directory (default: ./results)')
    
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
        DOCINDEX_PATH = os.path.join(RESULTS_DIR, 'DocIndex')
    
    # Get model from environment if not specified
    model = "gpt-5.1"
    # if not model:
    #     api_provider = os.getenv("API_PROVIDER", "ollama").lower()
    #     if api_provider == "ollama":
    #         model = os.getenv("OLLAMA_MODEL", "llama3.1:8b")
    #     else:
    #         model = "gpt-4o-2024-11-20"
    
    # Run RAG query
    result = asyncio.run(rag_query(args.query, model=model, doc_index_path=DOCINDEX_PATH))
    
    # Print summary
    if 'error' in result:
        print(f"\nError: {result['error']}")
    else:
        print(f"\n\nSummary:")
        print(f"  Query: {result['query']}")
        print(f"  Matched Documents: {len(result['matched_documents'])}")
        print(f"  Total Retrieved Nodes: {sum(len(r['node_ids']) for r in result['retrieved_nodes'])}")
        print(f"  Context Length: {result['context_length']} characters")


if __name__ == "__main__":
    main()
