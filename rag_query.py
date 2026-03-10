#!/usr/bin/env python3
"""
RAG Query Script using PageIndex DocIndex

This script performs reasoning-based RAG by:
1. Matching queries to keywords in DocIndex
2. Loading corresponding document structures
3. Performing tree search to find relevant nodes
4. Extracting context and generating answers

Usage:
    python rag_query.py --query "What are the conclusions in this document?"
    python rag_query.py --query "Explain the methodology" --model llama3.1:8b
    python rag_query.py --query "What is discussed about X?" --results-dir ./custom_results

Requirements:
    - DocIndex file must exist at ./results/DocIndex (or specified path)
    - Document structure JSON files must exist in results directory
    - Original PDF files should be available for text extraction (optional)
"""

import os
import json
import argparse
import asyncio
from collections import defaultdict
from typing import List, Dict, Any, Optional
from pageindex.utils import ChatGPT_API, ChatGPT_API_async, get_model_name, extract_json
import openai
from dotenv import load_dotenv

load_dotenv()

# Configuration
RESULTS_DIR = './results'
DOCINDEX_PATH = os.path.join(RESULTS_DIR, 'DocIndex')


def load_docindex() -> Dict[str, List[str]]:
    """Load DocIndex from file."""
    if not os.path.exists(DOCINDEX_PATH):
        print(f"DocIndex not found at {DOCINDEX_PATH}")
        print("Please run run_pageindex.py first to generate document structures and DocIndex.")
        return {}
    
    with open(DOCINDEX_PATH, 'r', encoding='utf-8') as f:
        doc_index = json.load(f)
    
    # Convert to defaultdict for easier handling
    if isinstance(doc_index, dict):
        doc_index = defaultdict(list, doc_index)
    
    return doc_index


def load_document_structure(file_path: str) -> Dict[str, Any]:
    """Load document structure from JSON file."""
    # Handle relative paths
    if not os.path.isabs(file_path):
        file_path = os.path.join(RESULTS_DIR, os.path.basename(file_path))
    
    if not os.path.exists(file_path):
        print(f"Warning: Document structure file not found: {file_path}")
        return None
    
    with open(file_path, 'r', encoding='utf-8') as f:
        structure = json.load(f)
    
    return structure


def match_query_to_keywords(query: str, doc_index: Dict[str, List[str]], model: Optional[str] = None) -> List[str]:
    """
    Match query to keywords in DocIndex using LLM.
    Returns list of document file paths that match the query.
    """
    if not doc_index:
        return []
    
    # Get all keywords
    keywords = list(doc_index.keys())
    
    if not keywords:
        return []
    
    # Use LLM to find relevant keywords
    prompt = f"""You are given a query and a list of keywords from a document index.
Your task is to identify which keywords are relevant to answering the query.

Query: {query}

Keywords:
{json.dumps(keywords, indent=2)}

Please reply in the following JSON format:
{{
    "thinking": "<Your thinking process on which keywords are relevant>",
    "relevant_keywords": ["keyword1", "keyword2", ...]
}}

Return ONLY the JSON object. Do not include any other text."""

    try:
        response = ChatGPT_API(model=model, prompt=prompt)
        
        # Extract JSON from response
        result = extract_json(response)
        relevant_keywords = result.get('relevant_keywords', [])
    except Exception as e:
        print(f"Error matching keywords: {e}")
        print(f"Response: {response}")
        # Fallback: return all keywords if matching fails
        relevant_keywords = keywords
    
    # Collect all document paths for relevant keywords
    matched_docs = set()
    for keyword in relevant_keywords:
        if keyword in doc_index:
            matched_docs.update(doc_index[keyword])
    
    return list(matched_docs)


def create_node_mapping(tree: List[Dict[str, Any]], doc_path: Optional[str] = None) -> Dict[str, Dict[str, Any]]:
    """
    Create a mapping from node_id to node for easy lookup.
    Handles both list and dict tree structures.
    Also adds page_index if available.
    """
    node_map = {}
    node_counter = 0
    
    def traverse(nodes, parent_id=None):
        nonlocal node_counter
        if isinstance(nodes, list):
            for idx, node in enumerate(nodes):
                if isinstance(node, dict):
                    # Generate node_id if missing
                    if 'node_id' not in node:
                        node['node_id'] = f"{node_counter:04d}"
                        node_counter += 1
                    
                    node_id = node['node_id']
                    
                    # Add page_index if start_index exists
                    if 'start_index' in node:
                        node['page_index'] = node['start_index']
                    
                    node_map[node_id] = node.copy()
                    
                    if 'nodes' in node and node['nodes']:
                        traverse(node['nodes'], parent_id=node_id)
        elif isinstance(nodes, dict):
            if 'node_id' not in nodes:
                nodes['node_id'] = f"{node_counter:04d}"
                node_counter += 1
            
            node_id = nodes['node_id']
            if 'start_index' in nodes:
                nodes['page_index'] = nodes['start_index']
            
            node_map[node_id] = nodes.copy()
            
            if 'nodes' in nodes and nodes['nodes']:
                traverse(nodes['nodes'], parent_id=node_id)
    
    # Handle different tree structures
    if isinstance(tree, dict):
        if 'structure' in tree:
            traverse(tree['structure'])
        else:
            traverse(tree)
    else:
        traverse(tree)
    
    return node_map


def remove_fields(data: Any, fields: List[str] = ['text']) -> Any:
    """Remove specified fields from tree structure."""
    if isinstance(data, dict):
        return {k: remove_fields(v, fields) for k, v in data.items() if k not in fields}
    elif isinstance(data, list):
        return [remove_fields(item, fields) for item in data]
    else:
        return data


def print_tree(tree: List[Dict[str, Any]], indent: int = 0):
    """Print tree structure in a readable format."""
    if isinstance(tree, dict):
        tree = [tree]
    
    for node in tree:
        title = node.get('title', 'Unknown')
        node_id = node.get('node_id', 'N/A')
        summary = node.get('summary', '')[:100] if node.get('summary') else ''
        
        print('  ' * indent + f"- {title} (ID: {node_id})")
        if summary:
            print('  ' * (indent + 1) + f"  Summary: {summary}...")
        
        if 'nodes' in node and node['nodes']:
            print_tree(node['nodes'], indent + 1)


def print_wrapped(text: str, width: int = 80):
    """Print text with word wrapping."""
    import textwrap
    print(textwrap.fill(text, width=width))


async def tree_search(query: str, tree: List[Dict[str, Any]], model: Optional[str] = None) -> Dict[str, Any]:
    """
    Perform tree search to find relevant nodes for the query.
    Similar to the notebook's tree search step.
    """
    # Remove text fields to reduce token usage
    tree_without_text = remove_fields(tree.copy(), fields=['text'])
    
    search_prompt = f"""
You are given a question and a tree structure of a document.
Each node contains a node id, node title, and a corresponding summary.
Your task is to find all nodes that are likely to contain the answer to the question.

Question: {query}

Document tree structure:
{json.dumps(tree_without_text, indent=2)}

Please reply in the following JSON format:
{{
    "thinking": "<Your thinking process on which nodes are relevant to the question>",
    "node_list": ["node_id_1", "node_id_2", ..., "node_id_n"]
}}
Directly return the final JSON structure. Do not output anything else.
"""

    try:
        response = await ChatGPT_API_async(model=model, prompt=search_prompt)
        
        # Extract JSON from response
        result = extract_json(response)
        
        return result
    except Exception as e:
        print(f"Error in tree search: {e}")
        print(f"Response: {response}")
        return {"thinking": "", "node_list": []}


async def extract_context(node_map: Dict[str, Dict[str, Any]], node_ids: List[str], doc_path: Optional[str] = None) -> str:
    """
    Extract text content from retrieved nodes.
    If text is not available, try to extract from PDF using start_index and end_index.
    """
    from pageindex.utils import get_text_of_pdf_pages
    
    context_parts = []
    for node_id in node_ids:
        if node_id in node_map:
            node = node_map[node_id]
            
            # Try different field names for text content
            text = node.get('text') or node.get('content') or node.get('summary', '')
            
            # If no text available, try to extract from PDF
            if not text and doc_path and 'start_index' in node and 'end_index' in node:
                try:
                    # Try to find original PDF path
                    pdf_path = doc_path.replace('_structure.json', '.pdf')
                    if not os.path.exists(pdf_path):
                        # Try without structure suffix
                        pdf_path = doc_path.replace('structure.json', '.pdf')
                    
                    if os.path.exists(pdf_path):
                        import PyPDF2
                        pdf_reader = PyPDF2.PdfReader(pdf_path)
                        start_idx = node['start_index'] - 1  # Convert to 0-based
                        end_idx = node['end_index']
                        
                        page_texts = []
                        for page_num in range(start_idx, min(end_idx, len(pdf_reader.pages))):
                            page = pdf_reader.pages[page_num]
                            page_texts.append(page.extract_text())
                        
                        text = '\n\n'.join(page_texts)
                except Exception as e:
                    print(f"  Warning: Could not extract text from PDF for node {node_id}: {e}")
            
            # Fallback to summary or title if no text
            if not text:
                text = node.get('summary', '') or node.get('title', '')
            
            if text:
                # Add title as header
                title = node.get('title', f'Node {node_id}')
                context_parts.append(f"## {title}\n\n{text}")
    
    return "\n\n---\n\n".join(context_parts)


async def generate_answer(query: str, context: str, model: Optional[str] = None) -> str:
    """Generate answer based on query and context."""
    answer_prompt = f"""
Answer the question based on the context:

Question: {query}
Context: {context}

Provide a clear, concise answer based only on the context provided.
"""

    try:
        response = await ChatGPT_API_async(model=model, prompt=answer_prompt)
        return response.strip()
    except Exception as e:
        print(f"Error generating answer: {e}")
        return "Error generating answer."


async def rag_query(query: str, model: Optional[str] = None, doc_index_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Main RAG query function.
    
    Args:
        query: User query/question
        model: LLM model to use (defaults to environment setting)
        doc_index_path: Optional path to DocIndex file
    
    Returns:
        Dictionary with query results including matched docs, retrieved nodes, and answer
    """
    # Load DocIndex
    global DOCINDEX_PATH
    if doc_index_path:
        DOCINDEX_PATH = doc_index_path
    
    doc_index = load_docindex()
    
    if not doc_index:
        return {
            "error": "No DocIndex found. Please run run_pageindex.py first.",
            "query": query
        }
    
    print(f"Query: {query}\n")
    print("=" * 80)
    
    # Match query to keywords
    print("\nStep 1: Matching query to keywords in DocIndex...")
    matched_docs = match_query_to_keywords(query, doc_index, model=model)
    
    if not matched_docs:
        return {
            "error": "No documents found matching the query.",
            "query": query,
            "matched_documents": []
        }
    
    print(f"Found {len(matched_docs)} matching document(s):")
    for doc in matched_docs:
        print(f"  - {doc}")
    
    # Load document structures
    print("\nStep 2: Loading document structures...")
    all_trees = []
    all_node_maps = []
    
    for doc_path in matched_docs:
        structure = load_document_structure(doc_path)
        if structure:
            # Extract tree structure (handle different formats)
            if isinstance(structure, dict):
                tree = structure.get('structure', structure)
            else:
                tree = structure
            
            if tree:
                all_trees.append({
                    'path': doc_path,
                    'tree': tree
                })
                node_map = create_node_mapping(tree, doc_path=doc_path)
                all_node_maps.append({
                    'path': doc_path,
                    'node_map': node_map
                })
                print(f"  Loaded: {os.path.basename(doc_path)} ({len(node_map)} nodes)")
    
    if not all_trees:
        return {
            "error": "Could not load any document structures.",
            "query": query,
            "matched_documents": matched_docs
        }
    
    # Perform tree search for each document
    print("\nStep 3: Performing reasoning-based tree search...")
    all_retrieved_nodes = []
    
    for i, doc_info in enumerate(all_trees):
        print(f"\nSearching in: {os.path.basename(doc_info['path'])}")
        search_result = await tree_search(query, doc_info['tree'], model=model)
        
        thinking = search_result.get('thinking', 'N/A')
        print(f"\nReasoning Process:")
        print_wrapped(thinking)
        
        node_ids = search_result.get('node_list', [])
        print(f"\nRetrieved Nodes:")
        node_map = all_node_maps[i]['node_map']
        for node_id in node_ids:
            if node_id in node_map:
                node = node_map[node_id]
                page_info = node.get('page_index', node.get('start_index', 'N/A'))
                print(f"  Node ID: {node['node_id']}\t Page: {page_info}\t Title: {node.get('title', 'Unknown')}")
        
        all_retrieved_nodes.append({
            'path': doc_info['path'],
            'node_ids': node_ids,
            'thinking': thinking
        })
    
    # Extract context from retrieved nodes
    print("\nStep 4: Extracting context from retrieved nodes...")
    all_contexts = []
    
    for i, doc_info in enumerate(all_node_maps):
        node_ids = all_retrieved_nodes[i]['node_ids']
        context = await extract_context(doc_info['node_map'], node_ids, doc_path=doc_info['path'])
        all_contexts.append({
            'path': doc_info['path'],
            'context': context
        })
        print(f"  Extracted {len(context)} characters from {os.path.basename(doc_info['path'])}")
    
    # Combine all contexts
    combined_context = "\n\n---\n\n".join([c['context'] for c in all_contexts])
    
    # Generate answer
    print("\nStep 5: Generating answer...")
    answer = await generate_answer(query, combined_context, model=model)
    
    print("\n" + "=" * 80)
    print("\nAnswer:")
    print_wrapped(answer)
    
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
    parser.add_argument('--model', type=str, default=None, help='LLM model to use (defaults to API_PROVIDER setting)')
    parser.add_argument('--docindex', type=str, default=None, help='Path to DocIndex file (default: ./results/DocIndex)')
    parser.add_argument('--results-dir', type=str, default='./results', help='Results directory (default: ./results)')
    
    args = parser.parse_args()
    
    # Update global paths
    global RESULTS_DIR, DOCINDEX_PATH
    RESULTS_DIR = args.results_dir
    if args.docindex:
        DOCINDEX_PATH = args.docindex
    else:
        DOCINDEX_PATH = os.path.join(RESULTS_DIR, 'DocIndex')
    
    # Get model from environment if not specified
    model = args.model
    if not model:
        import os
        api_provider = os.getenv("API_PROVIDER", "ollama").lower()
        if api_provider == "ollama":
            model = os.getenv("OLLAMA_MODEL", "llama3.1:8b")
        else:
            model = "gpt-4o-2024-11-20"
    
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
