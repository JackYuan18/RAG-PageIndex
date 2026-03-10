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
from collections import defaultdict
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv

# Add parent directory to path to ensure imports work
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from pageindex.utils import ChatGPT_API, ChatGPT_API_async, get_model_name, extract_json
import openai

load_dotenv()

# Configuration - paths relative to project root


def load_docindex(docindex_path: Optional[str] = None) -> Dict[str, List[str]]:
    """Load DocIndex from file."""
    # Use provided path or default to module-level DOCINDEX_PATH
    path = docindex_path 
    
    if not os.path.exists(path):
        print(f"DocIndex not found at {path}")
        print("Please run run_pageindex.py first to generate document structures and DocIndex.")
        return {}
    
    print(f"Loading DocIndex from {path}")
    with open(path, 'r', encoding='utf-8') as f:
        doc_index = json.load(f)
    
    # Convert to defaultdict for easier handling
    if isinstance(doc_index, dict):
        doc_index = defaultdict(list, doc_index)
    
    print(f"DocIndex loaded successfully with {len(doc_index)} keywords")
    return doc_index


def load_document_structure(file_path: str, results_dir: Optional[str] = None, project_root: Optional[str] = None) -> Dict[str, Any]:
    """Load document structure from JSON file."""
    # Calculate project root if not provided
    if project_root is None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(script_dir)
    
    # Calculate results dir if not provided
    if results_dir is None:
        results_dir = os.path.join(project_root, 'results')
    
    # Handle relative paths - check multiple possible locations
    if not os.path.isabs(file_path):
        # Try relative to results directory first
        potential_path = os.path.join(results_dir, os.path.basename(file_path))
        if os.path.exists(potential_path):
            file_path = potential_path
        else:
            # Try relative to project root
            potential_path = os.path.join(project_root, file_path.lstrip('./'))
            if os.path.exists(potential_path):
                file_path = potential_path
            else:
                # Try as-is (might be relative to current working directory)
                if not os.path.exists(file_path):
                    file_path = potential_path
    
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
        print(f"Relevant keywords: {relevant_keywords}")
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
    You are given a question and a hierarchical tree structure of a document.
    The tree structure has parent nodes that may contain child nodes (nested in a "nodes" field).
    Each node contains a node id, node title, and a corresponding summary.

    Your task is to find all nodes that are likely to contain the answer to the question.

    IMPORTANT RULES:
    1. If a parent node is relevant, you may also need to include its child nodes for complete context
    2. However, only include child nodes if they are also relevant to the question
    3. You can select both parent and child nodes if both are relevant
    4. Consider the hierarchical relationship: parent nodes often provide overview, child nodes provide details

    Question: {query}

    Document tree structure:
    {json.dumps(tree_without_text, indent=2)}

    Please reply in the following JSON format:
    {{
        "thinking": "<Your thinking process on which nodes are relevant, including consideration of parent-child relationships>",
        "node_list": ["node_id_1", "node_id_2", ..., "node_id_n"]
    }}

    Return the node_ids of all relevant nodes (both parents and children if relevant).
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


async def extract_context(node_map: Dict[str, Dict[str, Any]], node_ids: List[str], doc_path: Optional[str] = None, results_dir: Optional[str] = None, project_root: Optional[str] = None) -> str:
    """
    Extract text content from retrieved nodes.
    If text is not available, try to extract from PDF using start_index and end_index.
    """
    from pageindex.utils import get_text_of_pdf_pages
    
    # Calculate project root if not provided
    if project_root is None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(script_dir)
    
    # Calculate results dir if not provided
    if results_dir is None:
        results_dir = os.path.join(project_root, 'results')
    
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
                    # Handle both absolute and relative paths
                    if not os.path.isabs(doc_path):
                        doc_path_full = os.path.join(results_dir, os.path.basename(doc_path))
                    else:
                        doc_path_full = doc_path
                    
                    pdf_path = doc_path_full.replace('_structure.json', '.pdf')
                    if not os.path.exists(pdf_path):
                        # Try without structure suffix
                        pdf_path = doc_path_full.replace('structure.json', '.pdf')
                    
                    # Also try in project root or tests/pdfs directory
                    if not os.path.exists(pdf_path):
                        pdf_name = os.path.basename(pdf_path)
                        # Try tests/pdfs directory
                        test_pdf_path = os.path.join(project_root, 'tests', 'pdfs', pdf_name)
                        if os.path.exists(test_pdf_path):
                            pdf_path = test_pdf_path
                        # Try results directory
                        elif os.path.exists(os.path.join(results_dir, pdf_name)):
                            pdf_path = os.path.join(results_dir, pdf_name)
                    
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
    answer_prompt = f"""You are a helpful assistant answering queries based on provided document context.

User query: {query}

Relevant Context from Documents:
{context}

Instructions:
1. Answer the query directly and naturally, as if you're explaining to someone who asked
2. Use the context to provide specific details, examples, or explanations
3. If the query asks "how", explain the process or method
4. If the query asks "what", provide definitions or descriptions
5. If the query asks "why", explain reasons or motivations
6. Structure your answer to directly address what was asked
7. If information is not available in the context, acknowledge this but provide what you can
8. Write in a clear, natural, and conversational tone
9. Use the exact terminology and phrasing from the context when appropriate

Answer:"""

    try:
        response = await ChatGPT_API_async(model=model, prompt=answer_prompt)
        return response.strip()
    except Exception as e:
        print(f"Error generating answer: {e}")
        return "Error generating answer."


def extract_tree_and_node_map(structure: Dict[str, Any], doc_path: str, all_trees: List[Dict[str, Any]], all_node_maps: List[Dict[str, Any]]) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Extract tree structure and node mapping from document structure."""
   
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
    return all_trees, all_node_maps

def print_retrieved_nodes(node_ids: List[str], node_map: Dict[str, Dict[str, Any]]) -> None:
    """Print retrieved nodes in a readable format."""
    print(f"\nRetrieved Nodes:")
    for node_id in node_ids:
        if node_id in node_map:
            node = node_map[node_id]
            print(f"  Node ID: {node['node_id']}\t Page: {node.get('page_index', node.get('start_index', 'N/A'))}\t Title: {node.get('title', 'Unknown')}")

        
if __name__ == "__main__":
    main()
