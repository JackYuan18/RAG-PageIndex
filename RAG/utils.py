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
from urllib.parse import quote
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

async def combine_answers(query: str, all_trees_node_maps_with_answers: List[Dict[str, Any]], model: Optional[str] = None) -> tuple[str, List[Dict[str, str]]]:
    """
    Combine answers from multiple documents into a single answer with inline citations.
    Returns (answer_text, citation_sources) where citation_sources is [{name, url}, ...] for the UI.
    """
    combined_answer = {}
    citation_sources = []
    seen_basenames = set()
    for doc_info in all_trees_node_maps_with_answers:
        doc_path = doc_info.get('doc_path') or doc_info.get('path', '')
        answer = doc_info.get('answer', '')
        if not answer.strip():
            continue
        combined_answer[doc_path] = answer
        base = _doc_basename_for_citation(doc_path)
        if base != "document" and base not in seen_basenames:
            seen_basenames.add(base)
            citation_sources.append({"name": base, "url": f"/docs/{quote(base)}"})

    if not combined_answer:
        return "No answer could be generated from the retrieved contexts.", citation_sources

    valid_filenames = [s['name'] for s in citation_sources]
    prompt = {}
    prompt['system_prompt'] = f"""
    You are a rigorous scholar combining answers from multiple documents into a single answer with inline citations.

    Task:
    You are given a table of answers (keyed by document path) and a query.
    Compile them into one structured response.

    CITATION RULE (mandatory):
    - Place citations immediately after each sentence or claim they support.
    - When a claim is supported by ONE source, use: [Source: filename]
    - When a claim is supported by MULTIPLE sources, cite ALL of them: [Source: filename1, filename2, ...]
    - Valid filenames (use exactly as shown): {valid_filenames}
    - Do not invent filenames; use only the ones above.
    - Preserve existing [Source: ...] citations from per-document answers; when merging overlapping content from multiple docs, combine their citations into one multi-source citation.

    Instructions:
    - Merge overlapping content; avoid repetition.
    - Use clear section headers (## Heading).
    - Keep the answer concise and well-organized.
    - Every factual claim must have a citation listing all supporting sources.

    Directly return the final answer. Do not output anything else.
    """
    prompt['user_prompt'] = f"""
    Query: {query}

    Table of answers (each may already contain [Source: filename] citations):
    {json.dumps(combined_answer, indent=2)}
    """
    try:
        response = await ChatGPT_API_async(model=model, prompt=prompt)
        return response.strip(), citation_sources
    except Exception as e:
        print(f"Error generating answer: {e}")
        return "Error generating answer.", citation_sources


# Configuration - paths relative to project root
async def generate_answer_for_each_context(
    query: str, 
    all_trees_node_maps: List[Dict[str, Any]], 
    model: Optional[str] = None):
    """Generate answer for each context with inline [Source: filename] citations."""
    def _build_context(doc_info: Dict[str, Any]) -> str:
        text = doc_info.get("context", "")
        title = doc_info.get("title", "")
        authors = doc_info.get("authors", "")
        abstract = doc_info.get("abstract", "")
        parts = []
        if title:
            parts.append(f"Title: {title}\n")
        if authors:
            parts.append(f"Authors: {authors}\n")
        if abstract:
            parts.append(f"Abstract: {abstract}\n")
        if text:
            parts.append(f"Context: {text}\n")
        return "".join(parts)

    to_run: List[Dict[str, Any]] = []
    for doc_info in all_trees_node_maps:
        if not str(doc_info.get("context", "")).strip():
            doc_info["answer"] = ""
        else:
            to_run.append(doc_info)
    if not to_run:
        return all_trees_node_maps

    async def _one(doc_info: Dict[str, Any]) -> None:
        context = _build_context(doc_info)
        doc_path = doc_info.get("doc_path") or doc_info.get("path", "")
        # print(f"Context: {context}")
        
        answer = await generate_answer_with_citations(
            query, context, doc_path=doc_path, model=model
        )
        # print(f"Answer: {answer}")
        doc_info["answer"] = answer

    results = await asyncio.gather(
        *(_one(d) for d in to_run),
        return_exceptions=True,
    )
    for doc_info, res in zip(to_run, results):
        if isinstance(res, Exception):
            print(f"generate_answer_for_each_context error for {doc_info.get('path')}: {res}")
            doc_info["answer"] = ""

    return all_trees_node_maps

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
    Returns list of document file paths for the json tree files that match the query.
    """
    if not doc_index:
        return []
    
    # Get all keywords
    keywords = list(doc_index.keys())
    
    if not keywords:
        return []
    
    # Use LLM to find relevant keywords
    prompt = {}
    prompt['system_prompt'] = f"""
    You are a helpful assistant that identifies which keywords are relevant to answering a query.

    Task:
    You are given a query and a list of keywords from a document index.
    Your task is to identify which keywords are relevant to answering the query.

    IMPORTANT: You must reply in the following JSON format:
    {{
        "thinking": "<Your thinking process on which keywords are relevant>",
        "relevant_keywords": ["keyword1", "keyword2", ...]
    }}
    Directly return the final JSON structure. Do not output anything else.
    """
    prompt['user_prompt'] = f"""
    Query: {query}
    Keywords: {json.dumps(keywords, indent=2)}
    """
    

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


def create_node_mapping(tree: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
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
            for node in nodes:
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


def print_wrapped(text: str, width: int = 80, return_text: bool = False):
    """Print text with word wrapping."""
    import textwrap
    wrapped = textwrap.fill(text, width=width)
    if return_text:
        return wrapped
    print(wrapped)


async def tree_search(query: str, doc_info: Dict[str, Any], model: Optional[str] = None) -> Dict[str, Any]:
    """
    Perform tree search to find relevant nodes for the query.
    Similar to the notebook's tree search step.
    """
    # Remove text fields to reduce token usage
    tree = doc_info.get('tree')
    tree_without_text = remove_fields(tree.copy(), fields=['text'])
    

    prompt = {}
    prompt['system_prompt'] = f"""
    You are given a hierarchical tree structure of a document.

    The tree structure has parent nodes that may contain child nodes (nested in a "nodes" field).
    Each node contains a node id, node title, and a corresponding summary.

    Your task is to find all nodes that are likely to contain the answer to the question.

    IMPORTANT RULES:
    1. If a parent node is relevant, you may also need to include its child nodes for complete context
    2. However, only include child nodes if they are also relevant to the question
    3. You can select both parent and child nodes if both are relevant
    4. Consider the hierarchical relationship: parent nodes often provide overview, child nodes provide details

    IMPORTANT: You must reply in the following JSON format:
    {{
        "thinking": "<Your thinking process on which nodes are relevant>",
        "node_list": ["node_id1", "node_id2", ...]
    }}
    Directly return the final JSON structure. Do not output anything else.
    """
    prompt['user_prompt'] = f"""
    Query: {query}
    Document tree structure: {json.dumps(tree_without_text, indent=2)}
    """

    

    try:
        response = await ChatGPT_API_async(model=model, prompt=prompt)
        
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
            try:
                text = node.get('text') or node.get('content') or node.get('summary', '')
                title = node.get('title', f'Node {node_id}')
                context_parts.append(f"## Title: {title}\n\nText: {text}")
            
            except Exception as e:
                raise Exception(f"Error extracting text from node {node_id}: {e}")
    return "\n\n---\n\n".join(context_parts)


async def generate_answer(query: str, context: str, model: Optional[str] = None) -> str:
    """Generate answer based on query and context."""
    prompt = {}
    prompt['system_prompt'] = f"""
    You are a helpful assistant answering queries based on provided document context.
    You are a structured assistant.

    Always:
    - Use clear section headers. 
    - Separate each section with a horizontal rule.
    - Break answers into steps 
    - Each steps should be a single sentence or a short paragraph.
    - Keep responses concise and organized
    - Use markdown formatting to make the answer more readable, if appropriate.

    Task:
    You are given a query and a context from a document.
    Your task is to answer the query based on the context.

    Instructions:
    1. Answer the query directly and naturally, as if you're explaining to someone who asked
    2. Use the context to provide specific details, examples, or explanations
    3. If the query asks "how", explain the process or method
    4. If the query asks "what", provide definitions or descriptions
    5. If the query asks "why", explain reasons or motivations
    6. If the query asks "who", provide the names of the authors
    7. Structure your answer to directly address what was asked
    8. If information is not available in the context, acknowledge this but provide what you can
    9. Write in a clear, natural, and conversational tone
    10. Use the exact terminology and phrasing from the context when appropriate

    Directly return the final answer. Do not output anything else.
    """
    prompt['user_prompt'] = f"""
    User query: {query}
    Relevant Context from Documents: {context}
    """
    

    try:
        response = await ChatGPT_API_async(model=model, prompt=prompt)
        return response.strip()
    except Exception as e:
        print(f"Error generating answer: {e}")
        return "Error generating answer."
        
def _doc_basename_for_citation(doc_path: Optional[str]) -> str:
    """Get PDF basename for citation; derive from structure path if needed."""
    if not doc_path:
        return "document"
    base = os.path.basename(str(doc_path))
    if base.endswith("_structure.json"):
        return base.replace("_structure.json", ".pdf")
    if base.endswith("structure.json"):
        return base.replace("structure.json", ".pdf")
    return base


async def generate_answer_with_citations(query: str, context: str, doc_path: Optional[str] = None, model: Optional[str] = None) -> str:
    """Generate answer based on query and context, with inline [Source: filename] citations."""
    doc_basename = _doc_basename_for_citation(doc_path)
    prompt = {}
    prompt['system_prompt'] = f"""
    You are a rigorous scholar answering queries based on provided document context.
    You answer with structures and inline citations.

    Always:
    - Use clear section headers (## Heading).
    - Break answers into steps; each step should be a single sentence or short paragraph.
    - Keep responses concise and organized.
    - Use markdown formatting where appropriate.

    CITATION RULE (mandatory):
    - Place [Source: {doc_basename}] immediately after each sentence or claim that comes from this context.
    - Use this exact format; the filename must be exactly: {doc_basename}

    Instructions:
    1. Answer the query directly and naturally.
    2. Use the context to provide specific details, examples, or explanations.
    3. Structure your answer to directly address what was asked.
    4. If information is not available in the context, acknowledge this but provide what you can.
    5. Write in a clear, natural, conversational tone.

    Directly return the final answer. Do not output anything else.
    """
    prompt['user_prompt'] = f"""
    User query: {query}
    Relevant Context from Documents: {context}
    """
    

    try:
        response = await ChatGPT_API_async(model=model, prompt=prompt)
        return response.strip()
    except Exception as e:
        print(f"Error generating answer: {e}")
        return "Error generating answer."


def extract_tree_and_node_map(structure: Dict[str, Any], structure_path: str, all_trees_node_maps: List[Dict[str, Any]]) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Extract tree structure and node mapping from document structure."""
   
    if structure:
        # Extract tree structure (handle different formats)
        if isinstance(structure, dict):
            tree = structure.get('structure', structure)
            title = structure.get('doc_title', '')
            authors = structure.get('doc_authors', '')
            doc_path = structure.get('doc_path', structure_path)
            abstract = structure.get('doc_abstract', '')
        else:
            tree = structure
            doc_path = structure_path
        if tree:
            node_map = create_node_mapping(tree)
            all_trees_node_maps.append({
                'path': structure_path,
                'title': title,
                'authors': authors,
                'tree': tree,
                'node_map': node_map,
                'doc_path': doc_path,
                'doc_abstract': abstract
            })
            
            # all_node_maps.append({
            #     'path': structure_path,
                
            #     'doc_path': doc_path
            # })
            print(f"  Loaded: {os.path.basename(structure_path)} ({len(node_map)} nodes)")
    return all_trees_node_maps

def print_retrieved_nodes(node_ids: List[str], node_map: Dict[str, Dict[str, Any]], return_text: bool = False) -> str:
    """Print retrieved nodes in a readable format."""
    lines = ["\nRetrieved Nodes:"]
    for node_id in node_ids:
        if node_id in node_map:
            node = node_map[node_id]
            lines.append(f"  Node ID: {node['node_id']}\t Page: {node.get('page_index', node.get('start_index', 'N/A'))}\t Title: {node.get('title', 'Unknown')}")
    result = "\n".join(lines)
    if return_text:
        return result
    print(result)

        
if __name__ == "__main__":
    main()
