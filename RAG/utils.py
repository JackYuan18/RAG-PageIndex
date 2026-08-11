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
import threading
from urllib.parse import quote
from collections import defaultdict
from typing import List, Dict, Any, Optional

ConversationTurn = Dict[str, str]
DEFAULT_MAX_CONVERSATION_TURNS = 10


def normalize_conversation_history(
    history: Optional[List[Dict[str, Any]]],
    *,
    max_turns: int = DEFAULT_MAX_CONVERSATION_TURNS,
) -> List[ConversationTurn]:
    """Validate and trim conversation history to recent turns."""
    if not history:
        return []
    normalized: List[ConversationTurn] = []
    for item in history:
        if not isinstance(item, dict):
            continue
        role = str(item.get("role", "")).strip().lower()
        content = str(item.get("content", "")).strip()
        if role not in ("user", "assistant") or not content:
            continue
        normalized.append({"role": role, "content": content})
    if max_turns > 0 and len(normalized) > max_turns:
        normalized = normalized[-max_turns:]
    return normalized


def format_conversation_for_prompt(
    history: Optional[List[Dict[str, Any]]],
    *,
    max_turns: int = DEFAULT_MAX_CONVERSATION_TURNS,
) -> str:
    """Format prior turns for inclusion in LLM prompts."""
    turns = normalize_conversation_history(history, max_turns=max_turns)
    if not turns:
        return ""
    lines = []
    for turn in turns:
        label = "User" if turn["role"] == "user" else "Assistant"
        lines.append(f"{label}: {turn['content']}")
    return "\n\n".join(lines)


def build_retrieval_query(
    query: str,
    history: Optional[List[Dict[str, Any]]],
    *,
    max_turns: int = 6,
) -> str:
    """Combine recent user turns with the current query for keyword matching."""
    turns = normalize_conversation_history(history, max_turns=max_turns)
    if not turns:
        return query
    parts = [t["content"] for t in turns if t["role"] == "user"][-3:]
    parts.append(query)
    combined = " ".join(p.strip() for p in parts if p.strip()).strip()
    return combined or query
from dotenv import load_dotenv

# Add parent directory to path to ensure imports work
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from pageindex.utils import (
    ChatGPT_API,
    ChatGPT_API_async,
    get_model_name,
    extract_json,
    cosine_sim,
    ConfigLoader,
    get_openai_client,
    _embed_parts_ollama,
    _load_cached_embedding,
    _save_cached_embedding,
)
import openai

load_dotenv()

_warm_embed_lock = threading.Lock()

async def combine_answers(
    query: str,
    all_trees_node_maps_with_answers: List[Dict[str, Any]],
    model: Optional[str] = None,
    conversation_history: Optional[List[Dict[str, Any]]] = None,
) -> tuple[str, List[Dict[str, str]]]:
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
    history_block = format_conversation_for_prompt(conversation_history)
    history_section = (
        f"\n    Previous conversation:\n    {history_block}\n"
        if history_block
        else ""
    )
    prompt['user_prompt'] = f"""
    {history_section}
    Current query: {query}

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
    model: Optional[str] = None,
    conversation_history: Optional[List[Dict[str, Any]]] = None,
):
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
            query,
            context,
            doc_path=doc_path,
            model=model,
            conversation_history=conversation_history,
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

def load_docindex(docindex_path: Optional[str] = None, quiet: bool = False) -> Dict[str, List[str]]:
    """Load DocIndex from file."""
    path = docindex_path

    if not os.path.exists(path):
        if not quiet:
            print(f"DocIndex not found at {path}")
            print("Please run run_pageindex.py first to generate document structures and DocIndex.")
        return {}

    if not quiet:
        print(f"Loading DocIndex from {path}")
    with open(path, "r", encoding="utf-8") as f:
        doc_index = json.load(f)

    if isinstance(doc_index, dict):
        doc_index = defaultdict(list, doc_index)

    if not quiet:
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


def _default_ollama_embed_model() -> str:
    try:
        from pageindex.model_registry import get_rag_embed_model
        return get_rag_embed_model()
    except Exception:
        try:
            return ConfigLoader().load({}).ollama_embed_model
        except Exception:
            return os.getenv("OLLAMA_EMBED_MODEL", "mxbai-embed-large")


def _embed_texts_batched(
    texts: list[str],
    embed_model: str,
    progress_callback: Optional[Any] = None,
) -> list[list[float]]:
    """Embed short texts with API batching and disk cache (keywords / queries)."""
    embed_model = str(embed_model or "").strip() or "mxbai-embed-large"
    texts = ["" if t is None else str(t) for t in (texts or [])]
    if not texts:
        return []

    out: list[list[float] | None] = [None] * len(texts)
    to_fetch: list[str] = []
    fetch_idx: list[int] = []
    for i, t in enumerate(texts):
        cached = _load_cached_embedding(embed_model, t)
        if cached is not None:
            out[i] = cached
        else:
            to_fetch.append(t)
            fetch_idx.append(i)

    if to_fetch:
        cached_count = len(texts) - len(to_fetch)
        if progress_callback:
            progress_callback(
                f"  Embedding {len(to_fetch)} text(s) "
                f"({cached_count} already cached, model={embed_model})..."
            )
        client = get_openai_client("ollama")
        fetched = _embed_parts_ollama(client, embed_model, to_fetch)
        for i, emb in zip(fetch_idx, fetched):
            _save_cached_embedding(embed_model, texts[i], emb)
            out[i] = emb
        if progress_callback:
            progress_callback(f"  Finished embedding {len(to_fetch)} text(s).")

    return [e or [] for e in out]


def warm_docindex_keyword_embeddings(
    doc_index: Dict[str, List[str]],
    embed_model: Optional[str] = None,
    progress_callback: Optional[Any] = None,
) -> int:
    """Pre-embed DocIndex keywords so query-time Step 1 only embeds the query."""
    with _warm_embed_lock:
        keywords = list((doc_index or {}).keys())
        if not keywords:
            return 0

        embed_model = str(embed_model or _default_ollama_embed_model()).strip() or "mxbai-embed-large"
        uncached = [k for k in keywords if _load_cached_embedding(embed_model, k) is None]
        if not uncached:
            if progress_callback:
                progress_callback(
                    f"  DocIndex keywords already embedded ({len(keywords)} keywords)."
                )
            return 0

        if progress_callback:
            progress_callback(
                f"  Pre-warming {len(uncached)} keyword embedding(s) "
                f"({len(keywords) - len(uncached)} cached)..."
            )
        _embed_texts_batched(uncached, embed_model, progress_callback=progress_callback)
        return len(uncached)


def _cached_keyword_embeddings(keywords: list[str], embed_model: str) -> list[list[float]]:
    return [_load_cached_embedding(embed_model, k) or [] for k in keywords]


def match_query_to_keywords_llm(
    query: str, doc_index: Dict[str, List[str]], model: Optional[str] = None
) -> List[str]:
    """Match query to DocIndex keywords using an LLM."""
    if not doc_index:
        return []

    keywords = list(doc_index.keys())
    if not keywords:
        return []

    prompt = {}
    prompt["system_prompt"] = """
    You are a helpful assistant that identifies which keywords are relevant to answering a query.

    Task:
    You are given a query and a list of keywords from a document index.
    Your task is to identify which keywords are relevant to answering the query.

    IMPORTANT: You must reply in the following JSON format:
    {
        "thinking": "<Your thinking process on which keywords are relevant>",
        "relevant_keywords": ["keyword1", "keyword2", ...]
    }
    Directly return the final JSON structure. Do not output anything else.
    """
    prompt["user_prompt"] = f"""
    Query: {query}
    Keywords: {json.dumps(keywords, indent=2)}
    """

    try:
        response = ChatGPT_API(model=model, prompt=prompt)
        result = extract_json(response)
        relevant_keywords = result.get("relevant_keywords", [])
        print(f"Relevant keywords (llm): {relevant_keywords}")
    except Exception as e:
        print(f"Error matching keywords: {e}")
        relevant_keywords = keywords

    matched_docs = set()
    for keyword in relevant_keywords:
        if keyword in doc_index:
            matched_docs.update(doc_index[keyword])
    return list(matched_docs)


def match_query_to_keywords_embed(
    query: str,
    doc_index: Dict[str, List[str]],
    embed_model: Optional[str] = None,
    top_k: int = 8,
    min_similarity: float = 0.30,
    progress_callback: Optional[Any] = None,
) -> List[str]:
    """Match query to DocIndex keywords via embedding cosine similarity."""
    if not doc_index:
        return []

    keywords = list(doc_index.keys())
    if not keywords:
        return []

    embed_model = str(embed_model or _default_ollama_embed_model()).strip() or "mxbai-embed-large"
    top_k = max(1, int(top_k or 8))
    min_similarity = float(min_similarity)

    uncached_kw = [
        k for k in keywords if _load_cached_embedding(embed_model, k) is None
    ]
    if uncached_kw:
        if progress_callback:
            progress_callback(
                f"  Embedding {len(uncached_kw)} keyword(s) "
                f"({len(keywords) - len(uncached_kw)} cached)..."
            )
        _embed_texts_batched(uncached_kw, embed_model, progress_callback=progress_callback)

    if progress_callback:
        progress_callback("  Embedding query...")
    q_emb = _embed_texts_batched([query], embed_model, progress_callback=progress_callback)[0]
    kw_embs = _cached_keyword_embeddings(keywords, embed_model)

    if progress_callback:
        progress_callback("  Scoring keyword similarity...")

    scored = [(kw, cosine_sim(q_emb, kw_emb)) for kw, kw_emb in zip(keywords, kw_embs)]
    scored.sort(key=lambda x: -x[1])

    relevant_keywords = [kw for kw, sim in scored if sim >= min_similarity][:top_k]
    if not relevant_keywords and scored:
        relevant_keywords = [scored[0][0]]

    scored_display = [
        (kw, round(sim, 3))
        for kw, sim in scored
        if kw in relevant_keywords
    ]
    print(
        f"Relevant keywords (embed, model={embed_model}, "
        f"top_k={top_k}, min_sim={min_similarity}): {scored_display}"
    )
    if progress_callback:
        progress_callback(f"  Selected {len(relevant_keywords)} keyword(s).")

    scored_map = {kw: sim for kw, sim in scored}
    doc_best_sim: Dict[str, float] = {}
    for keyword in relevant_keywords:
        sim = scored_map.get(keyword, 0.0)
        for path in doc_index.get(keyword, []):
            doc_best_sim[path] = max(doc_best_sim.get(path, 0.0), sim)
    return sorted(doc_best_sim.keys(), key=lambda p: -doc_best_sim[p])


def match_query_to_keywords(
    query: str,
    doc_index: Dict[str, List[str]],
    model: Optional[str] = None,
    method: str = "embed",
    embed_model: Optional[str] = None,
    top_k: int = 8,
    min_similarity: float = 0.30,
    progress_callback: Optional[Any] = None,
) -> List[str]:
    """
    Match query to keywords in DocIndex.
    Returns list of document file paths for the json tree files that match the query.
    """
    method = (method or "embed").strip().lower()
    if method == "llm":
        return match_query_to_keywords_llm(query, doc_index, model=model)
    if method == "embed":
        return match_query_to_keywords_embed(
            query,
            doc_index,
            embed_model=embed_model,
            top_k=top_k,
            min_similarity=min_similarity,
            progress_callback=progress_callback,
        )
    raise ValueError(f"Unknown keyword match method: {method!r} (use 'embed' or 'llm')")


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


async def tree_search(
    query: str,
    doc_info: Dict[str, Any],
    model: Optional[str] = None,
    conversation_history: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """
    Perform tree search to find relevant nodes for the query.
    Similar to the notebook's tree search step.
    """
    tree = doc_info.get("tree")
    tree_without_text = remove_fields(tree.copy(), fields=["text"])

    prompt = {}
    prompt["system_prompt"] = f"""
    You are given a hierarchical tree structure of a document.

    The tree structure has parent nodes that may contain child nodes (nested in a "nodes" field).
    Each node contains a node id, node title, and a corresponding summary.

    NOTE ON SUMMARY FORMATS:
    - Some nodes may also include a "summary_payload" object with a "type" discriminator.
      - type="mmr": summary is extractive and may contain concatenated high-signal snippets (exact phrases).
      - type="llm": summary is abstractive.
    - When type="mmr", prefer matching on concrete terms, entities, and exact phrases in the snippets.

    Your task is to find all nodes that are likely to contain the answer to the question.
    The current query may be a follow-up; use the previous conversation to interpret it.

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
    history_block = format_conversation_for_prompt(conversation_history)
    history_section = (
        f"Previous conversation:\n{history_block}\n\n"
        if history_block
        else ""
    )
    prompt["user_prompt"] = f"""
    {history_section}Current query: {query}
    Document tree structure: {json.dumps(tree_without_text, indent=2)}
    """

    response = None
    try:
        response = await ChatGPT_API_async(model=model, prompt=prompt)
        if not response or str(response).strip() == "Error":
            return {"thinking": "", "node_list": []}
        result = extract_json(response)
        if not isinstance(result, dict):
            return {"thinking": "", "node_list": []}
        return result
    except Exception as e:
        print(f"Error in tree search: {e}")
        if response is not None:
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


async def generate_answer_with_citations(
    query: str,
    context: str,
    doc_path: Optional[str] = None,
    model: Optional[str] = None,
    conversation_history: Optional[List[Dict[str, Any]]] = None,
) -> str:
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
    1. Answer the current query directly and naturally.
    2. Use prior conversation for context when the current query is a follow-up.
    3. Use the context to provide specific details, examples, or explanations.
    4. Structure your answer to directly address what was asked.
    5. If information is not available in the context, acknowledge this but provide what you can.
    6. Write in a clear, natural, conversational tone.

    Directly return the final answer. Do not output anything else.
    """
    history_block = format_conversation_for_prompt(conversation_history)
    history_section = (
        f"Previous conversation:\n{history_block}\n\n"
        if history_block
        else ""
    )
    prompt['user_prompt'] = f"""
    {history_section}Current user query: {query}
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
    # This module provides helpers; run `RAG/rag_query.py` to execute a query end-to-end.
    raise SystemExit("Run `python3 RAG/rag_query.py --query \"...\"` instead.")
