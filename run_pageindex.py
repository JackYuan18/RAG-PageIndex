import argparse
import json
import os
import time
from collections import defaultdict
from typing import Any, Dict, List

from pageindex import *
from pageindex.page_index_md import md_to_tree
from datetime import datetime

def _finish_step(step_timings: List[Dict[str, Any]], label: str, t0: float) -> None:
    elapsed = time.perf_counter() - t0
    sec = round(elapsed, 3)
    step_timings.append({"step": label, "seconds": sec})
    print(f"  {label} — completed in {elapsed:.2f} s")


def _print_processing_summary(
    doc_name: str, step_timings: List[Dict[str, Any]], t_start: float
) -> None:
    total = time.perf_counter() - t_start
    print("")
    print(f"========== Processing complete: {doc_name} ==========")
    print(f"Total processing time: {total:.2f} s")
    for item in step_timings:
        print(f"  {item['step']}: {item['seconds']:.2f} s")
    print("=" * max(60, 42 + len(doc_name)))

def normalize_doc_index_to_sets(doc_index):
    """
    Normalize DocIndex values to sets for internal update logic.
    Accepts dict/defaultdict containing list/set/str values.
    """
    normalized = defaultdict(set)
    if not isinstance(doc_index, dict):
        return normalized
    for kw, value in doc_index.items():
        if isinstance(value, set):
            normalized[kw] = set(value)
        elif isinstance(value, list):
            normalized[kw] = set(value)
        elif isinstance(value, str):
            normalized[kw] = {value}
        elif value is None:
            normalized[kw] = set()
        else:
            normalized[kw] = {str(value)}
    return normalized

def serialize_doc_index_for_json(doc_index):
    """
    Convert DocIndex values to JSON-serializable lists.
    """
    serializable = {}
    for kw, value in doc_index.items():
        if isinstance(value, set):
            serializable[kw] = sorted(list(value))
        elif isinstance(value, list):
            serializable[kw] = value
        elif isinstance(value, str):
            serializable[kw] = [value]
        else:
            serializable[kw] = [str(value)]
    return serializable
def keywords_are_similar(kw1, kw2, model=None):
    prompt = {}
    prompt['system_prompt'] = f"""
    You are an expert in determining whether two keywords have similar meanings.
    Return a boolen variable only.
    True if they are similar in meaning, False otherwise.
    """
    prompt['user_prompt'] = f"""
    You are given two keywords: {kw1} and {kw2}.
    Directly return the boolean variable.
    
    """

    response = ChatGPT_API(model, prompt)
    return response

def make_merged_keyword(kw1, kw2, model=None):
    
    prompt = {}
    prompt['system_prompt'] = f"""
    You are an expert in creating a concise new keyword which has the same or similar meaning as the two given keywords.
    Return the new keyword. The new keyword should have the least deviation from the two given keywords.
    """
    prompt['user_prompt'] = f"""
    You are given two keywords: {kw1} and {kw2}.
    Directly return the new keyword.
    """
    
    response = ChatGPT_API(model, prompt)
    return response
def merge_keywords(keywords, doc_index, model=None): 

    updated_keywords = keywords.copy()
    updated_doc_index = doc_index.copy()
    sim_cache = {}

    for i, new_kw in enumerate(keywords):
        for existing_kw in list(doc_index.keys()):
            # print(f'new_kw: {new_kw}, existing_kw: {existing_kw}')
            # Use AI to determine whether similar in meaning
            if new_kw == existing_kw:
                continue
            cache_key = (new_kw, existing_kw)
            if cache_key not in sim_cache:
                sim_cache[cache_key] = keywords_are_similar(new_kw, existing_kw, model=model)
            is_sim = sim_cache[cache_key]
            if is_sim == 'True':
                print(f"similar: {new_kw} and {existing_kw}, {is_sim}")
                # Use AI to create a concise merged keyword
                merged_kw = make_merged_keyword(new_kw, existing_kw, model=model)
                # Update doc_index (replace existing_kw with merged_kw, preserve value)
                updated_doc_index[merged_kw] = doc_index.pop(existing_kw)
                # Update all occurrences in updated_keywords to merged_kw
                updated_keywords[i] = merged_kw      
                break
        
    # Remove duplicates and return
    return updated_keywords, updated_doc_index


def process_document(pdf_path, output_dir, opt, update_docindex=True):
    step_timings: List[Dict[str, Any]] = []
    doc_index = defaultdict(set)
    t_process = time.perf_counter()
    doc_name = os.path.basename(pdf_path)

    try:
        toc_with_page_number = page_index_main(
            pdf_path, opt, step_timings=step_timings
        )

        pdf_name = os.path.splitext(os.path.basename(pdf_path))[0]
        output_dir = "./results"
        output_file = f"{output_dir}/{pdf_name}_structure.json"
        os.makedirs(output_dir, exist_ok=True)

        print("Step 11: Saving structure JSON...")
        t0 = time.perf_counter()
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(toc_with_page_number, f, indent=2)
        print(f"Tree structure saved to: {output_file}")
        _finish_step(step_timings, "Step 11: Saving structure JSON", t0)

        if update_docindex:
            doc_index_path = os.path.join(output_dir, "DocIndex.json")

            print("Step 12: Loading DocIndex and removing old references for this document...")
            t0 = time.perf_counter()
            if os.path.exists(doc_index_path):
                try:
                    with open(doc_index_path, "r", encoding="utf-8") as f:
                        loaded_dict = json.load(f)
                        doc_index = normalize_doc_index_to_sets(loaded_dict)
                except Exception as e:
                    print(f"Warning: Could not load existing DocIndex, starting fresh. ({e})")
                    doc_index = defaultdict(set)

            keys_to_remove = []
            for key, val_set in list(doc_index.items()):
                if output_file in val_set:
                    val_set.discard(output_file)
                if not val_set:
                    keys_to_remove.append(key)
            for key in keys_to_remove:
                del doc_index[key]

            keywords = []
            if "keywords" in toc_with_page_number:
                keywords = toc_with_page_number["keywords"].split(",")
            print(f"Keywords: {keywords}")
            _finish_step(
                step_timings,
                "Step 12: Load DocIndex and prune old references",
                t0,
            )

            print("Step 13: Merging keywords into DocIndex...")
            t0 = time.perf_counter()
            merged_keywords, doc_index = merge_keywords(
                keywords, doc_index, model=opt.model
            )
            for kw in merged_keywords:
                doc_index[kw].add(output_file)
            _finish_step(step_timings, "Step 13: Merging keywords into DocIndex", t0)

            print("Step 14: Writing DocIndex.json...")
            t0 = time.perf_counter()
            with open(doc_index_path, "w", encoding="utf-8") as f:
                json.dump(
                    serialize_doc_index_for_json(doc_index),
                    f,
                    indent=2,
                    ensure_ascii=False,
                )
            _finish_step(step_timings, "Step 14: Saving DocIndex.json", t0)

        _print_processing_summary(doc_name, step_timings, t_process)
        return True, doc_index, step_timings, round(time.perf_counter() - t_process, 3)
    except Exception as e:
        print(f"Error processing PDF {pdf_path}: {e}")
        import traceback

        traceback.print_exc()
        return False, doc_index, step_timings, round(time.perf_counter() - t_process, 3)
    
if __name__ == "__main__":
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Process PDF or Markdown document and generate structure')
    parser.add_argument('--pdf_path', type=str, default = '/home/zyuan/NSTSCE_Bot/NSTSCE/Database/AVSC Best Practice for Developing ADS.pdf', help='Path to the PDF file')
    # parser.add_argument('--pdf_path', type=str, default = '/home/zyuan/NSTSCE_Bot/PageIndex/Database/Communication-aware_Distributed_Gaussian_Process_Regression_Algorithms_for_Real-time_Machine_Learning.pdf', help='Path to the PDF file')
    parser.add_argument('--md_path', type=str, help='Path to the Markdown file')

    # parser.add_argument('--model', type=str, default='gpt-5.1', help='Model to use')
    parser.add_argument('--model', type=str, default='qwen', help='Model to use')

    parser.add_argument('--toc-check-pages', type=int, default=20, 
                      help='Number of pages to check for table of contents (PDF only)')
    parser.add_argument('--max-pages-per-node', type=int, default=10,
                      help='Maximum number of pages per node (PDF only)')
    parser.add_argument('--max-tokens-per-node', type=int, default=20000,
                      help='Maximum number of tokens per node (PDF only)')

    parser.add_argument('--if-add-node-id', type=str, default='yes',
                      help='Whether to add node id to the node')
    parser.add_argument('--if-add-node-summary', type=str, default='yes',
                      help='Whether to add summary to the node')
    parser.add_argument('--if-add-parent-node-summary', type=str, default='yes',
                      help='Whether to add summary to the parent node')
    parser.add_argument('--if-add-doc-description', type=str, default='yes',
                      help='Whether to add doc description to the doc')
    parser.add_argument('--if-add-doc-abstract', type=str, default='yes',
                      help='Whether to add doc abstract to the doc')
    parser.add_argument('--if-add-node-text', type=str, default='yes',
                      help='Whether to add text to the node')

    parser.add_argument('--update_docindex', action='store_true',
                      help='Whether to update the docindex')
    # Markdown specific arguments
    parser.add_argument('--if-thinning', type=str, default='no',
                      help='Whether to apply tree thinning for markdown (markdown only)')
    parser.add_argument('--thinning-threshold', type=int, default=5000,
                      help='Minimum token threshold for thinning (markdown only)')
    parser.add_argument('--summary-token-threshold', type=int, default=200,
                      help='Token threshold for generating summaries (markdown only)')
    parser.add_argument('--timings-out', type=str, default=None,
                      help='Optional path to write JSON timings')
    args = parser.parse_args()
    
    output_dir = './results'

    print(f'Using model: {args.model}')
    # Validate that exactly one file type is specified
    if not args.pdf_path and not args.md_path:
        raise ValueError("Either --pdf_path or --md_path must be specified")
    if args.pdf_path and args.md_path:
        raise ValueError("Only one of --pdf_path or --md_path can be specified")
    
    if args.pdf_path:
        
        # Validate PDF file
        if not args.pdf_path.lower().endswith('.pdf'):
            raise ValueError("PDF file must have .pdf extension")
        if not os.path.isfile(args.pdf_path):
            raise ValueError(f"PDF file not found: {args.pdf_path}")
            
        # Process PDF file
        # Configure optionsdoc_index_path = os.path.join(output_dir, "DocIndex.json")
        opt = config(
            model=args.model,
            toc_check_page_num=args.toc_check_pages,
            max_page_num_each_node=args.max_pages_per_node,
            max_token_num_each_node=args.max_tokens_per_node,
            if_add_node_id=args.if_add_node_id,
            if_add_node_summary=args.if_add_node_summary,
            if_add_parent_node_summary=args.if_add_parent_node_summary,
            if_add_doc_description=args.if_add_doc_description,
            if_add_doc_abstract=args.if_add_doc_abstract,
            if_add_node_text=args.if_add_node_text
        )
        success, doc_index, step_timings, total_seconds = process_document(args.pdf_path, output_dir, opt, args.update_docindex)
        # Process the PDF
        # toc_with_page_number = page_index_main(args.pdf_path, opt)
        print(f'Parsing done, saving to file... {success}')
        if args.timings_out:
            payload = {
                # INSERT_YOUR_CODE 
                "test_datetime": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "doc": args.pdf_path,
                "success": bool(success),
                "total_seconds": total_seconds,
                "step_timings": step_timings,
            }
            try:
                os.makedirs(os.path.dirname(args.timings_out) or ".", exist_ok=True)
                with open(args.timings_out, "w", encoding="utf-8") as f:
                    json.dump(payload, f, indent=2, ensure_ascii=False)
                print(f"Wrote timings to: {args.timings_out}")
            except Exception as e:
                print(f"Warning: could not write timings JSON: {e}")
        
        
        
      