#!/usr/bin/env python3
"""
Build DocIndex for all documents in the Database directory.

This script processes all PDF and Markdown files in a specified directory,
generates their structure files, and creates/updates the DocIndex.
"""

import argparse
import os
import json
import glob
from pageindex.model_registry import get_indexing_chat_model, get_indexing_embed_model
from pageindex.page_index_md import md_to_tree
from run_pageindex import *
from collections import defaultdict
import asyncio

if __name__ == "__main__":
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Process all documents in Database directory and build DocIndex')
    _project_root = os.path.dirname(os.path.abspath(__file__))
    _default_database_dir = os.path.join(_project_root, "Database")
    parser.add_argument(
        '--database-dir',
        type=str,
        default=_default_database_dir,
        help='Path to the Database directory containing documents (default: ./Database)',
    )
    parser.add_argument('--model', type=str, default=get_indexing_chat_model(), help='Model to use (default: llm_models.yaml indexing.chat_model)')
    
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
                      
    # Markdown specific arguments
    parser.add_argument('--if-thinning', type=str, default='no',
                      help='Whether to apply tree thinning for markdown (markdown only)')
    parser.add_argument('--thinning-threshold', type=int, default=5000,
                      help='Minimum token threshold for thinning (markdown only)')
    parser.add_argument('--summary-token-threshold', type=int, default=200,
                      help='Token threshold for generating summaries (markdown only)')
    
    parser.add_argument('--output-dir', type=str, default='./results',
                      help='Output directory for structure files and DocIndex')
    
    parser.add_argument(
        "--pageindex-ai-mode",
        dest="pageindex_ai_mode",
        choices=["llm", "rule", "auto"],
        type=str,
        default="rule",
        help="Control PAGEINDEX_AI_MODE for this run: llm=1, rule=0, auto=leave env unchanged",
    )
    parser.add_argument(
        '--node-summary-method',
        '--summary-method',
        dest='summary_method',
        choices=['llm', 'mmr'],
        default='llm',
        help='Leaf node summarization backend (opt.summary_method); default: llm',
    )
    parser.add_argument(
        '--parent-summary-method',
        dest='parent_summary_method',
        choices=['llm', 'mmr'],
        default='llm',
        help='Parent node summarization backend (opt.parent_summary_method); default: llm',
    )
    parser.add_argument(
        '--abstract-method',
        dest='abstract_method',
        choices=['llm', 'mmr'],
        default='llm',
        help='Document abstract backend (opt.abstract_method); default: llm',
    )
    parser.add_argument(
        '--keyword-method',
        dest='keyword_method',
        choices=['llm', 'rich', 'embed_mmr'],
        default="embed_mmr",
        help='Keyword generation backend (opt.keyword_method); omit to use config / env',
    )
    parser.add_argument(
        '--keyword-merge-method',
        dest='keyword_merge_method',
        choices=['llm', 'embed'],
        default='embed',
        help='DocIndex keyword merge backend (llm=slow, embed=scalable ANN over embeddings)',
    )
    args = parser.parse_args()
    print(f'Using model: {args.model}')
    # Validate database directory
    if not os.path.isdir(args.database_dir):
        raise ValueError(f"Database directory not found: {args.database_dir}")
    
    # Find all PDF and Markdown files
    pdf_files = glob.glob(os.path.join(args.database_dir, '*.pdf'))
    
    
    total_files = len(pdf_files) 
    
    if total_files == 0:
        print(f"No PDF or Markdown files found in {args.database_dir}")
        exit(0)
    
    print(f"\n{'='*80}")
    print(f"Found {len(pdf_files)} PDF file(s)")
    print(f"Total: {total_files} file(s)")
    print(f"{'='*80}\n")
    
    # Initialize DocIndex
    output_dir = args.output_dir
  
    
    # Configure options for PDF processing (merge with pageindex/config.yaml)
    if args.pageindex_ai_mode == 'llm':
        keyword_merge_method = 'llm'
    else:
        keyword_merge_method = args.keyword_merge_method

    user_dict = {
        'model': args.model,
        'ollama_embed_model': get_indexing_embed_model(),
        'toc_check_page_num': args.toc_check_pages,
        'max_page_num_each_node': args.max_pages_per_node,
        'max_token_num_each_node': args.max_tokens_per_node,
        'if_add_node_id': args.if_add_node_id,
        'if_add_node_summary': args.if_add_node_summary,
        'if_add_parent_node_summary': args.if_add_parent_node_summary,
        'if_add_doc_description': args.if_add_doc_description,
        'if_add_doc_abstract': args.if_add_doc_abstract,
        'if_add_node_text': args.if_add_node_text,
        'summary_method': args.summary_method,
        'parent_summary_method': args.parent_summary_method,
        'abstract_method': args.abstract_method,
        'keyword_method': args.keyword_method,
        'keyword_merge_method': keyword_merge_method,
    }
    opt = ConfigLoader().load(user_dict)
    opt = config(**{**vars(opt), 'ai_mode': args.pageindex_ai_mode})
    
    # Process all PDF files
    pdf_success = 0
    pdf_failed = 0
    failed_doc = []
    for pdf_path in pdf_files:
        # Default to rule-based PageIndex internals (PAGEINDEX_AI_MODE=0).
        # If it fails, fall back to LLM mode (PAGEINDEX_AI_MODE=1) once.
        success, doc_index, _step_timings, _total_seconds = process_document(
            pdf_path, output_dir, opt, update_docindex=True
        )
        if not success and args.pageindex_ai_mode == "rule":
            print("Rule-based mode failed; retrying with LLM mode...")
            opt_llm = config(**{**vars(opt), "ai_mode": "llm"})
            success, doc_index, _step_timings, _total_seconds = process_document(
                pdf_path, output_dir, opt_llm, update_docindex=True
            )
        if success:
            pdf_success += 1
        else:
            pdf_failed += 1
            failed_doc.append(pdf_path)

        print(f'Total keywords in DocIndex: {len(doc_index)}')
        
        # Print summary
        print(f"\n{'='*80}")
        print("SUMMARY")
        print(f"{'='*80}")
        print(f"PDF files: {pdf_success} succeeded, {pdf_failed} failed")
        print(f"{'='*80}\n")

    # INSERT_YOUR_CODE
    if pdf_failed > 0:
        print("\nThe following PDF files failed to process:")
        for pdf_path in failed_doc:
            print(f"- {pdf_path}")
