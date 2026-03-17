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
from pageindex import *
from pageindex.page_index_md import md_to_tree
from run_pageindex import *
from collections import defaultdict
import asyncio



def process_pdf(pdf_path, output_dir, opt, doc_index, model):
    """Process a single PDF file and update DocIndex."""
    print(f"\n{'='*80}")
    print(f"Processing PDF: {os.path.basename(pdf_path)}")
    print(f"{'='*80}")
    
    try:
        # Process the PDF
        toc_with_page_number = page_index_main(pdf_path, opt)
        print('Parsing done, saving to file...')
        
        # Save results
        pdf_name = os.path.splitext(os.path.basename(pdf_path))[0]    
        output_file = f'{output_dir}/{pdf_name}_structure.json'
        os.makedirs(output_dir, exist_ok=True)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(toc_with_page_number, f, indent=2)
        
        print(f'Tree structure saved to: {output_file}')
        
        # Get keywords from the tree structure
        keywords = []
        if 'keywords' in toc_with_page_number:
            keywords_value = toc_with_page_number['keywords']
            if isinstance(keywords_value, list):
                keywords = keywords_value
            elif isinstance(keywords_value, str):
                # Parse string format
                keywords = [kw.strip() for kw in keywords_value.split(',') if kw.strip()]
        
        
        if keywords:
            # Merge keywords with existing DocIndex
            merged_keywords, doc_index = merge_keywords(keywords, doc_index, model=model)
            
            # # Remove current file from all existing keywords first (cleanup stale associations)
            # keywords_to_remove = []
            # for kw, file_list in doc_index.items():
            #     if isinstance(file_list, list):
            #         if output_file in file_list:
            #             file_list.remove(output_file)
            #             if not file_list:  # If list is now empty, mark for removal
            #                 keywords_to_remove.append(kw)
            #     elif isinstance(file_list, str):  # Handle old format
            #         if file_list == output_file:
            #             keywords_to_remove.append(kw)
            # for kw in keywords_to_remove:
            #     del doc_index[kw]
            
            # Add new keywords to the DocIndex
            for kw in merged_keywords:
                if output_file not in doc_index[kw]:  # Avoid duplicates
                    doc_index[kw].append(output_file)
            # print(f'DocIndex: {doc_index}')
        return True, doc_index
    except Exception as e:
        print(f"Error processing PDF {pdf_path}: {e}")
        import traceback
        traceback.print_exc()
        return False, doc_index

def process_markdown(md_path, output_dir, opt, doc_index, model, args):
    """Process a single Markdown file and update DocIndex."""
    print(f"\n{'='*80}")
    print(f"Processing Markdown: {os.path.basename(md_path)}")
    print(f"{'='*80}")
    
    try:
        # Process the markdown
        from pageindex.utils import ConfigLoader
        config_loader = ConfigLoader()
        
        # Create options dict with user args
        user_opt = {
            'model': model,
            'if_add_node_summary': args.if_add_node_summary,
            'if_add_doc_description': args.if_add_doc_description,
            'if_add_node_text': args.if_add_node_text,
            'if_add_node_id': args.if_add_node_id
        }
        
        # Load config with defaults from config.yaml
        md_opt = config_loader.load(user_opt)
        
        toc_with_page_number = asyncio.run(md_to_tree(
            md_path=md_path,
            if_thinning=args.if_thinning.lower() == 'yes',
            min_token_threshold=args.thinning_threshold,
            if_add_node_summary=md_opt.if_add_node_summary,
            summary_token_threshold=args.summary_token_threshold,
            model=md_opt.model,
            if_add_doc_description=md_opt.if_add_doc_description,
            if_add_node_text=md_opt.if_add_node_text,
            if_add_node_id=md_opt.if_add_node_id
        ))
        
        print('Parsing done, saving to file...')
        
        # Save results
        md_name = os.path.splitext(os.path.basename(md_path))[0]    
        output_file = f'{output_dir}/{md_name}_structure.json'
        os.makedirs(output_dir, exist_ok=True)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(toc_with_page_number, f, indent=2, ensure_ascii=False)
        
        print(f'Tree structure saved to: {output_file}')
        
        # Get keywords
        keywords = []
        if 'keywords' in toc_with_page_number:
            keywords_value = toc_with_page_number['keywords']
            if isinstance(keywords_value, list):
                keywords = keywords_value
            elif isinstance(keywords_value, str):
                # Parse string format
                keywords = [kw.strip() for kw in keywords_value.split(',') if kw.strip()]
        
        print(f'Keywords: {keywords}')
        
        if keywords:
            # Merge keywords with existing DocIndex
            merged_keywords, doc_index = merge_keywords(keywords, doc_index, model=model)
            
            # Remove current file from all existing keywords first (cleanup stale associations)
            keywords_to_remove = []
            for kw, file_list in doc_index.items():
                if isinstance(file_list, list):
                    if output_file in file_list:
                        file_list.remove(output_file)
                        if not file_list:  # If list is now empty, mark for removal
                            keywords_to_remove.append(kw)
                elif isinstance(file_list, str):  # Handle old format
                    if file_list == output_file:
                        keywords_to_remove.append(kw)
            for kw in keywords_to_remove:
                del doc_index[kw]
            
            # Add new keywords to the DocIndex
            for kw in merged_keywords:
                if output_file not in doc_index[kw]:  # Avoid duplicates
                    doc_index[kw].append(output_file)
        
        return True
    except Exception as e:
        print(f"Error processing Markdown {md_path}: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Process all documents in Database directory and build DocIndex')
    parser.add_argument('--database-dir', type=str, 
                       default='/home/zyuan/NSTSCE_Bot/PageIndex/Database',
                       help='Path to the Database directory containing documents')
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
                      
    # Markdown specific arguments
    parser.add_argument('--if-thinning', type=str, default='no',
                      help='Whether to apply tree thinning for markdown (markdown only)')
    parser.add_argument('--thinning-threshold', type=int, default=5000,
                      help='Minimum token threshold for thinning (markdown only)')
    parser.add_argument('--summary-token-threshold', type=int, default=200,
                      help='Token threshold for generating summaries (markdown only)')
    
    parser.add_argument('--output-dir', type=str, default='./results',
                      help='Output directory for structure files and DocIndex')
    
    args = parser.parse_args()
    print(f'Using model: {args.model}')
    # Validate database directory
    if not os.path.isdir(args.database_dir):
        raise ValueError(f"Database directory not found: {args.database_dir}")
    
    # Find all PDF and Markdown files
    pdf_files = glob.glob(os.path.join(args.database_dir, '*.pdf'))
    md_files = glob.glob(os.path.join(args.database_dir, '*.md')) + \
               glob.glob(os.path.join(args.database_dir, '*.markdown'))
    
    total_files = len(pdf_files) + len(md_files)
    
    if total_files == 0:
        print(f"No PDF or Markdown files found in {args.database_dir}")
        exit(0)
    
    print(f"\n{'='*80}")
    print(f"Found {len(pdf_files)} PDF file(s) and {len(md_files)} Markdown file(s)")
    print(f"Total: {total_files} file(s)")
    print(f"{'='*80}\n")
    
    # Initialize DocIndex
    output_dir = args.output_dir
    doc_index_path = os.path.join(output_dir, "DocIndex.json")
    doc_index = defaultdict(list)
    
    # Try to load existing DocIndex
    if os.path.exists(doc_index_path):
        try:
            with open(doc_index_path, "r", encoding="utf-8") as f:
                loaded_dict = json.load(f)
                # Convert to defaultdict to handle new keywords
                doc_index = defaultdict(list, loaded_dict)
                # Convert old format (string values) to new format (list values)
                for kw, value in list(doc_index.items()):
                    if isinstance(value, str):
                        doc_index[kw] = [value]
            print(f"Loaded existing DocIndex with {len(doc_index)} keywords")
        except Exception as e:
            print(f"Warning: Could not load existing DocIndex, starting fresh. ({e})")
    
    # Configure options for PDF processing
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
    
    # Process all PDF files
    pdf_success = 0
    pdf_failed = 0
    failed_doc = []
    for pdf_path in pdf_files:
        success, doc_index = process_pdf(pdf_path, output_dir, opt, doc_index, args.model)
        if success:
            pdf_success += 1
        else:
            pdf_failed += 1
            failed_doc.append(pdf_path)
        # Save the final DocIndex
        print(f"\n{'='*80}")
        print("Saving final DocIndex...")
        print(f"{'='*80}")
        
        os.makedirs(output_dir, exist_ok=True)
        with open(doc_index_path, "w", encoding="utf-8") as f:
            json.dump(doc_index, f, indent=2, ensure_ascii=False)
        
        print(f'\nDocIndex saved to: {doc_index_path}')
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
