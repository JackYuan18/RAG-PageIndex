import argparse
import os
import json
from pageindex import *
from pageindex.page_index_md import md_to_tree
from collections import defaultdict
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

    for i, new_kw in enumerate(keywords):
        for existing_kw in list(doc_index.keys()):
            # print(f'new_kw: {new_kw}, existing_kw: {existing_kw}')
            # Use AI to determine whether similar in meaning
            if new_kw == existing_kw:
                continue
            if keywords_are_similar(new_kw, existing_kw, model=model)=='True':
                print(f"similar: {new_kw} and {existing_kw}, {keywords_are_similar(new_kw, existing_kw, model=model)}")
                # Use AI to create a concise merged keyword
                merged_kw = make_merged_keyword(new_kw, existing_kw, model=model)
                # Update doc_index (replace existing_kw with merged_kw, preserve value)
                updated_doc_index[merged_kw] = doc_index.pop(existing_kw)
                # Update all occurrences in updated_keywords to merged_kw
                updated_keywords[i] = merged_kw      
                break
        
    # Remove duplicates and return
    return updated_keywords, updated_doc_index

if __name__ == "__main__":
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Process PDF or Markdown document and generate structure')
    parser.add_argument('--pdf_path', type=str, default = '/home/zyuan/NSTSCE_Bot/NSTSCE/Database/Communication-aware_Distributed_Gaussian_Process_Regression_Algorithms_for_Real-time_Machine_Learning.pdf', help='Path to the PDF file')
    # parser.add_argument('--pdf_path', type=str, default = '/home/zyuan/NSTSCE_Bot/NSTSCE/Database/Communication-aware_Distributed_Gaussian_Process_Regression_Algorithms_for_Real-time_Machine_Learning.pdf', help='Path to the PDF file')
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
                      
    # Markdown specific arguments
    parser.add_argument('--if-thinning', type=str, default='no',
                      help='Whether to apply tree thinning for markdown (markdown only)')
    parser.add_argument('--thinning-threshold', type=int, default=5000,
                      help='Minimum token threshold for thinning (markdown only)')
    parser.add_argument('--summary-token-threshold', type=int, default=200,
                      help='Token threshold for generating summaries (markdown only)')
    args = parser.parse_args()
    

 
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
        # Configure options
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

        # Process the PDF
        toc_with_page_number = page_index_main(args.pdf_path, opt)
        print('Parsing done, saving to file...')
        
        # Save results
        pdf_name = os.path.splitext(os.path.basename(args.pdf_path))[0]    
        output_dir = './results'
        output_file = f'{output_dir}/{pdf_name}_structure.json'
        os.makedirs(output_dir, exist_ok=True)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(toc_with_page_number, f, indent=2)
        
        print(f'Tree structure saved to: {output_file}')
        
        # Update DocIndex
        print('Updating DocIndex...')
        doc_index_path = os.path.join(output_dir, "DocIndex")
        doc_index = defaultdict(list)

        # Try to load if exists
        if os.path.exists(doc_index_path):
            try:
                with open(doc_index_path, "r", encoding="utf-8") as f:
                    loaded_dict = json.load(f)
                    # Convert to defaultdict to handle new keywords
                    doc_index = defaultdict(list, loaded_dict)
            except Exception as e:
                print(f"Warning: Could not load existing DocIndex, starting fresh. ({e})")

        # Get the list of keywords from the tree structure
        keywords = []
        if 'keywords' in toc_with_page_number:
            keywords = toc_with_page_number['keywords'].split(',')
        print(f'Keywords: {keywords}')

        merged_keywords, doc_index = merge_keywords(keywords, doc_index, model=args.model)

        
        # Add new keywords to the DocIndex
        # Register each keyword in the DocIndex
        for kw in merged_keywords:
            doc_index[kw].append(output_file)

        # Save the updated DocIndex
        with open(doc_index_path, "w", encoding="utf-8") as f:
            json.dump(doc_index, f, indent=2, ensure_ascii=False)
        
        print(f'Updated DocIndex saved to: {doc_index_path}')
            
    elif args.md_path:
        # Validate Markdown file
        if not args.md_path.lower().endswith(('.md', '.markdown')):
            raise ValueError("Markdown file must have .md or .markdown extension")
        if not os.path.isfile(args.md_path):
            raise ValueError(f"Markdown file not found: {args.md_path}")
            
        # Process markdown file
        print('Processing markdown file...')
        
        # Process the markdown
        import asyncio
        
        # Use ConfigLoader to get consistent defaults (matching PDF behavior)
        from pageindex.utils import ConfigLoader
        config_loader = ConfigLoader()
        
        # Create options dict with user args
        user_opt = {
            'model': args.model,
            'if_add_node_summary': args.if_add_node_summary,
            'if_add_doc_description': args.if_add_doc_description,
            'if_add_node_text': args.if_add_node_text,
            'if_add_node_id': args.if_add_node_id
        }
        
        # Load config with defaults from config.yaml
        opt = config_loader.load(user_opt)
        
        toc_with_page_number = asyncio.run(md_to_tree(
            md_path=args.md_path,
            if_thinning=args.if_thinning.lower() == 'yes',
            min_token_threshold=args.thinning_threshold,
            if_add_node_summary=opt.if_add_node_summary,
            summary_token_threshold=args.summary_token_threshold,
            model=opt.model,
            if_add_doc_description=opt.if_add_doc_description,
            if_add_node_text=opt.if_add_node_text,
            if_add_node_id=opt.if_add_node_id
        ))
        
        print('Parsing done, saving to file...')
        
        # Save results
        md_name = os.path.splitext(os.path.basename(args.md_path))[0]    
        output_dir = './results'
        output_file = f'{output_dir}/{md_name}_structure.json'
        os.makedirs(output_dir, exist_ok=True)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(toc_with_page_number, f, indent=2, ensure_ascii=False)
        
        # Generate keywords if doc_description exists but keywords don't
        
        
        
        print(f'Tree structure saved to: {output_file}')

        keywords = toc_with_page_number['keywords']
        print(f'Keywords: {keywords}')
        
        print('Updating DocIndex...')
        doc_index_path = os.path.join(output_dir, "DocIndex.json")
        doc_index = {}

        # Try to load if exists
        if os.path.exists(doc_index_path):
            try:
                with open(doc_index_path, "r", encoding="utf-8") as f:
                    doc_index = json.load(f)
            except Exception as e:
                print(f"Warning: Could not load existing DocIndex, starting fresh. ({e})")


        # Register each keyword in the DocIndex
        for kw in keywords:
            doc_index[kw] = output_file

        # Save the updated DocIndex
        with open(doc_index_path, "w", encoding="utf-8") as f:
            json.dump(doc_index, f, indent=2, ensure_ascii=False)
        print(f'Updated DocIndex saved to: {doc_index_path}')