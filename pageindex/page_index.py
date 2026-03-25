import os
import json
import copy
import math
import random
import re
import time
from typing import Any, Dict, List, Optional
from .utils import *
import os
from concurrent.futures import ThreadPoolExecutor, as_completed


################### check title in page #########################################################
async def check_title_appearance(item, page_list, start_index=1, model=None):    
    title=item['title']
    if 'physical_index' not in item or item['physical_index'] is None:
        return {'list_index': item.get('list_index'), 'answer': 'no', 'title':title, 'page_number': None}
    
    
    page_number = item['physical_index']
    page_text = page_list[page_number-start_index][0]

    prompt = {}
    prompt['system_prompt'] = f"""
    You are a helpful assistant that checks if a section title appears on a specific page.

    CRITICAL JSON RULES:
    - Use double quotes (") for all strings
    - Escape control characters: use \n for newline, \t for tab
    - Keep "thinking" field SHORT (max 50 words)
    - Replace any newlines/tabs in strings with spaces or escape them
    - DO NOT include escaped quotes like \\"


    VERIFICATION RULES:
    1. Check if the title appears ANYWHERE on the page
    2. Use fuzzy matching:
    - Ignore extra/missing spaces
    - Ignore case differences (uppercase/lowercase)
    - Match if the core words match
    3. The title should be recognizable, even if formatting differs
    4. Answer "yes" if you can find a clear match, "no" if not

    Reply format:
    {{
        
        "thinking": <why do you think the section appears or starts in the page_text>
        "answer": "yes or no" (yes if the section appears or starts in the page_text, no otherwise)
    }}
    IMPORTANT:
    - Output ONLY the JSON object
    - Keep thinking field very short or omit it
    - Escape all control characters properly
    - Do not include explanations outside JSON
    """

    prompt['user_prompt'] = f"""
    Verify if a section title appears on a specific page.

    Title: "{title}"
    Page text: {page_text}
    """
    response = await ChatGPT_API_async(model=model, prompt=prompt)
   
    response = extract_json(response)
    if 'answer' in response:
        answer = response['answer']
    else:
        answer = 'no'
    return {'list_index': item['list_index'], 'answer': answer, 'title': title, 'page_number': page_number}


async def check_title_appearance_in_start(title, page_text, model=None, logger=None):    
    
    prompt = {}
    prompt['system_prompt'] = f"""
    You are a helpful assistant that checks if a section title appears at the BEGINNING of a page.
    
    CRITICAL: Title must be at the START (first 20% of content). If there's significant content before the title, answer "no".

    JSON RULES:
    - Use double quotes (") only
    - Escape control characters (\\n, \\t)
    - Output ONLY JSON, no explanations

    Fuzzy matching rules:
    - Ignore spaces differences
    - Ignore case differences
    - Match if core words match

    reply format:
    {{
        "thinking": <why do you think the section appears or starts in the page_text>
        "start_begin": "yes or no" (yes if the section starts in the beginning of the page_text, no otherwise)
    }}
    Directly return the final JSON structure. Do not output anything else."
    """

    prompt['user_prompt'] = f"""
    
    Check if a section title appears at the BEGINNING of a page.


    The given section title is {title}.
    The given page_text is {page_text}.
    
    
    """
  
    response = ChatGPT_API(model=model, prompt=prompt)
    
    response = extract_json(response)
    if logger:
        logger.info(f"Response: {response}")
    return response.get("start_begin", "no")


async def check_title_appearance_in_start_concurrent(structure, page_list, model=None, logger=None):
    if logger:
        logger.info("Checking title appearance in start concurrently")
    
    # skip items without physical_index
    for item in structure:
        if item.get('physical_index') is None:
            item['appear_start'] = 'no'

    # only for items with valid physical_index
    tasks = []
    valid_items = []
    for item in structure:
        if item.get('physical_index') is not None:
            page_text = page_list[item['physical_index'] - 1][0]
            tasks.append(check_title_appearance_in_start(item['title'], page_text, model=model, logger=logger))
            valid_items.append(item)

    results = await asyncio.gather(*tasks, return_exceptions=True)
    for item, result in zip(valid_items, results):
        if isinstance(result, Exception):
            if logger:
                logger.error(f"Error checking start for {item['title']}: {result}")
            item['appear_start'] = 'no'
        else:
            item['appear_start'] = result

    return structure


def toc_detector_single_page(content, model=None):
    
    
    prompt = {}
    prompt['system_prompt'] = f"""
    Your job is to detect if there is a table of content provided in the given text.

    CRITICAL DEFINITION: A table of contents is a DEDICATED LISTING PAGE that shows:
    - Section titles/chapters
    - Page numbers where each section starts
    - Usually formatted as "Section Title ................... Page Number"
    - Appears BEFORE the main content begins

    IMPORTANT: The following are NOT table of contents:
    - Section headings within the document (e.g., "I. INTRODUCTION", "A. Notations")
    - Abstract, summary, or introduction text
    - Notation lists, figure lists, table lists
    - Bibliography or references sections
    - Just having section headings does NOT mean there's a TOC

    A research paper with section headings but NO dedicated TOC page should be detected as "no".

    Return the following JSON format:
    {{
        "thinking": <why do you think there is a table of content in the given text>
        "toc_detected": "<yes or no>",
    }}

    Directly return the final JSON structure. Do not output anything else.
    Please note: abstract,summary, notation list, figure list, table list, etc. are not table of contents.
    
    """

    prompt['user_prompt'] = f"""
    Given text: {content}

    """
    
    response = ChatGPT_API(model=model, prompt=prompt)
    # print(f'toc_detector_single_page response: {response}')
    # INSERT_YOUR_CODE
    # Check and remove any trailing comma in the response if present

    response = re.sub(r',\s*([\]}])\s*$', r'\1', response)

    json_content = extract_json(response)    
    # print(f'toc_detector_single_page response: {response}')
    # print(f'toc_detector_single_page json_content: {json_content}')
    return json_content['toc_detected']


def check_if_toc_extraction_is_complete(content, toc, model=None):
    
    
    prompt = {}
    prompt['system_prompt'] = f"""
    You are a helpful assistant that checks if the table of contents is complete.
    
    You are given a partial document and a table of contents.
    
    We must reply in the following format:
    {{
        "thinking": <why do you think the table of contents is complete or not>
        "completed": "yes" or "no"
    }}
    Directly return the final JSON structure. Do not output anything else.
    """

    prompt['user_prompt'] = f"""
    Document: {content}
    Table of contents: {toc}
    """
  
    
    
    response = ChatGPT_API(model=model, prompt=prompt)
    
   

    print(f'check_if_toc_extraction_is_complete response: {response}')
    json_content = extract_json(response)
    return json_content['completed']


def check_if_toc_transformation_is_complete(content, toc, model=None):
    
    prompt = {}
    prompt['system_prompt'] = f"""
    You are a helpful assistant that checks if the cleaned table of contents is complete.
    
    You are given a raw table of contents and a cleaned table of contents.
    
    We must reply in the following format:
    {{
        "thinking": <why do you think the cleaned table of contents is complete or not>
        "completed": "yes" or "no"
    }}
    Directly return the final JSON structure. Do not output anything else.
    
    IMPORTANT: You must respond with the format above.
    Do not include any explanation, reasoning, or additional text.
    Do not use markdown formatting or code blocks.
    Just return the final JSON structure.
    
    """

    prompt['user_prompt'] = f"""
    Raw Table of contents: {content}
    Cleaned Table of contents: {toc}
    """
    
    
    response = ChatGPT_API(model=model, prompt=prompt)
    json_content = extract_json(response)
    return json_content['completed'], json_content['thinking']

def extract_toc_content(content, model=None):
    
    prompt = {}
    prompt['system_prompt'] = f"""
    You are a helpful assistant that extracts the full table of contents from the given text.
    Replace ... with ... in the given text.

    Directly return the full table of contents content. Do not output anything else. """

    prompt['user_prompt'] = f"""
    Given text: {content}
    """
    
    response, finish_reason = ChatGPT_API_with_finish_reason(model=model, prompt=prompt)
    
    if_complete,_ = check_if_toc_transformation_is_complete(content, response, model)
    if if_complete == "yes" and finish_reason == "finished":
        return response
    
    chat_history = [
        {"role": "user", "content": prompt['user_prompt']}, 
        {"role": "assistant", "content": response},    
    ]
    prompt = f"""please continue the generation of table of contents , directly output the remaining part of the structure"""
    new_response, finish_reason = ChatGPT_API_with_finish_reason(model=model, prompt=prompt, chat_history=chat_history)
    response = response + new_response
    if_complete, _ = check_if_toc_transformation_is_complete(content, response, model)
    
    while not (if_complete == "yes" and finish_reason == "finished"):
        chat_history = [
            {"role": "user", "content": prompt}, 
            {"role": "assistant", "content": response},    
        ]
        prompt = f"""please continue the generation of table of contents , directly output the remaining part of the structure"""
        new_response, finish_reason = ChatGPT_API_with_finish_reason(model=model, prompt=prompt, chat_history=chat_history)
        response = response + new_response
        if_complete = check_if_toc_transformation_is_complete(content, response, model)
        
        # Optional: Add a maximum retry limit to prevent infinite loops
        if len(chat_history) > 5:  # Arbitrary limit of 10 attempts
            raise Exception('Failed to complete table of contents after maximum retries')
    
    return response

def detect_page_index(toc_content, model=None,logger=None):
    print('start detect_page_index')
    

    prompt = {}
    prompt['system_prompt'] = f"""
    You are a strict classifier that decides whether a TABLE OF CONTENTS already contains page numbers.

    TASK:
    You will be given only the TOC text. Determine if it contains PAGE NUMBERS (not other types of numbers).

    -WHAT ARE PAGE NUMBERS:
    Page numbers in a TOC are:
    - Small integers (typically 1-500 for most documents)
    - Located at the RIGHT END of lines (often aligned in a column)
    - Separated from section titles by dots, spaces, or alignment: "Section Title .......... 5"
    - Usually appear on MOST lines of the TOC
    - May include Roman numerals for front matter (i, ii, iii, iv, v, etc.)

    WHAT ARE NOT PAGE NUMBERS (COMMON FALSE POSITIVES):
    - Reference citations: "(refer to SAE J2808)", "ISO 12345", "IEEE 802.11"
    - Section numbers: "1.2.3", "Chapter 5", "Section 2.1"
    - Dates: "2024", "1999", "March 15"
    - Standards/Regulations: "J2808", "802.11", "ISO 9001"
    - Large numbers: "2808", "12345", "80211" (unless clearly sequential page numbers)
    - Numbers in parentheses at end: "(J2808)", "(2024)", "(Section 2.1)"
    - Numbers that are part of section titles: "Chapter 5", "Part 3"

    DETECTION RULES:
    1. Look for a PATTERN: Do MOST lines end with small integers (1-500) or Roman numerals?
    2. Check FORMATTING: Are numbers separated from titles by dots/spaces/alignment?
    3. Check SEQUENCE: Do numbers appear sequential or near-sequential?
    4. IGNORE: Numbers in parentheses, large numbers (>1000), alphanumeric codes (like "J2808")

    EXAMPLES:

    GOOD (has page numbers):
    "INTRODUCTION ................ 1
    METHODS ...................... 5
    RESULTS ..................... 10"

    BAD (no page numbers - just section numbers):
    "1. INTRODUCTION
    2. METHODS
    3. RESULTS"

    BAD (has reference citations, not page numbers):
    "Introduction (refer to SAE J2808)
    Methods (ISO 12345)
    Results (IEEE 802.11)"

    OUTPUT FORMAT (MUST BE VALID JSON):
    {{
    "thinking": "<brief explanation of your decision>",
    "page_index_given_in_toc": "yes" or "no"
    }}

    HARD RULES:
    - "page_index_given_in_toc" MUST be exactly "yes" or "no"
    - If numbers are citations, standards, or section numbers → answer "no"
    - Only answer "yes" if you see a clear pattern of page numbers (small integers, right-aligned, sequential)
    - Output ONLY the JSON object. No extra text, no markdown, no explanations outside JSON.
    """

    prompt['user_prompt'] = f"""
    Given text: {toc_content}
    """



    response = ChatGPT_API(model=model, prompt=prompt)
    json_content = extract_json(response)
    logger.info(f'toc_content: {toc_content}, detect_page_index response: {json_content}')
    return json_content['page_index_given_in_toc']

def toc_extractor(page_list, toc_page_list, model,logger=None):
    def transform_dots_to_colon(text):       # change dots to colon for better readability
        text = re.sub(r'\.{5,}', ': ', text)
        # Handle dots separated by spaces
        text = re.sub(r'(?:\. ){5,}\.?', ': ', text)
        return text
    
    toc_content = ""
    for page_index in toc_page_list:
        toc_content += page_list[page_index][0]
    toc_content = transform_dots_to_colon(toc_content)
    has_page_index = detect_page_index(toc_content, model=model,logger=logger)
    
    return {
        "toc_content": toc_content,
        "page_index_given_in_toc": has_page_index
    }




def toc_index_extractor(toc, content, model=None):
    print('start toc_index_extractor')
    prompt = {}
    prompt['system_prompt'] = f"""
    You are a JSON formatter. Your ONLY job is to output valid JSON.
    Do not explain, do not comment, do not add any text outside the JSON.

    Task:
    Match section titles to their page locations in a document.

    CRITICAL MATCHING RULES:
    1. Find the EXACT page where each section TITLE appears
    2. Look for the section title at the BEGINNING of a page (usually near the top)
    3. Use fuzzy matching:
    - Ignore extra spaces
    - Ignore case differences
    - Match partial titles if the beginning matches
    4. The section title should appear prominently, not buried in text
    5. If you cannot find a clear match, do NOT assign a physical_index

    MATCHING PROCESS:
    For each section title:
    1. Search through all pages for the title
    2. Look for the title near the START of a page (first 20% of page content)
    3. Match should be clear and unambiguous
    4. If found, use the <physical_index_X> tag from that page
    5. If NOT found, omit physical_index (do not guess)

    JSON OUTPUT RULES:
    - Use double quotes (") for all strings
    - Only include physical_index if you found a clear match
    - Output ONLY valid JSON array

    Output format:
    [
        {{"structure": "1", "title": "Introduction", "physical_index": "<physical_index_3>"}},
        {{"structure": "1.1", "title": "Background", "physical_index": "<physical_index_4>"}},
        {{"structure": "1.2", "title": "Related Work"}}  ← No physical_index if not found
    ]

    Output ONLY the JSON array, nothing else."""

    prompt['user_prompt'] = f"""
    Table of contents: {toc}
    Document pages: {content}
    """
    
    response = ChatGPT_API(model=model, prompt=prompt)
    json_content = extract_json(response)    
    return json_content



def toc_transformer(toc_content, model=None, logger=None):
    print('start toc_transformer')
    
    prompt = {}
    prompt['system_prompt'] = f"""
    You are a JSON formatter. Your ONLY job is to output valid JSON.
    Do not explain, do not comment, do not add any text outside the JSON.
    
    Your task is to continue the table of contents json structure, directly output the remaining part of the json structure.

    
    You are given a table of contents, You job is to transform the whole table of content into a JSON format included table_of_contents.

    structure is the numeric system which represents the index of the hierarchy section in the table of contents. For example, the first section has structure index 1, the first subsection has structure index 1.1, the second subsection has structure index 1.2, etc.

    The response should be in the following JSON format: 
    {{
    "table_of_contents": [
        {{
            "structure": <structure index, "x.x.x" or None> (string),
            "title": <title of the section>,
            "page": <page number or None>,
        }},
        ...
        ],
    }}
    IMPORTANT: You should transform the full table of contents in one go.
    Directly return the final JSON structure as specified in the format.
    Do not output anything else. """

    prompt['user_prompt'] = f"""
    Given table of contents: {toc_content}
    """


    
    last_complete, finish_reason = ChatGPT_API_with_finish_reason(model=model, prompt=prompt)
    
    if_complete,_ = check_if_toc_transformation_is_complete(toc_content, last_complete, model)
    if if_complete == "yes" and finish_reason == "finished":
        # print(f'toc_transformer response: {last_complete}')
        last_complete = extract_json(last_complete)
        
        cleaned_response=convert_page_to_int(last_complete['table_of_contents'])
        return cleaned_response
    print('Done for the first part')
    cnt = 1
    last_complete = get_json_content(last_complete)
    while not (if_complete == "yes" and finish_reason == "finished") and cnt < 20:
        cnt += 1
        # print(f'Working on the {cnt}th part')
        # print(f'last_complete: {last_complete}, finish_reason: {finish_reason}, if_complete: {if_complete}, thinking: {_}')
        position = last_complete.rfind('}')
        if position != -1:
            last_complete = last_complete[:position+2]
        
        prompt = {}
        prompt['system_prompt'] = f"""
        You are a JSON formatter. Your ONLY job is to output valid JSON.
        Do not explain, do not comment, do not add any text outside the JSON.

        Task:
        You are given a raw table of contents and a incomplete transformed table of contents json structure.
        Continue the table of contents json structure, directly output the remaining part of the json structure.
        
        REQUIRED JSON FORMAT:
        {{
        "table_of_contents": [
            {{
                "structure": <structure index, "x.x.x" or None> (string),
                "title": <title of the section>,
                "page": <page number or None>,
            }},
            ...
            ],
        }}"""

        prompt['user_prompt'] = f"""
        Raw table of contents: {toc_content}
        
        Incomplete transformed table of contents json structure: {last_complete}
        
        Please continue the json structure, directly output the remaining part of the json structure.
        """
 
        
        new_complete, finish_reason = ChatGPT_API_with_finish_reason(model=model, prompt=prompt)
        

        if new_complete.startswith('```json'):
            new_complete =  get_json_content(new_complete)
            last_complete = last_complete+new_complete

        if_complete,_ = check_if_toc_transformation_is_complete(toc_content, last_complete, model)
        logger.info(f'toc_content: {toc_content}')
        logger.info(f'toc_transformer new_complete: {new_complete}, if_complete: {if_complete}, thinking: {_}')
    if cnt >= 20:
        raise Exception('toc_transformer failed to complete')
    last_complete = json.loads(last_complete)

    cleaned_response=convert_page_to_int(last_complete['table_of_contents'])
    return cleaned_response
    



def find_toc_pages(start_page_index, page_list, opt, logger=None):
    print('start find_toc_pages')
    last_page_is_yes = False
    toc_page_list = []
    i = start_page_index
    
    while i < len(page_list):
        # Only check beyond max_pages if we're still finding TOC pages
        if i >= opt.toc_check_page_num and not last_page_is_yes:
            break
        detected_result = toc_detector_single_page(page_list[i][0],model=opt.model)
        if detected_result == 'yes':
            if logger:
                logger.info(f'Page {i} has toc')
            toc_page_list.append(i)
            last_page_is_yes = True
        elif detected_result == 'no' and last_page_is_yes:
            if logger:
                logger.info(f'Found the last page with toc: {i-1}')
            break
        i += 1
    
    if not toc_page_list and logger:
        logger.info('No toc found')
        
    return toc_page_list

def remove_page_number(data):
    if isinstance(data, dict):
        data.pop('page_number', None)  
        for key in list(data.keys()):
            if 'nodes' in key:
                remove_page_number(data[key])
    elif isinstance(data, list):
        for item in data:
            remove_page_number(item)
    return data

def extract_matching_page_pairs(toc_page, toc_physical_index, start_page_index):
    pairs = []
    for phy_item in toc_physical_index:
        for page_item in toc_page:
            if phy_item.get('title') == page_item.get('title'):
                physical_index = phy_item.get('physical_index')
                if physical_index is not None and int(physical_index) >= start_page_index:
                    pairs.append({
                        'title': phy_item.get('title'),
                        'page': page_item.get('page'),
                        'physical_index': physical_index
                    })
    return pairs


def calculate_page_offset(pairs):
    differences = []
    for pair in pairs:
        try:  
            physical_index = pair['physical_index']
            page_number = pair['page']
            difference = physical_index - page_number
            differences.append(difference)
        except (KeyError, TypeError):
            print(f'KeyError: {KeyError}, TypeError: {TypeError}')
            continue
    
    if not differences:
        return None
    
    difference_counts = {}
    for diff in differences:
        difference_counts[diff] = difference_counts.get(diff, 0) + 1
    
    most_common = max(difference_counts.items(), key=lambda x: x[1])[0]
    
    return most_common

def add_page_offset_to_toc_json(data, offset):
    if offset is None:
        offset = 0
    for i in range(len(data)):
        if data[i].get('page') is not None and isinstance(data[i]['page'], int):
            data[i]['physical_index'] = data[i]['page'] + offset
            del data[i]['page']
    
    return data



def page_list_to_group_text(page_contents, token_lengths, max_tokens=20000, overlap_page=1):    
    num_tokens = sum(token_lengths)
    
    if num_tokens <= max_tokens:
        # merge all pages into one text
        page_text = "".join(page_contents)
        return [page_text]
    
    subsets = []
    current_subset = []
    current_token_count = 0

    expected_parts_num = math.ceil(num_tokens / max_tokens)
    average_tokens_per_part = math.ceil(((num_tokens / expected_parts_num) + max_tokens) / 2)
    
    for i, (page_content, page_tokens) in enumerate(zip(page_contents, token_lengths)):
        if current_token_count + page_tokens > average_tokens_per_part:

            subsets.append(''.join(current_subset))
            # Start new subset from overlap if specified
            overlap_start = max(i - overlap_page, 0)
            current_subset = page_contents[overlap_start:i]
            current_token_count = sum(token_lengths[overlap_start:i])
        
        # Add current page to the subset
        current_subset.append(page_content)
        current_token_count += page_tokens

    # Add the last subset if it contains any pages
    if current_subset:
        subsets.append(''.join(current_subset))
    
    print('divide page_list to groups', len(subsets))
    return subsets

def add_page_number_to_toc(part, structure, model=None):
    prompt = {}
    prompt['system_prompt'] = f"""
    You are a JSON formatter. Your ONLY job is to output valid JSON.
    Do not explain, do not comment, do not add any text outside the JSON.


    Task:
    You are given an JSON structure of a document and a partial part of the document. 
    Your task is to check if the title that is described in the structure is started in the partial given document.

    The provided text contains tags like <physical_index_X> and <physical_index_X> to indicate the physical location of the page X. 

    CRITICAL REQUIREMENTS:
    1. The "physical_index" field MUST ALWAYS be present in every JSON object
    2. If the full target section starts in the partial given document:
       - Set "start": "yes"
       - Set "physical_index": "<physical_index_X>" (use the exact tag format from the document)
    3. If the full target section does NOT start in the partial given document:
       - Set "start": "no"
       - Set "physical_index": null (use null, not None, not missing)

    REQUIRED JSON FORMAT:
        [
            {{
                "structure": <structure index, "x.x.x" or None> (string),
                "title": <title of the section>,
                "start": "<yes or no>",
                "physical_index": "<physical_index_X> (keep the format)" or None (MUST BE PRESENT - never omit this field)
            }},
            ...
        ]    
    IMPORTANT:
    - Every object MUST have the "physical_index" field
    - Use null (not None, not missing) when section is not found
    - Keep the exact format "<physical_index_X>" when found
    - The given structure contains results from previous parts - preserve them and only add results for the current part
    - Do not change previous results, only add new ones

    Output ONLY the complete JSON array, nothing else."""

    prompt['user_prompt'] = f"""
    Current Partial Document: {part}
    Given Structure: {json.dumps(structure, indent=2)}
    """

    current_json_raw = ChatGPT_API(model=model, prompt=prompt)
    
    json_result = extract_json(current_json_raw)
    
    for item in json_result:
        if 'start' in item:
            del item['start']
    return json_result


def remove_first_physical_index_section(text):
    """
    Removes the first section between <physical_index_X> and <physical_index_X> tags,
    and returns the remaining text.
    """
    pattern = r'<physical_index_\d+>.*?<physical_index_\d+>'
    match = re.search(pattern, text, re.DOTALL)
    if match:
        # Remove the first matched section
        return text.replace(match.group(0), '', 1)
    return text

### add verify completeness
def generate_toc_continue(toc_content, part, model="gpt-4o-2024-11-20"):
    print('start generate_toc_continue')
    prompt = {}
    prompt['system_prompt'] = f"""
        You are a JSON formatter. Your ONLY job is to output valid JSON.

        CRITICAL JSON RULES:
        1. Use double quotes (") for ALL strings. NEVER use single quotes.
        2. Escape special characters in string values (\\n for newline, \\t for tab, \\" for quote)
        3. Replace newlines/tabs in titles with spaces
        4. Output ONLY the JSON array - no explanations, no markdown

        Task: Continue the tree structure from previous part using the given text.

        Structure format: numeric hierarchy (e.g., "1", "1.1", "1.2")
        Title: Extract original title, replace newlines/tabs with spaces
        Physical index: Extract from <physical_index_X> tags

        Output format (valid JSON only):
        [
            {{"structure": "2", "title": "Section Title", "physical_index": "<physical_index_5>"}},
            {{"structure": "2.1", "title": "Subsection Title", "physical_index": "<physical_index_6>"}}
        ]

        Output ONLY the additional JSON array items, nothing else."""
    prompt['user_prompt'] = f"""
    Given text: {part}
    Previous tree structure: {json.dumps(toc_content, indent=2)}
    """


    
    response, finish_reason = ChatGPT_API_with_finish_reason(model=model, prompt=prompt)
    
    if finish_reason == 'finished':
        return extract_json(response)
    else:
        raise Exception(f'finish reason: {finish_reason}')
    
### add verify completeness
def generate_toc_init(part, model=None):
    print('start generate_toc_init')
    prompt = {}
    prompt['system_prompt'] = f"""
        You are a JSON formatter. Your ONLY job is to output valid JSON.

        CRITICAL JSON RULES:
        1. Use double quotes (") for ALL strings. NEVER use single quotes.
        2. Escape special characters in string values (\\n for newline, \\t for tab, \\" for quote)
        3. Replace newlines/tabs in titles with spaces
        4. Output ONLY the JSON array - no explanations, no markdown

        Task: From the given text, extract ONLY section or chapter HEADINGS to build a table of contents.

        Structure format: numeric hierarchy (e.g., "1", "1.1", "1.2")
        
        Title: Extract original title, replace newlines/tabs with spaces
        Physical index: Extract from <physical_index_X> tags

        Output format (valid JSON only):
        [
            {{"structure": "1", "title": "Section Title", "physical_index": "<physical_index_1>"}},
            {{"structure": "1.1", "title": "Subsection Title", "physical_index": "<physical_index_2>"}}
        ]

        Output ONLY the JSON array items, nothing else.

        You are a strict JSON generator.


"""

    prompt['user_prompt'] = f"""
    Given text: {part}
    """
    
    response, finish_reason = ChatGPT_API_with_finish_reason(model=model, prompt=prompt)
   
    if finish_reason == 'finished':
         return extract_json(response)
    else:
        raise Exception(f'finish reason: {finish_reason}')

def process_no_toc(page_list, start_index=1, model=None, logger=None):
    page_contents=[]
    token_lengths=[]
    for page_index in range(start_index, start_index+len(page_list)):
        page_text = f"<physical_index_{page_index}>\n{page_list[page_index-start_index][0]}\n<physical_index_{page_index}>\n\n"
        page_contents.append(page_text)
        token_lengths.append(count_tokens(page_text, model))
    group_texts = page_list_to_group_text(page_contents, token_lengths)
    logger.info(f'len(group_texts): {len(group_texts)}')

    toc_with_page_number= generate_toc_init(group_texts[0], model)
    logger.info(f'generate_toc_init: {toc_with_page_number}')
    for group_text in group_texts[1:]:
        toc_with_page_number_additional = generate_toc_continue(toc_with_page_number, group_text, model)    
        toc_with_page_number.extend(toc_with_page_number_additional)
    logger.info(f'generate_toc: {toc_with_page_number}')

    toc_with_page_number = convert_physical_index_to_int(toc_with_page_number)
    logger.info(f'convert_physical_index_to_int: {toc_with_page_number}')

    return toc_with_page_number

def process_toc_no_page_numbers(toc_content, toc_page_list, page_list,  start_index=1, model=None, logger=None):
    page_contents=[]
    token_lengths=[]
    toc_content = toc_transformer(toc_content, model, logger)
    logger.info(f'toc_transformer: {toc_content}')
    for page_index in range(start_index, start_index+len(page_list)):
        page_text = f"<physical_index_{page_index}>\n{page_list[page_index-start_index][0]}\n<physical_index_{page_index}>\n\n"
        page_contents.append(page_text)
        token_lengths.append(count_tokens(page_text, model))
    
    group_texts = page_list_to_group_text(page_contents, token_lengths)
    logger.info(f'len(group_texts): {len(group_texts)}')

    toc_with_page_number=copy.deepcopy(toc_content)
    for group_text in group_texts:
        toc_with_page_number = add_page_number_to_toc(group_text, toc_with_page_number, model)
    logger.info(f'add_page_number_to_toc: {toc_with_page_number}')

    toc_with_page_number = convert_physical_index_to_int(toc_with_page_number)
    logger.info(f'convert_physical_index_to_int: {toc_with_page_number}')

    return toc_with_page_number



def process_toc_with_page_numbers(toc_content, toc_page_list, page_list, toc_check_page_num=None, model=None, logger=None):
    
    toc_with_page_number = toc_transformer(toc_content, model, logger)
    logger.info(f'toc_with_page_number: {toc_with_page_number}')

    toc_no_page_number = remove_page_number(copy.deepcopy(toc_with_page_number))
    
    start_page_index = toc_page_list[-1] + 1
    main_content = ""
    for page_index in range(start_page_index, min(start_page_index + toc_check_page_num, len(page_list))):
        main_content += f"<physical_index_{page_index+1}>\n{page_list[page_index][0]}\n<physical_index_{page_index+1}>\n\n"

    toc_with_physical_index = toc_index_extractor(toc_no_page_number, main_content, model)
    logger.info(f'toc_with_physical_index: {toc_with_physical_index}')

    toc_with_physical_index = convert_physical_index_to_int(toc_with_physical_index)
    logger.info(f'toc_with_physical_index: {toc_with_physical_index}')

    matching_pairs = extract_matching_page_pairs(toc_with_page_number, toc_with_physical_index, start_page_index)
    logger.info(f'matching_pairs: {matching_pairs}')

    offset = calculate_page_offset(matching_pairs)
    logger.info(f'offset: {offset}')

    toc_with_page_number = add_page_offset_to_toc_json(toc_with_page_number, offset)
    logger.info(f'toc_with_page_number: {toc_with_page_number}')

    toc_with_page_number = process_none_page_numbers(toc_with_page_number, page_list, model=model)
    logger.info(f'toc_with_page_number: {toc_with_page_number}')

    return toc_with_page_number



##check if needed to process none page numbers
def process_none_page_numbers(toc_items, page_list, start_index=1, model=None):
    for i, item in enumerate(toc_items):
        if "physical_index" not in item:
            # logger.info(f"fix item: {item}")
            # Find previous physical_index
            prev_physical_index = 0  # Default if no previous item exists
            for j in range(i - 1, -1, -1):
                if toc_items[j].get('physical_index') is not None:
                    prev_physical_index = toc_items[j]['physical_index']
                    break
            
            # Find next physical_index
            next_physical_index = -1  # Default if no next item exists
            for j in range(i + 1, len(toc_items)):
                if toc_items[j].get('physical_index') is not None:
                    next_physical_index = toc_items[j]['physical_index']
                    break

            page_contents = []
            for page_index in range(prev_physical_index, next_physical_index+1):
                # Add bounds checking to prevent IndexError
                list_index = page_index - start_index
                if list_index >= 0 and list_index < len(page_list):
                    page_text = f"<physical_index_{page_index}>\n{page_list[list_index][0]}\n<physical_index_{page_index}>\n\n"
                    page_contents.append(page_text)
                else:
                    continue

            item_copy = copy.deepcopy(item)
            del item_copy['page']
            result = add_page_number_to_toc(page_contents, item_copy, model)
            print(f'result: {result}')
            if isinstance(result[0]['physical_index'], str) and result[0]['physical_index'].startswith('<physical_index'):
                item['physical_index'] = int(result[0]['physical_index'].split('_')[-1].rstrip('>').strip())
                del item['page']
    
    return toc_items




def check_toc(page_list, opt=None,logger=None):
    toc_page_list = find_toc_pages(start_page_index=0, page_list=page_list, opt=opt)
    if len(toc_page_list) == 0:
        print('no toc found')
        return {'toc_content': None, 'toc_page_list': [], 'page_index_given_in_toc': 'no'}
    else:
        print('toc found')
        logger.info(f'toc_page_list: {toc_page_list}')
        toc_json = toc_extractor(page_list, toc_page_list, opt.model, logger=logger)

        if toc_json['page_index_given_in_toc'] == 'yes':
            print('index found')
            return {'toc_content': toc_json['toc_content'], 'toc_page_list': toc_page_list, 'page_index_given_in_toc': 'yes'}
        else:
            current_start_index = toc_page_list[-1] + 1
            
            while (toc_json['page_index_given_in_toc'] == 'no' and 
                   current_start_index < len(page_list) and 
                   current_start_index < opt.toc_check_page_num):
                
                additional_toc_pages = find_toc_pages(
                    start_page_index=current_start_index,
                    page_list=page_list,
                    opt=opt
                )
                
                if len(additional_toc_pages) == 0:
                    break

                additional_toc_json = toc_extractor(page_list, additional_toc_pages, opt.model, logger)
                if additional_toc_json['page_index_given_in_toc'] == 'yes':
                    print('index found')
                    return {'toc_content': additional_toc_json['toc_content'], 'toc_page_list': additional_toc_pages, 'page_index_given_in_toc': 'yes'}

                else:
                    current_start_index = additional_toc_pages[-1] + 1
            print('index not found')
            return {'toc_content': toc_json['toc_content'], 'toc_page_list': toc_page_list, 'page_index_given_in_toc': 'no'}






################### fix incorrect toc #########################################################
def single_toc_item_index_fixer(section_title, content, model="gpt-4o-2024-11-20"):
    prompt = {}
    prompt['system_prompt'] = f"""
    You are a JSON formatter. Your ONLY job is to output valid JSON.
    You are given a section title and several pages of a document, your job is to find the physical index of the start page of the section in the partial document.

    The provided pages contains tags like <physical_index_X> and <physical_index_X> to indicate the physical location of the page X.

    Reply in a JSON format:
    {
        "thinking": <explain which page, started and closed by <physical_index_X>, contains the start of this section>,
        "physical_index": "<physical_index_X>" (keep the format)
    }
    Directly return the final JSON structure. Do not output anything else."""

    prompt['user_prompt'] = f"""
    Section Title: {section_title}
    Document pages: {content}
    """


    
    response = ChatGPT_API(model=model, prompt=prompt)
    
    json_content = extract_json(response)    
    return convert_physical_index_to_int(json_content['physical_index'])



async def fix_incorrect_toc(toc_with_page_number, page_list, incorrect_results, start_index=1, model=None, logger=None):
    print(f'start fix_incorrect_toc with {len(incorrect_results)} incorrect results')
    incorrect_indices = {result['list_index'] for result in incorrect_results}
    
    end_index = len(page_list) + start_index - 1
    
    incorrect_results_and_range_logs = []
    # Helper function to process and check a single incorrect item
    async def process_and_check_item(incorrect_item):
        list_index = incorrect_item['list_index']
        
        # Check if list_index is valid
        if list_index < 0 or list_index >= len(toc_with_page_number):
            # Return an invalid result for out-of-bounds indices
            return {
                'list_index': list_index,
                'title': incorrect_item['title'],
                'physical_index': incorrect_item.get('physical_index'),
                'is_valid': False
            }
        
        # Find the previous correct item
        prev_correct = None
        for i in range(list_index-1, -1, -1):
            if i not in incorrect_indices and i >= 0 and i < len(toc_with_page_number):
                physical_index = toc_with_page_number[i].get('physical_index')
                if physical_index is not None:
                    prev_correct = physical_index
                    break
        # If no previous correct item found, use start_index
        if prev_correct is None:
            prev_correct = start_index - 1
        
        # Find the next correct item
        next_correct = None
        for i in range(list_index+1, len(toc_with_page_number)):
            if i not in incorrect_indices and i >= 0 and i < len(toc_with_page_number):
                physical_index = toc_with_page_number[i].get('physical_index')
                if physical_index is not None:
                    next_correct = physical_index
                    break
        # If no next correct item found, use end_index
        if next_correct is None:
            next_correct = end_index
        
        incorrect_results_and_range_logs.append({
            'list_index': list_index,
            'title': incorrect_item['title'],
            'prev_correct': prev_correct,
            'next_correct': next_correct
        })

        page_contents=[]
        for page_index in range(prev_correct, next_correct+1):
            # Add bounds checking to prevent IndexError
            list_index = page_index - start_index
            if list_index >= 0 and list_index < len(page_list):
                page_text = f"<physical_index_{page_index}>\n{page_list[list_index][0]}\n<physical_index_{page_index}>\n\n"
                page_contents.append(page_text)
            else:
                continue
        content_range = ''.join(page_contents)
        
        physical_index_int = single_toc_item_index_fixer(incorrect_item['title'], content_range, model)
        
        # Check if the result is correct
        check_item = incorrect_item.copy()
        check_item['physical_index'] = physical_index_int
        check_result = await check_title_appearance(check_item, page_list, start_index, model)

        return {
            'list_index': list_index,
            'title': incorrect_item['title'],
            'physical_index': physical_index_int,
            'is_valid': check_result['answer'] == 'yes'
        }

    # Process incorrect items concurrently
    tasks = [
        process_and_check_item(item)
        for item in incorrect_results
    ]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    for item, result in zip(incorrect_results, results):
        if isinstance(result, Exception):
            print(f"Processing item {item} generated an exception: {result}")
            continue
    results = [result for result in results if not isinstance(result, Exception)]

    # Update the toc_with_page_number with the fixed indices and check for any invalid results
    invalid_results = []
    for result in results:
        if result['is_valid']:
            # Add bounds checking to prevent IndexError
            list_idx = result['list_index']
            if 0 <= list_idx < len(toc_with_page_number):
                toc_with_page_number[list_idx]['physical_index'] = result['physical_index']
            else:
                # Index is out of bounds, treat as invalid
                invalid_results.append({
                    'list_index': result['list_index'],
                    'title': result['title'],
                    'physical_index': result['physical_index'],
                })
        else:
            invalid_results.append({
                'list_index': result['list_index'],
                'title': result['title'],
                'physical_index': result['physical_index'],
            })

    logger.info(f'incorrect_results_and_range_logs: {incorrect_results_and_range_logs}')
    logger.info(f'invalid_results: {invalid_results}')

    return toc_with_page_number, invalid_results



async def fix_incorrect_toc_with_retries(toc_with_page_number, page_list, incorrect_results, start_index=1, max_attempts=3, model=None, logger=None):
    print('start fix_incorrect_toc')
    fix_attempt = 0
    current_toc = toc_with_page_number
    current_incorrect = incorrect_results

    while current_incorrect:
        print(f"Fixing {len(current_incorrect)} incorrect results")
        
        current_toc, current_incorrect = await fix_incorrect_toc(current_toc, page_list, current_incorrect, start_index, model, logger)
                
        fix_attempt += 1
        if fix_attempt >= max_attempts:
            logger.info("Maximum fix attempts reached")
            break
    
    return current_toc, current_incorrect




################### verify toc #########################################################
async def verify_toc(page_list, list_result, start_index=1, N=None, model=None):
    print('start verify_toc')
    # Find the last non-None physical_index
    last_physical_index = None
    for item in reversed(list_result):
        if item.get('physical_index') is not None:
            last_physical_index = item['physical_index']
            break
    
    # Early return if we don't have valid physical indices
    if last_physical_index is None or last_physical_index < len(page_list)/2:
        return 0, []
    
    # Determine which items to check
    if N is None:
        print('check all items')
        sample_indices = range(0, len(list_result))
    else:
        N = min(N, len(list_result))
        print(f'check {N} items')
        sample_indices = random.sample(range(0, len(list_result)), N)

    # Prepare items with their list indices
    indexed_sample_list = []
    for idx in sample_indices:
        item = list_result[idx]
        # Skip items with None physical_index (these were invalidated by validate_and_truncate_physical_indices)
        if item.get('physical_index') is not None:
            item_with_index = item.copy()
            item_with_index['list_index'] = idx  # Add the original index in list_result
            indexed_sample_list.append(item_with_index)

    # Run checks concurrently
    tasks = [
        check_title_appearance(item, page_list, start_index, model)
        for item in indexed_sample_list
    ]
    results = await asyncio.gather(*tasks)
    
    # Process results
    correct_count = 0
    incorrect_results = []
    for result in results:
        if result['answer'] == 'yes':
            correct_count += 1
        else:
            incorrect_results.append(result)
    
    # Calculate accuracy
    checked_count = len(results)
    accuracy = correct_count / checked_count if checked_count > 0 else 0
    print(f"accuracy: {accuracy*100:.2f}%")
    return accuracy, incorrect_results





################### main process #########################################################
async def meta_processor(page_list, mode=None, toc_content=None, toc_page_list=None, start_index=1, opt=None, logger=None):
    print(mode)
    print(f'start_index: {start_index}')
    
    if mode == 'process_toc_with_page_numbers':
        toc_with_page_number = process_toc_with_page_numbers(toc_content, toc_page_list, page_list, toc_check_page_num=opt.toc_check_page_num, model=opt.model, logger=logger)
    elif mode == 'process_toc_no_page_numbers':
        toc_with_page_number = process_toc_no_page_numbers(toc_content, toc_page_list, page_list, model=opt.model, logger=logger)
    else:
        toc_with_page_number = process_no_toc(page_list, start_index=start_index, model=opt.model, logger=logger)
            
    toc_with_page_number = [item for item in toc_with_page_number if item.get('physical_index') is not None] 
    
    toc_with_page_number = validate_and_truncate_physical_indices(
        toc_with_page_number, 
        len(page_list), 
        start_index=start_index, 
        logger=logger
    )
    
    accuracy, incorrect_results = await verify_toc(page_list, toc_with_page_number, start_index=start_index, model=opt.model)
        
    logger.info({
        'mode': 'process_toc_with_page_numbers',
        'accuracy': accuracy,
        'incorrect_results': incorrect_results
    })
    if accuracy == 1.0 and len(incorrect_results) == 0:
        return toc_with_page_number
    if accuracy > 0.6 and len(incorrect_results) > 0:
        toc_with_page_number, incorrect_results = await fix_incorrect_toc_with_retries(toc_with_page_number, page_list, incorrect_results,start_index=start_index, max_attempts=3, model=opt.model, logger=logger)
        return toc_with_page_number
    else:
        if mode == 'process_toc_with_page_numbers':
            return await meta_processor(page_list, mode='process_toc_no_page_numbers', toc_content=toc_content, toc_page_list=toc_page_list, start_index=start_index, opt=opt, logger=logger)
        elif mode == 'process_toc_no_page_numbers':
            return await meta_processor(page_list, mode='process_no_toc', start_index=start_index, opt=opt, logger=logger)
        else:
            # INSERT_YOUR_CODE
            # Save toc_with_page_number to a JSON file for debugging
            debug_file_path = "debug/toc_with_page_number_debug.json"
            try:
                with open(debug_file_path, "w", encoding="utf-8") as f:
                    json.dump(toc_with_page_number, f, ensure_ascii=False, indent=2)
                if logger:
                    logger.info(f"toc_with_page_number saved to {debug_file_path}")
                else:
                    print(f"toc_with_page_number saved to {debug_file_path}")
            except Exception as e:
                if logger:
                    logger.error(f"Failed to save toc_with_page_number to file: {e}")
                else:
                    print(f"Failed to save toc_with_page_number to file: {e}")
            raise Exception('Processing failed')
        
 
async def process_large_node_recursively(node, page_list, opt=None, logger=None):
    node_page_list = page_list[node['start_index']-1:node['end_index']]
    token_num = sum([page[1] for page in node_page_list])
    
    if node['end_index'] - node['start_index'] > opt.max_page_num_each_node and token_num >= opt.max_token_num_each_node:
        print('large node:', node['title'], 'start_index:', node['start_index'], 'end_index:', node['end_index'], 'token_num:', token_num)

        node_toc_tree = await meta_processor(node_page_list, mode='process_no_toc', start_index=node['start_index'], opt=opt, logger=logger)
        node_toc_tree = await check_title_appearance_in_start_concurrent(node_toc_tree, page_list, model=opt.model, logger=logger)
        
        # Filter out items with None physical_index before post_processing
        valid_node_toc_items = [item for item in node_toc_tree if item.get('physical_index') is not None]
        
        if valid_node_toc_items and node['title'].strip() == valid_node_toc_items[0]['title'].strip():
            node['nodes'] = post_processing(valid_node_toc_items[1:], node['end_index'])
            node['end_index'] = valid_node_toc_items[1]['start_index'] if len(valid_node_toc_items) > 1 else node['end_index']
        else:
            node['nodes'] = post_processing(valid_node_toc_items, node['end_index'])
            node['end_index'] = valid_node_toc_items[0]['start_index'] if valid_node_toc_items else node['end_index']
        
    if 'nodes' in node and node['nodes']:
        tasks = [
            process_large_node_recursively(child_node, page_list, opt, logger=logger)
            for child_node in node['nodes']
        ]
        await asyncio.gather(*tasks)
    
    return node

async def tree_parser(page_list, opt, doc=None, logger=None):
    check_toc_result = check_toc(page_list, opt, logger=logger)
    logger.info(check_toc_result)

    if check_toc_result.get("toc_content") and check_toc_result["toc_content"].strip() and check_toc_result["page_index_given_in_toc"] == "yes":
        toc_with_page_number = await meta_processor(
            page_list, 
            mode='process_toc_with_page_numbers', 
            start_index=1, 
            toc_content=check_toc_result['toc_content'], 
            toc_page_list=check_toc_result['toc_page_list'], 
            opt=opt,
            logger=logger)
    else:
        toc_with_page_number = await meta_processor(
            page_list, 
            mode='process_no_toc', 
            start_index=1, 
            opt=opt,
            logger=logger)

    toc_with_page_number = add_preface_if_needed(toc_with_page_number)
    toc_with_page_number = await check_title_appearance_in_start_concurrent(toc_with_page_number, page_list, model=opt.model, logger=logger)
    
    # Filter out items with None physical_index before post_processings
    valid_toc_items = [item for item in toc_with_page_number if item.get('physical_index') is not None]
    
    toc_tree = post_processing(valid_toc_items, len(page_list))
    tasks = [
        process_large_node_recursively(node, page_list, opt, logger=logger)
        for node in toc_tree
    ]
    await asyncio.gather(*tasks)
    
    return toc_tree

import asyncio

async def extract_title_and_authors(page_list, max_pages_to_inspect=10, model=None):
    """
    Use an AI model to extract the document title and authors from the first `max_pages_to_inspect` pages.

    Returns a dict: {'title': ..., 'authors': ...}
    """
    # Grab the first N pages' text for context
    pages_to_check = page_list[:max_pages_to_inspect]
    page_texts = [p[0] if isinstance(p, (list, tuple)) and len(p) > 0 else str(p) for p in pages_to_check]
    document_start = "\n".join(page_texts).strip()

    prompt = {}
    prompt['system_prompt'] = """
        You are a helpful assistant. Your job is to robustly extract the TITLE of the document and the AUTHOR(S) from the provided beginning pages of a scientific/technical/academic document.

        CRITICAL INSTRUCTIONS AND RULES FOR JSON OUTPUT:
        - Output ONLY a JSON object, nothing else.
        - The JSON object must follow:
        {
            "title": "<most likely document title (string, never blank)>",
            "authors": "<concise author string or comma-separated list (string, can be blank if not found)>"
        }
        - Do NOT add explanation or comments, ONLY the JSON object.
        - If title or authors cannot be found, set their values as an empty string ("").

        HOW TO FIND TITLE AND AUTHORS:
        - The title is usually at the very top, before the abstract, often in a large font and may span multiple lines.
        - Exclude "abstract", "introduction", "contents", table of contents text, or headers/footers.
        - Authors may be on the next line(s) under the title or presented as a list, possibly including email addresses or affiliations.
        - If there are multiple potential titles, pick the most prominent/central one.
        - Authors should not include affiliations, only actual author names or author emails (if available).

        Your only task: Output a valid JSON object with the best guess at these two fields.
        """

    prompt['user_prompt'] = f"""
        The following is the beginning text from a document: {document_start}


        Please extract and output only the JSON with fields:
        - "title" (string, best guess at main title, never blank if anything reasonable exists)
        - "authors" (string, comma-separated author names, blank if not found)

        If you cannot robustly determine authors, leave it as an empty string.
        """

    # Call the ChatGPT model (sync or async, depending on available import)
    # We'll assume an async ChatGPT_API_async like elsewhere in this codebase
    response = await ChatGPT_API_async(model=model, prompt=prompt)

    out = extract_json(response)
    # Ensure both keys are present (for robust fallback)
    
    if "title" not in out:
        out["title"] = ""
    if "authors" not in out:
        out["authors"] = ""
    return out


def _record_step_time(
    step_timings: Optional[List[Dict[str, Any]]], label: str, t0: float
) -> None:
    elapsed = time.perf_counter() - t0
    sec = round(elapsed, 3)
    if step_timings is not None:
        step_timings.append({"step": label, "seconds": sec})
    print(f"  {label} — completed in {elapsed:.2f} s")


def page_index_main(doc, opt=None, step_timings: Optional[List[Dict[str, Any]]] = None):
    logger = JsonLogger(doc)

    is_valid_pdf = (
        (isinstance(doc, str) and os.path.isfile(doc) and doc.lower().endswith(".pdf")) or 
        isinstance(doc, BytesIO)
    )
    if not is_valid_pdf:
        raise ValueError("Unsupported input type. Expected a PDF file path or BytesIO object.")

    print("Step 1: Parsing PDF...")
    t0 = time.perf_counter()
    page_list = get_page_tokens(doc)
    logger.info({'page_list': page_list})
    logger.info({'total_page_number': len(page_list)})
    logger.info({'total_token': sum([page[1] for page in page_list])})
    _record_step_time(step_timings, "Step 1: Parsing PDF", t0)

    print("Step 2: Extracting title and authors...")
    t0 = time.perf_counter()
    doc_title_authors = asyncio.run(extract_title_and_authors(page_list, model=opt.model))
    logger.info({'extracted_title': doc_title_authors.get('title'), 'extracted_authors': doc_title_authors.get('authors')})
    _record_step_time(step_timings, "Step 2: Extracting title and authors", t0)

    async def page_index_builder():
        print("Step 3: Building tree structure...")
        t0 = time.perf_counter()
        structure = await tree_parser(page_list, opt, doc=doc, logger=logger)
        _record_step_time(step_timings, "Step 3: Building tree structure", t0)

        print("Step 4: Adding node ids...")
        t0 = time.perf_counter()
        write_node_id(structure)
        _record_step_time(step_timings, "Step 4: Adding node ids", t0)

        print("Step 5: Adding node text...")
        t0 = time.perf_counter()
        add_node_text(structure, page_list)
        _record_step_time(step_timings, "Step 5: Adding node text", t0)

        print("Step 6: Generating summaries for structure...")
        t0 = time.perf_counter()
        await generate_summaries_for_structure(structure, model=opt.model)
        _record_step_time(step_timings, "Step 6: Generating node summaries", t0)

        print("Step 7: Generating parent node summaries...")
        t0 = time.perf_counter()
        await generate_parent_node_summaries_for_structure(structure, model=opt.model)
        _record_step_time(step_timings, "Step 7: Generating parent node summaries", t0)

        print("Step 8: Generating document description...")
        t0 = time.perf_counter()
        clean_structure = create_clean_structure_for_description(structure)
        doc_description = generate_doc_description(clean_structure, model=opt.model)
        _record_step_time(step_timings, "Step 8: Generating document description", t0)

        print("Step 9: Generating document abstract...")
        t0 = time.perf_counter()
        clean_structure = create_clean_structure_for_abstract(structure)
        doc_abstract = generate_doc_abstract(clean_structure, model=opt.model)
        _record_step_time(step_timings, "Step 9: Generating document abstract", t0)

        print("Step 10: Generating keywords...")
        t0 = time.perf_counter()
        keywords = generate_keywords(doc_abstract, model=opt.model)
        _record_step_time(step_timings, "Step 10: Generating keywords", t0)

        return {
                'doc_name': get_pdf_name(doc),
                'doc_path': doc,
                'doc_title': doc_title_authors.get('title'),
                'doc_authors': doc_title_authors.get('authors'),
                'keywords': keywords,
                'doc_description': doc_description,
                'doc_abstract': doc_abstract,
                'structure': structure 
                }


    return asyncio.run(page_index_builder())


def page_index(doc, model=None, toc_check_page_num=None, max_page_num_each_node=None, max_token_num_each_node=None,
               if_add_node_id=None, if_add_node_summary=None, if_add_doc_description=None, if_add_node_text=None):
    
    user_opt = {
        arg: value for arg, value in locals().items()
        if arg != "doc" and value is not None
    }
    opt = ConfigLoader().load(user_opt)
    return page_index_main(doc, opt)


def validate_and_truncate_physical_indices(toc_with_page_number, page_list_length, start_index=1, logger=None):
    """
    Validates and truncates physical indices that exceed the actual document length.
    This prevents errors when TOC references pages that don't exist in the document (e.g. the file is broken or incomplete).
    """
    if not toc_with_page_number:
        return toc_with_page_number
    
    max_allowed_page = page_list_length + start_index - 1
    truncated_items = []
    
    for i, item in enumerate(toc_with_page_number):
        if item.get('physical_index') is not None:
            original_index = item['physical_index']
            if original_index > max_allowed_page:
                item['physical_index'] = None
                truncated_items.append({
                    'title': item.get('title', 'Unknown'),
                    'original_index': original_index
                })
                if logger:
                    logger.info(f"Removed physical_index for '{item.get('title', 'Unknown')}' (was {original_index}, too far beyond document)")
    
    if truncated_items and logger:
        logger.info(f"Total removed items: {len(truncated_items)}")
        
    print(f"Document validation: {page_list_length} pages, max allowed index: {max_allowed_page}")
    if truncated_items:
        print(f"Truncated {len(truncated_items)} TOC items that exceeded document length")
     
    return toc_with_page_number