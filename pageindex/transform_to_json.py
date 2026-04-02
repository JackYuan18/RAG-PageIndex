import re
import json
import ast


def transform_text_to_json(text: str) -> list[dict]:
    """
    Transforms document text with <physical_index_N> markers into a structured
    JSON list of sections and subsections.

    The markers appear as standalone tags (e.g. <physical_index_1>) that delimit
    page boundaries. Each page's content sits between two consecutive markers.

    Args:
        text: Raw document text containing <physical_index_N> markers and
              numbered section headings.

    Returns:
        List of dicts with keys: "structure", "title", "physical_index"
    """

    # Unescape double-escaped newlines (common when stored as Python list literal)
    text = text.replace('\\n', '\n')

    # Split on the standalone <physical_index_N> markers
    # Pattern captures the marker name and the text that follows it
    parts = re.split(r'<(physical_index_\d+)>', text)
    # parts = [pre-first-marker, marker1, content1, marker2, content2, ...]

    # Build list of (physical_index, page_text) pairs
    pages = []
    for i in range(1, len(parts) - 1, 2):
        physical_index = parts[i]
        page_text = parts[i + 1]
        pages.append((physical_index, page_text))

    # Regex patterns for section headings (tried in order; first match wins)
    section_patterns = [
        # Sub-subsections with extra spaces: "2.1.1  SAE Publications"
        re.compile(r'^(\d+(?:\.\d+){2,})\s{1,10}(.+)$'),
        # Top-level numbered sections: "1. Scope", "10. Abbreviations"
        re.compile(r'^(\d+(?:\.\d+)*)\.\s+(.+)$'),
        # Subsections without trailing dot: "1.1 Purpose", "5.3.4 Utilize..."
        re.compile(r'^(\d+\.\d+(?:\.\d+)*)\s+(.+)$'),
        # Appendix sections: "APPENDIX A. Quick Look"
        re.compile(r'^(APPENDIX\s+[A-Z](?:\.\d+)?)\.\s+(.+)$'),
        # Appendix subsections: "C.1 Standard Statistical Measures"
        re.compile(r'^([A-Z]\.\d+)\s+(.+)$'),
    ]

    # Skip lines that look like footnotes, references, URLs, or addresses
    skip_patterns = [
        re.compile(r'^\d{1,2}\s{1,4}[\u201c"a-z]'),  # footnotes: "1  Other relevant..."
        re.compile(r'^\[\d+\]'),                        # references: "[1] NHTSA..."
        re.compile(r'https?://'),                       # URLs
        re.compile(r'^\d+\s+Commonwealth'),             # street address
        re.compile(r'^\d+\(\d'),                        # journal volume: "35(2), ..."
        re.compile(r'^\+1\s'),                          # phone number
    ]

    results = []
    seen = set()  # Deduplicate across pages

    for physical_index, page_text in pages:
        lines = page_text.split('\n')
        for line in lines:
            line = line.strip()
            if not line:
                continue
            if any(p.match(line) for p in skip_patterns):
                continue

            for pattern in section_patterns:
                match = pattern.match(line)
                if match:
                    structure = match.group(1).strip()
                    title = match.group(2).strip()

                    # Collapse OCR-introduced extra spaces ("Perform ance" → "Performance")
                    title = re.sub(r'\s{2,}', ' ', title)

                    # Skip long sentences that look like body text, not headings
                    if title.endswith('.') and len(title) > 80:
                        continue

                    key = (structure, title)
                    if key not in seen:
                        seen.add(key)
                        results.append({
                            "structure": structure,
                            "title": title,
                            "physical_index": f"<{physical_index}>"
                        })
                    break  # Only match one pattern per line

    return results


def load_text(input_path: str) -> str:
    """
    Reads the input file. Handles:
      - Python list literal wrapping a string (e.g. ['...']).
      - Plain text.
    """
    with open(input_path, "r", encoding="utf-8") as f:
        raw = f.read()

    stripped = raw.strip()
    if stripped.startswith("['") or stripped.startswith('["'):
        items = ast.literal_eval(stripped)
        return items[0] if items else ""
    return raw


def main():
    input_path = "group_texts.txt"
    output_path = "output.json"

    text = load_text(input_path)
    sections = transform_text_to_json(text)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(sections, f, indent=2, ensure_ascii=False)

    print(f"Extracted {len(sections)} sections -> {output_path}")
    for s in sections:
        print(f"  [{s['structure']}] {s['title']}  ({s['physical_index']})")


if __name__ == "__main__":
    main()
