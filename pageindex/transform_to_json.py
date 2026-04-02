import re
import json
import ast


# ---------------------------------------------------------------------------
# Patterns used in both detection and extraction
# ---------------------------------------------------------------------------

HEADING_RE = re.compile(
    r'^('
    r'\d+(?:\.\d+){2,}\s{1,10}.+'          # deep sub-subsections: 2.1.1  Title
    r'|\d+(?:\.\d+)*\.\s+.+'               # top-level: 1. Scope
    r'|\d+\.\d+(?:\.\d+)*\s+.+'            # subsections: 1.1 Purpose
    r'|APPENDIX\s+[A-Z](?:\.\d+)?\s*\.\s+.+'  # APPENDIX A. Title
    r'|[A-Z]\.\d+\s+.+'                    # appendix sub: C.1 Title
    r')$'
)

RUNNING_HEADER_RE = re.compile(r'AVSC Best Practice for Developing')  # adapt per doc


def _clean_lines(page_text: str) -> list[str]:
    """Return non-empty, stripped lines from a page, dropping running headers."""
    return [
        l.strip() for l in page_text.split('\n')
        if l.strip() and not RUNNING_HEADER_RE.search(l)
    ]


# ---------------------------------------------------------------------------
# TOC page detection
# ---------------------------------------------------------------------------

TOC_KEYWORDS = re.compile(r'table\s+of\s+contents', re.IGNORECASE)
TOC_HEADING_DENSITY_THRESHOLD = 0.70   # fraction of lines that look like headings


def detect_toc_pages(pages: list[tuple[str, str]]) -> set[str]:
    """
    Automatically identify TOC pages by two complementary signals:
      1. High heading-line density (≥ threshold): almost every line is a heading.
      2. Presence of a 'Table of Contents' keyword anywhere on the page.
    A page is flagged as a TOC if BOTH signals are present, OR if density alone
    is extremely high (≥ 0.90) — catching TOCs that omit the label.
    """
    toc_pages = set()
    for marker, content in pages:
        lines = _clean_lines(content)
        if not lines:
            continue
        heading_count = sum(1 for l in lines if HEADING_RE.match(l))
        density = heading_count / len(lines)
        has_keyword = bool(TOC_KEYWORDS.search(content))

        if density >= 0.90 or (density >= TOC_HEADING_DENSITY_THRESHOLD and has_keyword):
            toc_pages.add(marker)

    return toc_pages


# ---------------------------------------------------------------------------
# Main extraction
# ---------------------------------------------------------------------------

SECTION_PATTERNS = [
    re.compile(r'^(\d+(?:\.\d+){2,})\s{1,10}(.+)$'),           # 2.1.1  Title
    re.compile(r'^(\d+(?:\.\d+)*)\.\s+(.+)$'),                  # 1. Title
    re.compile(r'^(\d+\.\d+(?:\.\d+)*)\s+(.+)$'),               # 1.1 Title
    re.compile(r'^(APPENDIX\s+[A-Z](?:\.\d+)?)\s*\.\s+(.+)$'),  # APPENDIX A. Title
    re.compile(r'^([A-Z]\.\d+)\s+(.+)$'),                       # C.1 Title
]

SKIP_PATTERNS = [
    re.compile(r'^\d{1,2}\s{1,4}[\u201c"a-z]'),  # footnotes
    re.compile(r'^\[\d+\]'),                        # references
    re.compile(r'https?://'),                       # URLs
    re.compile(r'^\d+\s+Commonwealth'),             # address
    re.compile(r'^\d+\(\d'),                        # journal volume
    re.compile(r'^\+1\s'),                          # phone
    RUNNING_HEADER_RE,
]


def transform_text_to_json(text: str) -> list[dict]:
    """
    Transforms document text with <physical_index_N> markers into a structured
    JSON list of section headings.

    TOC pages are detected automatically based on heading-line density and/or
    the presence of a 'Table of Contents' label — no hard-coded page index.

    Args:
        text: Raw document text containing <physical_index_N> markers.

    Returns:
        List of dicts with keys: "structure", "title", "physical_index"
    """

    text = text.replace('\\n', '\n')

    # Parse pages
    parts = re.split(r'<(physical_index_\d+)>', text)
    pages = [(parts[i], parts[i + 1]) for i in range(1, len(parts) - 1, 2)]

    # Auto-detect and skip TOC pages
    toc_pages = detect_toc_pages(pages)

    results = []
    seen = set()

    for physical_index, page_text in pages:
        if physical_index in toc_pages:
            continue

        for line in page_text.split('\n'):
            line = line.strip()
            if not line:
                continue
            if any(p.search(line) for p in SKIP_PATTERNS):
                continue

            for pattern in SECTION_PATTERNS:
                m = pattern.match(line)
                if m:
                    structure = m.group(1).strip()
                    title = re.sub(r'\s{2,}', ' ', m.group(2).strip())

                    # Skip long body-text sentences misidentified as headings
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
                    break

    return results


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def load_text(input_path: str) -> str:
    """Load file, unwrapping Python list literals if needed."""
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
