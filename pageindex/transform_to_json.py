import re
import json
import ast


# ---------------------------------------------------------------------------
# Roman numeral conversion
# ---------------------------------------------------------------------------

ROMAN_VALUES = [
    (1000, 'M'), (900, 'CM'), (500, 'D'), (400, 'CD'),
    (100,  'C'), (90,  'XC'), (50,  'L'), (40,  'XL'),
    (10,   'X'), (9,   'IX'), (5,   'V'), (4,   'IV'), (1, 'I'),
]

def roman_to_int(s: str) -> int | None:
    s = s.upper().strip()
    valid = set('IVXLCDM')
    if not s or not all(c in valid for c in s):
        return None
    result, prev = 0, 0
    for ch in reversed(s):
        val = next((v for v, r in ROMAN_VALUES if r == ch), None)
        if val is None:
            return None
        result += val if val >= prev else -val
        prev = val
    return result if result > 0 else None


# ---------------------------------------------------------------------------
# Structure normalizers
# ---------------------------------------------------------------------------

class NumericNormalizer:
    """
    Handles numeric-style docs. Passes numeric structures through unchanged,
    but converts Appendix entries to continuation numbers after the last
    regular section.

    e.g. after sections 1-10:
      "APPENDIX A"  -> "11"
      "APPENDIX B"  -> "12"
      "C.1"         -> "13.1"   (where APPENDIX C = 13)
      "C.2"         -> "13.2"
    """
    def __init__(self):
        self._max_top_level = 0
        self._appendix_map: dict[str, int] = {}

    def normalize(self, structure: str) -> str:
        # Plain integer top-level section — track the max
        if re.match(r'^\d+$', structure):
            self._max_top_level = max(self._max_top_level, int(structure))
            return structure

        # Dotted numeric (1.1, 2.1.1, etc.) — pass through unchanged
        if re.match(r'^\d+\.\d', structure):
            return structure

        # "APPENDIX A" -> next number after last section
        m = re.match(r'^APPENDIX\s+([A-Z])$', structure)
        if m:
            letter = m.group(1)
            if letter not in self._appendix_map:
                self._max_top_level += 1
                self._appendix_map[letter] = self._max_top_level
            return str(self._appendix_map[letter])

        # "C.1" appendix subsection -> "13.1"
        m = re.match(r'^([A-Z])\.(\d+)$', structure)
        if m:
            letter, sub = m.group(1), m.group(2)
            if letter not in self._appendix_map:
                self._max_top_level += 1
                self._appendix_map[letter] = self._max_top_level
            return f"{self._appendix_map[letter]}.{sub}"

        return structure  # fallback


class RomanNormalizer:
    """
    Handles Roman-numeral-style docs (IEEE papers).
    Letter subsections reset their counter under each new top-level section.

      "I"   -> "1"
      "II"  -> "2"
      "A"   -> "1.1"
      "B"   -> "1.2"
    """
    def __init__(self):
        self._section = 0
        self._sub_counter = 0

    def normalize(self, structure: str) -> str:
        roman_val = roman_to_int(structure)

        if roman_val is not None and (len(structure) > 1 or structure in ('I', 'V', 'X')):
            self._section = roman_val
            self._sub_counter = 0
            return str(roman_val)

        if len(structure) == 1 and structure.isupper():
            roman_val = roman_to_int(structure)
            if roman_val and structure in ('I', 'V', 'X') and self._section == 0:
                self._section = roman_val
                self._sub_counter = 0
                return str(roman_val)
            self._sub_counter += 1
            return f"{self._section}.{self._sub_counter}"

        return structure


# ---------------------------------------------------------------------------
# Heading style definitions
# ---------------------------------------------------------------------------

STYLES = {
    "numeric": {
        "patterns": [
            re.compile(r'^(\d+(?:\.\d+){2,})\s{1,10}(.+)$'),            # 2.1.1  Title
            re.compile(r'^(\d+(?:\.\d+)*)\.\s+(.+)$'),                   # 1. Title
            re.compile(r'^(\d+\.\d+(?:\.\d+)*)\s+(.+)$'),                # 1.1 Title
            re.compile(r'^(APPENDIX\s+[A-Z](?:\.\d+)?)\s*\.\s+(.+)$'),  # APPENDIX A. Title
            re.compile(r'^([A-Z]\.\d+)\s+(.+)$'),                        # C.1 Title
        ],
        "toc_heading_re": re.compile(
            r'^\d+(?:\.\d+)*[\.\s]|^APPENDIX\s+[A-Z]'
        ),
        "references_marker": re.compile(
            r'^\s*(?:\d+\.)?\s*references?\s*$', re.IGNORECASE
        ),
        "references_inline": re.compile(r'REFERENCES', re.IGNORECASE),
        "normalizer": NumericNormalizer,
    },
    "roman": {
        "patterns": [
            re.compile(r'^([IVX]+)\.\s+(.+)$'),
            re.compile(r'^([A-Z])\.\s+(.+)$'),
        ],
        "toc_heading_re": re.compile(r'^[IVX]+\.\s|^[A-Z]\.\s'),
        "references_marker": re.compile(
            r'^[IVX]*\.*\s*references?\s*$', re.IGNORECASE
        ),
        "references_inline": re.compile(r'REFERENCES', re.IGNORECASE),
        "normalizer": RomanNormalizer,
    },
}

UNIVERSAL_SKIP = [
    re.compile(r'^\[\d+\]'),
    re.compile(r'https?://'),
    re.compile(r'^\+1\s'),
    re.compile(r'^\d+\s+Commonwealth'),
    re.compile(r'^\d+\(\d'),
]

TOC_KEYWORDS = re.compile(r'table\s+of\s+contents', re.IGNORECASE)
TOC_DENSITY_THRESHOLD = 0.70
TOC_DENSITY_HIGH = 0.90


# ---------------------------------------------------------------------------
# Auto-detection helpers
# ---------------------------------------------------------------------------

def detect_style(pages: list[tuple[str, str]]) -> str:
    counts = {name: 0 for name in STYLES}
    for _, content in pages:
        for line in content.split('\n'):
            line = line.strip()
            for name, style in STYLES.items():
                if style["toc_heading_re"].match(line):
                    counts[name] += 1
    return max(counts, key=counts.get)


def detect_toc_pages(pages: list[tuple[str, str]], style_name: str) -> set[str]:
    heading_re = STYLES[style_name]["toc_heading_re"]
    toc_pages = set()
    for marker, content in pages:
        lines = [l.strip() for l in content.split('\n') if l.strip()]
        if not lines:
            continue
        density = sum(1 for l in lines if heading_re.match(l)) / len(lines)
        has_keyword = bool(TOC_KEYWORDS.search(content))
        if density >= TOC_DENSITY_HIGH or (density >= TOC_DENSITY_THRESHOLD and has_keyword):
            toc_pages.add(marker)
    return toc_pages


def find_references_start(pages: list[tuple[str, str]], style_name: str) -> str | None:
    style = STYLES[style_name]
    ref_re = style["references_marker"]
    ref_inline = style["references_inline"]
    for marker, content in pages:
        for line in content.split('\n'):
            stripped = line.strip()
            if ref_re.match(stripped):
                return marker
            if ref_inline.search(stripped) and 'REFERENCES' in stripped[-20:].upper():
                return marker
    return None


# ---------------------------------------------------------------------------
# Main extraction
# ---------------------------------------------------------------------------

def transform_text_to_json(text: str) -> list[dict]:
    """
    Transforms document text with <physical_index_N> markers into a structured
    JSON list of section headings with fully numeric structure values.

    Structure is always numeric:
      Numeric docs:  "1", "1.1", "APPENDIX A" -> "11", "C.1" -> "13.1"
      Roman docs:    "I" -> "1", "A" -> "1.1", "B" -> "1.2"

    Args:
        text: Raw document text containing <physical_index_N> markers.

    Returns:
        List of dicts with keys: "structure", "title", "physical_index"
    """

    text = text.replace('\\n', '\n')

    parts = re.split(r'<(physical_index_\d+)>', text)
    pages = [(parts[i], parts[i + 1]) for i in range(1, len(parts) - 1, 2)]

    style_name = detect_style(pages)
    style = STYLES[style_name]
    toc_pages = detect_toc_pages(pages, style_name)
    refs_start = find_references_start(pages, style_name)
    normalizer = style["normalizer"]()

    past_references = False
    results = []
    seen = set()

    for physical_index, page_text in pages:
        if physical_index in toc_pages:
            continue
        if refs_start and physical_index == refs_start:
            past_references = True
        if past_references:
            continue

        for line in page_text.split('\n'):
            line = line.strip()
            if not line:
                continue
            if any(p.search(line) for p in UNIVERSAL_SKIP):
                continue

            for pattern in style["patterns"]:
                m = pattern.match(line)
                if m:
                    raw_structure = m.group(1).strip()
                    title = re.sub(r'\s{2,}', ' ', m.group(2).strip())

                    if title.endswith('.') and len(title) > 80:
                        continue

                    structure = normalizer.normalize(raw_structure)

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
    with open(input_path, "r", encoding="utf-8") as f:
        raw = f.read()
    stripped = raw.strip()
    if stripped.startswith("['") or stripped.startswith('["'):
        items = ast.literal_eval(stripped)
        return items[0] if items else ""
    return raw


def main():
    import sys
    input_path = sys.argv[1] if len(sys.argv) > 1 else "group_texts.txt"
    output_path = sys.argv[2] if len(sys.argv) > 2 else "output.json"

    text = load_text(input_path)
    sections = transform_text_to_json(text)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(sections, f, indent=2, ensure_ascii=False)

    print(f"Extracted {len(sections)} sections -> {output_path}")
    for s in sections:
        print(f"  [{s['structure']}] {s['title']}  ({s['physical_index']})")


if __name__ == "__main__":
    main()
