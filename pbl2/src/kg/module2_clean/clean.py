"""
Module 2 – Cleaning / Preprocessing (Revised)
--------------------------------------------
Converts raw HTML or plain-text files (from Module 1) into normalized,
clean text suitable for LLM-based entity extraction (Module 3).

Strict trimming rules (per specification):

  - Always remove everything BEFORE the line containing "- Patient Handouts"
    (inclusive of that line).
  - Always remove everything AFTER AND INCLUDING the first occurrence of
    "## Start Here".

These rules are applied on the cleaned text representation. For sources that
do not contain these markers, the cleaner behaves conservatively.

Outputs:
    data/processed/*.txt          (normalized text)
    data/processed/metadata.jsonl (checksum + provenance)
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Any

import hashlib
import html as html_lib
import json
import re
import time

from bs4 import BeautifulSoup
from slugify import slugify

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

RAW_DIR = Path("data/raw")
RAW_META_PATH = RAW_DIR / "metadata.jsonl"
OUT_DIR = Path("data/processed")
OUT_DIR.mkdir(parents=True, exist_ok=True)
META_PATH = OUT_DIR / "metadata.jsonl"

# ---------------------------------------------------------------------------
# Normalization & boilerplate removal utilities
# ---------------------------------------------------------------------------

MOJIBAKE_MAP = {
    "â": "’",
    "â": "-",
    "â": "-",
    "â": "‘",
    "â": "“",
    "â": "”",
    "â": "–",
    "â": "—",
    "â¢": "•",
    "â¦": "…",
    "Â": "",
}

BOILERPLATE_PATTERNS = [
    r"^An official website of the United States government$",
    r"^Here’s how you know$",
    r"^Official websites use .gov",
    r"^Secure .gov websites use HTTPS",
    r"^A lock \( \) or https:// means",
    r"^Share sensitive information only",
    r"^See also$",
    r"^External links$",
    r"^References$",
    r"^Navigation menu$",
    r"^Search$",
    r"^Skip to main content$",
]


def log(msg: str) -> None:
    ts = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    print(f"[{ts}] {msg}")


def checksum(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def fix_mojibake(text: str) -> str:
    for bad, good in MOJIBAKE_MAP.items():
        text = text.replace(bad, good)
    return text


def normalize_text(text: str) -> str:
    text = html_lib.unescape(text)
    text = fix_mojibake(text)
    text = text.replace("\u00A0", " ")
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def strip_boilerplate_lines(text: str) -> str:
    """Remove lines matching known boilerplate patterns (conservative)."""
    lines = [ln.rstrip() for ln in text.split("\n")]
    cleaned = []
    for ln in lines:
        stripped = ln.strip()
        if not stripped:
            cleaned.append("")
            continue
        if any(re.match(p, stripped, re.IGNORECASE) for p in BOILERPLATE_PATTERNS):
            continue
        cleaned.append(ln)
    return "\n".join(cleaned)


# ---------------------------------------------------------------------------
# Source-aware helpers
# ---------------------------------------------------------------------------

def is_medlineplus_html(soup: BeautifulSoup) -> bool:
    """Heuristic to detect MedlinePlus pages."""
    if soup.find("meta", attrs={"name": "DCTERMS.source", "content": "MedlinePlus"}):
        return True
    if soup.find("div", attrs={"id": "medlineplus"}):
        return True
    if soup.find("title") and "MedlinePlus" in soup.find("title").get_text():
        return True
    return False


def html_block_to_text(root) -> str:
    """
    Convert a subtree of HTML into structured plain text:
      - h1/h2/h3 -> markdown-style headings
      - p/li    -> paragraphs / simple bullet lines
    """
    blocks = []

    for tag in root.find_all(["h1", "h2", "h3", "p", "li"], recursive=True):
        txt = tag.get_text(" ", strip=True)
        if not txt:
            continue
        name = tag.name.lower()
        if name == "h1":
            blocks.append(f"# {txt}")
        elif name == "h2":
            blocks.append(f"## {txt}")
        elif name == "h3":
            blocks.append(f"### {txt}")
        elif name == "li":
            blocks.append(f"- {txt}")
        else:
            blocks.append(txt)

    text = "\n\n".join(blocks)
    text = normalize_text(text)
    text = strip_boilerplate_lines(text)
    return text.strip()


def clean_html_to_text(html_content: str) -> str:
    """
    Main HTML cleaning entrypoint.

    - Parses HTML with lxml.
    - Removes non-content tags.
    - Uses generic extraction for all sources (including MedlinePlus),
      followed by the global trimming rules.
    """
    soup = BeautifulSoup(html_content, "lxml")

    for tag in soup(
        [
            "script",
            "style",
            "aside",
            "nav",
            "footer",
            "header",
            "form",
            "noscript",
            "svg",
        ]
    ):
        tag.decompose()

    main = soup.find(attrs={"role": "main"}) or soup.find("main") or soup.body or soup
    text = html_block_to_text(main)
    text = apply_global_trimming_rules(text)
    return text


def clean_plain_text(content: str) -> str:
    """
    Cleaner for plain text inputs.

    Applies normalization, boilerplate stripping, then the same
    trimming rules used for HTML-derived text.
    """
    text = normalize_text(content)
    text = strip_boilerplate_lines(text)
    text = apply_global_trimming_rules(text)
    return text.strip()


# ---------------------------------------------------------------------------
# Global trimming rules (per user request)
# ---------------------------------------------------------------------------

def apply_global_trimming_rules(text: str) -> str:
    """
    Apply strict trimming rules in order:

      1) Remove everything BEFORE the line containing "- Patient Handouts"
         (including that line). If not found, no change from this rule.

      2) On the remaining text, remove everything AFTER AND INCLUDING
         the first occurrence of "## Start Here".
         If not found, no change from this rule.

    Final result is normalized again.
    """
    # Rule 1: drop everything up to and including "- Patient Handouts"
    lines = text.split("\n")
    cut_index = None
    for i, line in enumerate(lines):
        if re.search(r"^\s*-\s*patient handouts\s*$", line, flags=re.IGNORECASE):
            cut_index = i
            break

    if cut_index is not None:
        # Keep only content AFTER this marker line
        lines = lines[cut_index + 1 :]

    trimmed = "\n".join(lines)

    # Rule 2: cut after AND including first "## Start Here"
    match = re.search(
        r"^##\s*Start Here\s*$", trimmed, flags=re.IGNORECASE | re.MULTILINE
    )
    if match:
        # Remove from start_here line to end (inclusive)
        start_pos = match.start()
        trimmed = trimmed[:start_pos]

    trimmed = normalize_text(trimmed)
    return trimmed


# ---------------------------------------------------------------------------
# File-level processing
# ---------------------------------------------------------------------------

def derive_disease_and_source(stem: str) -> (str, str):
    """
    Derive disease prefix and source suffix from filename stem.

    Expected pattern from Module 1:
        {slug}_{source}
    """
    s = stem.lower()
    if "_" in s:
        disease_part, source_part = s.rsplit("_", 1)
    elif "-" in s:
        disease_part, source_part = s.rsplit("-", 1)
    else:
        disease_part, source_part = s, "unknown"

    disease = slugify(disease_part) or "unknown"
    source = slugify(source_part) or "unknown"
    return disease, source

def lookup_source_reliability(source_slug: str) -> float:
    """
    Look up source reliability from raw metadata or default heuristics.
    """
    default_map = {
        "wikipedia": 0.6,
        "medlineplus": 0.8,
        "pubmed": 1.0,
    }
    if RAW_META_PATH.exists():
        try:
            for line in RAW_META_PATH.read_text(encoding="utf-8").splitlines():
                rec = json.loads(line)
                if slugify(rec.get("source_type", "")) == source_slug:
                    return float(rec.get("source_reliability", default_map.get(source_slug, 0.5)))
        except Exception:
            pass
    return default_map.get(source_slug, 0.5)


def process_file(raw_path: Path) -> Dict[str, Any]:
    """
    Process a single raw file into cleaned text and write metadata.
    """
    try:
        raw_content = raw_path.read_text(encoding="utf-8", errors="replace")
    except Exception as e:
        log(f"[ERROR] Failed to read {raw_path}: {e}")
        return {
            "source_filename": raw_path.name,
            "error": f"read_failed: {e}",
        }

    if raw_path.suffix.lower() == ".html":
        cleaned = clean_html_to_text(raw_content)
    else:
        cleaned = clean_plain_text(raw_content)

    disease, source = derive_disease_and_source(raw_path.stem)
    out_name = f"{disease}_-_{source}.txt"
    out_path = OUT_DIR / out_name

    try:
        out_path.write_text(cleaned, encoding="utf-8")
    except Exception as e:
        log(f"[ERROR] Failed to write cleaned file for {raw_path.name}: {e}")
        return {
            "source_filename": raw_path.name,
            "processed_filename": out_name,
            "error": f"write_failed: {e}",
        }

    record: Dict[str, Any] = {
        "source_filename": raw_path.name,
        "processed_filename": out_name,
        "disease_slug": disease,
        "source_slug": source,
        "raw_checksum": checksum(raw_content),
        "clean_checksum": checksum(cleaned),
        "clean_length": len(cleaned),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "source_reliability": lookup_source_reliability(source),
    }

    try:
        with META_PATH.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    except Exception as e:
        log(f"[WARN] Failed to append metadata for {raw_path.name}: {e}")

    log(f"[OK] {raw_path.name} → {out_name}")
    return record


def process_all(raw_dir: Path = RAW_DIR) -> None:
    """Iterate through all files in raw_dir and clean them."""
    if not raw_dir.exists():
        log(f"[ERROR] Raw directory not found: {raw_dir}")
        return

    log("[INFO] Starting cleaning process...")
    n_total = 0
    n_processed = 0

    for path in sorted(raw_dir.glob("*")):
        if not path.is_file():
            continue
        if path.suffix.lower() not in (".html", ".txt"):
            continue

        n_total += 1
        rec = process_file(path)
        if "error" not in rec:
            n_processed += 1

    log(f"[INFO] Cleaning complete. Processed {n_processed}/{n_total} files.")


if __name__ == "__main__":
    process_all()
