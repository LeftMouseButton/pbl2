"""
Module 2 – Cleaning / Preprocessing
-----------------------------------
Converts raw HTML or plain-text files (from Module 1) into normalized,
clean text suitable for LLM-based entity extraction (LLM Step 1).

Input:
    data/raw/*.html or .txt   (from module1_crawler)
Output:
    data/processed/*.txt      (normalized text)
    data/processed/metadata.jsonl (checksum + provenance)
"""

from bs4 import BeautifulSoup
from pathlib import Path
import hashlib, json, html, os, re, time
from slugify import slugify

RAW_DIR = Path("data/raw")
OUT_DIR = Path("data/processed")
OUT_DIR.mkdir(parents=True, exist_ok=True)
META_PATH = OUT_DIR / "metadata.jsonl"

# ---------------------------------------------------------------------------
# Normalization & boilerplate removal utilities
# ---------------------------------------------------------------------------

MOJIBAKE_MAP = {
    "â": "’", "â": "‘", "â": "“", "â": "”",
    "â": "–", "â": "—", "â¢": "•", "â¦": "…", "Â": "",
}

BOILERPLATE_PATTERNS = [
    r"^An official website of the United States government$",
    r"^Here’s how you know$", r"^Official websites use .gov",
    r"^Secure .gov websites use HTTPS",
    r"^A lock \( \) or https:// means",
    r"^Share sensitive information only",
    r"^Basics$", r"^Summary$", r"^Start Here$", r"^Diagnosis and Tests$",
    r"^Prevention and Risk Factors$", r"^Treatments and Therapies$",
    r"^Learn More$", r"^Living With$", r"^Related Issues$",
    r"^Specifics$", r"^Genetics$", r"^See, Play and Learn$",
    r"^Videos and Tutorials$", r"^Research$", r"^Resources$",
    r"^Reference Desk$", r"^Find an Expert$", r"^For You$",
    r"^Children$", r"^Teenagers$", r"^Men$", r"^Patient Handouts$",
]


def checksum(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def fix_mojibake(text: str) -> str:
    for bad, good in MOJIBAKE_MAP.items():
        text = text.replace(bad, good)
    return text


def normalize_text(text: str) -> str:
    text = html.unescape(text)
    text = fix_mojibake(text)
    text = re.sub(r"\u00A0", " ", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def strip_boilerplate_lines(text: str) -> str:
    lines = [ln.strip() for ln in text.split("\n") if ln.strip()]
    cleaned = []
    for ln in lines:
        if any(re.match(p, ln, re.IGNORECASE) for p in BOILERPLATE_PATTERNS):
            continue
        cleaned.append(ln)
    return "\n\n".join(cleaned)


def skip_to_main_content(text: str) -> str:
    lower = text.lower()
    start_idx = None
    match = re.search(r"what is ", lower)
    if match:
        start_idx = match.start()
    elif "overview" in lower:
        start_idx = lower.find("overview")
    if start_idx is not None:
        text = text[start_idx:]
    return text


def remove_after_start_here(text: str) -> str:
    """Remove everything from '## Start Here' onward."""
    pattern = re.compile(r"^## Start Here", flags=re.MULTILINE | re.IGNORECASE)
    match = pattern.search(text)
    if match:
        text = text[:match.start()].strip()
    return text


# ---------------------------------------------------------------------------
# Cleaning logic
# ---------------------------------------------------------------------------

def clean_html_to_text(html_content: str) -> str:
    soup = BeautifulSoup(html_content, "lxml")

    # Remove non-content tags
    for tag in soup(["script", "style", "aside", "nav", "footer", "header", "form", "noscript", "svg"]):
        tag.decompose()

    main = soup.find(attrs={"role": "main"}) or soup.find("main") or soup
    blocks = []

    for tag in main.find_all(["h1", "h2", "h3", "p", "li"]):
        txt = tag.get_text(" ", strip=True)
        if not txt:
            continue
        if tag.name == "h1":
            blocks.append(f"# {txt}\n")
        elif tag.name == "h2":
            blocks.append(f"## {txt}\n")
        elif tag.name == "h3":
            blocks.append(f"### {txt}\n")
        else:
            blocks.append(txt)

    text = "\n\n".join(blocks)
    text = normalize_text(text)
    text = strip_boilerplate_lines(text)
    text = skip_to_main_content(text)
    text = remove_after_start_here(text)
    return text


def clean_plain_text(content: str) -> str:
    text = content.replace("\r", "\n")
    text = normalize_text(text)
    text = strip_boilerplate_lines(text)
    text = skip_to_main_content(text)
    text = remove_after_start_here(text)
    return text


def process_file(raw_path: Path) -> dict:
    content = raw_path.read_text(encoding="utf-8", errors="replace")
    if raw_path.suffix.lower() == ".html":
        cleaned = clean_html_to_text(content)
    else:
        cleaned = clean_plain_text(content)

    # Derive disease prefix and source suffix from the RIGHT side
    stem = raw_path.stem.lower()

    if "_" in stem:
        disease_part, source_part = stem.rsplit("_", 1)
    elif "-" in stem:
        disease_part, source_part = stem.rsplit("-", 1)
    else:
        disease_part, source_part = stem, "unknown"

    disease = slugify(disease_part)
    source = slugify(source_part)

    out_name = f"{disease}_-_{source}.txt"
    out_path = OUT_DIR / out_name
    out_path.write_text(cleaned, encoding="utf-8")

    record = {
        "source_filename": raw_path.name,
        "processed_filename": out_name,
        "checksum": checksum(content),
        "length": len(cleaned),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }

    with open(META_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"[OK] {raw_path.name} → {out_name}")
    return record




def process_all(raw_dir: Path = RAW_DIR):
    """Iterate through all files in raw_dir and clean them."""
    print("[INFO] Starting cleaning process...")
    for path in raw_dir.glob("*"):
        if path.suffix.lower() not in (".html", ".txt"):
            continue
        process_file(path)
    print("[INFO] Cleaning complete.")


if __name__ == "__main__":
    process_all()

