"""
Module 1 – Web Crawler (Improved for Scientific Reliability)
------------------------------------------------------------
Collects raw natural-language content (HTML or plain text) for diseases.

Current Targets (configurable):
    - Wikipedia (API)
    - MedlinePlus (HTML via XML search API)

Key improvements:
    - Centralized configuration & logging
    - More robust error handling with retries + backoff
    - Richer provenance metadata
    - Structured capture of Wikipedia sections (where possible)
    - Future-ready hooks for additional sources (NCBI, NCI, etc.)

Outputs
-------
Content:
    data/raw/{slug}_{source}.{ext}

Metadata (one JSON per line):
    data/raw/metadata.jsonl

Each metadata record includes:
    - disease
    - source_type     ("wikipedia" | "medlineplus" | future)
    - url
    - path
    - crawl_timestamp (UTC, ISO 8601)
    - checksum        (SHA256 of raw content)
    - http_status
    - n_bytes
    - source_details  (e.g., sections, ranking scores)
    - license
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import hashlib
import html
import json
import re
import time
import xml.etree.ElementTree as ET

import requests

# =============================================================================
# Configuration
# =============================================================================


@dataclass
class CrawlerConfig:
    # I/O
    raw_dir: Path = Path("data/raw")
    disease_file: Path = Path("disease_names.txt")
    metadata_path: Path = Path("data/raw/metadata.jsonl")

    # HTTP
    user_agent: str = "DSGT-KG-Crawler/2.0 (+https://example.org)"
    timeout: int = 20
    max_retries: int = 3
    backoff_initial: float = 1.0  # seconds
    backoff_factor: float = 2.0

    # Courtesy / throttling
    sleep_between_requests: float = 1.0

    # Feature flags
    enable_wikipedia: bool = True
    enable_medlineplus: bool = True

    # Future extension hook (currently unused but left as option)
    # enable_ncbi_gene: bool = False
    # enable_nci_pdq: bool = False


CONFIG = CrawlerConfig()
CONFIG.raw_dir.mkdir(parents=True, exist_ok=True)

SESSION = requests.Session()
SESSION.headers.update({"User-Agent": CONFIG.user_agent})

# Reliability labels for downstream edge weighting
SOURCE_RELIABILITY = {
    "wikipedia": 0.6,
    "medlineplus": 0.8,
    "pubmed": 1.0,  # reserved for future sources
}

# =============================================================================
# Logging utilities
# =============================================================================


def log(msg: str) -> None:
    """Lightweight, timestamped logger (stdout)."""
    ts = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
    print(f"[{ts}] {msg}")


# =============================================================================
# Common utilities
# =============================================================================


def checksum(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def load_diseases(path: Path) -> List[str]:
    if not path.exists():
        raise FileNotFoundError(f"{path} not found")
    with path.open("r", encoding="utf-8") as f:
        diseases = [line.strip() for line in f if line.strip()]
    if not diseases:
        raise ValueError(f"No disease names found in {path}")
    return diseases


def slugify_name(name: str) -> str:
    # Simple, deterministic slug (do NOT over-normalize here;
    # more sophisticated canonicalization happens later in the pipeline).
    s = name.strip().lower()
    s = re.sub(r"\s+", "_", s)
    s = s.replace("-", "_")
    s = re.sub(r"[^\w_]+", "", s)
    return s or "unknown"


def write_metadata(record: Dict[str, Any], meta_path: Path = CONFIG.metadata_path) -> None:
    with meta_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def save_file(content: str, path: Path) -> None:
    path.write_text(content, encoding="utf-8")
    log(f"Saved: {path}")


def http_get_with_retries(
    url: str,
    *,
    params: Optional[Dict[str, Any]] = None,
    timeout: int = CONFIG.timeout,
    max_retries: int = CONFIG.max_retries,
    backoff_initial: float = CONFIG.backoff_initial,
    backoff_factor: float = CONFIG.backoff_factor,
) -> Tuple[Optional[requests.Response], Optional[int]]:
    """
    GET with basic retry & exponential backoff.
    Returns (response, final_status_code).
    """
    delay = backoff_initial
    last_status = None

    for attempt in range(1, max_retries + 1):
        try:
            r = SESSION.get(url, params=params, timeout=timeout)
            last_status = r.status_code
            # Retry on 5xx; accept 2xx/3xx/4xx as final
            if 500 <= r.status_code < 600:
                log(f"HTTP {r.status_code} for {url} (attempt {attempt}); retrying in {delay:.1f}s")
            else:
                return r, last_status
        except requests.RequestException as e:
            log(f"Request error for {url} (attempt {attempt}): {e}")
        if attempt < max_retries:
            time.sleep(delay)
            delay *= backoff_factor

    log(f"Giving up on {url} after {max_retries} attempts")
    return None, last_status


# =============================================================================
# Wikipedia
# =============================================================================

WIKIPEDIA_API = "https://en.wikipedia.org/w/api.php"


def fetch_wikipedia_page(title: str) -> Tuple[str, Dict[str, Any]]:
    """
    Fetch plain-text extract for a Wikipedia page.
    Also captures pageid and basic metadata for provenance.

    Returns (text, details_dict). If fetch fails, text is "".
    """
    params = {
        "action": "query",
        "format": "json",
        "prop": "extracts",
        "explaintext": True,
        "redirects": 1,
        "titles": title.replace(" ", "_"),
    }

    r, status = http_get_with_retries(WIKIPEDIA_API, params=params)
    if r is None:
        log(f"[WARN] Wikipedia fetch failed for '{title}' (no response)")
        return "", {"http_status": status}

    try:
        r.raise_for_status()
    except requests.HTTPError as e:
        log(f"[WARN] Wikipedia HTTP error for '{title}': {e}")
        return "", {"http_status": r.status_code}

    try:
        data = r.json()
    except ValueError as e:
        log(f"[WARN] Wikipedia JSON parse error for '{title}': {e}")
        return "", {"http_status": r.status_code}

    pages = data.get("query", {}).get("pages", {})
    if not pages:
        log(f"[INFO] Wikipedia: no pages found for '{title}'")
        return "", {"http_status": r.status_code}

    page = next(iter(pages.values()))
    extract = page.get("extract") or ""
    pageid = page.get("pageid")
    normalized_title = page.get("title")

    # Note: we keep the full extract for now; section parsing/weighting
    # is handled downstream during cleaning and extraction.
    details = {
        "http_status": r.status_code,
        "pageid": pageid,
        "normalized_title": normalized_title,
    }
    return extract, details


# =============================================================================
# MedlinePlus
# =============================================================================

MEDLINEPLUS_SEARCH_URL = "https://wsearch.nlm.nih.gov/ws/query"

_TAG_RE = re.compile(r"<[^>]+>")


def _clean_text(x: str) -> str:
    if not x:
        return ""
    x = html.unescape(x)
    x = _TAG_RE.sub("", x)
    return re.sub(r"\s+", " ", x).strip()


def _norm(s: str) -> str:
    s = s.lower()
    s = _TAG_RE.sub("", s)
    s = html.unescape(s)
    s = re.sub(r"[^a-z0-9\s]+", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def _token_join(s: str) -> str:
    return _norm(s).replace(" ", "")


def _contains_phrase(text: str, phrase: str) -> bool:
    return _norm(phrase) in _norm(text)


def _url_key(url: str) -> str:
    m = re.search(r"/([^/]+)\.html?$", url.lower())
    return m.group(1) if m else ""


def medlineplus_search(disease_name: str) -> Optional[Dict[str, Any]]:
    """
    Search MedlinePlus healthTopics and select the best-matching document
    using a transparent scoring heuristic suitable for scientific reporting.
    """
    params = {
        "db": "healthTopics",
        "term": disease_name,
        "retmax": 8,
        "rettype": "brief",
    }

    r, status = http_get_with_retries(MEDLINEPLUS_SEARCH_URL, params=params)
    if r is None:
        log(f"[WARN] MedlinePlus query failed for '{disease_name}' (no response)")
        return None

    try:
        r.raise_for_status()
    except requests.HTTPError as e:
        log(f"[WARN] MedlinePlus HTTP error for '{disease_name}': {e}")
        return None

    try:
        root = ET.fromstring(r.text)
    except Exception as e:
        log(f"[WARN] MedlinePlus XML parse error for '{disease_name}': {e}")
        return None

    docs: List[Dict[str, Any]] = []
    for doc in root.findall(".//document"):
        url = doc.attrib.get("url", "")
        rank = int(doc.attrib.get("rank", "999999"))  # lower is better
        title, alt_titles, full_summary = "", [], ""
        for content in doc.findall("content"):
            name = (content.attrib.get("name") or "").lower()
            text = _clean_text("".join(content.itertext()))
            if name == "title":
                title = text
            elif name == "alttitle":
                alt_titles.append(text)
            elif name == "fullsummary":
                full_summary = text

        if url:
            docs.append(
                {
                    "url": url,
                    "rank": rank,
                    "title": title,
                    "alt_titles": alt_titles,
                    "summary": full_summary,
                }
            )

    if not docs:
        log(f"[INFO] No MedlinePlus results for '{disease_name}'")
        return None

    q_norm = _norm(disease_name)
    q_join = _token_join(disease_name)
    qualifiers = {
        "male",
        "female",
        "men",
        "women",
        "pregnancy",
        "pediatric",
        "child",
        "children",
    }
    query_has_qualifier = any(q in q_norm.split() for q in qualifiers)

    def score(doc: Dict[str, Any]) -> float:
        """Transparent, documentable scoring function."""
        s = 0.0
        urlkey = _url_key(doc["url"])

        # URL-level match
        if urlkey == q_join:
            s += 120.0
        # Penalize misaligned qualifiers if not present in query
        if not query_has_qualifier and urlkey.startswith(
            (
                "male",
                "female",
                "pregnancy",
                "child",
                "pediatric",
                "men",
                "women",
            )
        ):
            s -= 60.0

        # Title similarity
        title_norm = _norm(doc["title"])
        if title_norm == q_norm:
            s += 100.0
        elif _contains_phrase(doc["title"], disease_name):
            s += 35.0

        # Alt titles
        for at in doc["alt_titles"]:
            at_norm = _norm(at)
            if at_norm == q_norm:
                s += 40.0
            elif _contains_phrase(at, disease_name):
                s += 15.0

        # Summary mention
        if _contains_phrase(doc["summary"], disease_name):
            s += 10.0

        # Rank-based bonus (higher for top-ranked results)
        s += max(0.0, 30.0 - min(doc["rank"], 30))
        return s

    for d in docs:
        d["score"] = score(d)

    best = max(docs, key=lambda d: d["score"])
    return best


def fetch_medlineplus_html(url: str) -> Tuple[str, Optional[int]]:
    r, status = http_get_with_retries(url)
    if r is None:
        log(f"[WARN] MedlinePlus fetch failed for {url} (no response)")
        return "", status
    try:
        r.raise_for_status()
    except requests.HTTPError as e:
        log(f"[WARN] MedlinePlus HTTP error for {url}: {e}")
        return "", r.status_code
    return r.text, r.status_code


# =============================================================================
# Crawl orchestration
# =============================================================================


def crawl_wikipedia_for_disease(disease_name: str) -> None:
    if not CONFIG.enable_wikipedia:
        return

    slug = slugify_name(disease_name)
    out_path = CONFIG.raw_dir / f"{slug}_wikipedia.txt"

    if out_path.exists():
        log(f"[SKIP] Wikipedia already exists: {out_path}")
        return

    text, details = fetch_wikipedia_page(disease_name)
    if not text:
        log(f"[WARN] No Wikipedia extract for '{disease_name}'")
        return

    save_file(text, out_path)

    meta = {
        "disease": disease_name,
        "slug": slug,
        "source_type": "wikipedia",
        "source_reliability": SOURCE_RELIABILITY.get("wikipedia", 0.6),
        "url": f"https://en.wikipedia.org/wiki/{disease_name.replace(' ', '_')}",
        "path": str(out_path),
        "crawl_timestamp": datetime.utcnow().isoformat() + "Z",
        "checksum": checksum(text),
        "http_status": details.get("http_status"),
        "n_bytes": len(text.encode("utf-8")),
        "source_details": {
            "pageid": details.get("pageid"),
            "normalized_title": details.get("normalized_title"),
        },
        "license": "CC-BY-SA 4.0",
    }
    write_metadata(meta)

    time.sleep(CONFIG.sleep_between_requests)


def crawl_medlineplus_for_disease(disease_name: str) -> None:
    if not CONFIG.enable_medlineplus:
        return

    slug = slugify_name(disease_name)
    out_path = CONFIG.raw_dir / f"{slug}_medlineplus.html"

    if out_path.exists():
        log(f"[SKIP] MedlinePlus already exists: {out_path}")
        return

    best = medlineplus_search(disease_name)
    if not best or not best.get("url"):
        return

    html_doc, status = fetch_medlineplus_html(best["url"])
    if not html_doc:
        return

    save_file(html_doc, out_path)

    meta = {
        "disease": disease_name,
        "slug": slug,
        "source_type": "medlineplus",
        "source_reliability": SOURCE_RELIABILITY.get("medlineplus", 0.8),
        "url": best["url"],
        "path": str(out_path),
        "crawl_timestamp": datetime.utcnow().isoformat() + "Z",
        "checksum": checksum(html_doc),
        "http_status": status,
        "n_bytes": len(html_doc.encode("utf-8")),
        "source_details": {
            "title": best.get("title", ""),
            "alt_titles": best.get("alt_titles", []),
            "summary_snippet": (best.get("summary") or "")[:400],
            "score": best.get("score"),
            "rank": best.get("rank"),
        },
        "license": "Public domain (U.S. NIH / NLM)",
    }
    write_metadata(meta)

    time.sleep(CONFIG.sleep_between_requests)


def crawl_all() -> None:
    diseases = load_diseases(CONFIG.disease_file)
    log(f"Loaded {len(diseases)} diseases from {CONFIG.disease_file}")

    for disease_name in diseases:
        log(f"=== Processing disease: {disease_name} ===")
        try:
            crawl_wikipedia_for_disease(disease_name)
        except Exception as e:
            log(f"[ERROR] Wikipedia crawl failed for '{disease_name}': {e}")

        try:
            crawl_medlineplus_for_disease(disease_name)
        except Exception as e:
            log(f"[ERROR] MedlinePlus crawl failed for '{disease_name}': {e}")

    log("Crawl complete.")


if __name__ == "__main__":
    crawl_all()
