"""
Module 1 – Web Crawler
-----------------------
Collects raw natural-language content (HTML or plain text) for diseases.
Targets: Wikipedia (API) and MedlinePlus (HTML).
Saves results under data/raw/ with provenance metadata (for Module 2 cleaning).

Output:
    data/raw/{slug}_{source}.{ext}
    data/raw/metadata.jsonl  (records url, timestamp, checksum)
"""

from pathlib import Path
from datetime import datetime
import hashlib, json, requests, time, re, html, xml.etree.ElementTree as ET

# --- Configuration -------------------------------------------------------------
RAW_DIR = Path("data/raw")
RAW_DIR.mkdir(parents=True, exist_ok=True)
META_PATH = RAW_DIR / "metadata.jsonl"
DISEASE_FILE = Path("disease_names.txt")

HEADERS = {"User-Agent": "DSGT-KG-Crawler/1.3 (+https://example.org)"}


# --- Utility functions ---------------------------------------------------------
def checksum(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def load_diseases() -> list[str]:
    if not DISEASE_FILE.exists():
        raise FileNotFoundError(f"{DISEASE_FILE} not found")
    with open(DISEASE_FILE, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def slugify(name: str) -> str:
    return name.lower().replace(" ", "_").replace("-", "_")


def fetch_wikipedia_text(title: str) -> str:
    api = "https://en.wikipedia.org/w/api.php"
    params = {
        "action": "query",
        "prop": "extracts",
        "explaintext": True,
        "titles": title.replace(" ", "_"),
        "format": "json",
    }
    try:
        r = requests.get(api, params=params, headers=HEADERS, timeout=15)
        r.raise_for_status()
        data = r.json()
        page = next(iter(data["query"]["pages"].values()))
        return page.get("extract", "")
    except Exception as e:
        print(f"[WARN] Wikipedia fetch failed for {title}: {e}")
        return ""


# ---------- Helpers for MedlinePlus selection ----------
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


def _url_key(url: str) -> str:
    m = re.search(r"/([^/]+)\.html?$", url.lower())
    return m.group(1) if m else ""


def _token_join(s: str) -> str:
    return _norm(s).replace(" ", "")


def _contains_phrase(text: str, phrase: str) -> bool:
    return _norm(phrase) in _norm(text)


# ---------- MedlinePlus Web Service ----------
def medlineplus_search(disease_name: str) -> dict | None:
    base_url = "https://wsearch.nlm.nih.gov/ws/query"
    params = {"db": "healthTopics", "term": disease_name, "retmax": 8, "rettype": "brief"}
    try:
        r = requests.get(base_url, params=params, headers=HEADERS, timeout=15)
        r.raise_for_status()
    except Exception as e:
        print(f"[WARN] MedlinePlus query failed for {disease_name}: {e}")
        return None

    try:
        root = ET.fromstring(r.text)
    except Exception as e:
        print(f"[WARN] XML parse error for {disease_name}: {e}")
        return None

    docs = []
    for doc in root.findall(".//document"):
        url = doc.attrib.get("url", "")
        rank = int(doc.attrib.get("rank", "999999"))
        title, alt_titles, full_summary = "", [], ""
        for content in doc.findall("content"):
            name = content.attrib.get("name", "").lower()
            text = _clean_text("".join(content.itertext()))
            if name == "title":
                title = text
            elif name == "alttitle":
                alt_titles.append(text)
            elif name == "fullsummary":
                full_summary = text

        docs.append({
            "url": url,
            "rank": rank,
            "title": title,
            "alt_titles": alt_titles,
            "summary": full_summary,
        })

    if not docs:
        print(f"[INFO] No MedlinePlus results for {disease_name}")
        return None

    # Scoring heuristic
    q_norm = _norm(disease_name)
    q_join = _token_join(disease_name)
    qualifiers = {"male", "female", "men", "women", "pregnancy", "pediatric", "child", "children"}
    query_has_qualifier = any(q in q_norm.split() for q in qualifiers)

    def score(d):
        s = 0
        urlkey = _url_key(d["url"])
        if urlkey == q_join:
            s += 120
        if not query_has_qualifier and urlkey.startswith(
            ("male", "female", "pregnancy", "child", "pediatric", "men", "women")
        ):
            s -= 60
        title_norm = _norm(d["title"])
        if title_norm == q_norm:
            s += 100
        elif _contains_phrase(d["title"], disease_name):
            s += 35
        for at in d["alt_titles"]:
            at_norm = _norm(at)
            if at_norm == q_norm:
                s += 40
            elif _contains_phrase(at, disease_name):
                s += 15
        if _contains_phrase(d["summary"], disease_name):
            s += 10
        s += max(0, 30 - min(d["rank"], 30))
        return s

    best = max(docs, key=score)
    return best


def fetch_medlineplus_html(url: str) -> str:
    try:
        r = requests.get(url, headers=HEADERS, timeout=15)
        r.raise_for_status()
        return r.text
    except Exception as e:
        print(f"[WARN] MedlinePlus fetch failed for {url}: {e}")
        return ""


# --- File I/O ------------------------------------------------------------------
def save_file(content: str, path: Path):
    path.write_text(content, encoding="utf-8")
    print(f"[OK] Saved: {path}")


def append_metadata(record: dict):
    with open(META_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


# --- Main crawl logic ----------------------------------------------------------
def crawl_all():
    diseases = load_diseases()
    print(f"[INFO] Loaded {len(diseases)} diseases from {DISEASE_FILE}")

    for disease_name in diseases:
        slug = slugify(disease_name)
        timestamp = datetime.utcnow().isoformat()

        # Wikipedia
        wiki_path = RAW_DIR / f"{slug}_wikipedia.txt"
        if wiki_path.exists():
            print(f"[SKIP] Wikipedia already exists: {wiki_path}")
        else:
            wiki_text = fetch_wikipedia_text(disease_name)
            if wiki_text:
                save_file(wiki_text, wiki_path)
                append_metadata({
                    "disease": disease_name,
                    "source_type": "wikipedia",
                    "url": f"https://en.wikipedia.org/wiki/{disease_name.replace(' ', '_')}",
                    "path": str(wiki_path),
                    "crawl_timestamp": timestamp,
                    "checksum": checksum(wiki_text),
                    "license": "CC-BY-SA 4.0",
                })
                time.sleep(1.2)

        # MedlinePlus
        med_path = RAW_DIR / f"{slug}_medlineplus.html"
        if med_path.exists():
            print(f"[SKIP] MedlinePlus already exists: {med_path}")
            continue

        mp = medlineplus_search(disease_name)
        if mp and mp.get("url"):
            html_doc = fetch_medlineplus_html(mp["url"])
            if html_doc:
                save_file(html_doc, med_path)
                append_metadata({
                    "disease": disease_name,
                    "source_type": "medlineplus",
                    "url": mp["url"],
                    "title": mp.get("title", ""),
                    "alt_titles": mp.get("alt_titles", []),
                    "summary_snippet": (mp.get("summary") or "")[:400],
                    "path": str(med_path),
                    "crawl_timestamp": timestamp,
                    "checksum": checksum(html_doc),
                    "license": "Public domain (NIH)",
                })
                time.sleep(1.2)

    print("[INFO] Crawl complete.")


if __name__ == "__main__":
    crawl_all()
