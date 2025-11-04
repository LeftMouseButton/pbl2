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
import hashlib, json, requests, time

# --- Configuration -------------------------------------------------------------
RAW_DIR = Path("data/raw")
RAW_DIR.mkdir(parents=True, exist_ok=True)
META_PATH = RAW_DIR / "metadata.jsonl"

HEADERS = {"User-Agent": "DSGT-KG-Crawler/1.0 (+https://example.org)"}

CANCER_SOURCES = {
    "breast_cancer": [
        ("wikipedia", "Breast_cancer"),
        ("medlineplus", "https://medlineplus.gov/breastcancer.html"),
    ],
    "lung_cancer": [
        ("wikipedia", "Lung_cancer"),
        ("medlineplus", "https://medlineplus.gov/lungcancer.html"),
    ],
    "colorectal_cancer": [
        ("wikipedia", "Colorectal_cancer"),
        ("medlineplus", "https://medlineplus.gov/colorectalcancer.html"),
    ],
    "leukemia": [
        ("wikipedia", "Leukemia"),
        ("medlineplus", "https://medlineplus.gov/leukemia.html"),
    ],
    "pancreatic_cancer": [
        ("wikipedia", "Pancreatic_cancer"),
        ("medlineplus", "https://medlineplus.gov/pancreaticcancer.html"),
    ],
    "prostate_cancer": [
        ("wikipedia", "Prostate_cancer"),
        ("medlineplus", "https://medlineplus.gov/prostatecancer.html"),
    ],
    "ovarian_cancer": [
        ("wikipedia", "Ovarian_cancer"),
        ("medlineplus", "https://medlineplus.gov/ovariancancer.html"),
    ],
    "stomach_cancer": [
        ("wikipedia", "Stomach_cancer"),
        ("medlineplus", "https://medlineplus.gov/stomachcancer.html"),
    ],
    "esophageal_cancer": [
        ("wikipedia", "Esophageal_cancer"),
        ("medlineplus", "https://medlineplus.gov/esophagealcancer.html"),
    ],
    "lymphoma": [
        ("wikipedia", "Lymphoma"),
        ("medlineplus", "https://medlineplus.gov/lymphoma.html"),
    ],
    "multiple_myeloma": [
        ("wikipedia", "Multiple_myeloma"),
        ("medlineplus", "https://medlineplus.gov/multiplemyeloma.html"),
    ],
    "chronic_myelogenous_leukemia": [
        ("wikipedia", "Chronic_myelogenous_leukemia"),
        ("medlineplus", "https://medlineplus.gov/chronicmyeloidleukemia.html"),
    ],
    "glioblastoma": [
        ("wikipedia", "Glioblastoma"),
        ("medlineplus", "https://medlineplus.gov/glioblastoma.html"),
    ],
    "medulloblastoma": [
        ("wikipedia", "Medulloblastoma"),
        ("medlineplus", "https://medlineplus.gov/medulloblastoma.html"),
    ],
    "thyroid_cancer": [
        ("wikipedia", "Thyroid_cancer"),
        ("medlineplus", "https://medlineplus.gov/thyroidcancer.html"),
    ],
    "adrenocortical_carcinoma": [
        ("wikipedia", "Adrenocortical_carcinoma"),
        ("medlineplus", "https://medlineplus.gov/adrenocorticalcarcinoma.html"),
    ],
    "cervical_cancer": [
        ("wikipedia", "Cervical_cancer"),
        ("medlineplus", "https://medlineplus.gov/cervicalcancer.html"),
    ],
    "endometrial_cancer": [
        ("wikipedia", "Endometrial_cancer"),
        ("medlineplus", "https://medlineplus.gov/endometrialcancer.html"),
    ],
    "neuroblastoma": [
        ("wikipedia", "Neuroblastoma"),
        ("medlineplus", "https://medlineplus.gov/neuroblastoma.html"),
    ],
    "melanoma": [
        ("wikipedia", "Melanoma"),
        ("medlineplus", "https://medlineplus.gov/melanoma.html"),
    ],
}


# --- Utility functions ---------------------------------------------------------
def checksum(s: str) -> str:
    """Compute SHA256 checksum for provenance."""
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def fetch_wikipedia_text(title: str) -> str:
    """Fetch clean plaintext from Wikipedia API."""
    api = "https://en.wikipedia.org/w/api.php"
    params = {
        "action": "query",
        "prop": "extracts",
        "explaintext": True,
        "titles": title,
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


def fetch_medlineplus_html(url: str) -> str:
    """Fetch raw HTML from MedlinePlus."""
    try:
        r = requests.get(url, headers=HEADERS, timeout=15)
        r.raise_for_status()
        return r.text
    except Exception as e:
        print(f"[WARN] MedlinePlus fetch failed for {url}: {e}")
        return ""


def save_file(content: str, path: Path):
    path.write_text(content, encoding="utf-8")
    print(f"[OK] Saved: {path}")


def append_metadata(record: dict):
    with open(META_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


# --- Main crawl logic ----------------------------------------------------------
def crawl_all():
    """
    Crawl all sources defined in CANCER_SOURCES.
    Produces text/html files and a metadata JSONL for provenance.
    """
    print("[INFO] Starting crawl...")
    for disease, sources in CANCER_SOURCES.items():
        for source_type, source_data in sources:
            timestamp = datetime.utcnow().isoformat()
            if source_type == "wikipedia":
                text = fetch_wikipedia_text(source_data)
                ext = "txt"
                url = f"https://en.wikipedia.org/wiki/{source_data}"
            else:
                text = fetch_medlineplus_html(source_data)
                ext = "html"
                url = source_data

            if not text:
                continue

            fname = f"{disease}_{source_type}.{ext}"
            path = RAW_DIR / fname
            save_file(text, path)

            meta = {
                "disease": disease,
                "source_type": source_type,
                "url": url,
                "path": str(path),
                "crawl_timestamp": timestamp,
                "checksum": checksum(text),
                "license": "unknown",
            }
            append_metadata(meta)
            time.sleep(1.5)  # rate limit
    print("[INFO] Crawl complete.")


if __name__ == "__main__":
    crawl_all()

