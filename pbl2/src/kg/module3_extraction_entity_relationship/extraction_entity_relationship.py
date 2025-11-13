#!/usr/bin/env python3
"""
extraction_entity_relationship.py
---------------------------------
Uses the Google AI Studio API (Gemini 2.5 Flash Live) to perform
structured entity and relationship extraction for knowledge-graph population.

Modes:
  • Single disease:
      python extraction_entity_relationship.py --disease breast-cancer
  • Batch (all diseases):
      python extraction_entity_relationship.py --all
  • Re-run cached results:
      python extraction_entity_relationship.py --all --force
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold

# ────────────────────────────────────────────────────────────────────────────── #
# CONFIGURATION
# ────────────────────────────────────────────────────────────────────────────── #

MODEL_NAME = "gemini-2.5-flash-lite"
DATA_PROCESSED_DIR = Path("data/processed")
EXAMPLE_JSON_PATH = Path("src/kg/module3_extraction_entity_relationship/example_entity_extraction.json")
SCHEMA_PATH = Path("schema/schema_keys.json")
OUTPUT_DIR = Path("data/json")
MAX_RETRIES = 3
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ────────────────────────────────────────────────────────────────────────────── #
# LOAD SCHEMA
# ────────────────────────────────────────────────────────────────────────────── #

def load_schema():
    if not SCHEMA_PATH.exists():
        sys.exit(f"❌ Schema file not found at {SCHEMA_PATH}")
    try:
        schema_json = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        sys.exit(f"❌ Failed to parse schema JSON: {e}")
    return json.dumps(schema_json, indent=2, ensure_ascii=False)

# ────────────────────────────────────────────────────────────────────────────── #
# PROMPT CONTENT
# ────────────────────────────────────────────────────────────────────────────── #

SCHEMA_CONTENT = load_schema()

PROMPT_CONTENT = f'''Task: Perform structured entity and relationship extraction for knowledge-graph population.

Input: The uploaded file(s) contain cleaned natural-language text describing a single disease.

Goal: Identify and organize all relevant biomedical entities and their relationships, and output as valid JSON following the schema below.

Schema:
{SCHEMA_CONTENT}

Instructions:
Use concise entity names (no full sentences in lists).
Include diseases as nodes in relationships.
Extract gene associations, treatments, and symptoms even if implicit.
Include at least 5 relationships connecting main entities (has_symptom, has_cause, has_risk_factor, treated_with, associated_gene, etc.).
If multiple files for the same disease are uploaded, merge their information.
Output only JSON, as a single .json file, no commentary.
'''

# ────────────────────────────────────────────────────────────────────────────── #
# HELPERS
# ────────────────────────────────────────────────────────────────────────────── #

def find_related_files(disease: str):
    files = [f for f in DATA_PROCESSED_DIR.glob(f"{disease}_-*.txt")]
    if not files:
        print(f"⚠️ No matching files found for prefix '{disease}_-'. Skipping.")
    return files


def upload_files(files):
    uploaded = []
    for path in files:
        print(f"⬆️ Uploading {path.name} ...")
        uploaded.append(genai.upload_file(path))
    return uploaded


def extract_prefixes():
    """Collect unique prefixes (disease names) from data/processed/."""
    names = {p.name.split("_-_")[0] for p in DATA_PROCESSED_DIR.glob("*.txt") if "_-_" in p.name}
    return sorted(names)


def process_once(disease, model):
    disease_files = find_related_files(disease)
    if not disease_files:
        return False
    if not EXAMPLE_JSON_PATH.exists():
        print(f"❌ Example JSON missing: {EXAMPLE_JSON_PATH}")
        return False

    combined_text = ""
    for path in disease_files:
        print(f"📖 Reading {path.name} ...")
        combined_text += f"\n\n--- SOURCE: {path.name} ---\n"
        combined_text += path.read_text(encoding="utf-8")

    example_json = EXAMPLE_JSON_PATH.read_text(encoding="utf-8")

    full_prompt = f"{PROMPT_CONTENT}\n\nExample JSON:\n{example_json}\n\nInput Text:\n{combined_text}"

    print(f"🧠 Sending extraction request for '{disease}' to {MODEL_NAME}...")
    try:
        response = model.generate_content(
            contents=[full_prompt],
            generation_config={"temperature": 0.2},
        )
    except Exception as e:
        print(f"❌ API error for {disease}: {e}")
        return False

    text = (response.text or "").strip()
    if not text:
        print(f"❌ Empty response for {disease}.")
        return False

    try:
        if text.startswith("```"):
            text = text.lstrip("`").removeprefix("json").strip()
        if text.endswith("```"):
            text = text.rstrip("`").strip()
        data = json.loads(text)
    except json.JSONDecodeError:
        print(f"❌ Invalid JSON for {disease}. Output preview:\n{text[:400]}...\n")
        return False

    out_path = OUTPUT_DIR / f"{disease}.json"
    out_path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"✅ Saved extraction: {out_path}")
    return True


def process_with_retry(disease, model):
    for attempt in range(1, MAX_RETRIES + 1):
        print(f"\n🔄 Attempt {attempt}/{MAX_RETRIES} for {disease}")
        ok = process_once(disease, model)
        if ok:
            return True, attempt
        if attempt < MAX_RETRIES:
            wait = 5 * attempt
            print(f"⏳ Retrying in {wait}s...")
            time.sleep(wait)
    return False, MAX_RETRIES

# ────────────────────────────────────────────────────────────────────────────── #
# MAIN
# ────────────────────────────────────────────────────────────────────────────── #

def main():
    p = argparse.ArgumentParser(description="Entity and relationship extraction using Gemini.")
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--disease", help="Disease prefix (before '_-').")
    g.add_argument("--all", action="store_true", help="Process all diseases.")
    p.add_argument("--force", action="store_true", help="Re-run even if JSON already exists.")
    args = p.parse_args()

    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        sys.exit("❌ Missing GOOGLE_API_KEY environment variable.")
    genai.configure(api_key=api_key)

    model = genai.GenerativeModel(MODEL_NAME)
    results = {}

    targets = [args.disease] if args.disease else extract_prefixes()
    if not targets:
        sys.exit("❌ No disease files found in data/processed/.")

    print(f"🔍 Target diseases: {', '.join(targets)}")

    for disease in targets:
        out_path = OUTPUT_DIR / f"{disease}.json"
        if out_path.exists() and not args.force:
            print(f"⚡ Skipping cached result for {disease} (use --force to re-run).")
            results[disease] = ("SKIPPED", 0)
            continue

        print("\n" + "=" * 80)
        print(f"🧩 Processing disease group: {disease}")
        print("=" * 80)
        success, attempts = process_with_retry(disease, model)
        results[disease] = ("SUCCESS" if success else "FAILED", attempts)

    print("\n" + "#" * 80)
    print("📊 SUMMARY")
    print("#" * 80)
    for disease, (status, attempts) in results.items():
        emoji = {"SUCCESS": "✅", "FAILED": "❌", "SKIPPED": "⚡"}[status]
        attempts_str = f"(Attempts: {attempts})" if attempts else ""
        print(f"{disease:<30} {emoji} {status:<8} {attempts_str}")
    print("#" * 80)

    failed = [d for d, (s, _) in results.items() if s == "FAILED"]
    if failed:
        print(f"⚠️ {len(failed)} disease(s) failed after {MAX_RETRIES} attempts: {', '.join(failed)}")
    else:
        print("🎉 All extractions completed successfully or were cached.")


if __name__ == "__main__":
    main()
