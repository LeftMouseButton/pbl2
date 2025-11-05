#!/usr/bin/env python3
"""
combine_json_files.py
---------------------
Combines all .json files in data/json/ into a single structured JSON file,
suitable for uploading to ChatGPT for knowledge-graph analysis.

Each input JSON must follow the schema:
{
  "disease_name": "",
  "synonyms": [],
  "summary": "",
  "causes": [],
  "risk_factors": [],
  "symptoms": [],
  "diagnosis": [],
  "treatments": [],
  "related_genes": [],
  "subtypes": []
}

Output:
    data/combined/all_diseases.json
"""

import json
from pathlib import Path

INPUT_DIR = Path("data/json")
OUTPUT_DIR = Path("data/combined")
OUTPUT_FILE = OUTPUT_DIR / "all_diseases.json"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def combine_json_files():
    combined = {"diseases": []}

    json_files = sorted(INPUT_DIR.glob("*.json"))
    if not json_files:
        print(f"No JSON files found in {INPUT_DIR}")
        return

    for file in json_files:
        try:
            data = json.loads(file.read_text(encoding="utf-8"))
            # Only include valid structured JSONs with a disease name
            if "disease_name" in data and isinstance(data, dict):
                combined["diseases"].append(data)
                print(f"✅ Added {file.name}")
            else:
                print(f"⚠️ Skipping {file.name} (missing 'disease_name')")
        except json.JSONDecodeError as e:
            print(f"❌ Error reading {file.name}: {e}")

    OUTPUT_FILE.write_text(json.dumps(combined, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n📦 Combined {len(combined['diseases'])} files → {OUTPUT_FILE}")

if __name__ == "__main__":
    combine_json_files()

