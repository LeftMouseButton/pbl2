"""
Auto-repair Validator for LLM Step-1 JSON Outputs
-------------------------------------------------
Checks and fixes structural issues in extracted JSONs before DB ingestion.
No backup files are produced — JSONs are overwritten in place.

Usage:
    python -m src.kg.tests.validate_extracted_json data/json/leukemia.json
    python -m src.kg.tests.validate_extracted_json data/json/
"""

from __future__ import annotations
from pathlib import Path
from typing import List
import sys, json, datetime, pydantic


# ---------------------------------------------------------------------
# Schema Definition
# ---------------------------------------------------------------------
class ExtractDoc(pydantic.BaseModel):
    disease_name: str
    synonyms: List[str] = []
    summary: str = ""
    causes: List[str] = []
    risk_factors: List[str] = []
    symptoms: List[str] = []
    diagnosis: List[str] = []
    treatments: List[str] = []
    related_genes: List[str] = []
    subtypes: List[str] = []


# ---------------------------------------------------------------------
# Validation + Auto-repair
# ---------------------------------------------------------------------
def repair_missing_keys(data: dict) -> dict:
    """Ensure all keys exist with proper types."""
    defaults = {
        "disease_name": "Unknown Disease",
        "synonyms": [],
        "summary": "",
        "causes": [],
        "risk_factors": [],
        "symptoms": [],
        "diagnosis": [],
        "treatments": [],
        "related_genes": [],
        "subtypes": [],
    }

    for key, default in defaults.items():
        if key not in data or data[key] is None:
            data[key] = default
        # Convert non-lists to single-element lists when needed
        if isinstance(default, list) and not isinstance(data[key], list):
            data[key] = [data[key]] if data[key] else []
    return data


def dump_json_compatible(model: pydantic.BaseModel) -> str:
    """Serialize BaseModel safely for both Pydantic v1 and v2."""
    # Pydantic v2.x
    if hasattr(model, "model_dump_json"):
        return model.model_dump_json(indent=2)
    # Pydantic v1.x
    return model.json(indent=2, ensure_ascii=False)



def validate_file(path: Path):
    print(f"🔍 Validating {path.name}...")
    try:
        text = path.read_text(encoding="utf-8")
        data = json.loads(text)
    except json.JSONDecodeError as e:
        print(f"❌ {path.name}: Invalid JSON — {e}")
        return False

    data = repair_missing_keys(data)

    try:
        doc = ExtractDoc.parse_obj(data)
    except pydantic.ValidationError as e:
        print(f"⚠️ {path.name}: Schema mismatch, attempting repair…")
        data = repair_missing_keys(data)
        try:
            doc = ExtractDoc.parse_obj(data)
        except Exception as e2:
            print(f"❌ {path.name}: Could not repair: {e2}")
            return False

    if not doc.disease_name.strip():
        doc.disease_name = "Unknown Disease"

    path.write_text(dump_json_compatible(doc), encoding="utf-8")
    print(f"✅ {path.name}: Validated and saved (in place)")
    return True


def validate_all(target: Path):
    if target.is_file():
        return validate_file(target)
    elif target.is_dir():
        files = list(target.glob("*.json"))
        if not files:
            print(f"No JSON files found in {target}")
            return False
        ok = [validate_file(f) for f in files]
        print(f"\nSummary: {sum(ok)}/{len(files)} passed (and fixed if needed).")
        return all(ok)
    else:
        print(f"Path not found: {target}")
        return False


# ---------------------------------------------------------------------
# CLI Entry
# ---------------------------------------------------------------------
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python validate_json.py <file_or_dir>")
        sys.exit(1)

    start = datetime.datetime.now()
    path = Path(sys.argv[1])
    ok = validate_all(path)
    print(f"\nCompleted in {(datetime.datetime.now() - start).total_seconds():.2f}s")
    sys.exit(0 if ok else 1)
