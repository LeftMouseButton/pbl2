"""
Auto-repair Validator for LLM Step-1 JSON Outputs
-------------------------------------------------
Checks and fixes structural issues in extracted JSONs before DB ingestion.
No backup files are produced — JSONs are overwritten in place.

Usage:
    python validate_json.py data/json/leukemia.json
    python validate_json.py data/json/
"""

from __future__ import annotations
from pathlib import Path
from typing import List, Dict, Any
import sys, json, datetime, pydantic

# ---------------------------------------------------------------------
# Load Schema from schema/schema_keys.json
# ---------------------------------------------------------------------
SCHEMA_PATH = Path("schema/schema_keys.json")

def load_schema() -> dict:
    """Load and return the JSON schema for validation."""
    if not SCHEMA_PATH.exists():
        sys.exit(f"❌ Schema file not found at {SCHEMA_PATH}")
    try:
        schema_json = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
        if not isinstance(schema_json, dict):
            sys.exit("❌ Schema file must contain a JSON object at the root level.")
        return schema_json
    except json.JSONDecodeError as e:
        sys.exit(f"❌ Failed to parse schema JSON: {e}")

SCHEMA_DEFAULTS = load_schema()

# ---------------------------------------------------------------------
# Dynamic Pydantic Model Creation
# ---------------------------------------------------------------------
def create_model_from_schema(schema: dict):
    """Dynamically create a Pydantic model from the schema dictionary."""
    fields = {}
    for key, val in schema.items():
        if isinstance(val, list):
            fields[key] = (List[str], [])
        elif isinstance(val, dict):
            fields[key] = (Dict[str, Any], {})
        elif isinstance(val, str):
            fields[key] = (str, "")
        else:
            fields[key] = (str, "")
    return pydantic.create_model("ExtractDoc", **fields)

ExtractDoc = create_model_from_schema(SCHEMA_DEFAULTS)

# ---------------------------------------------------------------------
# Validation + Auto-repair
# ---------------------------------------------------------------------
def repair_missing_keys(data: dict) -> dict:
    """Ensure all keys exist with proper types, using schema defaults."""
    for key, default in SCHEMA_DEFAULTS.items():
        if key not in data or data[key] is None:
            data[key] = default
        # Convert to list if default is a list
        if isinstance(default, list) and not isinstance(data[key], list):
            data[key] = [data[key]] if data[key] else []
        # Convert to dict if default is a dict
        if isinstance(default, dict) and not isinstance(data[key], dict):
            data[key] = {}
    return data


def dump_json_compatible(model: pydantic.BaseModel) -> str:
    """Serialize BaseModel safely for both Pydantic v1 and v2."""
    if hasattr(model, "model_dump_json"):
        return model.model_dump_json(indent=2)
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

    # Ensure disease_name is not empty
    if "disease_name" in data and not str(data["disease_name"]).strip():
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
