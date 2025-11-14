#!/usr/bin/env python3
"""
combine_json_files.py
---------------------
Combines all .json files in data/json/ into a single structured JSON file,
optionally performing ontology-based normalization before combining.

Features:
    ✅ Auto-detects all ontology files (.obo and .owl) in 'ontologies/' folder
    ✅ Uses fuzzy string matching for local normalization
    ✅ Writes clean all_diseases.json (optionally lowercase)
    ✅ Writes ontology_mapping.json grouped by file and key
    ✅ Prints & saves normalization summary (JSON + console)
    ✅ Exports unmatched terms to unmatched_terms.txt
    ✅ Supports --no-normalize and --no-lowercase flags
    ✅ Displays top 10 unmatched terms with closest ontology suggestions
"""

import json
import argparse
from pathlib import Path
from rapidfuzz import process, fuzz
from statistics import mean
from datetime import datetime
from collections import OrderedDict

try:
    from pronto import Ontology
except ImportError:
    raise SystemExit("❌ Please install dependencies: pip install pronto owlready2 rapidfuzz")

# --- Configuration --------------------------------------------------------
INPUT_DIR = Path("data/json")
OUTPUT_DIR = Path("data/combined")
ONTOLOGY_DIR = Path("ontologies")

OUTPUT_FILE = OUTPUT_DIR / "all_diseases.json"
MAPPING_FILE = OUTPUT_DIR / "ontology_mapping.json"
STATS_FILE = OUTPUT_DIR / "normalization_stats.json"
UNMATCHED_FILE = OUTPUT_DIR / "unmatched_terms.txt"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

FUZZY_CUTOFF = 85
TOP_UNMATCHED_DISPLAY = 10

# --- Ontology Loading -----------------------------------------------------
def detect_ontologies():
    """Automatically detect ontology files in ontologies/ folder."""
    if not ONTOLOGY_DIR.exists():
        print("⚠️  Ontology directory not found:", ONTOLOGY_DIR)
        return []
    files = [f for f in ONTOLOGY_DIR.glob("*") if f.suffix in (".obo", ".owl")]
    if not files:
        print("⚠️  No ontology files found in 'ontologies/' folder.")
    else:
        print(f"🔎 Found {len(files)} ontology file(s):")
        for f in files:
            print("   •", f.name)
    return files

def load_obo(path: Path, term_dict: dict):
    from pronto import Ontology
    print(f"📘 Parsing OBO ontology: {path.name}")
    ont = Ontology(path)
    for term in ont.terms():
        labels = [term.name] if term.name else []
        labels += [s.description for s in term.synonyms]
        for label in labels:
            if label:
                term_dict[label.lower()] = {"id": term.id, "name": term.name}
    return term_dict

def load_owl(path: Path, term_dict: dict):
    from owlready2 import get_ontology
    print(f"📗 Parsing OWL ontology: {path.name}")
    onto = get_ontology(str(path)).load()
    for cls in onto.classes():
        label = cls.label.first() if hasattr(cls, "label") and cls.label else cls.name
        if label:
            term_dict[label.lower()] = {"id": cls.iri, "name": label}
    return term_dict

def load_ontologies():
    """Load all ontology files (.obo or .owl) found in ontologies/, grouped by type."""
    term_dict_disease = {}
    term_dict_gene = {}
    ontology_files = detect_ontologies()

    for path in ontology_files:
        fname = path.name.lower()
        try:
            if path.suffix == ".obo":
                from pronto import Ontology
                ont = Ontology(path)
                for term in ont.terms():
                    labels = [term.name] if term.name else []
                    labels += [s.description for s in term.synonyms]
                    for label in labels:
                        if not label:
                            continue
                        label_lower = label.lower()
                        # classify by ontology filename
                        if any(x in fname for x in ["hgnc", "gene", "ensembl", "ncbi"]):
                            term_dict_gene[label_lower] = {"id": term.id, "name": term.name}
                        else:
                            term_dict_disease[label_lower] = {"id": term.id, "name": term.name}

            elif path.suffix == ".owl":
                from owlready2 import get_ontology
                onto = get_ontology(str(path)).load()
                for cls in onto.classes():
                    label = cls.label.first() if hasattr(cls, "label") and cls.label else cls.name
                    if not label:
                        continue
                    label_lower = label.lower()
                    if any(x in fname for x in ["hgnc", "gene", "ensembl", "ncbi"]):
                        term_dict_gene[label_lower] = {"id": cls.iri, "name": label}
                    else:
                        term_dict_disease[label_lower] = {"id": cls.iri, "name": label}
        except Exception as e:
            print(f"❌ Failed to load {path.name}: {e}")

    has_gene_ont = len(term_dict_gene) > 0
    has_disease_ont = len(term_dict_disease) > 0
    print(
        f"✅ Loaded {len(term_dict_disease)} disease terms "
        f"and {len(term_dict_gene)} gene terms "
        f"from {len(ontology_files)} file(s)"
    )
    if not has_gene_ont:
        print("⚠️  No gene ontology detected — related_genes will not be normalized.\n")
    elif not has_disease_ont:
        print("⚠️  No disease ontology detected — disease fields will not be normalized.\n")

    return {"disease": term_dict_disease, "gene": term_dict_gene}, ontology_files



# --- Normalization --------------------------------------------------------
def normalize_term(term, term_dict):
    """Return ontology match info for a single term."""
    if not term or not term_dict:
        return {"original": term, "normalized": term, "id": None, "score": 0, "matched": False}

    match, score, _ = process.extractOne(term.lower(), term_dict.keys(), scorer=fuzz.token_sort_ratio)
    if score >= FUZZY_CUTOFF:
        entry = term_dict[match]
        return {"original": term, "normalized": entry["name"], "id": entry["id"], "score": score, "matched": True}
    else:
        return {"original": term, "normalized": term, "id": None, "score": score, "matched": False}

def normalize_entity_lists(data, ontology_dicts, stats, mapping_dict, filename, lowercase=True):
    """Normalize entity lists using field-appropriate ontologies with safe fallbacks."""
    mapping_dict[filename] = mapping_dict.get(filename, {})
    term_dict_disease = ontology_dicts.get("disease", {}) or {}
    term_dict_gene = ontology_dicts.get("gene", {}) or {}

    for key in ["causes", "risk_factors", "symptoms", "diagnosis", "treatments", "related_genes", "subtypes"]:
        if key not in data or not isinstance(data[key], list):
            continue

        new_values = []
        key_mappings = []

        # choose ontology dictionary by field type
        if key == "related_genes":
            if len(term_dict_gene) == 0:
                # no gene ontology → skip normalization
                for item in data[key]:
                    if isinstance(item, str):
                        value = item.lower() if lowercase else item
                        new_values.append(value)
                        key_mappings.append({
                            "original": item,
                            "normalized": value,
                            "id": None,
                            "score": 0,
                            "matched": False,
                            "skipped": True
                        })
                data[key] = new_values
                mapping_dict[filename][key] = key_mappings
                continue
            field_dict = term_dict_gene
        else:
            field_dict = term_dict_disease

        for item in data[key]:
            if not isinstance(item, str):
                new_values.append(item)
                continue

            original_item = item

            # ---------------------------------------------------------------
            # Strip typical gene mutation suffixes before normalization
            # e.g., "msh3 gene mutation" -> "msh3"
            #       "pten gene"          -> "pten"
            #       "msh6 mutation"      -> "msh6"
            # ---------------------------------------------------------------
            if key == "related_genes" and isinstance(item, str):
                cleaned = item.lower()
                for suf in [
                    " gene mutation",
                    " mutation",
                    " gene",
                    " variant",
                ]:
                    if cleaned.endswith(suf):
                        cleaned = cleaned[: -len(suf)]
                item = cleaned.strip()

            # Perform ontology match using the cleaned term
            result = normalize_term(item, field_dict)
            result["original"] = original_item  # keep full original text in mapping

            key_mappings.append(result)
            stats["total"] += 1
            stats["scores"].append(result["score"])

            if result["matched"]:
                stats["matched"] += 1
                value = result["normalized"]    # ontology canonical label
            else:
                stats["unmatched"] += 1
                stats["unmatched_terms"].add(original_item)
                value = original_item           # fall back to original string

            # keep lowercase behavior consistent with rest of pipeline
            new_values.append(value.lower() if lowercase else value)


        data[key] = new_values
        mapping_dict[filename][key] = key_mappings

    return data

def create_matched_only_dataset(combined, mapping_file, output_dir, schema_path=Path("schema/schema_keys.json")):
    """
    Create a filtered dataset containing only ontology-matched terms,
    using schema/schema_keys.json to preserve field order and empty placeholders.
    """
    MATCHED_FILE = output_dir / "all_diseases_matched.json"
    matched_combined = {"diseases": []}

    # --- Load ontology mapping ---------------------------------------------------
    if not mapping_file.exists():
        print(f"⚠️  Ontology mapping file not found at: {mapping_file}")
        return

    with open(mapping_file, "r", encoding="utf-8") as f:
        mapping_data = json.load(f)

    # --- Load schema keys from /keys/schema_keys.json ----------------------------
    schema_keys = []
    if Path(schema_path).exists():
        try:
            with open(schema_path, "r", encoding="utf-8") as f:
                schema_json = json.load(f)
                schema_keys = list(schema_json.keys())
                print(f"✅ Loaded schema keys from {schema_path}")
        except Exception as e:
            print(f"⚠️  Failed to read schema file ({schema_path}): {e}")
    else:
        print(f"⚠️  Schema file not found at {schema_path}, using dynamic keys")
        all_possible = set()
        for fd in mapping_data.values():
            all_possible |= set(fd.keys())
        schema_keys = sorted(list(all_possible))

    # --- Build matched-only dataset ---------------------------------------------
    for filename, file_data in mapping_data.items():
        # find disease name from combined
        disease_name = next(
            (d.get("disease_name") for d in combined["diseases"]
             if d.get("_source_file") == filename),
            None
        )
        if not disease_name:
            continue

        new_entry = OrderedDict()
        new_entry["disease_name"] = disease_name
        has_matched = False

        for key in schema_keys:
            if key == "disease_name":
                continue
            matched_terms = []
            for m in file_data.get(key, []):
                if m.get("matched") is True:
                    norm = (m.get("normalized") or "").strip()
                    if norm:
                        matched_terms.append(norm.lower())
            new_entry[key] = matched_terms  # preserve empty
            if matched_terms:
                has_matched = True

        if has_matched:
            matched_combined["diseases"].append(new_entry)

    # --- Write output ------------------------------------------------------------
    if not matched_combined["diseases"]:
        print("⚠️  No matched entries found — check ontology_mapping.json or normalization output.")
    else:
        with open(MATCHED_FILE, "w", encoding="utf-8") as f:
            json.dump(matched_combined, f, indent=2, ensure_ascii=False)
        print(f"📄 Created matched-only dataset → {MATCHED_FILE}")


# --- Combination Logic ----------------------------------------------------
def combine_json_files(no_normalize=False, lowercase=True):
    combined = {"diseases": []}
    mapping_dict = {}
    stats = {"total": 0, "matched": 0, "unmatched": 0, "scores": [], "unmatched_terms": set()}

    ontology_dicts, ontology_files = ({}, []) if no_normalize else load_ontologies()

    json_files = sorted(INPUT_DIR.glob("*.json"))
    if not json_files:
        print(f"No JSON files found in {INPUT_DIR}")
        return

    for file in json_files:
        try:
            data = json.loads(file.read_text(encoding="utf-8"))
            if "disease_name" in data and isinstance(data, dict):
                # tag source filename so matched-only export can join correctly
                data["_source_file"] = file.name
                if lowercase:
                    data["disease_name"] = data["disease_name"].lower()
                if not no_normalize:
                    data = normalize_entity_lists(data, ontology_dicts, stats, mapping_dict, file.name, lowercase=lowercase)
                elif lowercase:
                    for key in ["causes", "risk_factors", "symptoms", "diagnosis", "treatments", "related_genes", "subtypes"]:
                        if key in data and isinstance(data[key], list):
                            data[key] = [v.lower() if isinstance(v, str) else v for v in data[key]]
                combined["diseases"].append(data)
                print(f"✅ Added {file.name}{' (no normalization)' if no_normalize else ''}")
            else:
                print(f"⚠️ Skipping {file.name} (missing 'disease_name')")
        except json.JSONDecodeError as e:
            print(f"❌ Error reading {file.name}: {e}")

    # Write clean combined output
    OUTPUT_FILE.write_text(json.dumps(combined, indent=2, ensure_ascii=False), encoding="utf-8")

    if no_normalize:
        print(f"\n📦 Combined {len(combined['diseases'])} files → {OUTPUT_FILE}")
        print("⚙️  Normalization skipped (--no-normalize)")
        return

    # Write ontology mapping
    MAPPING_FILE.write_text(json.dumps(mapping_dict, indent=2, ensure_ascii=False), encoding="utf-8")

    # --- Export matched-only dataset ---------------------------------
    create_matched_only_dataset(combined, MAPPING_FILE, OUTPUT_DIR)

    # --- Summary ----------------------------------------------------------
    if stats["total"] > 0:
        avg_score = mean(stats["scores"])
        match_rate = (stats["matched"] / stats["total"]) * 100
        summary = {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "ontology_files": [str(x) for x in ontology_files],
            "total_terms_processed": stats["total"],
            "matched_terms": stats["matched"],
            "unmatched_terms": stats["unmatched"],
            "match_rate_percent": round(match_rate, 2),
            "average_match_score": round(avg_score, 2),
            "lowercase_enabled": lowercase,
            "output_file": str(OUTPUT_FILE),
            "ontology_mapping_file": str(MAPPING_FILE),
            "unmatched_terms_file": str(UNMATCHED_FILE),
        }

        print("\n📊 Normalization Summary")
        print("────────────────────────────")
        print(f"Total terms processed : {summary['total_terms_processed']}")
        print(f"Matched to ontology   : {summary['matched_terms']} ({match_rate:.1f}%)")
        print(f"Unmatched terms       : {summary['unmatched_terms']}")
        print(f"Average match score   : {avg_score:.1f}")
        print(f"Ontologies loaded     : {len(ontology_files)}")

        STATS_FILE.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

        if stats["unmatched_terms"]:
            sorted_terms = sorted(stats["unmatched_terms"], key=str.lower)
            UNMATCHED_FILE.write_text("\n".join(sorted_terms), encoding="utf-8")
            print(f"\n🧾 Summary saved to {STATS_FILE}")
            print(f"🧠 Ontology mapping saved to {MAPPING_FILE}")
            print(f"🚫 Unmatched terms saved to {UNMATCHED_FILE}")

            print("\n🔍 Top Unmatched Terms (with closest ontology suggestions):")
            print("────────────────────────────────────────────────────────────")
            sample_terms = sorted_terms[:TOP_UNMATCHED_DISPLAY]
            # use the combined dictionaries for fuzzy suggestion search
            combined_terms = {}
            combined_terms.update(ontology_dicts.get("disease", {}))
            combined_terms.update(ontology_dicts.get("gene", {}))

            for term in sample_terms:
                if not combined_terms:
                    print(f"• {term}  →  (no ontology data available)")
                    continue

                suggestion, score, _ = process.extractOne(term.lower(), combined_terms.keys(), scorer=fuzz.token_sort_ratio)
                best = combined_terms.get(suggestion, {"id": "—", "name": suggestion})
                print(f"• {term}  →  {best['name']}  (score: {score:.1f}, id: {best['id']})")

        else:
            print(f"\n🧾 Summary saved to {STATS_FILE}")
            print(f"🧠 Ontology mapping saved to {MAPPING_FILE}")
            print("✅ All terms matched to ontology!")
    else:
        print("\n📊 No terms found for normalization.")

    print(f"\n📦 Combined {len(combined['diseases'])} files → {OUTPUT_FILE}")

# --- CLI Entry Point ------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Combine and optionally normalize biomedical JSON files using local ontologies."
    )
    parser.add_argument("--no-normalize", action="store_true", help="Skip ontology normalization.")
    parser.add_argument("--no-lowercase", action="store_true", help="Preserve original capitalization.")
    args = parser.parse_args()

    combine_json_files(no_normalize=args.no_normalize, lowercase=not args.no_lowercase)
