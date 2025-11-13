# src/kg/module6_analysis/loader/records_loader.py

"""
Record loading utilities for Module 6.

This module loads disease records from either:
1. A combined JSON file with a top-level "diseases" list.
2. A directory of individual JSON files (one per disease).

Each JSON object must contain at least "disease_name".
"""

from __future__ import annotations
import json
from pathlib import Path
from typing import List, Dict


def _load_combined(path: Path) -> List[Dict]:
    """
    Load a combined JSON file containing: {"diseases": [ ... ]}.
    """
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, dict) and "diseases" in data:
        return list(data["diseases"])
    raise ValueError("Combined file must contain a top-level 'diseases' list.")


def _load_dir(path: Path) -> List[Dict]:
    """
    Load multiple disease JSON files from a directory.
    """
    records: List[Dict] = []

    for p in sorted(path.glob("*.json")):
        try:
            obj = json.loads(p.read_text(encoding="utf-8"))
            if isinstance(obj, dict) and obj.get("disease_name"):
                records.append(obj)
        except json.JSONDecodeError:
            print(f"[WARN] Skipping invalid JSON: {p}")

    if not records:
        raise ValueError(f"No valid disease JSON files found in directory: {path}")

    return records


def load_records(input_path: str) -> List[Dict]:
    """
    Unified entry point.

    Parameters
    ----------
    input_path : str
        Either a JSON file path or a directory containing JSON files.

    Returns
    -------
    List[dict]
        A list of disease records.
    """
    path = Path(input_path)
    if not path.exists():
        raise FileNotFoundError(path)

    if path.is_dir():
        return _load_dir(path)

    return _load_combined(path)
