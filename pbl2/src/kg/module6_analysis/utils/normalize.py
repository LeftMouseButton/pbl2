# src/kg/module6_analysis/utils/normalize.py

"""
Normalization utilities.

The normalizer _norm() converts entity labels to canonical forms:
  - trim whitespace
  - collapse repeated spaces
  - lowercase everything

This ensures that nodes referring to the same biological concept
are merged correctly in the graph.
"""

def _norm(text: str) -> str:
    """
    Normalize node keys (case-insensitive, whitespace-normalized).

    Parameters
    ----------
    text : str
        Raw entity text from JSON files.

    Returns
    -------
    str
        Canonical node identifier suitable for graph keys.
    """
    if not text:
        return ""
    return " ".join(text.strip().split()).lower()
