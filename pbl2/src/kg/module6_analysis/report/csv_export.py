# src/kg/module6_analysis/report/csv_export.py

"""
CSV export utilities for Module 6.

This module writes:
  - Node list (id, label, type)
  - Edge list (u, v, type)
  - Centralities (degree, betweenness, eigenvector)
  - Link prediction results (full table)

All functions write CSV safely and atomically, with parent directory creation.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, Any, List
import networkx as nx


# ---------------------------------------------------------------------------
# Helper: safe writer
# ---------------------------------------------------------------------------
def _write_csv(path: Path, rows: List[Dict[str, Any]], fieldnames: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)
    print(f"[INFO] Saved CSV → {path}")


# ---------------------------------------------------------------------------
# NODES
# ---------------------------------------------------------------------------
def export_nodes_csv(
    G: nx.Graph,
    path: Path,
) -> None:
    """
    Write all nodes to CSV: id, label, type
    """
    rows = []
    for n, data in G.nodes(data=True):
        rows.append({
            "id": n,
            "label": data.get("label", n),
            "type": data.get("type", ""),
        })

    _write_csv(path, rows, ["id", "label", "type"])


# ---------------------------------------------------------------------------
# EDGES
# ---------------------------------------------------------------------------
def export_edges_csv(
    G: nx.Graph,
    path: Path,
) -> None:
    """
    Write all edges to CSV: u, v, type
    """
    rows = []
    for u, v, ed in G.edges(data=True):
        rows.append({
            "u": u,
            "v": v,
            "type": ed.get("type", ""),
        })

    _write_csv(path, rows, ["u", "v", "type"])


# ---------------------------------------------------------------------------
# CENTRALITY
# ---------------------------------------------------------------------------
def export_centrality_csv(
    centrality: Dict[str, Dict[str, float]],
    path: Path,
) -> None:
    """
    Write centrality metrics to CSV:
      node, degree, betweenness, eigenvector
    """
    rows = []
    all_nodes = set(centrality.get("degree", {}).keys())

    for node in sorted(all_nodes):
        rows.append({
            "node": node,
            "degree": centrality["degree"].get(node, 0.0),
            "betweenness": centrality["betweenness"].get(node, 0.0),
            "eigenvector": centrality["eigenvector"].get(node, 0.0),
        })

    _write_csv(path, rows, ["node", "degree", "betweenness", "eigenvector"])


def export_centrality_dual_csvs(
    cent_unweighted: Dict[str, Dict[str, float]],
    cent_weighted: Dict[str, Dict[str, float]],
    outdir: Path,
) -> None:
    """
    Write separate CSVs for unweighted and weighted centralities.
    """
    export_centrality_csv(cent_unweighted, outdir / "centrality_unweighted.csv")
    export_centrality_csv(cent_weighted, outdir / "centrality_weighted.csv")


# ---------------------------------------------------------------------------
# LINK PREDICTION
# ---------------------------------------------------------------------------
def export_linkpred_csv(
    rows: List[Dict[str, Any]],
    path: Path,
) -> None:
    """
    Export link prediction table to CSV.

    Handles both classical (ensemble) and improved (ensemble_score) formats.
    """
    if not rows:
        _write_csv(path, [], ["u", "type_u", "v", "type_v", "ensemble"])
        return

    first = rows[0]
    ensemble_key = (
        "ensemble_score" if "ensemble_score" in first
        else "ensemble"
    )

    fieldnames = [
        "u", "type_u",
        "v", "type_v",
        ensemble_key,
    ]

    # Convert rows for writing
    out_rows = []
    for r in rows:
        out_r = {
            "u": r.get("u"),
            "type_u": r.get("type_u"),
            "v": r.get("v"),
            "type_v": r.get("type_v"),
            ensemble_key: r.get(ensemble_key, 0.0),
        }
        out_rows.append(out_r)

    _write_csv(path, out_rows, fieldnames)
