# src/kg/module6_analysis/analytics/connectivity.py

"""
Connectivity analysis utilities.

This module provides tools for:
  - identifying connected components
  - extracting the giant component
  - summarizing graph connectivity statistics

Purely analytical: no visualization, no CLI, no file I/O.
"""

from __future__ import annotations
import networkx as nx
from typing import Any, Dict, List


def connectivity_summary(G: nx.Graph) -> Dict[str, Any]:
    """
    Compute high-level connectivity statistics for the graph.

    Parameters
    ----------
    G : nx.Graph
        The input knowledge graph.

    Returns
    -------
    Dict[str, Any]
        {
            "n_components" : int,
            "giant_nodes"  : int,
            "giant_fraction" : float,
            "n_isolates"   : int,
            "isolates"     : List[str],
            "giant"        : nx.Graph
        }
    """
    if G.number_of_nodes() == 0:
        return {
            "n_components": 0,
            "giant_nodes": 0,
            "giant_fraction": 0.0,
            "n_isolates": 0,
            "isolates": [],
            "giant": nx.Graph(),
        }

    comps = list(nx.connected_components(G))
    comps_sorted = sorted(comps, key=len, reverse=True)

    giant = G.subgraph(comps_sorted[0]).copy()
    frac = len(giant) / G.number_of_nodes()

    isolates = list(nx.isolates(G))
    isolate_labels = [G.nodes[n].get("label", n) for n in isolates]

    return {
        "n_components": len(comps),
        "giant_nodes": len(giant),
        "giant_fraction": frac,
        "n_isolates": len(isolates),
        "isolates": isolate_labels[:50],   # preview only
        "giant": giant,
    }
