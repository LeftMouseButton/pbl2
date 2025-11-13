# src/kg/module6_analysis/analytics/traversal.py

"""
Traversal utilities for Module 6.

This module implements:
  - BFS preview (depth-limited)
  - DFS preorder preview
  - Shortest-path demos between user-specified seeds

All routines are pure and return human-readable strings exactly matching
the original Module 6 output format.
"""

from __future__ import annotations
from typing import List, Tuple
import networkx as nx

from ..utils.normalize import _norm


def traversal_demo(
    G: nx.Graph,
    seeds: List[str]
) -> Tuple[str, str]:
    """
    Produce BFS and DFS textual previews for each seed.

    Parameters
    ----------
    G : nx.Graph
        Target graph.
    seeds : List[str]
        List of seed labels (raw strings from user).

    Returns
    -------
    (bfs_text, dfs_text) : Tuple[str, str]
        Multi-line strings ready for inclusion in markdown reports.
    """
    bfs_lines: List[str] = []
    dfs_lines: List[str] = []

    for s in seeds:
        key = _norm(s)

        if key not in G:
            bfs_lines.append(f"[seed missing] {s}")
            dfs_lines.append(f"[seed missing] {s}")
            continue

        # ---- BFS (depth ≤ 3) ----
        bfs_nodes = list(
            nx.bfs_tree(G, source=key, depth_limit=3).nodes()
        )
        bfs_labels = [G.nodes[n].get("label", n) for n in bfs_nodes]
        bfs_lines.append(f"Seed: {s}\n  " + " → ".join(bfs_labels[:20]))

        # ---- DFS preorder (first 20) ----
        dfs_nodes = list(
            nx.dfs_preorder_nodes(G, source=key)
        )[:20]
        dfs_labels = [G.nodes[n].get("label", n) for n in dfs_nodes]
        dfs_lines.append(f"Seed: {s}\n  " + " → ".join(dfs_labels))

    return "\n\n".join(bfs_lines), "\n\n".join(dfs_lines)


def shortest_path_demos(
    G: nx.Graph,
    seeds: List[str]
) -> List[str]:
    """
    Compute shortest paths between each pair of provided seed nodes.

    Parameters
    ----------
    G : nx.Graph
    seeds : List[str]
        Raw seed labels (e.g., ["Breast cancer", "Lung cancer"]).

    Returns
    -------
    List[str]
        Each entry is either:
          "A → B → C → D"
        or:
          "No path between X and Y"
    """
    results: List[str] = []

    # Normalize only seeds that exist in the graph
    keys = [
        _norm(s) for s in seeds
        if _norm(s) in G
    ]

    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            u = keys[i]
            v = keys[j]

            try:
                sp = nx.shortest_path(G, u, v)
                labels = [G.nodes[n].get("label", n) for n in sp]
                results.append(" → ".join(labels))
            except nx.NetworkXNoPath:
                label_u = G.nodes[u].get("label", u)
                label_v = G.nodes[v].get("label", v)
                results.append(f"No path between {label_u} and {label_v}")

    return results
