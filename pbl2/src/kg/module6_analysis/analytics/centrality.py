# src/kg/module6_analysis/analytics/centrality.py

"""
Centrality metric computations for the cancer knowledge graph.

This module computes:
  - Normalized degree centrality
  - Betweenness centrality (full or sampled)
  - Eigenvector centrality (with fallback to NumPy variant)

The output matches the structure used by the original monolithic analyse.py,
ensuring compatibility with reporting, visualization, and validation modules.
"""

from __future__ import annotations
from typing import Dict, List, Any
import random
import networkx as nx


def compute_centrality(
    G: nx.Graph,
    k_sample: int = 0,
    use_weights: bool = False,
) -> Dict[str, Dict[str, float]]:
    """
    Compute degree, betweenness, and eigenvector centrality.

    The function mirrors the behavior of the original Module 6 implementation.

    Parameters
    ----------
    G : nx.Graph
        Input graph.
    k_sample : int, optional
        If > 0: perform approximate betweenness centrality using a
        random sample of k nodes (faster on larger graphs).

    Returns
    -------
    Dict[str, Dict[str, float]]
        {
            "degree":      {node: score},
            "betweenness": {node: score},
            "eigenvector": {node: score}
        }
    """
    weight_kw = {"weight": "weight"} if use_weights else {}

    # ---- Degree centrality (normalized) ------------------------------------
    deg = dict(G.degree(weight=weight_kw.get("weight")))
    n = max(1, G.number_of_nodes() - 1)  # avoid division by zero
    deg_norm = {node: d / n for node, d in deg.items()}

    # ---- Betweenness centrality --------------------------------------------
    if k_sample and k_sample > 0:
        # Sample nodes deterministically via seed=0
        available_nodes = list(G.nodes())
        sample_size = min(k_sample, len(available_nodes))

        rng = random.Random(0)
        sample_nodes = rng.sample(available_nodes, sample_size)

        # NetworkX has two APIs depending on version; try both
        try:
            btw = nx.betweenness_centrality(
                G, normalized=True, nodes=sample_nodes, seed=0, **weight_kw
            )
        except TypeError:
            btw = nx.betweenness_centrality(
                G, k=sample_size, normalized=True, seed=0, **weight_kw
            )
    else:
        # Full betweenness
        btw = nx.betweenness_centrality(G, normalized=True, **weight_kw)

    # ---- Eigenvector centrality --------------------------------------------
    # Use the giant component only for convergence stability
    comps = list(nx.connected_components(G))
    if comps:
        largest = max(comps, key=len)
        giant = G.subgraph(largest)
    else:
        giant = G

    try:
        eig = nx.eigenvector_centrality(giant, max_iter=2000, **weight_kw)
    except nx.PowerIterationFailedConvergence:
        eig = nx.eigenvector_centrality_numpy(giant)

    # Expand eigenvector scores to the full graph (non-giant nodes → 0)
    eig_full = {node: eig.get(node, 0.0) for node in G.nodes()}

    return {
        "degree": deg_norm,
        "betweenness": btw,
        "eigenvector": eig_full,
    }
