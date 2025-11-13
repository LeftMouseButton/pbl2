# src/kg/module6_analysis/analytics/communities.py

"""
Community detection utilities.

This module includes:
  - Standard community detection (Louvain if available, else greedy modularity)
  - Consensus community detection combining multiple algorithms

All functions are pure: they read a graph and return structures.
"""

from __future__ import annotations
from typing import Dict, List, Set, Tuple

import networkx as nx


def detect_communities(
    G: nx.Graph,
    random_state: int = 42
) -> Tuple[Dict[str, int], List[Set[str]]]:
    """
    Detect communities using a primary algorithm:
      - Louvain (if available in current NetworkX)
      - otherwise greedy modularity

    Parameters
    ----------
    G : nx.Graph
        Input graph.
    random_state : int
        Seed used for Louvain (if supported).

    Returns
    -------
    node2comm : Dict[str, int]
        Mapping of node -> community ID.
    communities : List[Set[str]]
        List of community sets.
    """
    if hasattr(nx.algorithms.community, "louvain_communities"):
        comms = nx.algorithms.community.louvain_communities(G, seed=random_state)
    else:
        comms = list(nx.algorithms.community.greedy_modularity_communities(G))

    node2comm: Dict[str, int] = {}
    for cid, cset in enumerate(comms):
        for node in cset:
            node2comm[node] = cid

    return node2comm, comms


def consensus_community_detection(
    G: nx.Graph,
    random_state: int = 42
) -> Tuple[Dict[str, int], List[Set[str]]]:
    """
    Consensus community detection using multiple clustering algorithms:

      1. Louvain      (if available)
      2. Greedy modularity
      3. Label propagation

    The variant with the highest modularity score is chosen.

    Returns
    -------
    node2comm : Dict[str, int]
        Node-to-community mapping.
    communities : List[Set[str]]
        Final consensus communities.
    """
    all_results: List[List[Set[str]]] = []

    # Louvain
    try:
        if hasattr(nx.algorithms.community, "louvain_communities"):
            lv = list(nx.algorithms.community.louvain_communities(G, seed=random_state))
            all_results.append(lv)
    except Exception:
        pass

    # Greedy modularity
    try:
        gm = list(nx.algorithms.community.greedy_modularity_communities(G))
        all_results.append(gm)
    except Exception:
        pass

    # Label Propagation
    try:
        from networkx.algorithms.community.label_propagation import label_propagation_communities
        lp = list(label_propagation_communities(G))
        all_results.append(lp)
    except Exception:
        pass

    # Fallback
    if not all_results:
        return detect_communities(G, random_state=random_state)

    # Select the result with best modularity
    best_mod = -1.0
    best_comms = all_results[0]

    for comms in all_results:
        try:
            mod = nx.algorithms.community.modularity(G, comms)
            if mod > best_mod:
                best_mod = mod
                best_comms = comms
        except Exception:
            continue

    # Build node → community mapping
    node2comm = {}
    for cid, cset in enumerate(best_comms):
        for n in cset:
            node2comm[n] = cid

    return node2comm, best_comms
