# src/kg/module6_analysis/analytics/node_property.py

"""
Node property prediction via neighbor-majority voting.

This module implements the original Module 6 approach:
  - Randomly hide labels of a fraction of nodes
  - Predict missing labels using a majority vote among neighbors
  - Compute accuracy
  - Restore hidden labels afterward

This provides a simple sanity-check for how well node types are
recoverable from graph structure alone.
"""

from __future__ import annotations

import random
import networkx as nx
from collections import Counter
from typing import Dict, Any, List


def neighbor_majority_predict(
    G: nx.Graph,
    holdout_frac: float = 0.1,
    seed: int = 0
) -> Dict[str, Any]:
    """
    Perform neighbor-majority node type prediction.

    This is identical in behavior to the original analyse.py:

      - Sample `holdout_frac` of nodes at random
      - Temporarily hide their "type" attribute
      - Predict each hidden node using the majority type of its neighbors
      - If no neighbors have types, fall back to global majority
      - Compute accuracy and restore all labels

    Parameters
    ----------
    G : nx.Graph
        Target graph with a "type" attribute on each node.
    holdout_frac : float
        Fraction of nodes to hide (default 0.1).
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    Dict[str, Any]
        {
            "accuracy": float,
            "n_holdout": int,
            "preds": {node_key: predicted_type}
        }
    """
    rng = random.Random(seed)

    # Extract node → type mapping (copy)
    labels = nx.get_node_attributes(G, "type").copy()
    all_nodes = list(G.nodes())

    # Determine the holdout set
    holdout_size = max(1, int(len(all_nodes) * holdout_frac))
    holdout = set(rng.sample(all_nodes, holdout_size))

    # Remove labels for holdout nodes
    hidden_labels = {}
    for n in holdout:
        if n in labels:
            hidden_labels[n] = labels[n]
            del labels[n]

    # Global fallback class = most common type among remaining nodes
    global_majority_type = Counter(labels.values()).most_common(1)[0][0]

    # ---- Prediction function -----------------------------------------------
    def predict(node: str) -> str:
        neighbors = list(G.neighbors(node))
        if not neighbors:
            # No neighbors → fallback to global majority
            return global_majority_type

        # Count typed neighbors
        votes = Counter(labels.get(nb) for nb in neighbors if labels.get(nb) is not None)

        if votes:
            # Choose the most common neighbor type
            return votes.most_common(1)[0][0]

        # If neighbors exist but none has type, fallback to global majority
        return global_majority_type

    # ---- Predict labels for holdout nodes ----------------------------------
    preds = {n: predict(n) for n in holdout}

    # ---- Compute accuracy ---------------------------------------------------
    correct = sum(1 for n in holdout if preds[n] == hidden_labels.get(n))
    accuracy = correct / len(holdout)

    # ---- Restore labels -----------------------------------------------------
    labels.update(hidden_labels)

    return {
        "accuracy": accuracy,
        "n_holdout": len(holdout),
        "preds": preds,
    }
