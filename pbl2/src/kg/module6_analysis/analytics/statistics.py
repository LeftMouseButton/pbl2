# src/kg/module6_analysis/analytics/statistics.py

"""
Statistical validation utilities for Module 6.

This module performs:

1. Degree distribution analysis:
      - Fit power-law and exponential distributions (SciPy)
      - Compare AIC values
      - Determine whether the graph follows a power-law

2. Community quality:
      - Compute modularity of detected communities

3. Centrality correlations:
      - Spearman correlation between:
            • degree vs betweenness
            • degree vs eigenvector

Gracefully degrades if SciPy is unavailable.
"""

from __future__ import annotations
from typing import Dict, Any, List, Tuple

import networkx as nx

try:
    from scipy import stats
except Exception:
    stats = None


def statistical_validation(
    G: nx.Graph,
    node2comm: Dict[str, int],
    centrality: Dict[str, Dict[str, Dict[str, float]]],
) -> Dict[str, Any]:
    """
    Perform statistical validation of key graph properties.

    Parameters
    ----------
    G : nx.Graph
        Input graph.
    node2comm : Dict[str, int]
        Mapping from node to community ID (from community detection).
    centrality : Dict[str, Dict[str, float]]
        Output of compute_centrality():
            {
                "degree": {node: value},
                "betweenness": {...},
                "eigenvector": {...}
            }

    Returns
    -------
    Dict[str, Any]
        Contains:
          - degree_distribution {...}
          - community_quality {...}
          - centrality_correlations {...}
    """
    results: Dict[str, Any] = {}

    # ────────────────────────────────────────────────────────────────────────
    # 1. Degree distribution: power-law vs exponential (AIC comparison)
    # ────────────────────────────────────────────────────────────────────────
    degrees = [d for _, d in G.degree()]

    if len(degrees) > 10 and stats is not None:
        try:
            # Fit distributions
            powerlaw_params = stats.powerlaw.fit(degrees)
            exponential_params = stats.expon.fit(degrees)

            # Log-likelihoods
            powerlaw_ll = stats.powerlaw.logpdf(degrees, *powerlaw_params).sum()
            exponential_ll = stats.expon.logpdf(degrees, *exponential_params).sum()

            # AIC
            powerlaw_aic = 2 * len(powerlaw_params) - 2 * powerlaw_ll
            exponential_aic = 2 * len(exponential_params) - 2 * exponential_ll

            results["degree_distribution"] = {
                "power_law_aic": powerlaw_aic,
                "exponential_aic": exponential_aic,
                "favors_power_law": powerlaw_aic < exponential_aic,
            }
        except Exception:
            results["degree_distribution"] = {
                "note": "Could not fit distributions"
            }
    else:
        results["degree_distribution"] = {
            "note": "SciPy unavailable or insufficient sample size"
        }

    # ────────────────────────────────────────────────────────────────────────
    # 2. Community modularity
    # ────────────────────────────────────────────────────────────────────────
    # Convert mapping → list of lists
    comm_dict: Dict[int, List[str]] = {}
    for node, cid in node2comm.items():
        comm_dict.setdefault(cid, []).append(node)

    communities = [set(nodes) for nodes in comm_dict.values()]

    try:
        modularity = nx.algorithms.community.modularity(G, communities)
        results["community_quality"] = {"modularity": modularity}
    except Exception:
        results["community_quality"] = {"modularity": None}

    # ────────────────────────────────────────────────────────────────────────
    # 3. Centrality correlations (Spearman)
    # ────────────────────────────────────────────────────────────────────────
    # use unweighted centrality for correlations by default
    cent_unw = centrality.get("unweighted", centrality)
    cent_w = centrality.get("weighted", {})

    if stats is not None and len(cent_unw.get("degree", {})) > 5:
        try:
            deg_vals = [cent_unw["degree"][n] for n in G.nodes()]
            btw_vals = [cent_unw["betweenness"][n] for n in G.nodes()]
            eig_vals = [cent_unw["eigenvector"][n] for n in G.nodes()]

            deg_btw_corr = stats.spearmanr(deg_vals, btw_vals)
            deg_eig_corr = stats.spearmanr(deg_vals, eig_vals)

            results["centrality_correlations"] = {
                "degree_betweenness": {
                    "correlation": float(deg_btw_corr.correlation),
                    "p_value": float(deg_btw_corr.pvalue),
                },
                "degree_eigenvector": {
                    "correlation": float(deg_eig_corr.correlation),
                    "p_value": float(deg_eig_corr.pvalue),
                },
            }
        except Exception:
            results["centrality_correlations"] = {
                "note": "Could not compute correlations"
            }
    else:
        results["centrality_correlations"] = {
            "note": "SciPy unavailable or insufficient sample size"
        }

    if stats is not None and cent_w and len(cent_w.get("degree", {})) > 5:
        try:
            deg_vals_w = [cent_w["degree"][n] for n in G.nodes()]
            btw_vals_w = [cent_w["betweenness"][n] for n in G.nodes()]
            eig_vals_w = [cent_w["eigenvector"][n] for n in G.nodes()]

            deg_btw_corr_w = stats.spearmanr(deg_vals_w, btw_vals_w)
            deg_eig_corr_w = stats.spearmanr(deg_vals_w, eig_vals_w)

            results["centrality_correlations_weighted"] = {
                "degree_betweenness": {
                    "correlation": float(deg_btw_corr_w.correlation),
                    "p_value": float(deg_btw_corr_w.pvalue),
                },
                "degree_eigenvector": {
                    "correlation": float(deg_eig_corr_w.correlation),
                    "p_value": float(deg_eig_corr_w.pvalue),
                },
            }
        except Exception:
            results["centrality_correlations_weighted"] = {
                "note": "Could not compute weighted correlations"
            }
    else:
        results["centrality_correlations_weighted"] = {
            "note": "SciPy unavailable or insufficient sample size (weighted)"
        }

    return results
