# src/kg/module6_analysis/report/report_enhanced.py

"""
Enhanced Markdown report generation for Module 6.

This module produces a detailed, academic-style report including:

- Executive summary
- Graph statistics (size, density, degree stats)
- Connectivity summary (components, giant component, isolates)
- Community analysis (sizes, leaders)
- Centrality analysis (top-k hubs)
- Link prediction summary (top suggested edges)
- Statistical validation (degree distribution, correlations)
- Node property prediction performance
- Traversal demos (BFS/DFS) and shortest paths
- Brief biological interpretation section

The output is a Markdown string; no files are written here.
"""

from __future__ import annotations

from typing import Any, Dict, List, Set, Tuple, Optional

import networkx as nx

try:
    import numpy as np
except Exception:  # pragma: no cover
    np = None


def _topk_dict(
    scores: Dict[str, float],
    G: nx.Graph,
    giant: Optional[nx.Graph],
    k: int,
) -> List[Dict[str, Any]]:
    """
    Convert a centrality dictionary into a list of top-k rows with labels and types.
    """
    if not scores:
        return []

    # Use ranking over the provided scores; giant is for degree context only
    items = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:k]
    rows: List[Dict[str, Any]] = []
    for node, score in items:
        label = G.nodes[node].get("label", node)
        ntype = G.nodes[node].get("type")
        rows.append({
            "node": node,
            "label": label,
            "type": ntype,
            "score": score,
        })
    return rows


def _fmt_table(rows: List[Dict[str, Any]], cols: List[str]) -> str:
    """
    Format a list of dicts as a simple Markdown table.
    """
    if not rows:
        return "(none)"

    header = "| " + " | ".join(cols) + " |\n"
    header += "|" + "---|" * len(cols) + "\n"

    lines: List[str] = []
    for r in rows:
        lines.append("| " + " | ".join(str(r.get(c, "")) for c in cols) + " |")

    return header + "\n".join(lines)


def generate_enhanced_report(
    G: nx.Graph,
    stats: Any,
    connectivity: Dict[str, Any],
    cent: Dict[str, Dict[str, Dict[str, float]]],
    node2comm: Dict[str, int],
    comms: List[Set[str]],
    linkpred_rows: List[Dict[str, Any]],
    traversal_texts: Tuple[str, str],
    shortest_paths: List[str],
    top_k: int,
    npp_result: Optional[Dict[str, Any]] = None,
    validation_results: Optional[Dict[str, Any]] = None,
    title: str = "Enhanced Knowledge Graph Analysis Report",
) -> str:
    """
    Build a detailed Markdown report string.

    Parameters
    ----------
    G : nx.Graph
        The full knowledge graph.
    stats : Any
        Typically BuildStats (with attributes: n_nodes, n_edges, types),
        but any object with equivalent attributes or keys is accepted.
    connectivity : Dict[str, Any]
        Output of connectivity_summary().
    cent : Dict[str, Dict[str, Dict[str, float]]]
        Centralities from compute_centrality(), keyed by "unweighted" and "weighted".
    node2comm : Dict[str, int]
        Node → community mapping.
    comms : List[Set[str]]
        List of communities (sets of node IDs).
    linkpred_rows : List[Dict[str, Any]]
        Link prediction results.
    traversal_texts : Tuple[str, str]
        (bfs_text, dfs_text) from traversal_demo().
    shortest_paths : List[str]
        Output of shortest_path_demos().
    top_k : int
        How many top nodes to show in tables (upper bound, clipped internally).
    npp_result : Optional[Dict[str, Any]]
        Node property prediction result (accuracy, n_holdout, preds).
    validation_results : Optional[Dict[str, Any]]
        Statistical validation output from statistical_validation().
    title : str
        Report title.

    Returns
    -------
    markdown : str
        The full report as Markdown.
    """
    validation_results = validation_results or {}

    # Handle stats as dataclass or dict-like
    if hasattr(stats, "n_nodes"):
        n_nodes = stats.n_nodes
        n_edges = stats.n_edges
        types_counter = stats.types
    else:
        n_nodes = stats["n_nodes"]
        n_edges = stats["n_edges"]
        types_counter = stats["types"]

    by_type = " | ".join(f"{t}: {c}" for t, c in types_counter.most_common())

    # Connectivity
    giant = connectivity.get("giant", G)
    giant_fraction = connectivity.get("giant_fraction", 0.0)

    # Graph-level measures
    avg_deg = (2 * n_edges) / n_nodes if n_nodes else 0.0
    density = nx.density(G)

    cent_unw = cent.get("unweighted", cent)
    cent_w = cent.get("weighted", {})

    # Centrality statistics (if NumPy available)
    if np is not None and cent_unw and cent_unw.get("degree"):
        deg_values = list(cent_unw["degree"].values())
        deg_stats = {
            "mean": float(np.mean(deg_values)),
            "std": float(np.std(deg_values)),
            "max": float(max(deg_values)),
            "min": float(min(deg_values)),
        }

        comm_sizes = [len(c) for c in comms] if comms else []
        if comm_sizes:
            comm_stats = {
                "mean_size": float(np.mean(comm_sizes)),
                "std_size": float(np.std(comm_sizes)),
                "largest": int(max(comm_sizes)),
                "smallest": int(min(comm_sizes)),
            }
        else:
            comm_stats = {}
    else:
        deg_stats = {}
        comm_stats = {}

    bfs_text, dfs_text = traversal_texts

    # Top-k central nodes
    k_eff = min(top_k, 20)
    top_deg = _topk_dict(cent_unw.get("degree", {}), G, giant, k_eff)
    top_btw = _topk_dict(cent_unw.get("betweenness", {}), G, giant, k_eff)
    top_eig = _topk_dict(cent_unw.get("eigenvector", {}), G, giant, k_eff)

    top_deg_w = _topk_dict(cent_w.get("degree", {}), G, giant, k_eff)
    top_btw_w = _topk_dict(cent_w.get("betweenness", {}), G, giant, k_eff)
    top_eig_w = _topk_dict(cent_w.get("eigenvector", {}), G, giant, k_eff)

    # Community leaders table
    leaders: List[Dict[str, Any]] = []
    max_preview = min(12, len(comms))
    for cid, cset in enumerate(comms[:max_preview]):
        # Leader = node with highest degree inside giant if present
        def _deg(node: str) -> int:
            if node in giant:
                return giant.degree(node)
            return G.degree(node)

        if not cset:
            continue

        leader_node = max(cset, key=_deg)
        if leader_node in giant:
            leader_label = giant.nodes[leader_node].get("label", leader_node)
        else:
            leader_label = G.nodes[leader_node].get("label", leader_node)

        leaders.append({
            "community": cid,
            "leader": leader_label,
            "size": len(cset),
        })

    leaders_table = _fmt_table(leaders, ["community", "leader", "size"])
    degree_table = _fmt_table(top_deg, ["label", "type", "score"])
    betw_table = _fmt_table(top_btw, ["label", "type", "score"])
    eig_table = _fmt_table(top_eig, ["label", "type", "score"])

    # Determine link prediction table columns
    lp_table_md = ""
    if linkpred_rows:
        first = linkpred_rows[0]
        # Enhanced or basic ensemble key
        ensemble_key = "ensemble_score" if "ensemble_score" in first else "ensemble"
        cols = ["u", "type_u", "v", "type_v", ensemble_key]

        # Build preview rows (top_k)
        preview_rows: List[Dict[str, Any]] = []
        for r in linkpred_rows[:top_k]:
            preview_rows.append({
                "u": r.get("u"),
                "type_u": r.get("type_u"),
                "v": r.get("v"),
                "type_v": r.get("type_v"),
                ensemble_key: f"{r.get(ensemble_key, 0.0):.3f}",
            })

        lp_table_md = _fmt_table(preview_rows, cols)
    else:
        lp_table_md = "(none)"

    # ──────────────────────────────────────────────────────────────────────
    # Begin assembling Markdown
    # ──────────────────────────────────────────────────────────────────────
    sections: List[str] = []

    # Title + Executive Summary
    sections.append(f"# {title}\n")
    sections.append("## Executive Summary\n")
    sections.append(f"- **{n_nodes}** nodes and **{n_edges}** edges\n")
    sections.append(
        f"- **{connectivity['n_components']}** connected components "
        f"(giant component: {connectivity['giant_nodes']} nodes, "
        f"{giant_fraction:.1%} of all nodes)\n"
    )
    sections.append(f"- **{len(comms)}** communities detected\n")
    sections.append(f"- Average degree: **{avg_deg:.2f}**; density: **{density:.4f}**\n")

    if comm_stats:
        sections.append(
            f"- Community sizes – mean: **{comm_stats.get('mean_size', 0.0):.1f}**, "
            f"largest: **{comm_stats.get('largest', 0)}**, "
            f"smallest: **{comm_stats.get('smallest', 0)}**\n"
        )

    sections.append("\n")

    # Graph Summary
    sections.append("## Graph Summary\n")
    sections.append(f"- Nodes: **{n_nodes}**\n")
    sections.append(f"- Edges: **{n_edges}**\n")
    sections.append(f"- Types: {by_type}\n")
    sections.append(
        f"- Connected components: **{connectivity['n_components']}**; "
        f"giant component size: **{connectivity['giant_nodes']}** "
        f"({giant_fraction:.2%} of nodes)\n"
    )
    sections.append(f"- Average degree: **{avg_deg:.2f}**\n")
    sections.append(f"- Graph density: **{density:.4f}**\n")

    if connectivity.get("isolates"):
        iso_preview = ", ".join(connectivity["isolates"][:10])
        sections.append(f"- Isolates (preview): {iso_preview}\n")

    sections.append("\n")

    # Community Detection
    sections.append("## Community Detection\n")
    sections.append(f"- Detected **{len(comms)}** communities.\n\n")
    sections.append(leaders_table + "\n\n")

    # Centrality
    sections.append("## Centrality (Top Hubs)\n")
    sections.append("### Degree (Unweighted)\n")
    sections.append(degree_table + "\n\n")
    sections.append("### Betweenness (Unweighted)\n")
    sections.append(betw_table + "\n\n")
    sections.append("### Eigenvector (Unweighted)\n")
    sections.append(eig_table + "\n\n")

    if top_deg_w or top_btw_w or top_eig_w:
        sections.append("### Degree (Weighted)\n")
        sections.append(_fmt_table(top_deg_w, ["label", "type", "score"]) + "\n\n")
        sections.append("### Betweenness (Weighted)\n")
        sections.append(_fmt_table(top_btw_w, ["label", "type", "score"]) + "\n\n")
        sections.append("### Eigenvector (Weighted)\n")
        sections.append(_fmt_table(top_eig_w, ["label", "type", "score"]) + "\n\n")

    # Link Prediction
    sections.append("## Link Prediction (Top Suggestions)\n")
    sections.append(lp_table_md + "\n\n")

    # Statistical Validation
    sections.append("## Statistical Validation\n")

    if validation_results:
        dd = validation_results.get("degree_distribution", {})
        if "favors_power_law" in dd:
            if dd["favors_power_law"]:
                sections.append(
                    "- Degree distribution is **more consistent with a power-law** "
                    "than a simple exponential (AIC comparison).\n"
                )
            else:
                sections.append(
                    "- Degree distribution does **not strongly support a power-law** "
                    "over an exponential model (AIC comparison).\n"
                )

        corr_unw = validation_results.get("centrality_correlations", {})
        if "degree_betweenness" in corr_unw:
            db = corr_unw["degree_betweenness"]
            sections.append(
                f"- Spearman correlation (degree vs betweenness): "
                f"**r = {db['correlation']:.3f}**, p = {db['p_value']:.3g}\n"
            )
        if "degree_eigenvector" in corr_unw:
            de = corr_unw["degree_eigenvector"]
            sections.append(
                f"- Spearman correlation (degree vs eigenvector): "
                f"**r = {de['correlation']:.3f}**, p = {de['p_value']:.3g}\n"
            )
        corr_w = validation_results.get("centrality_correlations_weighted", {})
        if "degree_betweenness" in corr_w:
            dbw = corr_w["degree_betweenness"]
            sections.append(
                f"- Weighted Spearman (degree vs betweenness): "
                f"**r = {dbw['correlation']:.3f}**, p = {dbw['p_value']:.3g}\n"
            )
        if "degree_eigenvector" in corr_w:
            dew = corr_w["degree_eigenvector"]
            sections.append(
                f"- Weighted Spearman (degree vs eigenvector): "
                f"**r = {dew['correlation']:.3f}**, p = {dew['p_value']:.3g}\n"
            )
    else:
        sections.append("- Statistical validation was not performed.\n")

    sections.append("\n")

    # Node Property Prediction
    sections.append("## Node Property Prediction\n")
    if npp_result:
        sections.append(
            f"- Neighbor-majority accuracy: "
            f"**{npp_result['accuracy']:.2%}** on "
            f"{npp_result['n_holdout']} hidden nodes.\n"
        )
    else:
        sections.append("- Node property prediction was not computed.\n")

    sections.append("\n")

    # Traversal & Shortest Paths
    sections.append("## Traversal & Shortest Paths\n")

    if bfs_text:
        sections.append("### BFS (depth ≤ 3) from seeds\n")
        sections.append("```text\n" + bfs_text.strip() + "\n```\n\n")

    if dfs_text:
        sections.append("### DFS (preorder) from seeds\n")
        sections.append("```text\n" + dfs_text.strip() + "\n```\n\n")

    if shortest_paths:
        sections.append("### Shortest paths among seeds\n")
        sections.append("```text\n")
        for p in shortest_paths:
            sections.append(p + "\n")
        sections.append("```\n")
    else:
        sections.append("_No valid seed pairs or no paths found._\n")

    sections.append("\n")

    # Biological Interpretation (high-level)
    sections.append("## Biological Interpretation (High-Level)\n")

    if giant_fraction > 0.8:
        sections.append(
            "- The graph is highly connected, suggesting that the underlying "
            "disease–gene–treatment landscape is strongly interlinked. "
            "Many entities participate in shared pathways or therapeutic contexts.\n"
        )
    else:
        sections.append(
            "- The presence of multiple sizable components suggests distinct "
            "subnetworks, which may correspond to disease families, specialized "
            "treatment regimes, or disconnected data sources.\n"
        )

    if len(comms) > 10:
        sections.append(
            "- The rich community structure indicates multiple functional modules "
            "or thematic clusters, potentially corresponding to disease subtypes, "
            "shared genetic pathways, or co-occurring symptom groups.\n"
        )

    # Highlight top link prediction if available
    if linkpred_rows:
        top_lp = linkpred_rows[0]
        ensemble_key = "ensemble_score" if "ensemble_score" in top_lp else "ensemble"
        score_val = float(top_lp.get(ensemble_key, 0.0))
        sections.append(
            f"- The highest-scoring predicted relation is between "
            f"**{top_lp.get('u')}** and **{top_lp.get('v')}** "
            f"({top_lp.get('type_u')}–{top_lp.get('type_v')}), "
            f"with ensemble score ≈ **{score_val:.3f}**. "
            "This edge is a strong candidate for follow-up curation.\n"
        )

    sections.append("\n---\nGenerated by Module 6 (enhanced analysis).\n")

    return "".join(sections)
