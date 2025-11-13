# src/kg/module6_analysis/report/report_basic.py

"""
Basic Markdown report generation for Module 6.

This module produces the lightweight summary used in the original analyse.py:
  - Graph size stats
  - Node-type counts
  - Connectivity summary
  - Community summary
  - Centrality statistics (top-k only)
  - Traversal demos (BFS, DFS, shortest paths)

The output is a Markdown-formatted string.
"""

from __future__ import annotations

from typing import Dict, Any, List
import networkx as nx


def render_report(
    stats: Dict[str, Any],
    connectivity: Dict[str, Any],
    community: Dict[str, Any],
    centrality: Dict[str, Dict[str, float]],
    traversal_bfs_text: str,
    traversal_dfs_text: str,
    shortest_paths: List[str],
    title: str = "Knowledge Graph Summary",
    top_k: int = 10,
) -> str:
    """
    Produce a basic Markdown report.

    Parameters
    ----------
    stats : Dict[str, Any]
        BuildStats.asdict() → node counts, edge counts, type counts
    connectivity : Dict[str, Any]
        Output of connectivity_summary()
    community : Dict[str, Any]
        Contains modularity or similar metrics (from statistical_validation)
        or node2comm summary from community detection step.
    centrality : Dict[str, Dict[str, float]]
        Output of compute_centrality()
    traversal_bfs_text : str
        Multi-line BFS preview text
    traversal_dfs_text : str
        Multi-line DFS preview text
    shortest_paths : List[str]
        Shortest path examples
    title : str
        Report title
    top_k : int
        How many top nodes to list per centrality metric

    Returns
    -------
    md : str
        Markdown-formatted report
    """

    # ---------------------------------------------------------------------
    # Top-k central nodes (degree)
    # ---------------------------------------------------------------------
    deg_sorted = sorted(
        centrality["degree"].items(),
        key=lambda x: x[1],
        reverse=True
    )[:top_k]

    deg_lines = [
        f"- {node}: {value:.4f}"
        for node, value in deg_sorted
    ]

    # ---------------------------------------------------------------------
    # Basic header
    # ---------------------------------------------------------------------
    md = f"# {title}\n\n"

    # ---------------------------------------------------------------------
    # Graph build statistics
    # ---------------------------------------------------------------------
    md += "## Graph Statistics\n"
    md += f"- **Nodes**: {stats['n_nodes']}\n"
    md += f"- **Edges**: {stats['n_edges']}\n"
    md += f"- **Node Types**:\n"

    for t, c in stats["types"].items():
        md += f"  - {t}: {c}\n"

    md += "\n"

    # ---------------------------------------------------------------------
    # Connectivity
    # ---------------------------------------------------------------------
    md += "## Connectivity\n"
    md += f"- Connected components: **{connectivity['n_components']}**\n"
    md += f"- Giant component nodes: **{connectivity['giant_nodes']}**\n"
    md += f"- Fraction in giant component: **{connectivity['giant_fraction']:.3f}**\n"
    md += f"- Isolates: {connectivity['n_isolates']}\n"

    if connectivity["isolates"]:
        preview_iso = ", ".join(connectivity["isolates"][:10])
        md += f"  - Examples: {preview_iso}\n"

    md += "\n"

    # ---------------------------------------------------------------------
    # Communities
    # ---------------------------------------------------------------------
    md += "## Communities\n"

    if "modularity" in community.get("community_quality", {}):
        mod = community["community_quality"]["modularity"]
        if mod is not None:
            md += f"- Modularity: **{mod:.4f}**\n"
        else:
            md += "- Modularity: unavailable\n"
    else:
        # If communities given as node2comm only (earlier pipeline stage)
        md += "- Community detection completed\n"

    md += "\n"

    # ---------------------------------------------------------------------
    # Centrality overview
    # ---------------------------------------------------------------------
    md += "## Centrality (Top Nodes by Degree)\n"
    md += "\n".join(deg_lines) + "\n\n"

    # ---------------------------------------------------------------------
    # Traversal demos (BFS / DFS)
    # ---------------------------------------------------------------------
    md += "## BFS Traversal (Depth ≤ 3)\n"
    md += "```\n" + traversal_bfs_text.strip() + "\n```\n\n"

    md += "## DFS Traversal (Preorder)\n"
    md += "```\n" + traversal_dfs_text.strip() + "\n```\n\n"

    # ---------------------------------------------------------------------
    # Shortest paths
    # ---------------------------------------------------------------------
    md += "## Shortest Paths Between Seeds\n"
    if shortest_paths:
        md += "```\n"
        for line in shortest_paths:
            md += line + "\n"
        md += "```\n"
    else:
        md += "_No valid seed pairs or no paths found._\n"

    md += "\n"

    return md
