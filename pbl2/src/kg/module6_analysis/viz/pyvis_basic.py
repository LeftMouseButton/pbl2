# src/kg/module6_analysis/viz/pyvis_basic.py

"""
Basic PyVis visualization for Module 6.

This module provides:
  - export_pyvis_with_legend(): original Module 6 HTML visualization
  - export_pyvis(): minimal PyVis graph export

Both functions gracefully skip visualization if pyvis is not installed.
They are safe, side-effect free, and do not modify the graph.
"""

from __future__ import annotations
from pathlib import Path
from typing import Dict, Optional

import networkx as nx

from ..utils.constants import COLOR_BY_TYPE

try:
    from pyvis.network import Network
except Exception:  # pragma: no cover
    Network = None


def export_pyvis_with_legend(
    G: nx.Graph,
    path_html: Path,
    node2comm: Optional[Dict[str, int]] = None,
) -> None:
    """
    Create a PyVis HTML visualization with a compact legend of node types.

    Parameters
    ----------
    G : nx.Graph
        Input graph.
    path_html : Path
        Output HTML path.
    node2comm : Dict[str, int], optional
        Optional mapping node → community ID, for display in titles.

    Behavior
    --------
    - Uses original color scheme from Module 6.
    - Inserts a lightweight floating HTML legend.
    - Uses default PyVis physics and layout.
    - Skips silently if PyVis is not installed.
    """
    if Network is None:
        print("[INFO] pyvis not installed; skipping interactive visualization.")
        return

    net = Network(
        height="750px",
        width="100%",
        directed=False,
        notebook=False,
        bgcolor="#111",
        font_color="#EEE",
        heading=""
    )

    net.toggle_physics(True)

    # ---- Add nodes ----
    for n, data in G.nodes(data=True):
        node_type = data.get("type", "")
        label = data.get("label", n)

        title = f"{label}<br>type={node_type}"
        if node2comm is not None:
            title += f"<br>community={node2comm.get(n, -1)}"

        color = COLOR_BY_TYPE.get(node_type, "#cccccc")

        net.add_node(
            n,
            label=label,
            title=title,
            color=color,
        )

    # ---- Add edges ----
    for u, v, ed in G.edges(data=True):
        etype = ed.get("type", "")
        net.add_edge(u, v, title=etype, color="#888888")

    # ---- Generate HTML ----
    html = net.generate_html()

    # ---- Legend HTML block ----
    legend_html = """
    <div id="legend" style="
        position: fixed;
        top: 10px;
        right: 10px;
        background: rgba(0,0,0,0.8);
        border: 2px solid #333;
        border-radius: 8px;
        padding: 15px;
        color: white;
        font-family: Arial, sans-serif;
        font-size: 12px;
        z-index: 999;
        max-width: 200px;
    ">
        <div style="font-weight: bold; margin-bottom: 10px;
                    border-bottom: 1px solid #555; padding-bottom: 5px;">
            Node Types
        </div>
    """

    for node_type, color in COLOR_BY_TYPE.items():
        legend_html += f"""
        <div style="display: flex; align-items: center; margin-bottom: 5px;">
            <div style="
                width: 12px; height: 12px; background: {color};
                border: 1px solid #333; margin-right: 8px; border-radius: 2px;">
            </div>
            <span style="text-transform: capitalize;">
                {node_type.replace("_", " ")}
            </span>
        </div>
        """

    legend_html += "</div>"

    # ---- Insert legend before closing body ----
    html = html.replace("</body>", f"{legend_html}\n</body>")

    path_html.write_text(html, encoding="utf-8")
    print(f"[INFO] Basic PyVis HTML saved → {path_html}")


def export_pyvis(
    G: nx.Graph,
    path_html: Path,
    node2comm: Optional[Dict[str, int]] = None,
) -> None:
    """
    Minimal PyVis graph export without a legend.

    This function is preserved for compatibility with some legacy workflows.

    Parameters
    ----------
    G : nx.Graph
    path_html : Path
    node2comm : Optional[Dict[str, int]]
    """
    if Network is None:
        print("[INFO] pyvis not installed; skipping interactive visualization.")
        return

    net = Network(
        height="750px",
        width="100%",
        directed=False,
        notebook=False,
        bgcolor="#111",
        font_color="#EEE",
    )
    net.toggle_physics(True)

    # Nodes
    for n, data in G.nodes(data=True):
        node_type = data.get("type", "")
        label = data.get("label", n)
        title = f"{label}<br>type={node_type}"
        if node2comm is not None:
            title += f"<br>community={node2comm.get(n, -1)}"

        color = COLOR_BY_TYPE.get(node_type, "#cccccc")
        net.add_node(n, label=label, title=title, color=color)

    # Edges
    for u, v, ed in G.edges(data=True):
        etype = ed.get("type", "")
        net.add_edge(u, v, title=etype, color="#888888")

    net.write_html(str(path_html))
    print(f"[INFO] PyVis HTML saved → {path_html}")
