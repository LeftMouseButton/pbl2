# src/kg/module6_analysis/viz/pyvis_enhanced.py

"""
Enhanced PyVis visualization for Module 6

Features:
  - Python-side layout (spring_layout)
  - Centrality-scaled node sizes
  - Node colors by type
  - Edge colors by relation type
  - Node-type checkboxes (hide/show nodes)
  - Edge-type checkboxes (hide/show edges by type)
  - Search bar (highlight + auto-zoom)
  - Dark/Light mode toggle (including graph canvas background)
  - Movable legend (drag handle only) with boundary constraints
  - Resizable legend (bottom-right grip)
  - Physics enable/disable checkbox in legend
"""

from __future__ import annotations
from typing import Dict, Optional
from pathlib import Path
import networkx as nx

from ..utils.constants import COLOR_BY_TYPE

try:
    from pyvis.network import Network
except Exception:
    Network = None


# ---------------------------------------------------------------------
# Edge color map (including link prediction edges)
# ---------------------------------------------------------------------
EDGE_COLOR_MAP = {
    "treated_with": "#2ecc71",
    "associated_gene": "#9b59b6",
    "has_symptom": "#e74c3c",
    "has_diagnosis": "#3498db",
    "has_risk_factor": "#f1c40f",
    "has_cause": "#e67e22",
    "has_subtype": "#95a5a6",
    "interacts_with": "#8e44ad",
    "contraindicated_with": "#16a085",
    "correlated_with": "#c0392b",
    "targets": "#27ae60",
    "contributes_to": "#d35400",
}


# ---------------------------------------------------------------------
# Node size normalization
# ---------------------------------------------------------------------
def _normalize(values: Dict[str, float], scale_min: int = 10, scale_max: int = 55) -> Dict[str, float]:
    if not values:
        return {}
    lo, hi = min(values.values()), max(values.values())
    rng = (hi - lo) if hi > lo else 1.0
    return {
        k: scale_min + (v - lo) / rng * (scale_max - scale_min)
        for k, v in values.items()
    }


# ---------------------------------------------------------------------
# Main visualization
# ---------------------------------------------------------------------
def enhanced_pyvis_visualization(
    G: nx.Graph,
    path_html: Path,
    centrality: Optional[Dict[str, Dict[str, float]]] = None,
    node2comm: Optional[Dict[str, int]] = None,
    title: str = "Enhanced Knowledge Graph",
) -> None:
    """
    Export an enhanced PyVis HTML visualization.
    """
    if Network is None:
        print("[INFO] pyvis not installed; skipping enhanced visualization.")
        return

    # Base PyVis network
    net = Network(
        height="850px",
        width="100%",
        directed=False,
        bgcolor="#0d1117",
        font_color="#eeeeee",
    )

    # Python-side layout for stable initial positions
    print("[INFO] Computing spring_layout positions for PyVis export …")
    pos = nx.spring_layout(G, seed=42, dim=2)

    # Initial physics config (can be disabled via UI)
    net.set_options(
        """
        {
          "physics": {
            "enabled": true,
            "barnesHut": {
              "gravitationalConstant": -5000,
              "centralGravity": 0.20,
              "springLength": 130,
              "springConstant": 0.02,
              "damping": 0.09,
              "avoidOverlap": 0
            },
            "stabilization": {
              "enabled": true,
              "iterations": 300
            }
          }
        }
        """
    )

    # Node sizes
    if centrality:
        eig = centrality.get("eigenvector", {})
        size_map = _normalize(eig, 10, 55)
    else:
        size_map = {n: 25 for n in G.nodes()}

    # Add nodes with positions
    for n, data in G.nodes(data=True):
        label = data.get("label", n)
        ntype = data.get("type", "")
        color = COLOR_BY_TYPE.get(ntype, "#cccccc")
        size = size_map.get(n, 25)

        x, y = pos.get(n, (0.0, 0.0))
        # Option A: keep strong scaling so physics pulls things into a compact layout
        x *= 1000
        y *= 1000

        title_txt = f"{label}<br>type={ntype}"
        if node2comm:
            title_txt += f"<br>community={node2comm.get(n, -1)}"

        net.add_node(
            n,
            label=label,
            title=title_txt,
            color=color,
            size=size,
            type=ntype,
            x=x,
            y=y,
            physics=True,  # allow initial stabilization; we'll disable later in JS
        )

    # Add edges
    for u, v, ed in G.edges(data=True):
        etype = ed.get("type", "")
        color = EDGE_COLOR_MAP.get(etype, "#888888")
        net.add_edge(u, v, title=etype, color=color, width=2)

    # Generate HTML
    html = net.generate_html()

    # Insert header title
    header_html = f"""
<h1 style="
    text-align:center;
    color:#ddd;
    font-family:Arial, sans-serif;
    margin-top:20px;
">
{title}
</h1>
"""
    html = html.replace("<body>", "<body>\n" + header_html)

    # Theme CSS (dark mode default)
    theme_css = """
<style id="themeStyles">
body {
  background-color: #0d1117;
  color: #ddd;
}
#legend {
  background: rgba(0,0,0,0.85);
  border-color: #444;
  color: #fff;
}
#searchBox {
  background: #222;
  color: #eee;
  border-color: #555;
}
#darkLightToggle {
  background: #222;
  color: #eee;
  border-color: #555;
}
</style>
"""
    html = html.replace("<body>", "<body>\n" + theme_css)

    # Legend HTML with physics toggle
    legend_html = """
<!-- Dark/Light Toggle Button -->
<button id="darkLightToggle" style="
    position: fixed;
    top: 15px;
    right: 20px;
    z-index: 9999;
    padding: 8px 14px;
    border-radius: 6px;
    cursor: pointer;
    background: #222;
    color: #eee;
    border: 1px solid #555;
">🌙 Dark</button>

<div id="legend" style="
    position: fixed;
    bottom: 20px;
    right: 20px;
    padding: 0;
    background: rgba(0,0,0,0.85);
    border: 2px solid #444;
    border-radius: 10px;
    color: #fff;
    font-family: Arial, sans-serif;
    font-size: 12px;
    min-width: 260px;
    z-index: 9999;
    max-height: 450px;
">

    <div id="legendDragHandle" style="
        width: 100%;
        padding: 8px;
        background: rgba(255,255,255,0.05);
        border-bottom: 1px solid #555;
        cursor: grab;
        font-weight: bold;
        user-select: none;
    ">Legend</div>

    <div id="legendContent" style="padding:15px; max-height:380px; overflow:auto;">

        <div style="font-weight:bold; margin-bottom:10px;">Search Node</div>
        <input id="searchBox" placeholder="Type name…" style="
            width:95%;
            padding:6px;
            margin-bottom:12px;
            border-radius:5px;
            border:1px solid #555;
            background:#222;
            color:#eee;
        ">

        <div style="font-weight:bold; margin-bottom:10px;">Physics</div>
        <label style="display:block;margin-bottom:12px;cursor:pointer;">
            <input type="checkbox" id="physicsToggle" checked style="margin-right:6px;">
            <span>Enable physics (layout updates)</span>
        </label>

        <div style="font-weight:bold; margin-bottom:10px;">Node Types (Toggle)</div>
        <div id="nodeTypeControls"></div>

        <hr style="border-color:#666; margin:10px 0;">
        <div style="font-weight:bold; margin-bottom:10px;">Edge Types (Toggle)</div>
        <div id="edgeTypeControls"></div>
    </div>

    <div id="legendResizeHandle" style="
        position:absolute;
        width:14px;
        height:14px;
        bottom:2px;
        right:2px;
        cursor:se-resize;
        background:rgba(255,255,255,0.2);
        border-radius:3px;
    "></div>
</div>
"""

    # JavaScript: filters, search, theme, physics toggle, drag+resize
    js = """
<script>
// Helper: set canvas background to match theme
function applyGraphBackground(isDark) {
    var bg = isDark ? "#0d1117" : "#ffffff";
    var canvas = document.querySelector("canvas");
    if (canvas) {
        canvas.style.background = bg;
    }
}

// ======================================================================
// Node-type Filters (hide/show nodes only)
// ======================================================================
function addLegendControls() {
    if (typeof network === 'undefined') return setTimeout(addLegendControls, 200);

    var container = document.getElementById('nodeTypeControls');
    var allNodes = network.body.nodes;
    var typeColors = {};

    Object.values(allNodes).forEach(function(n) {
        var t = n.options.type || '';
        if (!typeColors[t]) {
            var col = (n.options.color && n.options.color.background)
                ? n.options.color.background
                : '#cccccc';
            typeColors[t] = col;
        }
    });

    Object.keys(typeColors).forEach(function(t) {
        var col = typeColors[t];
        var id = "toggle_" + t.replace(/[^a-zA-Z0-9]/g, '_');

        container.innerHTML += ''
            + '<label style="display:block;margin-bottom:6px;cursor:pointer;">'
            +   '<input type="checkbox" id="' + id + '" checked style="margin-right:6px;">'
            +   '<span style="color:' + col + '">' + t.replace(/_/g,' ') + '</span>'
            + '</label>';

        setTimeout(function() {
            var cb = document.getElementById(id);
            if (!cb) return;
            cb.onchange = function() {
                var visible = cb.checked;
                var nodeDS = network.body.data.nodes;
                var updates = [];

                // Batch updates to avoid repeated heavy work
                Object.values(network.body.nodes).forEach(function(n) {
                    if ((n.options.type || '') === t) {
                        updates.push({ id: n.id, hidden: !visible });
                    }
                });

                if (updates.length > 0) {
                    nodeDS.update(updates);
                }
                // No explicit redraw() here – vis handles it efficiently
            };
        }, 50);
    });
}
setTimeout(addLegendControls, 300);


// ======================================================================
// Edge-type Filters (hide/show edges by type)
// ======================================================================
function addEdgeFilters() {
    if (typeof network === 'undefined') return setTimeout(addEdgeFilters, 200);

    var container = document.getElementById('edgeTypeControls');
    var allEdges = network.body.edges;
    var typeColors = {};

    Object.values(allEdges).forEach(function(e) {
        var t = e.options.title || '';
        if (!typeColors[t]) {
            var col = e.options.color || '#cccccc';
            var colStr = (typeof col === 'string') ? col : (col.color || '#cccccc');
            typeColors[t] = colStr;
        }
    });

    Object.keys(typeColors).forEach(function(etype) {
        var color = typeColors[etype];
        var id = "edge_toggle_" + etype.replace(/[^a-zA-Z0-9]/g, '_');

        container.innerHTML += ''
            + '<label style="display:block;margin-bottom:6px;cursor:pointer;">'
            +   '<input type="checkbox" id="' + id + '" checked style="margin-right:6px;">'
            +   '<span style="color:' + color + '">' + etype.replace(/_/g,' ') + '</span>'
            + '</label>';

        setTimeout(function() {
            var cb = document.getElementById(id);
            if (!cb) return;
            cb.onchange = function() {
                var visible = cb.checked;
                var edgeDS = network.body.data.edges;
                var updates = [];

                Object.values(allEdges).forEach(function(edge) {
                    if ((edge.options.title || '') === etype) {
                        updates.push({ id: edge.id, hidden: !visible });
                    }
                });

                if (updates.length > 0) {
                    edgeDS.update(updates);
                }
                // Again, rely on internal redraw
            };
        }, 50);
    });
}
setTimeout(addEdgeFilters, 350);


// ======================================================================
// Search Bar with highlight + auto-zoom
// ======================================================================
function setupSearchBar() {
    if (typeof network === 'undefined') return setTimeout(setupSearchBar, 200);

    var box = document.getElementById("searchBox");
    var allNodes = network.body.nodes;

    box.addEventListener("input", function() {
        var q = box.value.trim().toLowerCase();

        if (q === "") {
            Object.values(allNodes).forEach(function(n) {
                var orig = n.options._origColor || n.options.color;
                n.setOptions({ color: orig });
            });
            return;
        }

        var found = null;

        Object.values(allNodes).forEach(function(n) {
            var label = (n.options.label || "").toLowerCase();
            var idStr = String(n.id).toLowerCase();

            if (label.includes(q) || idStr.includes(q)) {
                found = n;
                n.setOptions({
                    _origColor: n.options.color,
                    color: { background:"#ffcc00", border:"#ffaa00" }
                });
            } else {
                n.setOptions({
                    _origColor: n.options.color,
                    color: { background:"#444", border:"#222" }
                });
            }
        });

        if (found) {
            network.focus(found.id, {
                scale: 1.2,
                animation: { duration: 500, easingFunction: "easeInOutQuad" }
            });
        }
    });
}
setTimeout(setupSearchBar, 300);


// ======================================================================
// Dark/Light Mode Toggle (including canvas background)
// ======================================================================
function setupDarkLightToggle() {
    var btn = document.getElementById("darkLightToggle");
    var styleTag = document.getElementById("themeStyles");
    var mode = "dark";

    // ensure initial canvas matches dark theme
    applyGraphBackground(true);

    btn.onclick = function() {
        if (mode === "dark") {
            mode = "light";
            btn.textContent = "☀️ Light";
            styleTag.innerHTML = ''
                + 'body { background:#f4f4f4; color:#222; }'
                + '#legend { background:rgba(255,255,255,0.85);color:#222;border-color:#bbb; }'
                + '#searchBox { background:#fff;color:#000;border-color:#aaa; }'
                + '#darkLightToggle { background:#fff;color:#000;border-color:#aaa; }';
            applyGraphBackground(false);
        } else {
            mode = "dark";
            btn.textContent = "🌙 Dark";
            styleTag.innerHTML = ''
                + 'body { background:#0d1117; color:#ddd; }'
                + '#legend { background:rgba(0,0,0,0.85);color:#fff;border-color:#444; }'
                + '#searchBox { background:#222;color:#eee;border-color:#555; }'
                + '#darkLightToggle { background:#222;color:#eee;border-color:#555; }';
            applyGraphBackground(true);
        }
    };
}
setTimeout(setupDarkLightToggle, 300);


// ======================================================================
// Physics Toggle Checkbox
// ======================================================================
function setupPhysicsToggle() {
    if (typeof network === 'undefined') return setTimeout(setupPhysicsToggle, 200);
    var cb = document.getElementById("physicsToggle");
    if (!cb) return;

    cb.onchange = function() {
        var enable = cb.checked;
        network.setOptions({ physics: { enabled: enable } });
    };
}
setTimeout(setupPhysicsToggle, 350);


// ======================================================================
// Disable physics after initial stabilization
// ======================================================================
function disablePhysicsAfterStabilization() {
    if (typeof network === 'undefined') {
        setTimeout(disablePhysicsAfterStabilization, 200);
        return;
    }
    network.once("stabilizationIterationsDone", function () {
        network.setOptions({ physics: { enabled: false } });
        var cb = document.getElementById("physicsToggle");
        if (cb) cb.checked = false;
    });
}
disablePhysicsAfterStabilization();


// ======================================================================
// DRAGGABLE + RESIZABLE LEGEND (with boundaries)
// ======================================================================
function setupLegendDragResize() {
    var legend = document.getElementById("legend");
    var handle = document.getElementById("legendDragHandle");
    var resize = document.getElementById("legendResizeHandle");
    if (!legend || !handle || !resize) return;

    // Dragging
    var dragging = false, offsetX = 0, offsetY = 0;

    handle.addEventListener("mousedown", function(e) {
        dragging = true;
        handle.style.cursor = "grabbing";
        var rect = legend.getBoundingClientRect();
        offsetX = e.clientX - rect.left;
        offsetY = e.clientY - rect.top;
    });

    document.addEventListener("mousemove", function(e) {
        if (!dragging) return;
        var vw = window.innerWidth;
        var vh = window.innerHeight;
        var rect = legend.getBoundingClientRect();

        var left = e.clientX - offsetX;
        var top = e.clientY - offsetY;

        left = Math.max(0, Math.min(left, vw - rect.width));
        top = Math.max(0, Math.min(top, vh - rect.height));

        legend.style.left = left + "px";
        legend.style.top = top + "px";
        legend.style.right = "auto";
        legend.style.bottom = "auto";
        legend.style.position = "fixed";
    });

    document.addEventListener("mouseup", function() {
        dragging = false;  // FIX: JS boolean
        handle.style.cursor = "grab";
    });

    // Resizing
    var resizing = false, startW = 0, startH = 0, startX = 0, startY = 0;

    resize.addEventListener("mousedown", function(e) {
        e.preventDefault();
        resizing = true;
        var rect = legend.getBoundingClientRect();
        startW = rect.width;
        startH = rect.height;
        startX = e.clientX;
        startY = e.clientY;
    });

    document.addEventListener("mousemove", function(e) {
        if (!resizing) return;

        var newW = Math.max(260, startW + (e.clientX - startX));
        var newH = Math.max(220, startH + (e.clientY - startY));

        legend.style.width = newW + "px";
        legend.style.height = newH + "px";

        var inner = document.getElementById("legendContent");
        inner.style.maxHeight = (newH - 60) + "px";
    });

    document.addEventListener("mouseup", function() {
        resizing = false;
    });
}
setTimeout(setupLegendDragResize, 400);
</script>
"""

    html = html.replace("</body>", legend_html + js + "\n</body>")
    path_html.write_text(html, encoding="utf-8")
    print(f"[INFO] Enhanced PyVis HTML saved → {path_html}")
