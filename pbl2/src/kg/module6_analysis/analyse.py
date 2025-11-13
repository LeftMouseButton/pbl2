#!/usr/bin/env python3
"""
Module 6 – Graph Construction & Analysis (analyse.py)
----------------------------------------------------
Builds a heterogeneous cancer knowledge graph from project JSON outputs, then performs
a suite of analyses:

1) Path Finding & Shortest Paths
2) Connectivity Analysis  
3) Community Detection (Louvain ➜ fallback Greedy Modularity ➜ Label Propagation)
4) Centrality (degree, betweenness, eigenvector)
5) Link Prediction (Jaccard, Adamic–Adar, Preferential Attachment; ensemble; expanded edge types)
6) Node Property Prediction (neighbor-majority; holdout evaluation)
7) Statistical Validation (distribution fitting, correlation tests)
8) Traversal / Search (BFS/DFS demos from seeds)

Inputs
------
• A combined JSON file produced by Module 5, e.g. data/combined/all_diseases.json
  (schema: {"diseases": [ {disease_name, symptoms, treatments, related_genes, ...}, ... ]})
• OR: a directory of per-disease JSON files under data/json/ (same schema per file)

Outputs (default under --outdir, e.g., data/analysis/)
------------------------------------------------------
• report_module6.md                 # Enhanced analysis report with statistical validation
• graph.graphml                     # Full graph (labels & types as node/edge attributes)
• graph.html                        # Interactive PyVis visualization
• centrality.csv                    # Degree, betweenness, eigenvector (top-k and full)
• communities.csv                   # Node → community mapping
• link_predictions.csv              # Top predicted links (ensemble-ranked)

Enhanced Features
-----------------
• Statistical validation of key findings (distribution fitting, correlations)
• Enhanced link prediction covering all plausible edge types
• Consensus community detection using multiple algorithms
• Memory monitoring and optimization
• Enhanced visualization with centrality-based node sizing

Usage
-----
  python -m src.kg.module6_analysis.analyse \
    --input data/combined/all_diseases.json \
    --outdir data/analysis \
    --viz-html graph.html  \
    --graphml graph.graphml \
    --topk 30 \
    --seed "Breast cancer" \
    --seed "Lung cancer"  \
    --betweenness-sample 200  \
    --random-state 42 \
    --validation \
    --enhanced-viz \
    --memory-monitor

If --input points to a directory, the program will load all *.json within it.

Enhanced Usage
--------------
  python -m src.kg.module6_analysis.analyse \
    --input data/combined/all_diseases.json \
    --outdir data/analysis \
    --validation \
    --enhanced-viz \
    --memory-monitor \
    --max-nodes 5000 \
    --topk 30

For very large graphs, use --max-nodes to limit memory usage.
Program includes statistical validation when scipy is available.

Notes
-----
• Safe on mid-sized graphs (1–10k nodes). For very large graphs, use --betweenness-sample.
• Program is deterministic except for community detection (random seeding handled).
• Enhanced features require scipy, numpy, and psutil packages.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from collections import defaultdict
import itertools
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Set, Any, Tuple
from typing import Optional
import time
import gc
import random

import networkx as nx

try:
    import pandas as pd  # type: ignore
except Exception:  # pragma: no cover
    pd = None

try:
    from pyvis.network import Network  # type: ignore
except Exception:  # pragma: no cover
    Network = None

try:
    import numpy as np
except Exception:
    np = None

try:
    import psutil
except Exception:  # pragma: no cover
    psutil = None

try:
    from scipy import stats
except Exception:
    stats = None

# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Module 6 – Build and analyse the cancer KG.")
    p.add_argument("--input", required=True,
                   help="Path to combined JSON (top-level 'diseases') OR a directory of per-disease JSONs")
    p.add_argument("--outdir", default="data/analysis", help="Output directory (default: data/analysis)")
    p.add_argument("--viz-html", default="graph.html", help="Filename for PyVis HTML under outdir")
    p.add_argument("--graphml", default="graph.graphml", help="Filename for GraphML under outdir")
    p.add_argument("--topk", type=int, default=25, help="Top-k rows to include in CSVs & report tables")
    p.add_argument("--betweenness-sample", type=int, default=0,
                   help="If >0, approximate betweenness using K-node sample for speed")
    p.add_argument("--seed", action="append", default=[], help="Add a seed node for traversal & shortest-path demos (repeatable)")
    p.add_argument("--random-state", type=int, default=42, help="Random seed for community detection")
    p.add_argument("--max-nodes", type=int, default=0, 
                   help="Limit graph to N nodes (0 = no limit) for memory management")
    p.add_argument("--memory-monitor", action="store_true", 
                   help="Enable memory usage monitoring during analysis")
    p.add_argument("--validation", action="store_true",
                   help="Perform statistical validation of results (requires scipy)")
    p.add_argument("--enhanced-viz", action="store_true",
                   help="Use enhanced visualization with centrality-based sizing")
    return p.parse_args()

# ──────────────────────────────────────────────────────────────────────────────
# Enhanced Link Prediction
# ──────────────────────────────────────────────────────────────────────────────

def get_plausible_edge_types():
    """Define all plausible edge types between node categories."""
    return {
        ("disease", "gene"): "associated_gene",
        ("disease", "treatment"): "treated_with", 
        ("disease", "symptom"): "has_symptom",
        ("disease", "diagnosis"): "has_diagnosis",
        ("disease", "cause"): "has_cause",
        ("disease", "risk_factor"): "has_risk_factor",
        ("disease", "subtype"): "has_subtype",
        ("gene", "gene"): "interacts_with",  # Gene-gene interactions
        ("treatment", "treatment"): "contraindicated_with",  # Drug interactions
        ("symptom", "symptom"): "correlated_with",  # Symptom correlations
        ("gene", "treatment"): "targets",  # Gene-target interactions
        ("gene", "symptom"): "contributes_to"  # Genetic contributions to symptoms
    }

# ──────────────────────────────────────────────────────────────────────────────
# Enhanced Community Detection
# ──────────────────────────────────────────────────────────────────────────────

def consensus_community_detection(G: nx.Graph, random_state: int = 42) -> Tuple[Dict[str, int], List[Set[str]]]:
    """Perform consensus community detection using multiple algorithms."""
    all_communities = []
    
    # Method 1: Louvain
    try:
        if hasattr(nx.algorithms.community, "louvain_communities"):
            louvain_communities = list(nx.algorithms.community.louvain_communities(
                G, seed=random_state))
            all_communities.append(louvain_communities)
    except:
        pass
    
    # Method 2: Greedy modularity
    try:
        greedy_communities = list(nx.algorithms.community.greedy_modularity_communities(G))
        all_communities.append(greedy_communities)
    except:
        pass
    
    # Method 3: Label propagation
    try:
        from networkx.algorithms.community.label_propagation import label_propagation_communities
        label_communities = list(label_propagation_communities(G))
        all_communities.append(label_communities)
    except:
        pass
    
    if not all_communities:
        # Fallback to single method
        return detect_communities(G, random_state)
    
    # Use the method with highest modularity as consensus
    best_modularity = -1
    best_communities = all_communities[0]
    
    for communities in all_communities:
        try:
            modularity = nx.algorithms.community.modularity(G, communities)
            if modularity > best_modularity:
                best_modularity = modularity
                best_communities = communities
        except:
            continue
    
    # Map nodes to community
    node2comm = {}
    for i, community in enumerate(best_communities):
        for node in community:
            node2comm[node] = i
    
    return node2comm, best_communities

# ──────────────────────────────────────────────────────────────────────────────
# Statistical Validation
# ──────────────────────────────────────────────────────────────────────────────

def statistical_validation(G: nx.Graph, node2comm: Dict[str, int], 
                          centrality: Dict[str, Dict[str, float]]) -> Dict[str, Any]:
    """Perform statistical validation of key graph properties."""
    results = {}
    
    # Test if degree distribution follows power law
    degrees = [d for n, d in G.degree()]
    if len(degrees) > 10 and stats is not None:
        try:
            # Fit power law and exponential distributions
            powerlaw_params = stats.powerlaw.fit(degrees)
            exponential_params = stats.expon.fit(degrees)
            
            # Compare AIC values
            powerlaw_ll = stats.powerlaw.logpdf(degrees, *powerlaw_params).sum()
            exponential_ll = stats.expon.logpdf(degrees, *exponential_params).sum()
            
            powerlaw_aic = 2 * len(powerlaw_params) - 2 * powerlaw_ll
            exponential_aic = 2 * len(exponential_params) - 2 * exponential_ll
            
            results["degree_distribution"] = {
                "power_law_aic": powerlaw_aic,
                "exponential_aic": exponential_aic,
                "favors_power_law": powerlaw_aic < exponential_aic
            }
        except Exception:
            results["degree_distribution"] = {"note": "Could not fit distributions"}
    
    # Community quality metrics
    communities_list = defaultdict(list)
    for node, comm_id in node2comm.items():
        communities_list[comm_id].append(node)
    
    communities = list(communities_list.values())
    
    # Modularity calculation
    try:
        modularity = nx.algorithms.community.modularity(G, communities)
        results["community_quality"] = {"modularity": modularity}
    except:
        results["community_quality"] = {"modularity": None}
    
    # Centrality correlations
    if len(centrality["degree"]) > 5 and stats is not None:
        deg_values = [centrality["degree"][n] for n in G.nodes()]
        btw_values = [centrality["betweenness"][n] for n in G.nodes()]
        eig_values = [centrality["eigenvector"][n] for n in G.nodes()]
        
        try:
            deg_btw_corr = stats.spearmanr(deg_values, btw_values)
            deg_eig_corr = stats.spearmanr(deg_values, eig_values)
            
            results["centrality_correlations"] = {
                "degree_betweenness": {
                    "correlation": deg_btw_corr.correlation,
                    "p_value": deg_btw_corr.pvalue
                },
                "degree_eigenvector": {
                    "correlation": deg_eig_corr.correlation, 
                    "p_value": deg_eig_corr.pvalue
                }
            }
        except Exception:
            results["centrality_correlations"] = {"note": "Could not compute correlations"}
    
    return results

# ──────────────────────────────────────────────────────────────────────────────
# Enhanced Visualization
# ──────────────────────────────────────────────────────────────────────────────

def enhanced_pyvis_visualization(G: nx.Graph, path_html: Path, 
                                node2comm: Optional[Dict[str, int]] = None,
                                centrality: Optional[Dict[str, float]] = None):
    """Create enhanced interactive visualization with better styling and legend."""
    if Network is None:
        print("[INFO] pyvis not installed; skipping interactive visualization.")
        return
    
    net = Network(height="800px", width="100%", directed=False, 
                  notebook=False, bgcolor="#222222", font_color="#EEEEEE", heading="")
    
    # Configure physics
    net.set_options("""
    var options = {
      "physics": {
        "enabled": true,
        "stabilization": {"iterations": 100},
        "barnesHut": {
          "gravitationalConstant": -8000,
          "centralGravity": 0.3,
          "springLength": 95,
          "damping": 0.09
        }
      },
      "nodes": {
        "font": {"size": 14, "color": "#FFFFFF"},
        "borderWidth": 2,
        "shadow": true
      },
      "edges": {
        "color": {"color": "#888888", "highlight": "#FF6600"},
        "smooth": true,
        "width": 1,
        "shadow": true
      }
    }
    """)
    
    # Set node sizes based on centrality if provided
    max_size = 25
    min_size = 10
    
    for n, data in G.nodes(data=True):
        node_type = data.get("type", "")
        label = data.get("label", n)
        
        # Base styling
        color = COLOR_BY_TYPE.get(node_type, "#CCCCCC")
        title = f"<b>{label}</b><br>Type: {node_type}"
        
        # Size based on centrality
        size = min_size
        if centrality and n in centrality:
            values = list(centrality.values())
            max_centrality = max(values) if values else 1
            min_centrality = min(values) if values else 0
            range_centrality = max_centrality - min_centrality if max_centrality > min_centrality else 1
            
            normalized_centrality = (centrality[n] - min_centrality) / range_centrality
            size = min_size + (max_size - min_size) * normalized_centrality
        
        # Add community info to title
        if node2comm and n in node2comm:
            title += f"<br>Community: {node2comm[n]}"
        
        net.add_node(n, label=label, title=title, color=color, size=size)
    
    # Add edges with styling based on type
    edge_colors = {
        "treated_with": "#2ca02c",
        "associated_gene": "#9467bd", 
        "has_symptom": "#d62728",
        "has_diagnosis": "#17becf",
        "has_cause": "#8c564b",
        "has_risk_factor": "#ff7f0e",
        "has_subtype": "#7f7f7f",
        "interacts_with": "#e377c2",
        "targets": "#7f7f7f",
        "contributes_to": "#bcbd22"
    }
    
    for u, v, ed in G.edges(data=True):
        edge_type = ed.get("type", "")
        color = edge_colors.get(edge_type, "#888888")
        
        net.add_edge(u, v, title=edge_type, color=color, width=2)
    
    # Generate HTML with legend
    html_content = net.generate_html()
    
    # Add comprehensive legend
    legend_html = """
    <div id="legend" style="position: fixed; top: 10px; right: 10px; background: rgba(0,0,0,0.9); 
                         border: 2px solid #333; border-radius: 10px; padding: 15px; 
                         color: white; font-family: Arial, sans-serif; font-size: 12px; 
                         z-index: 999; max-width: 250px; box-shadow: 0 4px 8px rgba(0,0,0,0.3);">
        <div style="font-weight: bold; margin-bottom: 12px; border-bottom: 2px solid #555; padding-bottom: 8px; text-align: center;">
            Legend
        </div>
        
        <div style="margin-bottom: 15px;">
            <div style="font-weight: bold; margin-bottom: 8px; color: #FFD700;">Node Types:</div>
    """
    
    # Add node type legend items
    for node_type, color in COLOR_BY_TYPE.items():
        legend_html += f"""
            <div style="display: flex; align-items: center; margin-bottom: 4px; margin-left: 5px;">
                <div style="width: 14px; height: 14px; background: {color}; 
                            border: 1px solid #333; margin-right: 8px; border-radius: 3px;"></div>
                <span style="text-transform: capitalize; font-size: 11px;">{node_type.replace('_', ' ')}</span>
            </div>
        """
    
    legend_html += """
        </div>
        
        <div>
            <div style="font-weight: bold; margin-bottom: 8px; color: #FFD700;">Edge Types:</div>
    """
    
    # Add edge type legend items
    for edge_type, color in edge_colors.items():
        legend_html += f"""
            <div style="display: flex; align-items: center; margin-bottom: 4px; margin-left: 5px;">
                <div style="width: 14px; height: 14px; background: {color}; 
                            border: 1px solid #333; margin-right: 8px; border-radius: 3px;"></div>
                <span style="text-transform: capitalize; font-size: 11px;">{edge_type.replace('_', ' ')}</span>
            </div>
        """
    
    legend_html += """
        </div>
        
        <div style="margin-top: 10px; padding-top: 8px; border-top: 1px solid #555; font-size: 10px; text-align: center;">
            <div>🔍 Hover for details</div>
            <div>↔ Drag to move</div>
            <div>🔍 Scroll to zoom</div>
        </div>
    </div>
    """
    
    # Insert legend into HTML
    html_content = html_content.replace('</body>', f'{legend_html}\n</body>')
    
    # Write the enhanced HTML
    with open(path_html, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    print(f"[INFO] Enhanced visualization with comprehensive legend saved to {path_html}")

# ──────────────────────────────────────────────────────────────────────────────
# Memory Monitoring
# ──────────────────────────────────────────────────────────────────────────────

def monitor_memory(func):
    """Decorator to monitor memory usage of functions."""
    def wrapper(*args, **kwargs):
        if psutil:
            process = psutil.Process()
            mem_before = process.memory_info().rss / 1024 / 1024  # MB
        else:
            mem_before = 0
            
        start_time = time.time()
        result = func(*args, **kwargs)
        elapsed = time.time() - start_time
        
        if psutil:
            mem_after = process.memory_info().rss / 1024 / 1024  # MB
            print(f"[MEMORY] {func.__name__}: {mem_before:.1f}MB → {mem_after:.1f}MB (+{elapsed:.1f}s)")
        
        return result
    return wrapper

def optimize_memory():
    """Force garbage collection and memory optimization."""
    gc.collect()
    if psutil:
        process = psutil.Process()
        memory = process.memory_info().rss / 1024 / 1024
        print(f"[MEMORY] After GC: {memory:.1f}MB")

# ──────────────────────────────────────────────────────────────────────────────
# Loading & Graph Construction
# ──────────────────────────────────────────────────────────────────────────────

SCHEMA_KEYS = [
    "disease_name", "synonyms", "summary", "causes", "risk_factors",
    "symptoms", "diagnosis", "treatments", "related_genes", "subtypes"
]

NODE_TYPES = {
    "disease": "disease",
    "symptom": "symptom",
    "treatment": "treatment",
    "gene": "gene",
    "diagnosis": "diagnosis",
    "cause": "cause",
    "risk_factor": "risk_factor",
    "subtype": "subtype",
}

MAX_NODES_DEFAULT = 10000  # Default maximum nodes for memory management

EDGE_TYPES = {
    "has_symptom": ("disease", "symptom"),
    "treated_with": ("disease", "treatment"),
    "associated_gene": ("disease", "gene"),
    "has_diagnosis": ("disease", "diagnosis"),
    "has_cause": ("disease", "cause"),
    "has_risk_factor": ("disease", "risk_factor"),
    "has_subtype": ("disease", "subtype"),
}


def _norm(x: str) -> str:
    return " ".join((x or "").strip().split()).lower()


def _load_combined(path: Path) -> List[dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, dict) and "diseases" in data:
        return list(data["diseases"])
    raise ValueError("Combined file must contain a top-level 'diseases' list.")


def _load_dir(path: Path) -> List[dict]:
    items: List[dict] = []
    for p in sorted(path.glob("*.json")):
        try:
            obj = json.loads(p.read_text(encoding="utf-8"))
            if isinstance(obj, dict) and obj.get("disease_name"):
                items.append(obj)
        except json.JSONDecodeError:
            print(f"[WARN] Skipping invalid JSON: {p}")
    if not items:
        raise ValueError(f"No valid disease JSONs found in {path}")
    return items


def load_records(input_path: str) -> List[dict]:
    path = Path(input_path)
    if not path.exists():
        raise FileNotFoundError(path)
    if path.is_dir():
        return _load_dir(path)
    return _load_combined(path)


@dataclass
class BuildStats:
    n_nodes: int
    n_edges: int
    types: Counter


def build_graph(records: List[dict]) -> Tuple[nx.Graph, BuildStats]:
    G = nx.Graph()

    def add_node(name: str, ntype: str, **attrs):
        key = _norm(name)
        if not key:
            return
        G.add_node(key, label=(name or "").strip(), type=ntype, **attrs)

    def add_edge(a: str, b: str, etype: str):
        if not a or not b:
            return
        u, v = _norm(a), _norm(b)
        if not u or not v:
            return
        G.add_edge(u, v, type=etype)

    for rec in records:
        disease = (rec.get("disease_name") or "").strip()
        if not disease:
            continue
        add_node(disease, NODE_TYPES["disease"])  # disease node

        for trt in rec.get("treatments", []) or []:
            add_node(trt, NODE_TYPES["treatment"]); add_edge(disease, trt, "treated_with")
        for gen in rec.get("related_genes", []) or []:
            add_node(gen, NODE_TYPES["gene"]); add_edge(disease, gen, "associated_gene")
        for diag in rec.get("diagnosis", []) or []:
            add_node(diag, NODE_TYPES["diagnosis"]); add_edge(disease, diag, "has_diagnosis")
        for cause in rec.get("causes", []) or []:
            add_node(cause, NODE_TYPES["cause"]); add_edge(disease, cause, "has_cause")
        for rf in rec.get("risk_factors", []) or []:
            add_node(rf, NODE_TYPES["risk_factor"]); add_edge(disease, rf, "has_risk_factor")
        for st in rec.get("subtypes", []) or []:
            add_node(st, NODE_TYPES["subtype"]); add_edge(disease, st, "has_subtype")


    types = Counter(nx.get_node_attributes(G, "type").values())
    return G, BuildStats(G.number_of_nodes(), G.number_of_edges(), types)

# ──────────────────────────────────────────────────────────────────────────────
# Analyses
# ──────────────────────────────────────────────────────────────────────────────

def connectivity_summary(G: nx.Graph) -> Dict[str, Any]:
    comps = list(nx.connected_components(G))
    comps_sorted = sorted(comps, key=len, reverse=True)
    giant = G.subgraph(comps_sorted[0]).copy() if comps_sorted else G.copy()
    frac_giant = len(giant) / G.number_of_nodes() if G.number_of_nodes() else 0.0
    isolates = list(nx.isolates(G))
    return {
        "n_components": len(comps_sorted),
        "giant_nodes": len(giant),
        "giant_fraction": frac_giant,
        "n_isolates": len(isolates),
        "isolates": [G.nodes[n].get("label", n) for n in isolates[:50]],
        "giant": giant,
    }


def detect_communities(G: nx.Graph, random_state: int = 42) -> Tuple[Dict[str, int], List[Set[str]]]:
    if hasattr(nx.algorithms.community, "louvain_communities"):
        comms = nx.algorithms.community.louvain_communities(G, seed=random_state)
    else:
        comms = list(nx.algorithms.community.greedy_modularity_communities(G))
    node2comm: Dict[str, int] = {}
    for i, c in enumerate(comms):
        for n in c:
            node2comm[n] = i
    return node2comm, comms


def compute_centrality(G: nx.Graph, k_sample: int = 0) -> Dict[str, Dict[str, float]]:
    comps = list(nx.connected_components(G))
    giant = G.subgraph(max(comps, key=len)).copy() if comps else G

    deg = dict(G.degree())
    deg_norm = {n: d / (G.number_of_nodes() - 1) if G.number_of_nodes() > 1 else 0.0 for n, d in deg.items()}

    # Betweenness – support both explicit list and older API
    if k_sample and k_sample > 0:
        sample_nodes = random.Random(0).sample(list(G.nodes()), k=min(k_sample, G.number_of_nodes()))
        try:
            btw = nx.betweenness_centrality(G, normalized=True, nodes=sample_nodes, seed=0)  # new API
        except TypeError:
            btw = nx.betweenness_centrality(G, k=len(sample_nodes), normalized=True, seed=0)  # fallback
    else:
        btw = nx.betweenness_centrality(G, normalized=True)

    try:
        eig = nx.eigenvector_centrality(giant, max_iter=2000)
    except nx.PowerIterationFailedConvergence:
        eig = nx.eigenvector_centrality_numpy(giant)

    eig_full = {n: eig.get(n, 0.0) for n in G.nodes()}

    return {"degree": deg_norm, "betweenness": btw, "eigenvector": eig_full}


def link_prediction(G: nx.Graph, limit: int = 2000) -> List[Dict[str, Any]]:
    results: List[Dict[str, Any]] = []

    def plausible(u: str, v: str) -> bool:
        tu, tv = G.nodes[u].get("type"), G.nodes[v].get("type")
        return (tu == "disease" and tv in {"gene", "treatment", "symptom"}) or (
            tv == "disease" and tu in {"gene", "treatment", "symptom"}
        )

    lengths = dict(nx.all_pairs_shortest_path_length(G, cutoff=3))
    candidates: Set[Tuple[str, str]] = set()
    for u, dists in lengths.items():
        for v, dist in dists.items():
            if u < v and dist >= 2 and not G.has_edge(u, v) and plausible(u, v):
                candidates.add((u, v))
                if len(candidates) >= limit:
                    break
        if len(candidates) >= limit:
            break

    jac = {(u, v): s for u, v, s in nx.jaccard_coefficient(G, candidates)}
    aa = {(u, v): s for u, v, s in nx.adamic_adar_index(G, candidates)}
    a = {(u, v): s for u, v, s in nx.preferential_attachment(G, candidates)}

    for (u, v) in candidates:
        results.append({
            "u": G.nodes[u].get("label", u), "v": G.nodes[v].get("label", v),
            "type_u": G.nodes[u].get("type"), "type_v": G.nodes[v].get("type"),
            "jaccard": jac.get((u, v), 0.0),
            "adamic_adar": aa.get((u, v), 0.0),
            "pref_attach": pa.get((u, v), 0.0),
        })

    if results:
        for key in ("jaccard", "adamic_adar", "pref_attach"):
            vals = [r[key] for r in results]
            lo, hi = min(vals), max(vals)
            for r in results:
                r[f"{key}_n"] = 0.0 if hi == lo else (r[key] - lo) / (hi - lo)
        for r in results:
            r["ensemble"] = r["jaccard_n"] + r["adamic_adar_n"] + r["pref_attach_n"]
        results.sort(key=lambda x: x["ensemble"], reverse=True)
    return results


def improved_link_prediction(G: nx.Graph, limit: int = 2000) -> List[Dict[str, Any]]:
    """Enhanced link prediction with more edge types and features."""
    plausible_edges = get_plausible_edge_types()
    results = []
    
    # Generate candidates for all plausible edge types
    candidates_by_type = defaultdict(list)
    
    for u, v in itertools.combinations(G.nodes(), 2):
        if G.has_edge(u, v):
            continue
            
        type_u = G.nodes[u].get("type")
        type_v = G.nodes[v].get("type")
        
        # Check both directions for plausible edges
        edge_key = tuple(sorted([type_u, type_v]))
        if edge_key in [(k[0], k[1]) if k[0] <= k[1] else (k[1], k[0]) 
                       for k in plausible_edges.keys()]:
            candidates_by_type[edge_key].append((u, v))
    
    # Process each edge type separately
    for edge_type, candidates in candidates_by_type.items():
        if not candidates:
            continue
            
        # Limit candidates per edge type
        candidates = candidates[:limit // len(candidates_by_type)]
        
        # Compute similarity metrics
        jac = {(u, v): s for u, v, s in nx.jaccard_coefficient(G, candidates)}
        aa = {(u, v): s for u, v, s in nx.adamic_adar_index(G, candidates)}
        pa = {(u, v): s for u, v, s in nx.preferential_attachment(G, candidates)}
        
        for (u, v) in candidates:
            result = {
                "u": G.nodes[u].get("label", u),
                "v": G.nodes[v].get("label", v),
                "type_u": G.nodes[u].get("type"),
                "type_v": G.nodes[v].get("type"),
                "jaccard": jac.get((u, v), 0.0),
                "adamic_adar": aa.get((u, v), 0.0),
                "pref_attach": pa.get((u, v), 0.0),
                "edge_type": plausible_edges.get((type_u, type_v)) or 
                           plausible_edges.get((type_v, type_u), "")
            }
            results.append(result)
    
    # Normalize and rank
    if results:
        for metric in ["jaccard", "adamic_adar", "pref_attach"]:
            values = [r[metric] for r in results]
            min_val, max_val = min(values), max(values)
            range_val = max_val - min_val if max_val > min_val else 1.0
            
            for r in results:
                r[f"{metric}_normalized"] = (r[metric] - min_val) / range_val
        
        for r in results:
            r["ensemble_score"] = (r["jaccard_normalized"] + 
                                 r["adamic_adar_normalized"] + 
                                 r["pref_attach_normalized"]) / 3
    
    return sorted(results, key=lambda x: x.get("ensemble_score", 0), reverse=True)


def neighbor_majority_predict(G: nx.Graph, holdout_frac: float = 0.1, seed: int = 0) -> Dict[str, Any]:
    rng = random.Random(seed)
    labels = nx.get_node_attributes(G, "type")
    nodes = list(G.nodes())
    holdout_size = max(1, int(len(nodes) * holdout_frac))
    holdout = set(rng.sample(nodes, holdout_size))

    hidden_labels = {n: labels.pop(n) for n in list(labels.keys()) if n in holdout}

    def predict(n: str) -> str:
        neigh = list(G.neighbors(n))
        if not neigh:
            return Counter(labels.values()).most_common(1)[0][0]
        votes = Counter(labels.get(m) for m in neigh if labels.get(m) is not None)
        if votes:
            return votes.most_common(1)[0][0]
        return Counter(labels.values()).most_common(1)[0][0]

    preds: Dict[str, str] = {n: predict(n) for n in holdout}
    acc = sum(1 for n in holdout if preds[n] == hidden_labels[n]) / len(holdout)

    labels.update(hidden_labels)
    return {"accuracy": acc, "n_holdout": len(holdout), "preds": preds}


def traversal_demo(G: nx.Graph, seeds: List[str]) -> Tuple[str, str]:
    lines_bfs: List[str] = []
    lines_dfs: List[str] = []
    for s in seeds:
        key = _norm(s)
        if key not in G:
            lines_bfs.append(f"[seed missing] {s}")
            lines_dfs.append(f"[seed missing] {s}")
            continue
        bfs_order = list(nx.bfs_tree(G, source=key, depth_limit=3).nodes())
        dfs_order = list(nx.dfs_preorder_nodes(G, source=key))[:20]
        bfs_labels = [G.nodes[n].get("label", n) for n in bfs_order]
        dfs_labels = [G.nodes[n].get("label", n) for n in dfs_order]
        lines_bfs.append(f"Seed: {s}\n  " + " → ".join(bfs_labels[:20]))
        lines_dfs.append(f"Seed: {s}\n  " + " → ".join(dfs_labels))
    return "\n\n".join(lines_bfs), "\n\n".join(lines_dfs)


def shortest_path_demos(G: nx.Graph, seeds: List[str]) -> List[str]:
    paths: List[str] = []
    keys = [_norm(s) for s in seeds if _norm(s) in G]
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            u, v = keys[i], keys[j]
            try:
                sp = nx.shortest_path(G, u, v)
                labels = [G.nodes[n].get("label", n) for n in sp]
                paths.append(" → ".join(labels))
            except nx.NetworkXNoPath:
                paths.append(f"No path between {G.nodes[u]['label']} and {G.nodes[v]['label']}")
    return paths

# ──────────────────────────────────────────────────────────────────────────────
# Visualization (PyVis)
# ──────────────────────────────────────────────────────────────────────────────

COLOR_BY_TYPE = {
    "disease": "#1f77b4",
    "gene": "#9467bd",
    "treatment": "#2ca02c",
    "symptom": "#d62728",
    "diagnosis": "#17becf",
    "risk_factor": "#ff7f0e",
    "cause": "#8c564b",
    "subtype": "#7f7f7f",
}

def export_pyvis_with_legend(G: nx.Graph, path_html: Path, node2comm: Dict[str, int] | None = None):
    """Create interactive visualization with color legend."""
    if Network is None:
        print("[INFO] pyvis not installed; skipping interactive visualization.")
        return
    
    net = Network(height="750px", width="100%", directed=False, notebook=False, 
                  bgcolor="#111", font_color="#EEE", heading="")
    
    # Configure physics
    net.set_options("""
    var options = {
      "physics": {
        "enabled": true,
        "stabilization": {"iterations": 100},
        "barnesHut": {
          "gravitationalConstant": -8000,
          "centralGravity": 0.3,
          "springLength": 95,
          "damping": 0.09
        }
      },
      "nodes": {
        "font": {"size": 14, "color": "#FFFFFF"},
        "borderWidth": 2,
        "shadow": true
      },
      "edges": {
        "color": {"color": "#888888", "highlight": "#FF6600"},
        "smooth": true,
        "width": 1,
        "shadow": true
      }
    }
    """)
    
    # Add nodes
    for n, data in G.nodes(data=True):
        t = data.get("type", "")
        label = data.get("label", n)
        title = f"{label}<br>type={t}"
        if node2comm is not None:
            title += f"<br>community={node2comm.get(n, -1)}"
        color = COLOR_BY_TYPE.get(t, "#cccccc")
        net.add_node(n, label=label, title=title, color=color)

    # Add edges
    for u, v, ed in G.edges(data=True):
        et = ed.get("type", "")
        net.add_edge(u, v, title=et, color="#888888")

    # Generate HTML with legend
    html_content = net.generate_html()
    
    # Add legend to the HTML
    legend_html = """
    <div id="legend" style="position: fixed; top: 10px; right: 10px; background: rgba(0,0,0,0.8); 
                         border: 2px solid #333; border-radius: 8px; padding: 15px; 
                         color: white; font-family: Arial, sans-serif; font-size: 12px; 
                         z-index: 999; max-width: 200px;">
        <div style="font-weight: bold; margin-bottom: 10px; border-bottom: 1px solid #555; padding-bottom: 5px;">
            Node Types
        </div>
    """
    
    # Add legend items for each node type
    for node_type, color in COLOR_BY_TYPE.items():
        legend_html += f"""
        <div style="display: flex; align-items: center; margin-bottom: 5px;">
            <div style="width: 12px; height: 12px; background: {color}; 
                        border: 1px solid #333; margin-right: 8px; border-radius: 2px;"></div>
            <span style="text-transform: capitalize;">{node_type.replace('_', ' ')}</span>
        </div>
        """
    
    legend_html += "</div>"
    
    # Insert legend into HTML
    html_content = html_content.replace('</body>', f'{legend_html}\n</body>')
    
    # Write the enhanced HTML
    with open(path_html, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    print(f"[INFO] Enhanced visualization with legend saved to {path_html}")


def export_pyvis(G: nx.Graph, path_html: Path, node2comm: Dict[str, int] | None = None):
    if Network is None:
        print("[INFO] pyvis not installed; skipping interactive visualization.")
        return
    net = Network(height="750px", width="100%", directed=False, notebook=False, bgcolor="#111", font_color="#EEE")
    net.toggle_physics(True)

    for n, data in G.nodes(data=True):
        t = data.get("type", "")
        label = data.get("label", n)
        title = f"{label}<br>type={t}"
        if node2comm is not None:
            title += f"<br>community={node2comm.get(n, -1)}"
        color = COLOR_BY_TYPE.get(t, "#cccccc")
        net.add_node(n, label=label, title=title, color=color)

    for u, v, ed in G.edges(data=True):
        et = ed.get("type", "")
        net.add_edge(u, v, title=et, color="#888888")

    net.write_html(str(path_html))

# ──────────────────────────────────────────────────────────────────────────────
# Enhanced Report Generation
# ──────────────────────────────────────────────────────────────────────────────

def enhanced_report_generation(G: nx.Graph, outdir: Path, stats: BuildStats, 
                              conn: Dict[str, Any], cent: Dict[str, Dict[str, float]],
                              node2comm: Dict[str, int], comms: List[Set[str]],
                              linkpred_rows: List[Dict[str, Any]],
                              traversal_texts: Tuple[str, str], sp_texts: List[str], 
                              topk: int, npp_result: Dict[str, Any],
                              validation_results: Dict[str, Any]) -> str:
    """Generate enhanced analysis report with statistical insights."""
    
    # Basic statistics
    by_type = " | ".join(f"{t}: {c}" for t, c in stats.types.most_common())
    avg_deg = (2 * stats.n_edges) / stats.n_nodes if stats.n_nodes else 0.0
    density = nx.density(G)
    
    # Centrality statistics
    if np is not None:
        deg_stats = {
            'mean': np.mean(list(cent["degree"].values())),
            'std': np.std(list(cent["degree"].values())),
            'max': max(cent["degree"].values()),
            'min': min(cent["degree"].values())
        }
        comm_sizes = [len(c) for c in comms]
        comm_stats = {
            'mean_size': np.mean(comm_sizes),
            'std_size': np.std(comm_sizes),
            'largest': max(comm_sizes),
            'smallest': min(comm_sizes)
        }
    else:
        deg_stats = {}
        comm_stats = {}
    
    report_sections = [
        "# Enhanced Cancer Knowledge Graph Analysis Report\n",
        "## Executive Summary\n",
        f"- **{stats.n_nodes}** nodes and **{stats.n_edges}** edges\n",
        f"- **{conn['n_components']}** connected components (giant: {conn['giant_fraction']:.1%})\n",
        f"- **{len(comms)}** communities detected (avg size: {comm_stats.get('mean_size', 'N/A'):.1f})\n",
        f"- Average degree: **{avg_deg:.2f}**, Density: **{density:.4f}**\n"
    ]
    
    # Graph summary (similar to original)
    report_sections.extend([
        "\n## Graph Summary\n",
        f"Nodes: **{stats.n_nodes}**, Edges: **{stats.n_edges}**  \n",
        f"Types: {by_type}\n",
        f"Connected components: **{conn['n_components']}**, giant component size: **{conn['giant_nodes']}** (fraction={conn['giant_fraction']:.2%})  \n",
        f"Average degree: **{avg_deg:.2f}**, Graph density: **{density:.4f}**\n"
    ])
    
    if conn["isolates"]:
        report_sections.append(f"Isolates (preview): {', '.join(conn['isolates'][:10])}\n")

    # Communities
    report_sections.extend([
        "\n## Community Detection\n",
        f"Detected **{len(comms)}** communities.\n"
    ])
    
    leaders = []
    for i, cset in enumerate(comms[: min(12, len(comms))]):
        leader = max(cset, key=lambda n: conn["giant"].degree(n) if n in conn["giant"] else 0)
        leader_label = (
            conn["giant"].nodes[leader].get("label", leader)
            if leader in conn["giant"]
            else G.nodes[leader].get("label", leader)
        )
        leaders.append({"community": i, "leader": leader_label, "size": len(cset)})
    
    if leaders:
        header = "| " + " | ".join(["community", "leader", "size"]) + " |\n" + "|" + "---|" * 3 + "\n"
        lines = []
        for r in leaders:
            lines.append("| " + " | ".join(str(r.get(c, "")) for c in ["community", "leader", "size"]) + " |")
        report_sections.append(header + "\n".join(lines))

    # Centrality
    report_sections.extend([
        "\n## Centrality (Top Hubs)\n"
    ])
    
    def topk_dict(d: Dict[str, float], G: nx.Graph, k: int) -> List[Dict[str, Any]]:
        items = sorted(d.items(), key=lambda x: x[1], reverse=True)[:k]
        rows = [{"node": n, "label": G.nodes[n].get("label", n), "type": G.nodes[n].get("type"), "score": s} for n, s in items]
        return rows
    
    top_deg = topk_dict(cent["degree"], conn["giant"], min(topk, 20))
    top_btw = topk_dict(cent["betweenness"], conn["giant"], min(topk, 20))
    top_eig = topk_dict(cent["eigenvector"], conn["giant"], min(topk, 20))
    
    def fmt_table(rows: List[Dict[str, Any]], cols: List[str]) -> str:
        if not rows:
            return "(none)"
        header = "| " + " | ".join(cols) + " |\n" + "|" + "---|" * len(cols) + "\n"
        lines = []
        for r in rows:
            lines.append("| " + " | ".join(str(r.get(c, "")) for c in cols) + " |")
        return header + "\n".join(lines)
    
    report_sections.extend([
        "**Degree**\n" + fmt_table(top_deg, ["label", "type", "score"]),
        "\n**Betweenness**\n" + fmt_table(top_btw, ["label", "type", "score"]),
        "\n**Eigenvector**\n" + fmt_table(top_eig, ["label", "type", "score"])
    ])

    # Link prediction
    report_sections.extend([
        "\n## Link Prediction (Top Suggestions)\n"
    ])
    
    if linkpred_rows:
        if isinstance(linkpred_rows[0], dict) and 'ensemble_score' in linkpred_rows[0]:
            report_sections.append(fmt_table(linkpred_rows[:topk], ["u", "type_u", "v", "type_v", "ensemble_score"]))
        else:
            report_sections.append(fmt_table(linkpred_rows[:topk], ["u", "type_u", "v", "type_v", "ensemble"]))

    # Statistical validation section
    if validation_results:
        report_sections.extend([
            "\n## Statistical Validation\n",
            "### Degree Distribution Analysis\n"
        ])
        
        if "degree_distribution" in validation_results:
            dd = validation_results["degree_distribution"]
            if "favors_power_law" in dd:
                report_sections.append(f"- Degree distribution {'follows' if dd['favors_power_law'] else 'does not follow'} power law (AIC comparison)\n")
        
        if "centrality_correlations" in validation_results:
            corr = validation_results["centrality_correlations"]
            if "degree_betweenness" in corr:
                r = corr["degree_betweenness"]["correlation"]
                p = corr["degree_betweenness"]["p_value"]
                report_sections.append(f"- Degree-Betweenness correlation: **{r:.3f}** (p={p:.3f})\n")

    # Node property prediction
    report_sections.extend([
        "\n## Node Property Prediction\n"
    ])
    
    if npp_result:
        report_sections.append(f"Accuracy: **{npp_result['accuracy']:.2%}** on {npp_result['n_holdout']} hidden nodes.\n")
    else:
        report_sections.append("(Not computed)\n")

    # Traversal & shortest paths
    report_sections.extend([
        "\n## Traversal & Shortest Paths\n"
    ])
    
    bfs_text, dfs_text = traversal_texts
    if bfs_text:
        report_sections.append("**BFS (depth≤3) from seeds**\n" + "\n\n" + bfs_text)
    if dfs_text:
        report_sections.append("\n**DFS (preview)**\n" + "\n\n" + dfs_text)
    if sp_texts:
        report_sections.append("\n**Shortest paths among seeds**\n" + "\n".join(f"- {p}" for p in sp_texts))

    # Biological interpretation
    report_sections.extend([
        "\n## Biological Interpretation\n",
        "### Key Findings\n"
    ])
    
    # Identify key patterns
    if conn['giant_fraction'] > 0.8:
        report_sections.append("- High connectivity suggests strong interrelations among cancer entities\n")
    else:
        report_sections.append("- Modular structure indicates distinct cancer subtypes or mechanisms\n")
    
    if len(comms) > 10:
        report_sections.append("- Rich community structure reveals functional specialization\n")
    
    # Top predictions
    if linkpred_rows:
        top_prediction = linkpred_rows[0] if isinstance(linkpred_rows[0], dict) else linkpred_rows[0]
        report_sections.append(f"- **Top link prediction**: {top_prediction['u']} ↔ {top_prediction['v']} "
                              f"(score: {top_prediction.get('ensemble_score', top_prediction.get('ensemble', 0)):.3f})\n")

    report_sections.append("\n---\nGenerated by Enhanced Module 6 Analysis.\n")
    
    return "".join(report_sections)

# ──────────────────────────────────────────────────────────────────────────────
# Reporting helpers
# ──────────────────────────────────────────────────────────────────────────────

def to_df(rows: List[Dict[str, Any]]):
    if pd is None:
        return rows
    return pd.DataFrame(rows)


def topk_dict(d: Dict[str, float], G: nx.Graph, k: int) -> List[Dict[str, Any]]:
    items = sorted(d.items(), key=lambda x: x[1], reverse=True)[:k]
    rows = [{"node": n, "label": G.nodes[n].get("label", n), "type": G.nodes[n].get("type"), "score": s} for n, s in items]
    return rows


def write_csv(df_or_rows: Any, path: Path):
    if pd is None:
        with path.open("w", encoding="utf-8") as f:
            if isinstance(df_or_rows, list):
                for r in df_or_rows:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
            else:
                for r in df_or_rows.to_dict(orient="records"):
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
        return
    if isinstance(df_or_rows, list):
        df_or_rows = pd.DataFrame(df_or_rows)
    df_or_rows.to_csv(path, index=False)


def render_report(G: nx.Graph, outdir: Path, stats: BuildStats, conn: Dict[str, Any], cent: Dict[str, Dict[str, float]],
                  node2comm: Dict[str, int], comms: List[Set[str]],
                  linkpred_rows: List[Dict[str, Any]],
                  traversal_texts: Tuple[str, str], sp_texts: List[str], topk: int,
                  npp_result: Dict[str, Any] | None = None) -> str:
    by_type = " | ".join(f"{t}: {c}" for t, c in stats.types.most_common())
    top_deg = topk_dict(cent["degree"], conn["giant"], min(topk, 20))
    top_btw = topk_dict(cent["betweenness"], conn["giant"], min(topk, 20))
    top_eig = topk_dict(cent["eigenvector"], conn["giant"], min(topk, 20))

    bfs_text, dfs_text = traversal_texts

    def fmt_table(rows: List[Dict[str, Any]], cols: List[str]) -> str:
        if not rows:
            return "(none)"
        header = "| " + " | ".join(cols) + " |\n" + "|" + "---|" * len(cols) + "\n"
        lines = []
        for r in rows:
            lines.append("| " + " | ".join(str(r.get(c, "")) for c in cols) + " |")
        return header + "\n".join(lines)

    report = []
    report.append("# Module 6 – Graph Analysis Report\n")

    # Graph summary
    avg_deg = (2 * stats.n_edges) / stats.n_nodes if stats.n_nodes else 0.0
    density = nx.density(G)
    report.append("## Graph Summary\n")
    report.append(f"Nodes: **{stats.n_nodes}**, Edges: **{stats.n_edges}**  ")
    report.append(f"Types: {by_type}\n")
    report.append(f"Connected components: **{conn['n_components']}**, giant component size: **{conn['giant_nodes']}** (fraction={conn['giant_fraction']:.2%})  ")
    report.append(f"Average degree: **{avg_deg:.2f}**, Graph density: **{density:.4f}**\n")
    if conn["isolates"]:
        report.append(f"Isolates (preview): {', '.join(conn['isolates'][:10])}\n")

    # Communities
    report.append("\n## Community Detection\n")
    report.append(f"Detected **{len(comms)}** communities.\n")
    leaders = []
    for i, cset in enumerate(comms[: min(12, len(comms))]):
        leader = max(cset, key=lambda n: conn["giant"].degree(n) if n in conn["giant"] else 0)
        leader_label = (
            conn["giant"].nodes[leader].get("label", leader)
            if leader in conn["giant"]
            else G.nodes[leader].get("label", leader)
        )
        leaders.append({"community": i, "leader": leader_label, "size": len(cset)})
    report.append(fmt_table(leaders, ["community", "leader", "size"]))

    # Centrality
    report.append("\n## Centrality (Top Hubs)\n")
    report.append("**Degree**\n" + fmt_table(top_deg, ["label", "type", "score"]))
    report.append("\n**Betweenness**\n" + fmt_table(top_btw, ["label", "type", "score"]))
    report.append("\n**Eigenvector**\n" + fmt_table(top_eig, ["label", "type", "score"]))

    # Link prediction
    report.append("\n## Link Prediction (Top Suggestions)\n")
    report.append(fmt_table(linkpred_rows[:topk], ["u", "type_u", "v", "type_v", "ensemble"]))

    # Node property prediction
    report.append("\n## Node Property Prediction\n")
    if npp_result:
        report.append(f"Accuracy: **{npp_result['accuracy']:.2%}** on {npp_result['n_holdout']} hidden nodes.\n")
    else:
        report.append("(Not computed)\n")

    # Traversal & shortest paths
    report.append("\n## Traversal & Shortest Paths\n")
    if bfs_text:
        report.append("**BFS (depth≤3) from seeds**\n" + "\n\n" + bfs_text)
    if dfs_text:
        report.append("\n**DFS (preview)**\n" + "\n\n" + dfs_text)
    if sp_texts:
        report.append("\n**Shortest paths among seeds**\n" + "\n".join(f"- {p}" for p in sp_texts))

    report.append("\n---\nGenerated by Module 6 (analyse.py).\n")
    return "\n".join(report)

# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    outdir = Path(args.outdir)
    start_time = time.time()
    outdir.mkdir(parents=True, exist_ok=True)

    print("📥 Loading records …")
    records = load_records(args.input)
    
    if args.max_nodes > 0:
        records = records[:args.max_nodes]
        print(f"📏 Limited to {args.max_nodes} records for memory management")
    
    if args.memory_monitor:
        print("🔍 Memory monitoring enabled")
    if args.validation:
        print("📊 Statistical validation enabled")

    print("🧱 Building graph …")
    G, stats = build_graph(records)
    print(f"✅ Graph built: {stats.n_nodes} nodes, {stats.n_edges} edges. Types: {stats.types}")

    print("🔗 Connectivity analysis …")
    conn = connectivity_summary(G)
    print(f"   Components: {conn['n_components']} | Giant: {conn['giant_nodes']} ({conn['giant_fraction']:.2%}) | Isolates: {conn['n_isolates']}")

    print("🧩 Community detection …")
    node2comm, comms = detect_communities(G, random_state=args.random_state)
    print(f"   Detected communities: {len(comms)}")

    # Use consensus community detection if available
    if args.validation or args.enhanced_viz:
        print("🧩 Using consensus community detection …")
        node2comm, comms = consensus_community_detection(G, random_state=args.random_state)
        print(f"   Consensus communities: {len(comms)}")

    print("📈 Centrality metrics …")
    cent = compute_centrality(G, k_sample=args.betweenness_sample)
    
    # Memory optimization after centrality computation
    if args.memory_monitor:
        optimize_memory()
    
    # Console Top-5 summaries
    def print_top5(name: str, scores: Dict[str, float]):
        top5 = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)[:5]
        lines = [f"   {i+1:>2}. {G.nodes[n].get('label', n)} ({scores[n]:.4f})" for i, (n, _) in enumerate(top5)]
        print(f"   Top 5 by {name}:")
        print("\n".join(lines))
    print_top5("degree", cent["degree"])
    print_top5("betweenness", cent["betweenness"])
    print_top5("eigenvector", cent["eigenvector"])

    print("🔮 Link prediction (local similarity indices) …")
    if args.validation:
        linkpred_rows = improved_link_prediction(G, limit=4000)
        print("   Using enhanced link prediction with expanded edge types")
    else:
        linkpred_rows = link_prediction(G, limit=4000)
    
    if linkpred_rows:
        print("   Top 5 link suggestions (ensemble):")
        for i, r in enumerate(linkpred_rows[:5], 1):
            # FIXED: Handle both ensemble formats
            ensemble_score = r.get('ensemble_score', r.get('ensemble', 0))
            print(f"   {i:>2}. {r['u']} ↔ {r['v']}  [{r['type_u']}–{r['type_v']}]  ensemble={ensemble_score:.3f}")

    print("🏷️ Node property prediction (neighbor-majority holdout) …")
    npp = neighbor_majority_predict(G, holdout_frac=0.1, seed=0)
    print(f"   Accuracy: {npp['accuracy']:.2%} on {npp['n_holdout']} hidden nodes")

    # Statistical validation
    validation_results = {}
    if args.validation:
        print("📊 Performing statistical validation …")
        validation_results = statistical_validation(G, node2comm, cent)
        print(f"   Validation completed: {len(validation_results)} tests performed")

    print("🧭 Traversal demos (BFS/DFS, shortest paths) …")
    bfs_txt, dfs_txt = traversal_demo(G, args.seed)
    sp_txts = shortest_path_demos(G, args.seed)

    graphml_path = outdir / args.graphml
    nx.write_graphml(G, graphml_path)
    print(f"💾 Saved GraphML → {graphml_path}")

    html_path = outdir / args.viz_html
    if args.enhanced_viz:
        enhanced_pyvis_visualization(G, html_path, node2comm=node2comm, centrality=cent["degree"])
    else:
        export_pyvis_with_legend(G, html_path, node2comm=node2comm)
        if html_path.exists():
            print(f"🌐 Saved interactive HTML → {html_path}")

    # Tabular exports
    rows: List[Dict[str, Any]] = []
    if pd is not None:
        for metric, scores in cent.items():
            for n, s in scores.items():
                rows.append({
                    "node": n,
                    "label": G.nodes[n].get("label", n),
                    "type": G.nodes[n].get("type"),
                    "metric": metric,
                    "score": s,
                })
        centrality_df = pd.DataFrame(rows)
        communities_df = pd.DataFrame([{ "node": n,
                                         "label": G.nodes[n].get("label", n),
                                         "type": G.nodes[n].get("type"),
                                         "community": node2comm.get(n, -1)}
                                       for n in G.nodes()])
        
        # FIXED: Ensure consistent column names for CSV export
        if args.validation and linkpred_rows and 'ensemble_score' in linkpred_rows[0]:
            # Convert to original format for CSV compatibility
            linkpred_export = []
            for r in linkpred_rows:
                linkpred_export.append({
                    'u': r['u'],
                    'v': r['v'], 
                    'type_u': r['type_u'],
                    'type_v': r['type_v'],
                    'ensemble': r['ensemble_score']  # Use 'ensemble' for CSV
                })
            linkpred_df = pd.DataFrame(linkpred_export)
        else:
            linkpred_df = pd.DataFrame(linkpred_rows)
    else:
        for metric, scores in cent.items():
            for n, s in scores.items():
                rows.append({
                    "node": n,
                    "label": G.nodes[n].get("label", n),
                    "type": G.nodes[n].get("type"),
                    "metric": metric,
                    "score": s,
                })
        centrality_df = rows
        communities_df = [{"node": n, "community": node2comm.get(n, -1)} for n in G.nodes()]
        linkpred_df = linkpred_rows

    write_csv(centrality_df, outdir / "centrality.csv")
    write_csv(communities_df, outdir / "communities.csv")
    write_csv(linkpred_df, outdir / "link_predictions.csv")

    print("📝 Rendering report …")
    if args.validation:
        report_md = enhanced_report_generation(G, outdir, stats, conn, cent, node2comm, comms, 
                                             linkpred_rows, (bfs_txt, dfs_txt), sp_txts, 
                                             args.topk, npp_result=npp, validation_results=validation_results)
    else:
        report_md = render_report(G, outdir, stats, conn, cent, node2comm, comms, linkpred_rows, 
                                (bfs_txt, dfs_txt), sp_txts, args.topk, npp_result=npp)
    
    report_path = outdir / "report_module6.md"
    report_path.write_text(report_md, encoding="utf-8")
    print(f"📄 Saved report → {report_path}")

    elapsed = time.time() - start_time
    print(f"⏱️ Total execution time: {elapsed:.1f}s")
    print("✔️ Module 6 complete.")


if __name__ == "__main__":
    main()
