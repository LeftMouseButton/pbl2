#!/usr/bin/env python3
"""
Module 6 – Graph Construction & Analysis (analyse.py)
----------------------------------------------------
Builds a heterogeneous cancer knowledge graph from project JSON outputs, then
performs a suite of analyses:

1) Path Finding & Shortest Paths
2) Connectivity Analysis
3) Community Detection (Louvain ➜ fallback Greedy Modularity)
4) Centrality (degree, betweenness, eigenvector)
5) Link Prediction (Jaccard, Adamic–Adar, Preferential Attachment; ensemble)
6) Node Property Prediction (neighbor-majority; holdout evaluation)
7) Traversal / Search (BFS/DFS demos from seeds)

Inputs
------
• A combined JSON file produced by Module 5, e.g. data/combined/all_diseases.json
  (schema: {"diseases": [ {disease_name, symptoms, treatments, related_genes, ...}, ... ]})
• OR: a directory of per-disease JSON files under data/json/ (same schema per file)

Outputs (default under --outdir, e.g., data/analysis/)
------------------------------------------------------
• report_module6.md                 # Human-readable analysis report (paste-able into paper)
• graph.graphml                     # Full graph (labels & types as node/edge attributes)
• graph.html                        # Interactive PyVis visualization
• centrality.csv                    # Degree, betweenness, eigenvector (top-k and full)
• communities.csv                   # Node → community mapping
• link_predictions.csv              # Top predicted links (ensemble-ranked)

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
    --random-state 42

If --input points to a directory, the program will load all *.json within it.

Notes
-----
• Safe on mid-sized graphs (1–10k nodes). For very large graphs, use --betweenness-sample.
• Program is deterministic except for community detection (random seeding handled).
"""


from __future__ import annotations

import argparse
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Set, Any, Tuple
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
    return p.parse_args()

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
    pa = {(u, v): s for u, v, s in nx.preferential_attachment(G, candidates)}

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
    outdir.mkdir(parents=True, exist_ok=True)

    print("📥 Loading records …")
    records = load_records(args.input)

    print("🧱 Building graph …")
    G, stats = build_graph(records)
    print(f"✅ Graph built: {stats.n_nodes} nodes, {stats.n_edges} edges. Types: {stats.types}")

    print("🔗 Connectivity analysis …")
    conn = connectivity_summary(G)
    print(f"   Components: {conn['n_components']} | Giant: {conn['giant_nodes']} ({conn['giant_fraction']:.2%}) | Isolates: {conn['n_isolates']}")

    print("🧩 Community detection …")
    node2comm, comms = detect_communities(G, random_state=args.random_state)
    print(f"   Detected communities: {len(comms)}")

    print("📈 Centrality metrics …")
    cent = compute_centrality(G, k_sample=args.betweenness_sample)
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
    linkpred_rows = link_prediction(G, limit=4000)
    if linkpred_rows:
        print("   Top 5 link suggestions (ensemble):")
        for i, r in enumerate(linkpred_rows[:5], 1):
            print(f"   {i:>2}. {r['u']} ↔ {r['v']}  [{r['type_u']}–{r['type_v']}]  ensemble={r['ensemble']:.3f}")

    print("🏷️ Node property prediction (neighbor-majority holdout) …")
    npp = neighbor_majority_predict(G, holdout_frac=0.1, seed=0)
    print(f"   Accuracy: {npp['accuracy']:.2%} on {npp['n_holdout']} hidden nodes")

    print("🧭 Traversal demos (BFS/DFS, shortest paths) …")
    bfs_txt, dfs_txt = traversal_demo(G, args.seed)
    sp_txts = shortest_path_demos(G, args.seed)

    graphml_path = outdir / args.graphml
    nx.write_graphml(G, graphml_path)
    print(f"💾 Saved GraphML → {graphml_path}")

    html_path = outdir / args.viz_html
    export_pyvis(G, html_path, node2comm=node2comm)
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
    report_md = render_report(G, outdir, stats, conn, cent, node2comm, comms, linkpred_rows, (bfs_txt, dfs_txt), sp_txts, args.topk, npp_result=npp)
    report_path = outdir / "report_module6.md"
    report_path.write_text(report_md, encoding="utf-8")
    print(f"📄 Saved report → {report_path}")

    print("✔️ Module 6 complete.")


if __name__ == "__main__":
    main()
