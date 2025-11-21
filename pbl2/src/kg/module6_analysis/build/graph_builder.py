# src/kg/module6_analysis/build/graph_builder.py

"""
Graph construction utilities for Module 6.

This module is responsible ONLY for:
  - building a NetworkX graph from disease JSON records
  - assigning node labels and types
  - creating typed edges between entities
  - returning basic build statistics

It deliberately does NOT perform any analysis (centrality, communities, etc.),
keeping a clean separation of concerns.
"""

from __future__ import annotations

from dataclasses import dataclass
from collections import Counter
from typing import Any, Dict, List, Tuple

from pathlib import Path
import networkx as nx

from ..utils.constants import NODE_TYPES
from ..utils.normalize import _norm


@dataclass
class BuildStats:
    n_nodes: int
    n_edges: int
    types: Counter


def _add_node(G: nx.Graph, name: str, ntype: str, **attrs: Any) -> None:
    key = _norm(name)
    if not key:
        return

    if key in G:
        if not G.nodes[key].get("label"):
            G.nodes[key]["label"] = (name or "").strip()
        if "type" not in G.nodes[key]:
            G.nodes[key]["type"] = ntype
        for k, v in attrs.items():
            G.nodes[key].setdefault(k, v)
    else:
        G.add_node(
            key,
            label=(name or "").strip(),
            type=ntype,
            **attrs,
        )


def _add_edge(G: nx.Graph, a: str, b: str, etype: str, weight: float | None = None) -> None:
    if not a or not b:
        return

    u = _norm(a)
    v = _norm(b)
    if not u or not v:
        return

    if G.has_edge(u, v):
        # preserve the max weight if multiple contributions exist
        existing = G[u][v].get("weight")
        if weight is not None:
            if existing is None or weight > existing:
                G[u][v]["weight"] = weight
    else:
        attrs = {"type": etype}
        if weight is not None:
            attrs["weight"] = weight
        G.add_edge(u, v, **attrs)


def build_graph(records: List[Dict[str, Any]]) -> Tuple[nx.Graph, BuildStats]:
    G = nx.Graph()

    for rec in records:
        disease = (rec.get("disease_name") or "").strip()
        if not disease:
            continue
        edge_weights = rec.get("_edge_weights", {})

        # Disease node
        _add_node(G, disease, NODE_TYPES["disease"])

        # Treatments
        trt_weights = {w["value"]: w.get("weight") for w in edge_weights.get("treatments", [])} if edge_weights else {}
        for trt in (rec.get("treatments") or []):
            _add_node(G, trt, NODE_TYPES["treatment"])
            _add_edge(G, disease, trt, "treated_with", trt_weights.get(trt))

        # Genes
        gene_weights = {w["value"]: w.get("weight") for w in edge_weights.get("related_genes", [])} if edge_weights else {}
        for gen in (rec.get("related_genes") or []):
            _add_node(G, gen, NODE_TYPES["gene"])
            _add_edge(G, disease, gen, "associated_gene", gene_weights.get(gen))

        # Diagnosis
        diag_weights = {w["value"]: w.get("weight") for w in edge_weights.get("diagnosis", [])} if edge_weights else {}
        for diag in (rec.get("diagnosis") or []):
            _add_node(G, diag, NODE_TYPES["diagnosis"])
            _add_edge(G, disease, diag, "has_diagnosis", diag_weights.get(diag))

        # Causes
        cause_weights = {w["value"]: w.get("weight") for w in edge_weights.get("causes", [])} if edge_weights else {}
        for cause in (rec.get("causes") or []):
            _add_node(G, cause, NODE_TYPES["cause"])
            _add_edge(G, disease, cause, "has_cause", cause_weights.get(cause))

        # Risk factors
        rf_weights = {w["value"]: w.get("weight") for w in edge_weights.get("risk_factors", [])} if edge_weights else {}
        for rf in (rec.get("risk_factors") or []):
            _add_node(G, rf, NODE_TYPES["risk_factor"])
            _add_edge(G, disease, rf, "has_risk_factor", rf_weights.get(rf))

        # Subtypes
        st_weights = {w["value"]: w.get("weight") for w in edge_weights.get("subtypes", [])} if edge_weights else {}
        for st in (rec.get("subtypes") or []):
            _add_node(G, st, NODE_TYPES["subtype"])
            _add_edge(G, disease, st, "has_subtype", st_weights.get(st))

    types_counter = Counter(nx.get_node_attributes(G, "type").values())
    stats = BuildStats(
        n_nodes=G.number_of_nodes(),
        n_edges=G.number_of_edges(),
        types=types_counter,
    )

    return G, stats
