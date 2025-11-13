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


def _add_edge(G: nx.Graph, a: str, b: str, etype: str) -> None:
    if not a or not b:
        return

    u = _norm(a)
    v = _norm(b)
    if not u or not v:
        return

    G.add_edge(u, v, type=etype)


def build_graph(records: List[Dict[str, Any]]) -> Tuple[nx.Graph, BuildStats]:
    G = nx.Graph()

    for rec in records:
        disease = (rec.get("disease_name") or "").strip()
        if not disease:
            continue

        # Disease node
        _add_node(G, disease, NODE_TYPES["disease"])

        # Treatments
        for trt in (rec.get("treatments") or []):
            _add_node(G, trt, NODE_TYPES["treatment"])
            _add_edge(G, disease, trt, "treated_with")

        # Genes
        for gen in (rec.get("related_genes") or []):
            _add_node(G, gen, NODE_TYPES["gene"])
            _add_edge(G, disease, gen, "associated_gene")

        # Diagnosis
        for diag in (rec.get("diagnosis") or []):
            _add_node(G, diag, NODE_TYPES["diagnosis"])
            _add_edge(G, disease, diag, "has_diagnosis")

        # Causes
        for cause in (rec.get("causes") or []):
            _add_node(G, cause, NODE_TYPES["cause"])
            _add_edge(G, disease, cause, "has_cause")

        # Risk factors
        for rf in (rec.get("risk_factors") or []):
            _add_node(G, rf, NODE_TYPES["risk_factor"])
            _add_edge(G, disease, rf, "has_risk_factor")

        # Subtypes
        for st in (rec.get("subtypes") or []):
            _add_node(G, st, NODE_TYPES["subtype"])
            _add_edge(G, disease, st, "has_subtype")

    types_counter = Counter(nx.get_node_attributes(G, "type").values())
    stats = BuildStats(
        n_nodes=G.number_of_nodes(),
        n_edges=G.number_of_edges(),
        types=types_counter,
    )

    return G, stats
