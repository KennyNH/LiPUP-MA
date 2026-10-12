"""Graph analysis tools for simulation output json (networkx node-link)."""

from __future__ import annotations

import ast
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Tuple

try:
    from .landuse import MODIFIABLE_LANDUSE
except ImportError:
    from landuse import MODIFIABLE_LANDUSE


@dataclass
class GraphIndex:
    """In-memory index for fast graph queries.

    Purpose:
        Store original node/link lists and pre-built lookup maps.

    Fields:
        nodes: Node dictionaries from node-link JSON.
        links: Edge dictionaries from node-link JSON.
        node_by_id: Node lookup by raw id and stringified id.
        edges_by_type: Edge list grouped by `edge_type`.
    """

    nodes: List[Dict[str, Any]]
    links: List[Dict[str, Any]]
    node_by_id: Dict[Any, Dict[str, Any]]
    edges_by_type: Dict[str, List[Dict[str, Any]]]


def build_graph_index(graph_data: Dict[str, Any]) -> GraphIndex:
    """Build a GraphIndex from simulation node-link JSON.

    Args:
        graph_data: Dict with `nodes` and `links`.

    Returns:
        GraphIndex ready for downstream tool queries.
    """

    nodes = graph_data.get("nodes", [])
    links = graph_data.get("links", [])
    node_by_id: Dict[Any, Dict[str, Any]] = {}
    for n in nodes:
        nid = n.get("id")
        node_by_id[nid] = n
        node_by_id[str(nid)] = n
    edges_by_type: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for e in links:
        edges_by_type[e.get("edge_type", "")].append(e)
    return GraphIndex(nodes=nodes, links=links, node_by_id=node_by_id, edges_by_type=dict(edges_by_type))


def resolve_node(idx: GraphIndex, raw_id: Any) -> Dict[str, Any] | None:
    """Resolve a node by id with int/str fallback.

    Args:
        idx: Graph index.
        raw_id: Node id (int/str).

    Returns:
        Node dict if found, else None.
    """

    node = idx.node_by_id.get(raw_id)
    if node and node.get("node_type"):
        return node

    if isinstance(raw_id, str) and raw_id.isdigit():
        node2 = idx.node_by_id.get(int(raw_id))
        if node2:
            return node2
    if isinstance(raw_id, int):
        node3 = idx.node_by_id.get(str(raw_id))
        if node3 and node3.get("node_type"):
            return node3
    return node


def parse_relation_pair(relation_id: Any) -> Tuple[str, str] | None:
    """Parse a relation node id to a pair of AOI ids.

    Args:
        relation_id: Usually a string like "[aoi_a, aoi_b]" or a 2-item list/tuple.

    Returns:
        Tuple of two AOI id strings if valid, else None.
    """

    try:
        if isinstance(relation_id, str) and relation_id.startswith("["):
            arr = ast.literal_eval(relation_id)
        elif isinstance(relation_id, (list, tuple)):
            arr = relation_id
        else:
            return None
        if len(arr) != 2:
            return None
        return str(arr[0]), str(arr[1])
    except Exception:
        return None


def aoi_landuse_map(idx: GraphIndex) -> Dict[str, str]:
    """Return AOI -> current land-use mapping.

    Args:
        idx: Graph index.

    Returns:
        Dict where key is AOI id string and value is land-use code.
    """

    out: Dict[str, str] = {}
    for n in idx.nodes:
        if n.get("node_type") == "aoi_node":
            out[str(n.get("id"))] = str(n.get("urban_land_use") or n.get("use") or "")
    return out


def get_graph_overview(idx: GraphIndex) -> Dict[str, Any]:
    """Summarize overall graph and AOI land-use distribution.

    Args:
        idx: Graph index.

    Returns:
        Dict including node/edge counts, node/edge type histograms, and AOI land-use histogram.
    """

    node_types = Counter(str(n.get("node_type", "")) for n in idx.nodes)
    edge_types = Counter(str(e.get("edge_type", "")) for e in idx.links)
    landuse = Counter()
    for n in idx.nodes:
        if n.get("node_type") == "aoi_node":
            lu = str(n.get("urban_land_use") or n.get("use") or "unknown")
            landuse[lu] += 1
    return {
        "node_count": len(idx.nodes),
        "edge_count": len(idx.links),
        "node_types": dict(node_types),
        "edge_types": dict(edge_types),
        "aoi_landuse_distribution": dict(landuse),
    }


def get_candidate_aois(
    idx: GraphIndex,
    modifiable_landuse_types: Iterable[str] | None = None,
    min_opinion_count: int = 1,
) -> Dict[str, Any]:
    """Find AOIs that are both modifiable and discussed by residents.

    Args:
        idx: Graph index.
        modifiable_landuse_types: Allowed source land-use types. Defaults to project constants.
        min_opinion_count: Minimum `obj_reflection` count to keep an AOI.

    Returns:
        Dict with sorted candidate AOI ids, per-AOI opinion count, and total count.
    """

    modifiable = set(modifiable_landuse_types or MODIFIABLE_LANDUSE)
    obj_reflection = idx.edges_by_type.get("obj_reflection", [])
    opinion_count = defaultdict(int)

    for e in obj_reflection:
        target = e.get("target")
        n = resolve_node(idx, target)
        if not n or n.get("node_type") != "aoi_node":
            continue
        lu = n.get("urban_land_use") or n.get("use")
        if lu in modifiable:
            opinion_count[str(n.get("id"))] += 1

    candidates = [aoi for aoi, c in opinion_count.items() if c >= min_opinion_count]
    candidates.sort(key=lambda x: opinion_count[x], reverse=True)
    return {
        "candidate_aoi_ids": candidates,
        "opinion_count": {aoi: opinion_count[aoi] for aoi in candidates},
        "count": len(candidates),
    }


def get_aoi_opinions(idx: GraphIndex, aoi_ids: Iterable[str]) -> Dict[str, Any]:
    """Collect resident comments for selected AOIs.

    Args:
        idx: Graph index.
        aoi_ids: AOI ids to query.

    Returns:
        Dict keyed by AOI id with comment list (capped) and reflection count.
    """

    wanted = {str(x) for x in aoi_ids}
    output: Dict[str, Dict[str, Any]] = {k: {"comments": [], "count": 0} for k in wanted}
    for e in idx.edges_by_type.get("obj_reflection", []):
        n = resolve_node(idx, e.get("target"))
        target = str(n.get("id")) if n and n.get("node_type") == "aoi_node" else str(e.get("target"))
        if target not in wanted:
            continue
        comments = e.get("comments") or []
        if isinstance(comments, str):
            comments = [comments]
        output[target]["comments"].extend(comments)
        output[target]["count"] += 1
    for key in output:
        output[key]["comments"] = output[key]["comments"][:30]
    return {"opinions_by_aoi": output}


def get_planning_summary(idx: GraphIndex, modifiable_landuse_types: Iterable[str] | None = None) -> Dict[str, Any]:
    """Build ranked planning summary for modifiable AOIs.

    Args:
        idx: Graph index.
        modifiable_landuse_types: Optional override for modifiable land-use set.

    Returns:
        Dict with modifiable types, AOI count with opinions, and top-ranked AOI summaries.
    """

    modifiable = set(modifiable_landuse_types or MODIFIABLE_LANDUSE)
    aoi_info: Dict[str, Dict[str, Any]] = {}

    for n in idx.nodes:
        if n.get("node_type") != "aoi_node":
            continue
        aoi_id = str(n.get("id"))
        lu = str(n.get("urban_land_use") or n.get("use") or "")
        if lu not in modifiable:
            continue
        aoi_info[aoi_id] = {
            "name": n.get("name", ""),
            "landuse": lu,
            "opinion_count": 0,
            "comments": [],
        }

    for e in idx.edges_by_type.get("obj_reflection", []):
        n = resolve_node(idx, e.get("target"))
        target = str(n.get("id")) if n and n.get("node_type") == "aoi_node" else str(e.get("target"))
        if target not in aoi_info:
            if not target.isdigit() or str(int(target)) not in aoi_info:
                continue
            target = str(int(target))
        comments = e.get("comments") or []
        if isinstance(comments, str):
            comments = [comments]
        aoi_info[target]["opinion_count"] += 1
        aoi_info[target]["comments"].extend(comments[:3])

    ranked = sorted(aoi_info.items(), key=lambda kv: kv[1]["opinion_count"], reverse=True)
    top = []
    for aoi_id, info in ranked[:50]:
        top.append(
            {
                "aoi_id": aoi_id,
                "name": info["name"],
                "landuse": info["landuse"],
                "opinion_count": info["opinion_count"],
                "sample_comments": info["comments"][:2],
            }
        )

    return {
        "modifiable_types": sorted(modifiable),
        "aoi_with_opinions": sum(1 for _, info in ranked if info["opinion_count"] > 0),
        "ranked_aois": top,
    }


def get_aoi_profile(idx: GraphIndex, aoi_id: str) -> Dict[str, Any]:
    """Return detailed profile for one AOI.

    Args:
        idx: Graph index.
        aoi_id: AOI id string.

    Returns:
        Dict including AOI basics, opinions, and linked relations; or error dict if AOI is missing.
    """

    n = resolve_node(idx, aoi_id)
    if not n or n.get("node_type") != "aoi_node":
        return {"error": f"AOI {aoi_id} not found"}

    area = n.get("area")
    lu = n.get("urban_land_use") or n.get("use")

    opinions = get_aoi_opinions(idx, [aoi_id]).get("opinions_by_aoi", {}).get(str(aoi_id), {})

    linked_relations = []
    for e in idx.edges_by_type.get("aoi_relation", []):
        if str(e.get("source")) == str(aoi_id):
            linked_relations.append(str(e.get("target")))
        elif str(e.get("target")) == str(aoi_id):
            linked_relations.append(str(e.get("source")))

    return {
        "aoi_id": str(aoi_id),
        "name": n.get("name", ""),
        "urban_land_use": lu,
        "area": area,
        "position_count": len(n.get("positions") or []),
        "opinion_count": opinions.get("count", 0),
        "sample_comments": (opinions.get("comments") or [])[:8],
        "linked_relations": linked_relations[:20],
    }


def get_resident_mobility_summary(idx: GraphIndex, top_k: int = 10) -> Dict[str, Any]:
    """Summarize resident mobility load from mobility edges.

    Args:
        idx: Graph index.
        top_k: Number of residents to keep in ranking output.

    Returns:
        Dict with resident count and top residents by trip count and travel minutes.
    """

    move_counts = Counter()
    durations = defaultdict(float)

    for e in idx.edges_by_type.get("mobility", []):
        src = str(e.get("source"))
        move_counts[src] += 1
        if isinstance(e.get("mobility_minutes"), (int, float)):
            move_min = float(e.get("mobility_minutes") or 0.0)
        else:
            info = e.get("mobility_info")
            if isinstance(info, list):
                move_min = 0.0
                for item in info:
                    if isinstance(item, dict):
                        move_min += float(item.get("move_minute") or 0)
            elif isinstance(info, dict):
                move_min = float(info.get("move_minute") or 0)
            else:
                move_min = 0.0
        durations[src] += move_min

    top = []
    for rid, count in move_counts.most_common(top_k):
        top.append(
            {
                "resident_id": rid,
                "trip_count": count,
                "total_move_minutes": round(durations[rid], 2),
            }
        )

    return {
        "resident_count_with_trips": len(move_counts),
        "top_residents_by_trip_count": top,
    }
