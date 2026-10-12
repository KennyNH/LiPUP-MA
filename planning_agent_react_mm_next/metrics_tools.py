"""Planning metrics tools with graceful fallback."""

from __future__ import annotations

import json
import math
import pickle
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import Counter, defaultdict
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

import networkx as nx
from shapely.geometry import Polygon

try:
    from .graph_tools import GraphIndex, parse_relation_pair
    from .landuse import LANDUSE_LABELS, MODIFIABLE_LANDUSE, validate_plan_mapping
except ImportError:
    from graph_tools import GraphIndex, parse_relation_pair
    from landuse import LANDUSE_LABELS, MODIFIABLE_LANDUSE, validate_plan_mapping


SERVICE_TYPES = ("BS", "RC", "OF", "SC", "HS", "ID")
SATISFACTION_CANDIDATE_TYPES = ("BS", "RC", "PK", "OF", "SC", "HS")
DEFAULT_SERVICE_RADIUS = 500.0
DEFAULT_ID_SERVICE_RADIUS = 1000.0
DEFAULT_ECOLOGY_RADIUS = 300.0

LLMJSONFn = Callable[[str], Optional[Dict[str, Any]]]


class WalkingRoadDistance:
    """Road-network distance index built from map pickle lanes + AOI walking_positions."""

    def __init__(self, map_pickle_path: str):
        m = pickle.load(open(map_pickle_path, "rb"))
        lanes = m.get("lanes") or []
        aois = m.get("aois") or []
        self.aoi_walk_positions: Dict[str, List[Tuple[int, float]]] = {}
        self.aoi_centroid_xy: Dict[str, Tuple[float, float]] = {}
        self.aoi_anchor_xy: Dict[str, Tuple[float, float]] = {}
        self.lane_length: Dict[int, float] = {}
        self.lane_endpoint_xy: Dict[int, Tuple[Tuple[float, float], Tuple[float, float]]] = {}
        self._sp_cache: Dict[Tuple[str, str], float] = {}
        self._bridge_cache: Dict[Tuple[int, int], Tuple[str, str, float]] = {}
        self.endpoint_xy: Dict[str, Tuple[float, float]] = {}
        self.graph = nx.Graph()
        self.global_anchor_xy: Optional[Tuple[float, float]] = None

        for lane in lanes:
            lid = int(lane.get("id"))
            ll = float(lane.get("length") or 0.0)
            self.lane_length[lid] = max(ll, 0.0)
            s = f"{lid}:S"
            e = f"{lid}:E"
            # Allow both directions on the same lane segment for robust pedestrian distance.
            self.graph.add_edge(s, e, weight=self.lane_length[lid])
            c = lane.get("center_line") or {}
            nodes = c.get("nodes") or []
            if isinstance(nodes, list) and len(nodes) >= 2:
                p0, p1 = nodes[0], nodes[-1]
                try:
                    self.endpoint_xy[s] = (float(p0.get("x")), float(p0.get("y")))
                    self.endpoint_xy[e] = (float(p1.get("x")), float(p1.get("y")))
                    self.lane_endpoint_xy[lid] = (self.endpoint_xy[s], self.endpoint_xy[e])
                except Exception:
                    pass

        lane_ids = set(self.lane_length.keys())
        for lane in lanes:
            lid = int(lane.get("id"))
            e = f"{lid}:E"
            s = f"{lid}:S"
            for succ in lane.get("successors") or []:
                sid = succ.get("id")
                if sid is None:
                    continue
                sid = int(sid)
                if sid in lane_ids:
                    self.graph.add_edge(e, f"{sid}:S", weight=0.0)
            for pred in lane.get("predecessors") or []:
                pid = pred.get("id")
                if pid is None:
                    continue
                pid = int(pid)
                if pid in lane_ids:
                    self.graph.add_edge(s, f"{pid}:E", weight=0.0)

        for a in aois:
            aid = str(a.get("id"))
            wps = []
            for w in (a.get("walking_positions") or []):
                lid = w.get("lane_id")
                s = w.get("s")
                if lid is None or s is None:
                    continue
                lid = int(lid)
                if lid not in lane_ids:
                    continue
                s_f = float(s)
                s_f = max(0.0, min(s_f, self.lane_length.get(lid, s_f)))
                wps.append((lid, s_f))
            if wps:
                self.aoi_walk_positions[aid] = wps
            pts = a.get("positions") or []
            if isinstance(pts, list) and pts:
                xs, ys = [], []
                for p in pts:
                    try:
                        xs.append(float(p.get("x")))
                        ys.append(float(p.get("y")))
                    except Exception:
                        pass
                if xs and ys:
                    self.aoi_centroid_xy[aid] = (sum(xs) / len(xs), sum(ys) / len(ys))
            anchor = self.aoi_centroid_xy.get(aid)
            if anchor is None:
                anchor = self._anchor_from_lane_positions(a.get("walking_positions") or [])
            if anchor is None:
                anchor = self._anchor_from_lane_positions(a.get("driving_positions") or [])
            if anchor is not None:
                self.aoi_anchor_xy[aid] = anchor

        if self.endpoint_xy:
            vals = list(self.endpoint_xy.values())
            self.global_anchor_xy = (
                sum(p[0] for p in vals) / float(len(vals)),
                sum(p[1] for p in vals) / float(len(vals)),
            )

        self.node_comp: Dict[str, int] = {}
        self.comp_nodes: Dict[int, List[str]] = defaultdict(list)
        for cid, nodes in enumerate(nx.connected_components(self.graph)):
            for n in nodes:
                self.node_comp[n] = cid
                self.comp_nodes[cid].append(n)

    def _position_endpoint_costs(self, lane_id: int, s_val: float) -> List[Tuple[str, float]]:
        ll = self.lane_length.get(lane_id)
        if ll is None:
            return []
        s_val = max(0.0, min(float(s_val), ll))
        return [(f"{lane_id}:S", s_val), (f"{lane_id}:E", max(ll - s_val, 0.0))]

    def _shortest_nodes(self, n1: str, n2: str) -> Optional[float]:
        key = (n1, n2)
        if key in self._sp_cache:
            return self._sp_cache[key]
        try:
            d = float(nx.shortest_path_length(self.graph, n1, n2, weight="weight"))
            self._sp_cache[key] = d
            return d
        except Exception:
            return None

    @staticmethod
    def _euclidean_xy(p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
        return math.hypot(float(p1[0]) - float(p2[0]), float(p1[1]) - float(p2[1]))

    def _euclidean_nodes(self, n1: str, n2: str) -> Optional[float]:
        p1 = self.endpoint_xy.get(n1)
        p2 = self.endpoint_xy.get(n2)
        if p1 is None or p2 is None:
            return None
        return self._euclidean_xy(p1, p2)

    def _lane_point_xy(self, lane_id: int, s_val: float) -> Optional[Tuple[float, float]]:
        seg = self.lane_endpoint_xy.get(int(lane_id))
        ll = self.lane_length.get(int(lane_id))
        if seg is None or ll is None or ll <= 0:
            return None
        p0, p1 = seg
        t = max(0.0, min(float(s_val) / float(ll), 1.0))
        return (p0[0] + (p1[0] - p0[0]) * t, p0[1] + (p1[1] - p0[1]) * t)

    def _anchor_from_lane_positions(self, positions: List[Dict[str, Any]]) -> Optional[Tuple[float, float]]:
        pts: List[Tuple[float, float]] = []
        for item in positions or []:
            try:
                lid = int(item.get("lane_id"))
                s = float(item.get("s"))
            except Exception:
                continue
            p = self._lane_point_xy(lid, s)
            if p is not None:
                pts.append(p)
        if not pts:
            return None
        return (sum(p[0] for p in pts) / float(len(pts)), sum(p[1] for p in pts) / float(len(pts)))

    def _closest_bridge(self, c1: int, c2: int) -> Optional[Tuple[str, str, float]]:
        key = (c1, c2) if c1 <= c2 else (c2, c1)
        if key in self._bridge_cache:
            return self._bridge_cache[key]
        best: Optional[Tuple[str, str, float]] = None
        n1s = self.comp_nodes.get(c1) or []
        n2s = self.comp_nodes.get(c2) or []
        for n1 in n1s:
            p1 = self.endpoint_xy.get(n1)
            if p1 is None:
                continue
            for n2 in n2s:
                p2 = self.endpoint_xy.get(n2)
                if p2 is None:
                    continue
                d = self._euclidean_xy(p1, p2)
                if best is None or d < best[2]:
                    best = (n1, n2, d)
        if best is not None:
            self._bridge_cache[key] = best
        return best

    def _shortest_nodes_or_estimate(self, n1: str, n2: str) -> Tuple[Optional[float], str]:
        d = self._shortest_nodes(n1, n2)
        if d is not None:
            return d, "road_shortest_path"
        c1 = self.node_comp.get(n1)
        c2 = self.node_comp.get(n2)
        if c1 is not None and c2 is not None and c1 != c2:
            bridge = self._closest_bridge(c1, c2)
            if bridge is not None:
                b1, b2, bdist = bridge
                d1 = self._shortest_nodes(n1, b1)
                d2 = self._shortest_nodes(b2, n2)
                if d1 is None:
                    d1 = self._euclidean_nodes(n1, b1) or 0.0
                if d2 is None:
                    d2 = self._euclidean_nodes(b2, n2) or 0.0
                # Use a detour factor for disconnected-component bridge estimation.
                return (d1 + 1.35 * bdist + d2), "component_bridge_estimate"
        direct = self._euclidean_nodes(n1, n2)
        if direct is not None:
            return 1.35 * direct, "direct_endpoint_estimate"
        return None, "unavailable"

    def _centroid_fallback(self, aoi_src: str, aoi_dst: str) -> Tuple[Optional[float], str]:
        if str(aoi_src) == str(aoi_dst):
            return 0.0, "same_aoi"
        p1 = self.aoi_anchor_xy.get(str(aoi_src)) or self.aoi_centroid_xy.get(str(aoi_src))
        p2 = self.aoi_anchor_xy.get(str(aoi_dst)) or self.aoi_centroid_xy.get(str(aoi_dst))
        if p1 is None:
            p1 = self.global_anchor_xy
        if p2 is None:
            p2 = self.global_anchor_xy
        if p1 is None or p2 is None:
            return 0.0, "global_constant_estimate"
        return 1.35 * self._euclidean_xy(p1, p2), "aoi_anchor_estimate"

    def distance_aoi_to_aoi(self, aoi_src: str, aoi_dst: str) -> Optional[float]:
        d, _ = self.distance_aoi_to_aoi_with_meta(aoi_src, aoi_dst)
        return d

    def distance_aoi_to_aoi_with_meta(self, aoi_src: str, aoi_dst: str) -> Tuple[Optional[float], str]:
        src_wps = self.aoi_walk_positions.get(str(aoi_src)) or []
        dst_wps = self.aoi_walk_positions.get(str(aoi_dst)) or []
        if not src_wps or not dst_wps:
            return self._centroid_fallback(str(aoi_src), str(aoi_dst))
        best = None
        best_mode = "unavailable"
        for lid1, s1 in src_wps:
            ends1 = self._position_endpoint_costs(lid1, s1)
            if not ends1:
                continue
            for lid2, s2 in dst_wps:
                ends2 = self._position_endpoint_costs(lid2, s2)
                if not ends2:
                    continue
                for n1, c1 in ends1:
                    for n2, c2 in ends2:
                        dmid, mode = self._shortest_nodes_or_estimate(n1, n2)
                        if dmid is None:
                            continue
                        d = c1 + dmid + c2
                        if best is None or d < best:
                            best = d
                            best_mode = mode
        if best is None:
            return self._centroid_fallback(str(aoi_src), str(aoi_dst))
        return best, best_mode


def _build_aoi_polygons_xy(idx: GraphIndex):
    aoi_geom: Dict[str, Polygon] = {}
    aoi_landuse: Dict[str, str] = {}
    for n in idx.nodes:
        if n.get("node_type") != "aoi_node":
            continue
        aid = str(n.get("id"))
        pos = n.get("positions") or []
        coords = []
        for p in pos:
            try:
                coords.append((float(p["x"]), float(p["y"])))
            except Exception:
                pass
        if len(coords) >= 3:
            try:
                aoi_geom[aid] = Polygon(coords)
                aoi_landuse[aid] = str(n.get("urban_land_use") or n.get("use") or "")
            except Exception:
                continue
    return aoi_geom, aoi_landuse


def _resident_home_aois(idx: GraphIndex) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for n in idx.nodes:
        if n.get("node_type") != "resident_node":
            continue
        rid = str(n.get("id"))
        home = n.get("home_aoi_id")
        if home is not None:
            out[rid] = str(home)
    return out


def _build_landuse_spatial_index(aoi_geom: Dict[str, Polygon], aoi_landuse: Dict[str, str], landuse: str):
    geoms = [aoi_geom[aid] for aid, lu in aoi_landuse.items() if lu == landuse and aid in aoi_geom]
    if not geoms:
        return None
    return geoms


def _geometry_distance(src_geom: Polygon, dst_geom: Polygon) -> float:
    return float(src_geom.distance(dst_geom))


def _has_landuse_within_distance(src_geom: Polygon, index_obj, radius: float) -> bool:
    if index_obj is None:
        return False
    minx, miny, maxx, maxy = src_geom.bounds
    qminx, qminy, qmaxx, qmaxy = minx - radius, miny - radius, maxx + radius, maxy + radius
    for g in index_obj:
        gx1, gy1, gx2, gy2 = g.bounds
        # quick bbox rejection before exact geometry distance
        if gx2 < qminx or gx1 > qmaxx or gy2 < qminy or gy1 > qmaxy:
            continue
        if _geometry_distance(src_geom, g) <= radius:
            return True
    return False


def _safe_snippet(text: str, max_chars: int = 200) -> str:
    t = " ".join((text or "").split())
    if len(t) <= max_chars:
        return t
    return t[:max_chars] + "..."


def _parse_mobility_info(mobility_info: Any) -> Tuple[Optional[int], float]:
    if isinstance(mobility_info, list):
        if len(mobility_info) >= 2 and isinstance(mobility_info[0], (int, float)):
            try:
                return int(mobility_info[0]), float(mobility_info[1] or 0.0)
            except Exception:
                return None, 0.0
        if mobility_info and isinstance(mobility_info[0], dict):
            start_t = None
            total_move = 0.0
            for item in mobility_info:
                if not isinstance(item, dict):
                    continue
                if start_t is None and isinstance(item.get("start_t"), (int, float)):
                    start_t = int(item["start_t"])
                total_move += float(item.get("move_minute") or 0.0)
            return start_t, total_move
    if isinstance(mobility_info, dict):
        try:
            st = mobility_info.get("start_t")
            return (int(st) if isinstance(st, (int, float)) else None), float(mobility_info.get("move_minute") or 0.0)
        except Exception:
            return None, 0.0
    return None, 0.0


def _parse_mobility_edge(edge: Dict[str, Any]) -> Tuple[Optional[int], float, Optional[str], Optional[float]]:
    """Parse mobility fields from either new edge schema or legacy `mobility_info` schema."""
    st = edge.get("start_timestamp")
    mv = edge.get("mobility_minutes")
    intention = edge.get("intention")
    distance_m = edge.get("distance_m")

    start_t: Optional[int] = int(st) if isinstance(st, (int, float)) else None
    move_minute: float = float(mv or 0.0) if isinstance(mv, (int, float)) else 0.0

    # Backward-compatible fallback to legacy mobility_info.
    if start_t is None and move_minute <= 0.0:
        start_t, move_minute = _parse_mobility_info(edge.get("mobility_info"))

    if intention is not None:
        intention = str(intention)
    if isinstance(distance_m, (int, float)):
        distance_val: Optional[float] = float(distance_m)
    else:
        distance_val = None
    return start_t, move_minute, intention, distance_val


def _extract_resident_evidence(
    idx: GraphIndex,
    aoi_geom: Dict[str, Polygon],
    aoi_landuse: Dict[str, str],
) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}

    for n in idx.nodes:
        if n.get("node_type") == "resident_node":
            rid = str(n.get("id"))
            out[rid] = {
                "resident_id": rid,
                "home_aoi_id": str(n.get("home_aoi_id")) if n.get("home_aoi_id") is not None else None,
                "work_aoi_id": str(n.get("work_aoi_id")) if n.get("work_aoi_id") is not None else None,
                "trajectories": [],
                "reflections": [],
                "trip_target_landuse_count": Counter(),
                "reflection_landuse_count": Counter(),
            }

    for e in idx.edges_by_type.get("mobility", []):
        rid = str(e.get("source"))
        if rid not in out:
            continue
        pair = parse_relation_pair(e.get("target"))
        if not pair:
            continue
        a0, a1 = str(pair[0]), str(pair[1])
        if a0 not in aoi_geom or a1 not in aoi_geom:
            continue
        start_t, move_minute, intention, distance_m = _parse_mobility_edge(e)
        boundary_dist = _geometry_distance(aoi_geom[a0], aoi_geom[a1])
        # Prefer explicit mobility edge distance when available in new graph schema.
        if isinstance(distance_m, (int, float)) and float(distance_m) >= 0.0:
            dist = float(distance_m)
            dist_source = "mobility_edge_distance_m"
        else:
            dist = boundary_dist
            dist_source = "aoi_boundary_distance"
        rec = {
            "from_aoi_id": a0,
            "to_aoi_id": a1,
            "distance": dist,
            "distance_boundary": boundary_dist,
            "distance_source": dist_source,
            "move_minute": move_minute,
            "start_t": start_t,
            "intention": intention,
            "distance_m": distance_m,
            "to_landuse": aoi_landuse.get(a1, ""),
        }
        out[rid]["trajectories"].append(rec)
        out[rid]["trip_target_landuse_count"][aoi_landuse.get(a1, "")] += 1

    for e in idx.edges_by_type.get("obj_reflection", []):
        src = str(e.get("source"))
        tgt = str(e.get("target"))
        rid = src if src in out else (tgt if tgt in out else "")
        if rid not in out:
            continue
        target_aoi = tgt if rid == src else src
        comments = e.get("comments") or []
        if isinstance(comments, str):
            comments = [comments]
        landuse = aoi_landuse.get(target_aoi, "")
        out[rid]["reflections"].append(
            {
                "target_aoi_id": target_aoi,
                "target_landuse": landuse,
                "day": e.get("day"),
                "t": e.get("t"),
                # Keep full reflection comments for LLM-side preference inference.
                "comments": comments,
            }
        )
        out[rid]["reflection_landuse_count"][landuse] += 1

    # Counter -> dict for stable serialization
    for rid, rec in out.items():
        rec["trip_target_landuse_count"] = dict(rec["trip_target_landuse_count"])
        rec["reflection_landuse_count"] = dict(rec["reflection_landuse_count"])
    return out


def _parse_json_object_from_text(text: str) -> Optional[Dict[str, Any]]:
    t = (text or "").strip()
    if not t:
        return None
    t = re.sub(r"^```(?:json)?\s*", "", t)
    t = re.sub(r"\s*```$", "", t)
    try:
        obj = json.loads(t)
        return obj if isinstance(obj, dict) else None
    except Exception:
        pass
    m = re.search(r"\{[\s\S]*\}", t)
    if not m:
        return None
    try:
        obj = json.loads(m.group(0))
        return obj if isinstance(obj, dict) else None
    except Exception:
        return None


def _choose_preferences_heuristic(
    resident_evidence: Dict[str, Dict[str, Any]],
) -> Dict[str, List[str]]:
    prefs: Dict[str, List[str]] = {}
    global_score = Counter()
    for rid, ev in resident_evidence.items():
        score = Counter()
        for lu, c in (ev.get("trip_target_landuse_count") or {}).items():
            if lu in SATISFACTION_CANDIDATE_TYPES:
                score[lu] += int(c)
        for lu, c in (ev.get("reflection_landuse_count") or {}).items():
            if lu in SATISFACTION_CANDIDATE_TYPES:
                score[lu] += int(c) * 2
        signal = sum(score.values())
        if signal <= 1:
            want_n = 1
        elif signal <= 3:
            want_n = 2
        elif signal <= 6:
            want_n = 3
        elif signal <= 10:
            want_n = 4
        elif signal <= 16:
            want_n = 5
        else:
            want_n = 6
        want_n = max(1, min(6, want_n))
        choice = [lu for lu, _ in score.most_common(want_n)]
        prefs[rid] = choice
        global_score.update(score)
    fallback = [lu for lu, _ in global_score.most_common(3)] or list(SATISFACTION_CANDIDATE_TYPES[:3])
    for rid in resident_evidence:
        if len(prefs.get(rid) or []) < 1:
            merged = list(dict.fromkeys((prefs.get(rid) or []) + fallback))
            prefs[rid] = merged[:3]
    return prefs


def prepare_resident_preference_cache(
    idx: GraphIndex,
    resident_profiles: Dict[str, Dict[str, Any]] | None = None,
    llm_json_fn: Optional[LLMJSONFn] = None,
    llm_required: bool = False,
) -> Dict[str, Any]:
    """Build one-time resident service preference cache for satisfaction metric."""
    aoi_geom, aoi_landuse = _build_aoi_polygons_xy(idx)
    evidence = _extract_resident_evidence(idx, aoi_geom, aoi_landuse)
    resident_ids = sorted(evidence.keys(), key=lambda x: int(x) if x.isdigit() else x)
    profiles = resident_profiles or {}

    if llm_required and not llm_json_fn:
        raise RuntimeError("prepare_resident_preference_cache requires llm_json_fn when llm_required=True")

    prefs = _choose_preferences_heuristic(evidence)
    source = "heuristic"

    if llm_json_fn and resident_ids:
        payload = []
        for rid in resident_ids:
            ev = evidence[rid]
            bg = profiles.get(rid, {})
            payload.append(
                {
                    "resident_id": rid,
                    "home_aoi_id": ev.get("home_aoi_id"),
                    "work_aoi_id": ev.get("work_aoi_id"),
                    # Keep complete resident profile context (no truncation) to improve preference inference.
                    "background": bg if isinstance(bg, dict) else {"raw_profile": bg},
                    # Keep both aggregated and detailed mobility/reflection evidence.
                    "trip_target_landuse_count": ev.get("trip_target_landuse_count", {}),
                    "reflection_landuse_count": ev.get("reflection_landuse_count", {}),
                    "trajectories": ev.get("trajectories", []),
                    "reflections": ev.get("reflections", []),
                }
            )
        landuse_full = {k: LANDUSE_LABELS.get(k, k) for k in SATISFACTION_CANDIDATE_TYPES}
        prompt = (
            "You are an urban planning analysis assistant. For each resident, infer the most needed service land-use types from background, trajectories, and reflections.\n"
            f"Candidate types are restricted to: {list(SATISFACTION_CANDIDATE_TYPES)}.\n"
            f"Candidate type full names: {json.dumps(landuse_full, ensure_ascii=False)}.\n"
            "Select 1-6 types per resident, sorted by importance, without duplicates. Usually 2-4 types are enough.\n"
            "Think step by step internally, then output JSON only, without showing reasoning.\n"
            "Output format:\n"
            "{\n"
            "  \"resident_preferences\": {\n"
            "    \"resident_id\": [\"type1\", \"type2\"]\n"
            "  }\n"
            "}\n"
            f"Input data:\n{json.dumps(payload, ensure_ascii=False)}"
        )
        obj = llm_json_fn(prompt) or {}
        llm_map = obj.get("resident_preferences") if isinstance(obj, dict) else None
        if isinstance(llm_map, dict):
            merged = dict(prefs)
            valid_count = 0
            for rid in resident_ids:
                arr = llm_map.get(rid)
                if arr is None and rid.isdigit():
                    arr = llm_map.get(int(rid))
                if not isinstance(arr, list):
                    continue
                cleaned = []
                for v in arr:
                    lu = str(v)
                    if lu in SATISFACTION_CANDIDATE_TYPES and lu not in cleaned:
                        cleaned.append(lu)
                    if len(cleaned) >= 6:
                        break
                if len(cleaned) >= 1:
                    valid_count += 1
                    if len(cleaned) < 2:
                        refill = [x for x in merged.get(rid, []) if x not in cleaned]
                        cleaned.extend(refill[: max(0, 2 - len(cleaned))])
                    merged[rid] = cleaned[:6]
            if valid_count > 0:
                prefs = merged
                source = "llm+heuristic_fallback"

    if llm_required and source == "heuristic":
        raise RuntimeError("prepare_resident_preference_cache failed to produce LLM preferences")

    return {
        "resident_preferences": prefs,
        "source": source,
        "resident_count": len(resident_ids),
        "candidate_types": list(SATISFACTION_CANDIDATE_TYPES),
    }


def _normalize_preference_map(
    resident_preferences: Dict[str, List[str]] | None,
    resident_ids: Iterable[str],
) -> Dict[str, List[str]]:
    default_pref = list(SATISFACTION_CANDIDATE_TYPES[:3])
    out: Dict[str, List[str]] = {}
    for rid in resident_ids:
        arr = (resident_preferences or {}).get(rid) or []
        cleaned = []
        for v in arr:
            lu = str(v)
            if lu in SATISFACTION_CANDIDATE_TYPES and lu not in cleaned:
                cleaned.append(lu)
            if len(cleaned) >= 6:
                break
        if not cleaned:
            cleaned = default_pref
        out[rid] = cleaned
    return out


def _has_landuse_within_road_distance(
    src_aoi: str,
    target_aois: Iterable[str],
    radius: float,
    road_model: WalkingRoadDistance,
) -> bool:
    for tid in target_aois:
        if str(tid) == str(src_aoi):
            return True
        d = road_model.distance_aoi_to_aoi(str(src_aoi), str(tid))
        if d is not None and d <= radius:
            return True
    return False


def evaluate_accessibility_proxy(
    idx: GraphIndex,
    service_radius: float = DEFAULT_SERVICE_RADIUS,
    id_service_radius: float = DEFAULT_ID_SERVICE_RADIUS,
    ecology_radius: float = DEFAULT_ECOLOGY_RADIUS,
    landuse_override: Dict[str, str] | None = None,
    resident_preferences: Dict[str, List[str]] | None = None,
    road_distance_model: Optional[WalkingRoadDistance] = None,
) -> Dict[str, Any]:
    """Evaluate service/ecology/satisfaction from AOI geometry and optional road model.

    Distance basis:
    - Service + Satisfaction: road-network AOI-to-AOI distance when `road_distance_model` is available,
      otherwise boundary distance from AOI polygons (`aoi_node.positions`).
    - Ecology: AOI boundary distance in positions XY.
    """
    aoi_geom, aoi_landuse = _build_aoi_polygons_xy(idx)
    if landuse_override:
        for aid, lu in landuse_override.items():
            aid_s = str(aid)
            if aid_s in aoi_landuse:
                aoi_landuse[aid_s] = str(lu)
    resident_home = _resident_home_aois(idx)
    home_items = [(rid, hid) for rid, hid in resident_home.items() if hid in aoi_geom]
    if not home_items:
        return {
            "metrics": {"service": 0.0, "ecology": 0.0, "satisfaction": 0.0},
            "resident_sample": 0,
            "method": "positions_xy_boundary_distance",
        }

    idx_by_landuse = {lu: _build_landuse_spatial_index(aoi_geom, aoi_landuse, lu) for lu in set(("PK", "ID"))}
    aois_by_landuse = defaultdict(list)
    for aid, lu in aoi_landuse.items():
        aois_by_landuse[lu].append(aid)
    pref_map = _normalize_preference_map(resident_preferences, [rid for rid, _ in home_items])

    service_scores = []
    ecology_scores = []
    satisfaction_scores = []
    for rid, home_aoi in home_items:
        src = aoi_geom[home_aoi]

        reached = 0
        for lu in SERVICE_TYPES:
            radius_for_type = float(id_service_radius if lu == "ID" else service_radius)
            if road_distance_model is not None:
                ok = _has_landuse_within_road_distance(home_aoi, aois_by_landuse.get(lu, []), radius_for_type, road_distance_model)
            else:
                lu_idx = _build_landuse_spatial_index(aoi_geom, aoi_landuse, lu)
                ok = _has_landuse_within_distance(src, lu_idx, radius_for_type)
            if ok:
                reached += 1
        service_scores.append(reached / float(len(SERVICE_TYPES)))

        has_pk = _has_landuse_within_distance(src, idx_by_landuse.get("PK"), ecology_radius)
        has_id = _has_landuse_within_distance(src, idx_by_landuse.get("ID"), ecology_radius)
        cond_a = has_pk
        cond_b = not has_id
        if cond_a and cond_b:
            eco = 1.0
        elif cond_a or cond_b:
            eco = 0.5
        else:
            eco = 0.0
        ecology_scores.append(eco)

        pref_types = pref_map.get(rid, list(SATISFACTION_CANDIDATE_TYPES[:3]))
        sat_reach = 0
        for lu in pref_types:
            radius_for_type = float(id_service_radius if lu == "ID" else service_radius)
            if road_distance_model is not None:
                ok = _has_landuse_within_road_distance(home_aoi, aois_by_landuse.get(lu, []), radius_for_type, road_distance_model)
            else:
                lu_idx = _build_landuse_spatial_index(aoi_geom, aoi_landuse, lu)
                ok = _has_landuse_within_distance(src, lu_idx, radius_for_type)
            if ok:
                sat_reach += 1
        satisfaction_scores.append(sat_reach / float(max(len(pref_types), 1)))

    service = sum(service_scores) / len(service_scores)
    ecology = sum(ecology_scores) / len(ecology_scores)
    satisfaction = sum(satisfaction_scores) / len(satisfaction_scores)
    return {
        "metrics": {"service": round(service, 4), "ecology": round(ecology, 4), "satisfaction": round(satisfaction, 4)},
        "resident_sample": len(home_items),
        "distance_basis": {
            "service": "road_network_aoi_distance" if road_distance_model is not None else "aoi_boundary_distance",
            "satisfaction": "road_network_aoi_distance" if road_distance_model is not None else "aoi_boundary_distance",
            "ecology": "aoi_boundary_distance",
        },
        "params": {
            "service_radius_m": service_radius,
            "id_service_radius_m": id_service_radius,
            "ecology_radius_m": ecology_radius,
            "service_types": list(SERVICE_TYPES),
            "satisfaction_candidate_types": list(SATISFACTION_CANDIDATE_TYPES),
        },
    }


SUBJECTIVE_NEG_WORDS = [
    "bad", "poor", "worse", "worst", "inconvenient", "unsafe", "noisy", "crowded", "dirty", "pollution",
    "far", "farther", "lack", "lacks", "lacking", "without", "limited", "isolated", "difficult", "stress",
    "stressful", "congestion", "issue", "issues", "problem", "concern", "missing", "beyond", "dissatisfied",
    "uncomfortable", "not enough", "no nearby", "no immediate", "have to travel", "travel further",
    "travel farther", "longer trips", "longer drives", "could be improved", "needs improvement",
    "wish there were", "would appreciate", "need better", "need more", "rely on", "over 1.2 km",
    "拥堵", "吵", "脏", "污染", "不便", "不满意", "糟糕", "危险", "远", "缺少", "不足", "不方便",
]

SENTIMENT_POS_WORDS = [
    "good", "great", "nice", "convenient", "clean", "safe", "quiet", "pleasant", "comfortable", "satisfied",
    "smoothly", "well-suited", "well located", "easy access", "efficient", "acceptable", "appreciate",
    "方便", "不错", "满意", "舒适", "干净", "安全", "安静", "喜欢", "很好", "优质",
]

SENTIMENT_NEG_WORDS = SUBJECTIVE_NEG_WORDS + [
    "negative", "disappointed", "frustrating", "burden", "cost", "time-consuming", "too far", "not close",
]


def _count_terms(text: str, terms: List[str]) -> int:
    return sum(str(text or "").count(term) for term in terms)


def _score_subjective_interview_heuristic(ev: Dict[str, Any]) -> Optional[int]:
    texts = []
    for ref in ev.get("reflections") or []:
        comments = ref.get("comments") or []
        if isinstance(comments, str):
            comments = [comments]
        texts.extend([str(c) for c in comments])
    if not texts and not (ev.get("trajectories") or []):
        return None
    neg_comment_count = sum(1 for text in texts if _count_terms(str(text).lower(), SUBJECTIVE_NEG_WORDS) > 0)
    if neg_comment_count <= 0:
        return 5
    neg_ratio = neg_comment_count / max(len(texts), 1)
    if neg_ratio <= 0.35:
        return 4
    if neg_ratio <= 0.48:
        return 3
    if neg_ratio <= 0.55:
        return 2
    return 1


def _infer_subjective_interview_scores(
    resident_evidence: Dict[str, Dict[str, Any]],
    resident_profiles: Dict[str, Dict[str, Any]] | None = None,
    last_resident_evidence: Dict[str, Dict[str, Any]] | None = None,
    llm_json_fn: Optional[LLMJSONFn] = None,
    llm_required: bool = False,
    max_workers: int = 300,
) -> Dict[str, Dict[str, Any]]:
    profiles = resident_profiles or {}
    result: Dict[str, Dict[str, Any]] = {}
    for rid, ev in resident_evidence.items():
        score = _score_subjective_interview_heuristic(ev)
        result[rid] = {
            "score": score,
            "source": "heuristic" if score is not None else "none",
            "evidence": {
                "trajectory_count": len(ev.get("trajectories") or []),
                "reflection_count": sum(
                    len((ref.get("comments") if isinstance(ref.get("comments"), list) else [ref.get("comments")]) or [])
                    for ref in (ev.get("reflections") or [])
                ),
            },
        }

    if llm_required and not llm_json_fn:
        raise RuntimeError("_infer_subjective_interview_scores requires llm_json_fn when llm_required=True")
    if not llm_json_fn:
        return result

    landuse_full = {k: LANDUSE_LABELS.get(k, k) for k in sorted(LANDUSE_LABELS.keys())}

    def _score_one(rid: str) -> Tuple[str, Optional[Dict[str, Any]]]:
        ev = resident_evidence[rid]
        bg = profiles.get(rid, {})
        last_ev = (last_resident_evidence or {}).get(rid)
        payload = {
            "resident_id": rid,
            "home_aoi_id": ev.get("home_aoi_id"),
            "work_aoi_id": ev.get("work_aoi_id"),
            "background": bg if isinstance(bg, dict) else {"raw_profile": bg},
            "trip_count": len(ev.get("trajectories") or []),
            "avg_trip_distance": round(
                sum(float(t.get("distance") or 0.0) for t in (ev.get("trajectories") or []))
                / max(len(ev.get("trajectories") or []), 1),
                3,
            ),
            "trip_target_landuse_count": ev.get("trip_target_landuse_count", {}),
            "reflection_landuse_count": ev.get("reflection_landuse_count", {}),
            "trajectories": ev.get("trajectories", []),
            "reflections": ev.get("reflections", []),
        }
        if last_ev:
            payload["last_simulation_reflections"] = last_ev.get("reflections", [])
        prompt = (
            "You are an urban life evaluator. Based on resident background, trajectories, and reflections, score this resident's lived experience.\n"
            "Scores must be integers from 1 to 5, where 5 means highly satisfied and 1 means highly dissatisfied.\n"
            "When last_simulation_reflections are provided, use them as the previous simulation baseline for the same resident. "
            "If the current reflections show that problems raised in the previous simulation have been solved or clearly improved, reward that improvement with a higher satisfaction score when justified by the current evidence.\n"
            f"Land-use code full names: {json.dumps(landuse_full, ensure_ascii=False)}.\n"
            "Think internally, then output JSON only, without showing reasoning.\n"
            "Output format: {\"score\": 1-5, \"reason\": \"short reason\"}\n"
            f"Input data:\n{json.dumps(payload, ensure_ascii=False)}"
        )
        obj = llm_json_fn(prompt) or {}
        return rid, obj if isinstance(obj, dict) else None

    score_items: Dict[str, Dict[str, Any]] = {}
    resident_ids = sorted(resident_evidence.keys(), key=lambda x: int(x) if x.isdigit() else x)
    with ThreadPoolExecutor(max_workers=max(1, min(int(max_workers), max(len(resident_ids), 1)))) as ex:
        futures = [ex.submit(_score_one, rid) for rid in resident_ids]
        for fut in as_completed(futures):
            rid, item = fut.result()
            if isinstance(item, dict):
                score_items[rid] = item

    if llm_required and not score_items:
        raise RuntimeError("_infer_subjective_interview_scores failed: invalid LLM JSON")

    for rid, item in score_items.items():
        score = item.get("score")
        try:
            score_i = int(score)
        except Exception:
            continue
        if 1 <= score_i <= 5:
            result[rid] = {
                "score": score_i,
                "reason": str(item.get("reason") or ""),
                "source": "llm",
            }
    if llm_required:
        llm_cnt = sum(1 for v in result.values() if str(v.get("source")) == "llm")
        if llm_cnt <= 0:
            raise RuntimeError("_infer_subjective_interview_scores failed: no valid LLM scores")
    return result


def _score_reflection_text_sentiment_heuristic(text: str) -> int:
    """Heuristic sentiment score for a single reflection text in 5-level scale [-2, 2]."""
    t = str(text or "").strip().lower()
    if not t:
        return 0

    pos = _count_terms(t, SENTIMENT_POS_WORDS)
    neg = _count_terms(t, SENTIMENT_NEG_WORDS)
    delta = pos - neg
    if delta >= 2:
        return 2
    if delta == 1:
        return 1
    if delta == 0:
        return 0
    if delta == -1:
        return -1
    return -2


def _score_reflection_sentiment_by_resident(
    resident_evidence: Dict[str, Dict[str, Any]],
    last_resident_evidence: Dict[str, Dict[str, Any]] | None = None,
    llm_json_fn: Optional[LLMJSONFn] = None,
    sentiment_scale: int = 5,
    llm_required: bool = False,
    max_workers: int = 300,
) -> Dict[str, Dict[str, Any]]:
    """Return per-resident sentiment in 5-level scale [-2, 2]."""
    result: Dict[str, Dict[str, Any]] = {}

    # 1) Heuristic baseline for all residents.
    for rid, ev in resident_evidence.items():
        scores: List[int] = []
        for ref in ev.get("reflections") or []:
            comments = ref.get("comments") or []
            if isinstance(comments, str):
                comments = [comments]
            for c in comments:
                scores.append(_score_reflection_text_sentiment_heuristic(str(c)))
        avg_score = (sum(scores) / float(len(scores))) if scores else 0.0
        result[rid] = {
            "score_5": max(-2.0, min(2.0, float(avg_score))) if scores else None,
            "reflection_count": len(scores),
            "source": "heuristic" if scores else "none",
        }

    # 2) Optional LLM overwrite when available.
    if llm_required and not llm_json_fn:
        raise RuntimeError("_score_reflection_sentiment_by_resident requires llm_json_fn when llm_required=True")
    if not llm_json_fn:
        return result

    scale = 3 if int(sentiment_scale) == 3 else 5

    def _score_one(rid: str) -> Tuple[str, Optional[Dict[str, Any]]]:
        ev = resident_evidence[rid]
        last_ev = (last_resident_evidence or {}).get(rid)
        payload = {
            "resident_id": rid,
            "reflection_count": sum(
                len((r.get("comments") if isinstance(r.get("comments"), list) else [r.get("comments")]) or [])
                for r in (ev.get("reflections") or [])
            ),
            "reflections": ev.get("reflections", []),
        }
        if last_ev:
            payload["last_simulation_reflections"] = last_ev.get("reflections", [])
        if scale == 3:
            prompt = (
                "You are a sentiment analysis assistant for one urban resident's reflections.\n"
                "Classify overall reflection sentiment using 3 levels in [-1, 0, 1].\n"
                "-1 means negative; 0 mixed/neutral; 1 positive. Use only reflection texts. If evidence is sparse or mixed, prefer 0.\n"
                "When last_simulation_reflections are provided, use them as the previous simulation baseline for the same resident. "
                "If the current reflections show that problems raised in the previous simulation have been solved or clearly improved, reward that improvement with a more positive sentiment score when justified by the current evidence.\n"
                "Output JSON only in this format: {\"score_3\": -1..1, \"reason\": \"short\"}\n"
                f"Input:\n{json.dumps(payload, ensure_ascii=False)}"
            )
        else:
            prompt = (
                "You are a sentiment analysis assistant for one urban resident's reflections.\n"
                "Classify overall reflection sentiment using 5 levels in [-2, -1, 0, 1, 2].\n"
                "-2 means strongly negative; -1 negative; 0 mixed/neutral; 1 positive; 2 strongly positive.\n"
                "Use only reflection texts. If evidence is sparse or mixed, prefer 0.\n"
                "When last_simulation_reflections are provided, use them as the previous simulation baseline for the same resident. "
                "If the current reflections show that problems raised in the previous simulation have been solved or clearly improved, reward that improvement with a more positive sentiment score when justified by the current evidence.\n"
                "Output JSON only in this format: {\"score_5\": -2..2, \"reason\": \"short\"}\n"
                f"Input:\n{json.dumps(payload, ensure_ascii=False)}"
            )
        obj = llm_json_fn(prompt) or {}
        return rid, obj if isinstance(obj, dict) else None

    resident_ids = sorted(resident_evidence.keys(), key=lambda x: int(x) if x.isdigit() else x)
    score_items: Dict[str, Dict[str, Any]] = {}
    with ThreadPoolExecutor(max_workers=max(1, min(int(max_workers), max(len(resident_ids), 1)))) as ex:
        futures = [ex.submit(_score_one, rid) for rid in resident_ids]
        for fut in as_completed(futures):
            rid, item = fut.result()
            if isinstance(item, dict):
                score_items[rid] = item

    if llm_required and not score_items:
        raise RuntimeError("_score_reflection_sentiment_by_resident failed: invalid LLM JSON")

    for rid, item in score_items.items():
        raw = item.get("score_3") if scale == 3 else item.get("score_5")
        try:
            score = float(raw)
        except Exception:
            continue
        if scale == 3:
            score = max(-1.0, min(1.0, score))
        else:
            score = max(-2.0, min(2.0, score))
        result[rid] = {
            "score_5": score if scale == 5 else float(max(-2.0, min(2.0, score * 2.0))),
            "reflection_count": int(result[rid].get("reflection_count") or 0),
            "reason": str(item.get("reason") or ""),
            "source": "llm",
        }
    if llm_required:
        llm_cnt = sum(1 for v in result.values() if str(v.get("source")) == "llm")
        if llm_cnt <= 0:
            raise RuntimeError("_score_reflection_sentiment_by_resident failed: no valid LLM scores")
    return result


def _score5_to_score3(score5: float) -> int:
    if score5 <= -0.34:
        return -1
    if score5 >= 0.34:
        return 1
    return 0


def _summarize_subjective_scores(score_map: Dict[str, Dict[str, Any]], resident_total: int) -> Dict[str, Any]:
    scores = []
    for v in score_map.values():
        try:
            score = int(v.get("score"))
        except Exception:
            continue
        if score > 0:
            scores.append(score)
    return {
        "avg": (sum(scores) / len(scores)) if scores else 0.0,
        "distribution": Counter(scores),
        "non_none_count": len(scores),
        "none_count": max(0, int(resident_total) - len(scores)),
    }


def _summarize_reflection_sentiment(
    sentiment_by_resident: Dict[str, Dict[str, Any]],
    resident_total: int,
    sentiment_scale: int,
) -> Dict[str, Any]:
    residents_with_reflection = [
        rid for rid, rec in sentiment_by_resident.items() if int(rec.get("reflection_count") or 0) > 0
    ]
    score5_values = [
        float(sentiment_by_resident[rid].get("score_5"))
        for rid in residents_with_reflection
        if sentiment_by_resident[rid].get("score_5") is not None
    ]
    overall_score5 = (sum(score5_values) / len(score5_values)) if score5_values else 0.0
    score3_values = [_score5_to_score3(v) for v in score5_values]
    overall_score3 = (sum(score3_values) / len(score3_values)) if score3_values else 0.0
    scale = 3 if int(sentiment_scale) == 3 else 5
    return {
        "scale": scale,
        "sentiment_index": overall_score3 if scale == 3 else overall_score5,
        "overall_score5": overall_score5,
        "overall_score3": overall_score3,
        "score5_distribution": Counter(int(round(v)) for v in score5_values),
        "score3_distribution": Counter(score3_values),
        "residents_with_reflection": len(residents_with_reflection),
        "non_none_count": len(score5_values),
        "none_count": max(0, int(resident_total) - len(score5_values)),
    }


def evaluate_simulation_assessment(
    idx: GraphIndex,
    resident_profiles: Dict[str, Dict[str, Any]] | None = None,
    last_idx: Optional[GraphIndex] = None,
    llm_json_fn: Optional[LLMJSONFn] = None,
    interview_llm_json_fn: Optional[LLMJSONFn] = None,
    sentiment_llm_json_fn: Optional[LLMJSONFn] = None,
    sentiment_scale: int = 5,
    llm_required: bool = False,
    dist_cut: float = 0.0,
) -> Dict[str, Any]:
    """Simulation-result metrics (independent from planning accessibility metrics)."""
    aoi_geom, aoi_landuse = _build_aoi_polygons_xy(idx)
    evidence = _extract_resident_evidence(idx, aoi_geom, aoi_landuse)
    last_evidence: Dict[str, Dict[str, Any]] = {}
    if last_idx is not None:
        last_aoi_geom, last_aoi_landuse = _build_aoi_polygons_xy(last_idx)
        last_evidence = _extract_resident_evidence(last_idx, last_aoi_geom, last_aoi_landuse)

    all_distances = []
    trip_count_by_resident = {}
    for rid, ev in evidence.items():
        dists = [float(t.get("distance") or 0.0) for t in (ev.get("trajectories") or [])]
        all_distances.extend(dists)
        trip_count_by_resident[rid] = len(dists)
    raw_objective_avg_move_distance = (sum(all_distances) / len(all_distances)) if all_distances else 0.0
    dist_cut = max(0.0, min(float(dist_cut or 0.0), 0.999999))
    dist_cut_count = int(len(all_distances) * dist_cut) if all_distances and dist_cut > 0 else 0
    if dist_cut_count > 0:
        distance_values_for_avg = sorted(all_distances)[: len(all_distances) - dist_cut_count]
    else:
        distance_values_for_avg = all_distances
    objective_avg_move_distance = (
        (sum(distance_values_for_avg) / len(distance_values_for_avg)) if distance_values_for_avg else 0.0
    )

    interview_fn = interview_llm_json_fn or llm_json_fn
    sentiment_fn = sentiment_llm_json_fn or llm_json_fn

    heuristic_subjective_map = _infer_subjective_interview_scores(
        evidence,
        resident_profiles=resident_profiles,
        llm_json_fn=None,
        llm_required=False,
    )
    heuristic_subjective = _summarize_subjective_scores(heuristic_subjective_map, len(evidence))

    heuristic_sentiment_by_resident = _score_reflection_sentiment_by_resident(
        evidence,
        llm_json_fn=None,
        sentiment_scale=sentiment_scale,
        llm_required=False,
    )
    heuristic_sentiment = _summarize_reflection_sentiment(
        heuristic_sentiment_by_resident,
        len(evidence),
        sentiment_scale,
    )

    llm_metrics: Optional[Dict[str, Any]] = None
    llm_subjective_map: Dict[str, Dict[str, Any]] = {}
    llm_sentiment_by_resident: Dict[str, Dict[str, Any]] = {}
    if interview_fn or sentiment_fn or llm_required:
        llm_subjective_map = _infer_subjective_interview_scores(
            evidence,
            resident_profiles=resident_profiles,
            last_resident_evidence=last_evidence,
            llm_json_fn=interview_fn,
            llm_required=llm_required,
        )
        llm_subjective = _summarize_subjective_scores(llm_subjective_map, len(evidence))
        llm_sentiment_by_resident = _score_reflection_sentiment_by_resident(
            evidence,
            last_resident_evidence=last_evidence,
            llm_json_fn=sentiment_fn,
            sentiment_scale=sentiment_scale,
            llm_required=llm_required,
        )
        llm_sentiment = _summarize_reflection_sentiment(
            llm_sentiment_by_resident,
            len(evidence),
            sentiment_scale,
        )
        llm_metrics = {
            "objective_avg_move_distance": round(objective_avg_move_distance, 4),
            "subjective_interview_score": round(llm_subjective["avg"], 4),
            "reflection_sentiment_index": round(llm_sentiment["sentiment_index"], 4),
        }

    return {
        "metrics": {
            "objective_avg_move_distance": round(objective_avg_move_distance, 4),
            "subjective_interview_score": round(heuristic_subjective["avg"], 4),
            "reflection_sentiment_index": round(heuristic_sentiment["sentiment_index"], 4),
        },
        "metric_source": "heuristic",
        "llm_metrics": llm_metrics,
        "resident_sample": len(evidence),
        "trip_sample": len(all_distances),
        "distance_cut": {
            "enabled": dist_cut_count > 0,
            "ratio": round(dist_cut, 6),
            "removed_worst_trip_count": dist_cut_count,
            "kept_trip_count": len(distance_values_for_avg),
            "raw_objective_avg_move_distance": round(raw_objective_avg_move_distance, 4),
        },
        "subjective_score_non_none_resident_count": heuristic_subjective["non_none_count"],
        "subjective_score_none_resident_count": heuristic_subjective["none_count"],
        "subjective_score_distribution": {str(k): v for k, v in sorted(heuristic_subjective["distribution"].items())},
        "subjective_scores_by_resident": heuristic_subjective_map,
        "reflection_sentiment": {
            "selected_scale": heuristic_sentiment["scale"],
            "selected_range": [-1, 1] if heuristic_sentiment["scale"] == 3 else [-2, 2],
            "resident_sample_with_reflection": heuristic_sentiment["residents_with_reflection"],
            "non_none_resident_count": heuristic_sentiment["non_none_count"],
            "none_resident_count": heuristic_sentiment["none_count"],
            "overall_index_scale5": round(heuristic_sentiment["overall_score5"], 4),
            "overall_index_scale3": round(heuristic_sentiment["overall_score3"], 4),
            "distribution_scale5": {str(k): int(v) for k, v in sorted(heuristic_sentiment["score5_distribution"].items())},
            "distribution_scale3": {str(k): int(v) for k, v in sorted(heuristic_sentiment["score3_distribution"].items())},
            "by_resident": heuristic_sentiment_by_resident,
        },
        "llm_assessment": None if llm_metrics is None else {
            "metrics": llm_metrics,
            "compare_with_last_sim": last_idx is not None,
            "subjective_score_non_none_resident_count": llm_subjective["non_none_count"],
            "subjective_score_none_resident_count": llm_subjective["none_count"],
            "subjective_score_distribution": {str(k): v for k, v in sorted(llm_subjective["distribution"].items())},
            "subjective_scores_by_resident": llm_subjective_map,
            "reflection_sentiment": {
                "selected_scale": llm_sentiment["scale"],
                "selected_range": [-1, 1] if llm_sentiment["scale"] == 3 else [-2, 2],
                "resident_sample_with_reflection": llm_sentiment["residents_with_reflection"],
                "non_none_resident_count": llm_sentiment["non_none_count"],
                "none_resident_count": llm_sentiment["none_count"],
                "overall_index_scale5": round(llm_sentiment["overall_score5"], 4),
                "overall_index_scale3": round(llm_sentiment["overall_score3"], 4),
                "distribution_scale5": {str(k): int(v) for k, v in sorted(llm_sentiment["score5_distribution"].items())},
                "distribution_scale3": {str(k): int(v) for k, v in sorted(llm_sentiment["score3_distribution"].items())},
                "by_resident": llm_sentiment_by_resident,
            },
        },
        "trip_count_by_resident": trip_count_by_resident,
        "distance_definition": {
            "input": "mobility edge relation pair AOI polygons from graph aoi_node.positions",
            "function": "shapely Polygon.distance between trip start/end AOI boundaries",
            "method_note": "positions_xy_boundary_distance",
            "dist_cut_note": "objective_avg_move_distance removes the largest trip distances when distance_cut.enabled is true",
        },
        "method": "positions_xy_boundary_distance",
    }


def evaluate_plan_constraints(plan: Dict[str, str], aoi_landuse: Dict[str, str]) -> Dict[str, Any]:
    """Validate a proposed plan against land-use transition rules.

    Purpose:
        Filter illegal changes and provide diagnostics for rejected items.

    Args:
        plan: Proposed mapping `{aoi_id: target_landuse}`.
        aoi_landuse: Current mapping `{aoi_id: current_landuse}`.

    Returns:
        Dict with:
        - valid_plan: cleaned legal changes only.
        - changed_aoi_count: count of accepted AOI updates.
        - changed_by_target_landuse: histogram by target land-use code.
        - invalid_items: list of rejected entries and reasons.
    """
    cleaned = validate_plan_mapping(plan, aoi_landuse)
    invalid = []

    for raw_id, target in plan.items():
        aoi_id = str(raw_id)
        old = aoi_landuse.get(aoi_id)
        if not old:
            invalid.append({"aoi_id": aoi_id, "reason": "unknown_aoi"})
            continue
        if aoi_id not in cleaned:
            invalid.append({"aoi_id": aoi_id, "old": old, "new": target, "reason": "illegal_transition_or_no_change"})

    changed_area_count = len(cleaned)
    changed_by_target = defaultdict(int)
    for _, target in cleaned.items():
        changed_by_target[target] += 1

    return {
        "valid_plan": cleaned,
        "changed_aoi_count": changed_area_count,
        "changed_by_target_landuse": dict(changed_by_target),
        "invalid_items": invalid,
    }
