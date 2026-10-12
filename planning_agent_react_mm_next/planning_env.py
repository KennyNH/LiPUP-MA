"""Planning environment wrapper for pickle/pb temporary map iteration."""

from __future__ import annotations

import base64
import json
import mimetypes
import pickle
import re
import shutil
import hashlib
import math
import os
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from shapely.geometry import Polygon

try:
    from .graph_tools import GraphIndex, aoi_landuse_map, get_graph_overview, get_planning_summary, parse_relation_pair
    from .landuse import IMMUTABLE_LANDUSE, LANDUSE_LABELS, MODIFIABLE_LANDUSE
    from .llm_client import OpenAICompatibleClient
    from .map_tools import MapWorkspace
    from .metrics_tools import (
        WalkingRoadDistance,
        evaluate_accessibility_proxy,
        evaluate_simulation_assessment,
        prepare_resident_preference_cache,
    )
except ImportError:
    from graph_tools import GraphIndex, aoi_landuse_map, get_graph_overview, get_planning_summary, parse_relation_pair
    from landuse import IMMUTABLE_LANDUSE, LANDUSE_LABELS, MODIFIABLE_LANDUSE
    from llm_client import OpenAICompatibleClient
    from map_tools import MapWorkspace
    from metrics_tools import WalkingRoadDistance, evaluate_accessibility_proxy, evaluate_simulation_assessment, prepare_resident_preference_cache


@dataclass
class TempMapState:
    name: str
    pickle_path: Path
    pb_path: Path


class PlanningEnvironment:
    def __init__(
        self,
        repo_root: Path,
        run_dir: Path,
        graph_index: GraphIndex,
        init_pb_path: Path,
        init_pickle_path: Path,
        llm_client: Optional[OpenAICompatibleClient] = None,
        resident_profiles_path: Optional[Path] = None,
        resident_preferences: Optional[Dict[str, List[str]]] = None,
        importance_level3_ratio: float = 0.15,
        importance_level2_ratio: float = 0.25,
        service_radius_m: float = 500.0,
        id_service_radius_m: float = 1000.0,
        ecology_radius_m: float = 300.0,
        skip_simulation_assessment: bool = False,
    ):
        self.repo_root = repo_root
        self.run_dir = run_dir
        self.graph_index = graph_index
        self.init_pb_path = init_pb_path
        self.init_pickle_path = init_pickle_path
        self.base_landuse = self._load_landuse_from_pickle(self.init_pickle_path)
        self.aoi_id_prefix, self.aoi_id_suffix_width = self._infer_aoi_id_profile(set(self.base_landuse.keys()))
        self.modifiable_base_count = sum(1 for _aid, lu in self.base_landuse.items() if lu in MODIFIABLE_LANDUSE)
        self.max_changed_allowed = max(1, int(math.floor(self.modifiable_base_count * 0.30))) if self.modifiable_base_count > 0 else 0
        self.temp_state: Optional[TempMapState] = None
        self.llm_client = llm_client
        self.resident_profiles_path = resident_profiles_path
        self.precomputed_resident_preferences = resident_preferences or {}
        r3 = float(importance_level3_ratio)
        r2 = float(importance_level2_ratio)
        if r3 < 0:
            r3 = 0.0
        if r2 < 0:
            r2 = 0.0
        if r3 + r2 > 1.0:
            scale = 1.0 / max(r3 + r2, 1e-9)
            r3 *= scale
            r2 *= scale
        self.importance_level3_ratio = r3
        self.importance_level2_ratio = r2
        self.service_radius_m = float(service_radius_m)
        self.id_service_radius_m = float(id_service_radius_m)
        self.ecology_radius_m = float(ecology_radius_m)
        self.skip_simulation_assessment = bool(skip_simulation_assessment)
        self.resident_profiles = self._load_resident_profiles()
        self.road_distance_model = self._init_road_distance_model()
        self.geojson_snapshot_path = self.run_dir / "map_snapshot.geojson"
        self.aoi_geom, self.aoi_landuse, self.aoi_name = self._build_aoi_geometry_and_meta()
        self.aoi_suffix_index = self._build_aoi_suffix_index()
        self.preprocess_cache_dir = Path(os.environ.get("LIPUP_CACHE_DIR", str(self.repo_root / "outputs" / "preprocess_cache")))
        self.preprocess_cache_dir.mkdir(parents=True, exist_ok=True)
        self.process_dir = self.run_dir / "process_outputs"
        self.process_dir.mkdir(parents=True, exist_ok=True)
        self.apply_step_counter = 0
        self.summary_step_counter = 0
        self.related_step_counter = 0
        self.render_step_counter = 0
        self.zoom_step_counter = 0
        self.cache_meta: Dict[str, Any] = {}
        self.preference_cache = {}
        self.simulation_assessment = {}
        self.aoi_analysis = {}
        self.relation_pair_count = {}
        self._temp_metrics_cache: Dict[str, Dict[str, Any]] = {}
        # Parallel knobs (opt-in for LLM-related parallelism; deterministic defaults).
        self.preprocess_workers = max(1, int(os.getenv("PLANNING_PREPROCESS_WORKERS", "1")))
        self.enable_llm_preprocess_parallel = str(os.getenv("PLANNING_ENABLE_LLM_PREPROCESS_PARALLEL", "0")).strip() in {"1", "true", "True"}
        self.aoi_llm_batch_workers = max(
            1,
            int(os.getenv("PLANNING_AOI_LLM_BATCH_WORKERS", str(self.preprocess_workers))),
        )
        self._load_or_build_preprocess_cache()

    @staticmethod
    def _pp_log(msg: str) -> None:
        # Keep progress visible in long preprocessing sections.
        print(f"[planning-preprocess] {msg}")

    def init_temp_copy(self, name: str = "temp_plan") -> Dict[str, Any]:
        pkl = self.run_dir / f"{name}.pickle"
        pb = self.run_dir / f"{name}.pb"
        shutil.copy2(self.init_pickle_path, pkl)
        self.temp_state = TempMapState(name=name, pickle_path=pkl, pb_path=pb)
        return {"success": True, "temp_pickle_path": str(pkl), "temp_pb_path": str(pb), "pb_written": False}

    def current_landuse_from_pickle(self) -> Dict[str, str]:
        if not self.temp_state:
            raise RuntimeError("temp copy not initialized")
        m = pickle.load(open(self.temp_state.pickle_path, "rb"))
        aois = m.get("aois") or []
        out: Dict[str, str] = {}
        for a in aois:
            out[str(a.get("id"))] = str(a.get("urban_land_use") or "")
        return out

    def apply_plan(self, plan: Dict[str, str], export_pb: bool = False) -> Dict[str, Any]:
        if not self.temp_state:
            return {"error": "temp copy not initialized"}
        if not isinstance(plan, dict):
            return {"error": "plan must be object"}

        curr = self.current_landuse_from_pickle()
        valid: Dict[str, str] = {}
        ignored: List[Dict[str, Any]] = []
        for raw_id, target in plan.items():
            aid = self._normalize_aoi_id(raw_id)
            old = curr.get(aid)
            if not old:
                ignored.append({"aoi_id": str(raw_id), "reason": "unknown_aoi"})
                continue
            target = str(target).strip().upper()
            if old in IMMUTABLE_LANDUSE:
                ignored.append(
                    {
                        "aoi_id": aid,
                        "reason": "immutable_source",
                        "source_landuse": old,
                        "requested_target": target,
                        "message": f"source landuse `{old}` is immutable",
                    }
                )
                continue
            if target not in MODIFIABLE_LANDUSE:
                ignored.append(
                    {
                        "aoi_id": aid,
                        "reason": "invalid_target",
                        "source_landuse": old,
                        "requested_target": target,
                    }
                )
                continue
            if old == target:
                ignored.append(
                    {
                        "aoi_id": aid,
                        "reason": "no_change",
                        "source_landuse": old,
                        "requested_target": target,
                    }
                )
                continue
            valid[aid] = target
        if not valid:
            return {
                "error": "no_effective_changes",
                "message": "No valid AOI updates after filtering unknown/immutable/invalid/no-change inputs.",
                "ignored_count": len(ignored),
                "ignored_items": ignored,
            }

        changed_before = {aid for aid, lu in curr.items() if self.base_landuse.get(aid) != lu}
        projected = set(changed_before)
        for aid, new_lu in valid.items():
            if self.base_landuse.get(aid) == new_lu:
                projected.discard(aid)
            else:
                projected.add(aid)
        if self.max_changed_allowed > 0 and len(projected) > self.max_changed_allowed:
            return {
                "error": "max_changed_ratio_constraint_violated",
                "changed_if_applied": len(projected),
                "max_allowed_changed": self.max_changed_allowed,
                "modifiable_base_count": self.modifiable_base_count,
                "ratio_limit": 0.30,
                "ignored_count": len(ignored),
                "ignored_items": ignored,
                "violating_plan_items": [
                    {
                        "aoi_id": aid,
                        "source_landuse": curr.get(aid),
                        "requested_target": lu,
                    }
                    for aid, lu in sorted(valid.items())
                ],
            }

        # apply to pickle map dict
        m = pickle.load(open(self.temp_state.pickle_path, "rb"))
        aois = m.get("aois") or []
        changed = []
        for a in aois:
            aid = str(a.get("id"))
            if aid in valid:
                a["urban_land_use"] = valid[aid]
                changed.append(aid)
        m["aois"] = aois
        pickle.dump(m, open(self.temp_state.pickle_path, "wb"))
        pb_written = False
        if export_pb:
            self._dict_to_pb_path(self.temp_state.pickle_path, self.temp_state.pb_path)
            pb_written = True

        return {
            "success": True,
            "applied_count": len(changed),
            "applied_plan": {aid: valid[aid] for aid in changed},
            "ignored_count": len(ignored),
            "ignored_items": ignored,
            "temp_pickle_path": str(self.temp_state.pickle_path),
            "temp_pb_path": str(self.temp_state.pb_path),
            "pb_written": pb_written,
        }

    def evaluate_temp(self, service_radius: Optional[float] = None, id_service_radius: Optional[float] = None, ecology_radius: Optional[float] = None) -> Dict[str, Any]:
        if not self.temp_state:
            return {"error": "temp copy not initialized"}
        service_radius = float(self.service_radius_m if service_radius is None else service_radius)
        id_service_radius = float(self.id_service_radius_m if id_service_radius is None else id_service_radius)
        ecology_radius = float(self.ecology_radius_m if ecology_radius is None else ecology_radius)
        curr = self.current_landuse_from_pickle()
        sig = self._state_signature(curr)
        cached = self._temp_metrics_cache.get(sig)
        if cached is not None:
            return cached
        metrics = evaluate_accessibility_proxy(
            self.graph_index,
            service_radius=service_radius,
            id_service_radius=id_service_radius,
            ecology_radius=ecology_radius,
            landuse_override=curr,
            resident_preferences=(self.preference_cache or {}).get("resident_preferences"),
            road_distance_model=self.road_distance_model,
        )
        self._temp_metrics_cache[sig] = metrics
        return metrics

    def render_temp(
        self,
        output_name: str,
        label_bbox_rel: Optional[List[float]] = None,
        highlight_changed: bool = True,
    ) -> Dict[str, Any]:
        if not self.temp_state:
            return {"error": "temp copy not initialized"}
        curr = self.current_landuse_from_pickle()
        changed_ids = [aid for aid, lu in curr.items() if self.base_landuse.get(aid) != lu]
        overrides = {aid: curr[aid] for aid in changed_ids}
        ws = MapWorkspace(repo_root=self.repo_root, output_dir=self.run_dir)
        ws.load_map(self.temp_state.pb_path)
        return ws.render_map(
            output_name=output_name,
            proposed_overrides=overrides,
            highlight_aois=changed_ids if highlight_changed else None,
            label_aois=changed_ids[:150],
            label_bbox_rel=label_bbox_rel,
            use_relative_coords=True,
            show_network=True,
        )

    def planning_overview(self) -> Dict[str, Any]:
        graph_overview = get_graph_overview(self.graph_index)
        planning_summary = get_planning_summary(self.graph_index)
        base_metrics = evaluate_accessibility_proxy(
            self.graph_index,
            landuse_override=self.base_landuse,
            resident_preferences=(self.preference_cache or {}).get("resident_preferences"),
            road_distance_model=self.road_distance_model,
        )
        out = {
            "graph_overview": graph_overview,
            "planning_summary": planning_summary,
            "constraints": {
                "immutable_landuse": sorted(IMMUTABLE_LANDUSE),
                "modifiable_landuse": sorted(MODIFIABLE_LANDUSE),
                "max_changed_ratio": 0.30,
                "max_changed_count": self.max_changed_allowed,
                "modifiable_base_count": self.modifiable_base_count,
            },
            "base_metrics": base_metrics,
            "resident_preference_cache": self.preference_cache,
            "simulation_assessment": self.simulation_assessment,
            "preprocess_cache": self.cache_meta,
            "aoi_analysis_summary": {
                "aoi_count": len(self.aoi_analysis),
                "high_self_importance_count": sum(1 for x in self.aoi_analysis.values() if int(x.get("self_importance", 1)) >= 3),
                "high_neighbor_importance_count": sum(1 for x in self.aoi_analysis.values() if int(x.get("neighbor_importance", 1)) >= 3),
            },
        }
        if self.temp_state:
            out["temp_metrics"] = self.evaluate_temp()
            out["temp_diff"] = self.inspect_temp_diff()
        return out

    def evaluate_simulation_metrics(self) -> Dict[str, Any]:
        return self.simulation_assessment

    def evaluate_base(self, service_radius: Optional[float] = None, id_service_radius: Optional[float] = None, ecology_radius: Optional[float] = None) -> Dict[str, Any]:
        service_radius = float(self.service_radius_m if service_radius is None else service_radius)
        id_service_radius = float(self.id_service_radius_m if id_service_radius is None else id_service_radius)
        ecology_radius = float(self.ecology_radius_m if ecology_radius is None else ecology_radius)
        return evaluate_accessibility_proxy(
            self.graph_index,
            service_radius=service_radius,
            id_service_radius=id_service_radius,
            ecology_radius=ecology_radius,
            landuse_override=self.base_landuse,
            resident_preferences=(self.preference_cache or {}).get("resident_preferences"),
            road_distance_model=self.road_distance_model,
        )

    def query_top_degree_aois(self, top_n: int = 10, modifiable_only: bool = True) -> Dict[str, Any]:
        top_n = max(1, int(top_n))
        weighted: Dict[str, int] = {}
        for aid, cnt in self.relation_pair_count.items():
            a0, a1 = aid
            weighted[a0] = weighted.get(a0, 0) + cnt
            weighted[a1] = weighted.get(a1, 0) + cnt
        if bool(modifiable_only):
            weighted = {aid: c for aid, c in weighted.items() if self.base_landuse.get(aid) in MODIFIABLE_LANDUSE}
        ranked = sorted(weighted.items(), key=lambda kv: kv[1], reverse=True)[:top_n]
        return {
            "top_mobility_endpoint_aois": [
                {"aoi_id": k, "mobility_endpoint_count": int(v), "landuse": self.base_landuse.get(k, "")}
                for k, v in ranked
            ],
        }

    def query_importance_ids(self, importance_level: int, scope: str = "self") -> Dict[str, Any]:
        lv = int(importance_level)
        scope = "neighbor" if str(scope).lower().startswith("nei") else "self"
        key = "neighbor_importance" if scope == "neighbor" else "self_importance"
        ids = [aid for aid, rec in self.aoi_analysis.items() if int(rec.get(key, 1)) == lv]
        ids.sort()
        return {"scope": scope, "importance_level": lv, "aoi_ids": ids, "count": len(ids)}

    def query_aois_by_landuse(self, landuse_types: Any) -> Dict[str, Any]:
        """Query AOI ids grouped by requested land-use type(s).

        Input supports:
        - single string: "BS" or "Business Area" or "BS,SC"
        - list/tuple of strings
        """
        # Build case-insensitive alias map from both code and full-name.
        alias_to_code: Dict[str, str] = {}
        for code, name in LANDUSE_LABELS.items():
            alias_to_code[str(code).strip().upper()] = str(code)
            alias_to_code[str(name).strip().lower()] = str(code)

        raw_items: List[str] = []
        if isinstance(landuse_types, (list, tuple, set)):
            for x in landuse_types:
                if x is None:
                    continue
                raw_items.append(str(x))
        elif isinstance(landuse_types, str):
            s = str(landuse_types)
            if "," in s:
                raw_items.extend([p.strip() for p in s.split(",") if p.strip()])
            else:
                raw_items.append(s.strip())
        elif landuse_types is not None:
            raw_items.append(str(landuse_types))

        requested_codes: List[str] = []
        invalid_inputs: List[str] = []
        for item in raw_items:
            token = str(item).strip()
            if not token:
                continue
            code = alias_to_code.get(token.upper()) or alias_to_code.get(token.lower())
            if code is None:
                invalid_inputs.append(token)
                continue
            if code not in requested_codes:
                requested_codes.append(code)

        groups: Dict[str, Dict[str, Any]] = {}
        for code in requested_codes:
            ids = sorted([aid for aid, lu in self.base_landuse.items() if str(lu) == code])
            groups[code] = {
                "landuse": code,
                "landuse_name": LANDUSE_LABELS.get(code, code),
                "aoi_ids": ids,
                "count": len(ids),
            }

        return {
            "requested_landuse_types": requested_codes,
            "invalid_inputs": invalid_inputs,
            "groups": groups,
        }

    def query_aois_in_bbox(self, x1: float, y1: float, x2: float, y2: float) -> Dict[str, Any]:
        xmin, xmax = sorted([float(x1), float(x2)])
        ymin, ymax = sorted([float(y1), float(y2)])
        out = []
        for aid, g in self.aoi_geom.items():
            c = g.centroid
            if xmin <= c.x <= xmax and ymin <= c.y <= ymax:
                out.append(aid)
        out.sort()
        return {"bbox_abs": [xmin, ymin, xmax, ymax], "aoi_ids": out, "count": len(out)}

    def query_aoi_details(self, aoi_ids: List[Any]) -> Dict[str, Any]:
        ids = self._normalize_aoi_ids(aoi_ids)
        curr = self.current_landuse_from_pickle() if self.temp_state else self.base_landuse
        rows = []
        for aid in ids:
            g = self.aoi_geom.get(aid)
            if not g:
                continue
            c = g.centroid
            a = self.aoi_analysis.get(aid, {})
            initial_lu = self.base_landuse.get(aid, "")
            row = {
                "aoi_id": aid,
                "name": self.aoi_name.get(aid, ""),
                "landuse": {
                    "initial": initial_lu,
                    "current": curr.get(aid, initial_lu),
                },
                "position_centroid": {"x": round(float(c.x), 3), "y": round(float(c.y), 3)},
                "immutable": initial_lu in IMMUTABLE_LANDUSE,
                "neighbor_feedback": {
                    "total_reflections": int(a.get("neighbor_total_reflections", 0) or 0),
                    "negative_ratio": float(a.get("neighbor_negative_ratio", 0.0) or 0.0),
                    "summary": str(a.get("neighbor_summary") or a.get("neighbor_common_summary") or ""),
                    "importance": int(a.get("neighbor_importance", 1) or 1),
                },
            }
            if initial_lu not in IMMUTABLE_LANDUSE:
                row["self_feedback"] = {
                    "total_reflections": int(a.get("self_total_reflections", 0) or 0),
                    "negative_ratio": float(a.get("self_negative_ratio", 0.0) or 0.0),
                    "summary": str(a.get("self_summary") or a.get("self_common_summary") or ""),
                    "importance": int(a.get("self_importance", 1) or 1),
                }
            rows.append(row)
        return {"aoi_details": rows}

    def query_related_aois(self, aoi_id: Any, top_n: int = 3, radius_m: float = 100.0) -> Dict[str, Any]:
        qid = self._normalize_aoi_id(aoi_id)
        if qid not in self.aoi_geom:
            return {"error": f"unknown_aoi: {aoi_id}"}
        near = []
        qg = self.aoi_geom[qid]
        for aid, g in self.aoi_geom.items():
            if aid == qid:
                continue
            if qg.distance(g) <= float(radius_m):
                near.append(aid)
        near.sort()

        rel = []
        for (a0, a1), cnt in self.relation_pair_count.items():
            if a0 == qid and a1 not in near:
                rel.append((a1, cnt))
            elif a1 == qid and a0 not in near:
                rel.append((a0, cnt))
        rel_ranked = sorted(rel, key=lambda kv: kv[1], reverse=True)
        seen = set()
        far = []
        for aid, cnt in rel_ranked:
            if aid in seen:
                continue
            seen.add(aid)
            far.append({"aoi_id": aid, "relation_flow": cnt})
            if len(far) >= int(top_n):
                break

        near_ids = [qid] + near[:20]
        far_ids = [qid] + [x["aoi_id"] for x in far]
        self.related_step_counter += 1
        near_map = self.render_map_compact(
            map_version="initial",
            aoi_ids=near_ids,
            output_name=f"related_near_{self.related_step_counter:03d}.png",
            view_bbox_rel=None,
            highlight_changed=True,
        )
        far_map = self.render_map_compact(
            map_version="initial",
            aoi_ids=far_ids,
            output_name=f"related_far_{self.related_step_counter:03d}.png",
            view_bbox_rel=None,
            highlight_changed=True,
        )
        near_summary = self._summarize_aoi_group_text(near_ids, focus_aoi=qid, group_name="nearby", image_path=near_map.get("output_path"))
        far_summary = self._summarize_aoi_group_text(far_ids, focus_aoi=qid, group_name="mobility-linked", image_path=far_map.get("output_path"))
        output = {
            "query_aoi_id": qid,
            "nearby_aoi_ids": near,
            "nearby_summary": near_summary,
            "mobility_linked_topn": far,
            "mobility_summary": far_summary,
        }
        self._save_json(
            self.process_dir / f"related_{self.related_step_counter:03d}.json",
            {
                **output,
                "internal_near_image_path": near_map.get("output_path"),
                "internal_far_image_path": far_map.get("output_path"),
            },
        )
        return output

    def query_aoi_distances(self, aoi_id_1: Any, aoi_id_2: Any, distance_type: str = "both") -> Dict[str, Any]:
        a1 = self._normalize_aoi_id(aoi_id_1)
        a2 = self._normalize_aoi_id(aoi_id_2)
        g1 = self.aoi_geom.get(a1)
        g2 = self.aoi_geom.get(a2)
        if not g1 or not g2:
            return {"error": f"unknown_aoi_pair: {aoi_id_1},{aoi_id_2}"}
        mode = str(distance_type or "both").strip().lower()
        if mode == "boundary":
            mode = "straight_line"
        if mode not in {"both", "road", "straight_line"}:
            mode = "both"
        straight = float(g1.distance(g2)) if mode in {"both", "straight_line"} else None
        road = None
        road_mode = None
        if mode in {"both", "road"} and self.road_distance_model is not None:
            if hasattr(self.road_distance_model, "distance_aoi_to_aoi_with_meta"):
                road, road_mode = self.road_distance_model.distance_aoi_to_aoi_with_meta(a1, a2)
            else:
                road = self.road_distance_model.distance_aoi_to_aoi(a1, a2)
                road_mode = "road_shortest_path"
        return {
            "aoi_id_1": a1,
            "aoi_id_2": a2,
            "distance_type": mode,
            "straight_line_distance_m": None if straight is None else round(straight, 3),
            "road_distance_m": None if road is None else round(float(road), 3),
            "road_distance_mode": road_mode,
        }

    def summarize_region(
        self,
        x1: float,
        y1: float,
        x2: float,
        y2: float,
        output_name: Optional[str] = None,
    ) -> Dict[str, Any]:
        bbox = [float(x1), float(y1), float(x2), float(y2)]
        bbox_query = self.query_aois_in_bbox(*bbox)
        bbox_ids = bbox_query.get("aoi_ids", [])
        merged_ids = list(dict.fromkeys(bbox_ids))
        if not merged_ids:
            return {"error": "no_aoi_in_region"}
        # Summary tools always use initial map for stable spatial grounding.
        map_info = self.render_map_compact(
            map_version="initial",
            aoi_ids=merged_ids,
            output_name=output_name or self._next_output_name("region_zoom"),
            view_bbox_rel=bbox,
            highlight_changed=True,
        )
        summary_text = self._summarize_aoi_group_text(merged_ids, focus_aoi=None, group_name="selected region", image_path=map_info.get("output_path"))
        self.summary_step_counter += 1
        output = {
            "bbox_abs": bbox_query.get("bbox_abs"),
            "aoi_ids": merged_ids,
            "regional_summary": summary_text,
        }
        self._save_json(
            self.process_dir / f"summary_region_{self.summary_step_counter:03d}.json",
            {
                **output,
                "internal_image_path": map_info.get("output_path"),
            },
        )
        return output

    def summarize_aoi_list(
        self,
        aoi_ids: List[Any],
        output_name: Optional[str] = None,
    ) -> Dict[str, Any]:
        ids = self._normalize_aoi_ids(aoi_ids or [])
        if not ids:
            return {"error": "no_valid_aoi_ids"}
        # Summary tools always use initial map for stable spatial grounding.
        map_info = self.render_map_compact(
            map_version="initial",
            aoi_ids=ids,
            output_name=output_name or self._next_output_name("summary_aoi_list"),
            view_bbox_rel=None,
            highlight_changed=True,
        )
        summary_text = self._summarize_aoi_group_text(
            ids,
            focus_aoi=None,
            group_name="input AOI list",
            image_path=map_info.get("output_path"),
        )
        self.summary_step_counter += 1
        output = {
            "aoi_ids": ids,
            "list_summary": summary_text,
        }
        self._save_json(
            self.process_dir / f"summary_aoi_list_{self.summary_step_counter:03d}.json",
            {
                **output,
                "internal_image_path": map_info.get("output_path"),
            },
        )
        return output

    def render_map_compact(
        self,
        map_version: str,
        aoi_ids: Optional[List[Any]],
        output_name: Optional[str] = None,
        view_bbox_rel: Optional[List[float]] = None,
        highlight_changed: bool = True,
    ) -> Dict[str, Any]:
        use_temp = str(map_version).lower() in {"temp", "modified", "current"}
        if view_bbox_rel:
            self.zoom_step_counter += 1
            default_name = f"zoom_{self.zoom_step_counter:03d}.png"
        else:
            self.render_step_counter += 1
            prefix = "temp_map" if use_temp else "initial_map"
            default_name = f"{prefix}_{self.render_step_counter:03d}.png"
        requested_ids = self._normalize_aoi_ids(aoi_ids or [])
        # Keep full spatial context in rendering:
        # - Always render all AOIs (positions/colors complete).
        # - `aoi_ids` controls label targets (AOI suffix text), not visibility filter.
        visible = None
        label_ids = requested_ids
        # Auto-annotate modifiable AOIs only when AOI label list is not provided.
        annotate_modifiable = len(requested_ids) == 0
        changed = []
        overrides = None
        self._ensure_geojson_snapshot()
        ws = MapWorkspace(repo_root=self.repo_root, output_dir=self.run_dir)
        ws.load_map(self.geojson_snapshot_path)
        if use_temp:
            if not self.temp_state:
                self.init_temp_copy(name="temp_plan")
            curr = self.current_landuse_from_pickle()
            changed = [aid for aid, lu in curr.items() if self.base_landuse.get(aid) != lu]
            overrides = {aid: curr[aid] for aid in changed}
        return ws.render_map(
            output_name=output_name or default_name,
            visible_aois=visible if visible else None,
            proposed_overrides=overrides,
            highlight_aois=changed if (highlight_changed and changed) else None,
            label_aois=label_ids[:300] if label_ids else None,
            annotate_modifiable=annotate_modifiable,
            view_bbox_rel=view_bbox_rel,
            use_relative_coords=True,
            show_network=True,
        )

    def apply_plan_and_evaluate(self, plan: Dict[str, str], render_output_name: Optional[str] = None) -> Dict[str, Any]:
        if not self.temp_state:
            self.init_temp_copy(name="temp_plan")
        before = self.current_landuse_from_pickle()
        before_metrics = self.evaluate_temp().get("metrics", {})
        ap = self.apply_plan(plan, export_pb=False)
        if not ap.get("success"):
            return ap
        after = self.current_landuse_from_pickle()
        step_changes = []
        for aid, new_lu in ap.get("applied_plan", {}).items():
            step_changes.append({"aoi_id": aid, "from": before.get(aid), "to": new_lu})
        all_changes = []
        for aid, lu in after.items():
            if self.base_landuse.get(aid) != lu:
                all_changes.append({"aoi_id": aid, "from": self.base_landuse.get(aid), "to": lu})
        render = self.render_map_compact(
            map_version="temp",
            # Render modifiable planning area in temp map and highlight changed AOIs.
            # This keeps modified-step map consistent with "show all modifiable AOI ids".
            aoi_ids=None,
            output_name=render_output_name or self._next_output_name("temp_plan_step"),
            view_bbox_rel=None,
            highlight_changed=True,
        )
        after_metrics_full = self.evaluate_temp()
        after_metrics = after_metrics_full.get("metrics", {})
        baseline_metrics = self.evaluate_base().get("metrics", {})
        self.apply_step_counter += 1
        compact = {
            "success": True,
            "step_changes": step_changes,
            "all_changes_vs_initial": all_changes,
            "ignored_inputs": ap.get("ignored_items", []),
            "baseline_metrics": baseline_metrics,
            "before_metrics": before_metrics,
            "after_metrics": after_metrics,
            "render_map": {
                "shown_aoi_count": len([x["aoi_id"] for x in all_changes]),
                "image_attached": bool(render.get("output_path")),
            },
            "_image_path": render.get("output_path"),
        }
        output = {
            **compact,
            "_debug": {
                "ignored_count": ap.get("ignored_count", 0),
                "render_meta": render,
                "metrics_full": after_metrics_full,
            },
        }
        self._save_json(
            self.process_dir / f"apply_step_{self.apply_step_counter:03d}.json",
            output,
        )
        return compact

    def _next_output_name(self, prefix: str) -> str:
        self.render_step_counter += 1
        return f"{prefix}_{self.render_step_counter:03d}.png"

    @staticmethod
    def _state_signature(curr: Dict[str, str]) -> str:
        try:
            items = sorted(curr.items(), key=lambda kv: kv[0])
            return hashlib.sha1(json.dumps(items, ensure_ascii=False).encode("utf-8")).hexdigest()[:16]
        except Exception:
            return "na"

    def get_all_modifications(self) -> Dict[str, Any]:
        if not self.temp_state:
            return {"changed_count": 0, "changed_items": []}
        diff = self.inspect_temp_diff().get("changed_items", {})
        rows = [{"aoi_id": aid, "from": x.get("base"), "to": x.get("temp")} for aid, x in diff.items()]
        rows.sort(key=lambda x: x["aoi_id"])
        return {"changed_count": len(rows), "changed_items": rows}

    def inspect_temp_diff(self) -> Dict[str, Any]:
        if not self.temp_state:
            return {"error": "temp copy not initialized"}
        curr = self.current_landuse_from_pickle()
        diff = {}
        for aid, lu in curr.items():
            base_lu = self.base_landuse.get(aid)
            if base_lu != lu:
                diff[aid] = {"base": base_lu, "temp": lu}
        return {"changed_count": len(diff), "changed_items": diff}

    def _save_json(self, path: Path, obj: Dict[str, Any]) -> None:
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(path, "w", encoding="utf-8") as f:
                json.dump(obj, f, ensure_ascii=False, indent=2)
        except Exception:
            pass

    def _cache_key(self) -> str:
        chunks = []
        for p in [self.init_pickle_path, self.init_pb_path, self.resident_profiles_path]:
            if not p:
                chunks.append("none")
                continue
            pp = Path(p)
            if not pp.exists():
                chunks.append(str(pp))
                continue
            st = pp.stat()
            chunks.append(f"{pp.resolve()}|{int(st.st_mtime)}|{st.st_size}")
        chunks.append(f"graph_nodes={len(self.graph_index.nodes)}|graph_edges={len(self.graph_index.links)}")
        chunks.append(f"graph_sig={self._graph_signature()}")
        if self.precomputed_resident_preferences:
            payload = json.dumps(self.precomputed_resident_preferences, ensure_ascii=False, sort_keys=True)
            chunks.append(f"resident_pref_sig={hashlib.sha1(payload.encode('utf-8')).hexdigest()[:16]}")
        chunks.append(f"skip_simulation_assessment={int(self.skip_simulation_assessment)}")
        base = "||".join(chunks).encode("utf-8")
        return hashlib.sha1(base).hexdigest()[:16]

    def _graph_signature(self) -> str:
        h = hashlib.sha1()
        try:
            for n in self.graph_index.nodes:
                h.update(str(n.get("id")).encode("utf-8"))
                h.update(str(n.get("node_type")).encode("utf-8"))
            for e in self.graph_index.links:
                h.update(str(e.get("edge_type")).encode("utf-8"))
                t = e.get("target")
                h.update(type(t).__name__.encode("utf-8"))
                h.update(str(t).encode("utf-8"))
                h.update(str(e.get("source")).encode("utf-8"))
        except Exception:
            return "na"
        return h.hexdigest()[:16]

    @staticmethod
    def _encode_pair_counts(pair_count: Dict[tuple, int]) -> Dict[str, int]:
        out: Dict[str, int] = {}
        for (a0, a1), cnt in pair_count.items():
            out[f"{a0}|{a1}"] = int(cnt)
        return out

    @staticmethod
    def _decode_pair_counts(pair_count: Dict[str, Any]) -> Dict[tuple, int]:
        out: Dict[tuple, int] = {}
        for k, v in (pair_count or {}).items():
            if "|" not in str(k):
                continue
            a0, a1 = str(k).split("|", 1)
            out[(a0, a1)] = int(v)
        return out

    def _load_or_build_preprocess_cache(self) -> None:
        key = self._cache_key()
        cache_path = self.preprocess_cache_dir / f"preprocess_{key}.json"
        t0 = time.perf_counter()
        if cache_path.exists():
            try:
                with open(cache_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                self.preference_cache = data.get("preference_cache") or {}
                self.simulation_assessment = data.get("simulation_assessment") or {}
                self.aoi_analysis = data.get("aoi_analysis") or {}
                self.relation_pair_count = self._decode_pair_counts(data.get("relation_pair_count") or {})
                self.cache_meta = {
                    "cache_key": key,
                    "cache_path": str(cache_path),
                    "cache_hit": True,
                }
                self._pp_log(
                    f"cache hit: {cache_path.name} "
                    f"(elapsed={time.perf_counter() - t0:.2f}s)"
                )
                return
            except Exception:
                pass
        self._pp_log(f"cache miss: building preprocess cache -> {cache_path.name}")

        # Build lightweight graph frequency first; cheap and deterministic.
        t_rel = time.perf_counter()
        self.relation_pair_count = self._build_relation_pair_frequency()
        self._pp_log(f"relation_pair_count ready ({time.perf_counter() - t_rel:.2f}s)")

        # For LLM-backed preprocess, keep sequential calls (safer for client/session behavior).
        # For non-LLM preprocess, parallelize independent CPU tasks.
        if self.llm_client is not None and self.enable_llm_preprocess_parallel and self.preprocess_workers > 1:
            self._pp_log(
                f"LLM preprocess parallel enabled: workers={self.preprocess_workers} "
                "(preference/simulation/aoi-analysis)"
            )
            with ThreadPoolExecutor(max_workers=min(3, self.preprocess_workers)) as ex:
                f_pref = ex.submit(self._init_preference_cache)
                f_sim = None if self.skip_simulation_assessment else ex.submit(self._init_simulation_assessment)
                f_aoi = ex.submit(self._init_aoi_analysis)
                self.preference_cache = f_pref.result()
                self.simulation_assessment = self._init_simulation_assessment() if f_sim is None else f_sim.result()
                self.aoi_analysis = f_aoi.result()
            self._pp_log("LLM parallel preprocess tasks finished")
        elif self.llm_client is not None:
            t_pref = time.perf_counter()
            self._pp_log("building resident preference cache (LLM-enabled)...")
            self.preference_cache = self._init_preference_cache()
            self._pp_log(f"resident preference cache ready ({time.perf_counter() - t_pref:.2f}s)")

            t_sim = time.perf_counter()
            if self.skip_simulation_assessment:
                self._pp_log("skipping simulation assessment (configured)...")
            else:
                self._pp_log("building simulation assessment (LLM-enabled)...")
            self.simulation_assessment = self._init_simulation_assessment()
            self._pp_log(f"simulation assessment ready ({time.perf_counter() - t_sim:.2f}s)")

            t_aoi = time.perf_counter()
            self._pp_log("building AOI analysis (LLM-enabled)...")
            self.aoi_analysis = self._init_aoi_analysis()
            self._pp_log(f"AOI analysis ready ({time.perf_counter() - t_aoi:.2f}s)")
        else:
            self._pp_log("LLM disabled: running preprocess tasks in parallel...")
            with ThreadPoolExecutor(max_workers=3) as ex:
                f_pref = ex.submit(self._init_preference_cache)
                f_sim = None if self.skip_simulation_assessment else ex.submit(self._init_simulation_assessment)
                f_aoi = ex.submit(self._init_aoi_analysis)
                self.preference_cache = f_pref.result()
                self.simulation_assessment = self._init_simulation_assessment() if f_sim is None else f_sim.result()
                self.aoi_analysis = f_aoi.result()
            self._pp_log("parallel preprocess tasks finished")

        data = {
            "cache_key": key,
            "preference_cache": self.preference_cache,
            "simulation_assessment": self.simulation_assessment,
            "aoi_analysis": self.aoi_analysis,
            "relation_pair_count": self._encode_pair_counts(self.relation_pair_count),
        }
        self._save_json(cache_path, data)
        self.cache_meta = {
            "cache_key": key,
            "cache_path": str(cache_path),
            "cache_hit": False,
        }
        self._pp_log(
            f"cache saved: {cache_path.name} "
            f"(total_elapsed={time.perf_counter() - t0:.2f}s)"
        )

    @staticmethod
    def _dict_to_pb_path(path_dict: Path, path_pb: Path) -> None:
        from map_utils import dict2pb_path

        dict2pb_path(str(path_dict), str(path_pb))

    def _load_resident_profiles(self) -> Dict[str, Dict[str, Any]]:
        p = self.resident_profiles_path
        if not p or (not p.exists()):
            return {}
        try:
            with open(p, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            return {}
        out: Dict[str, Dict[str, Any]] = {}
        if isinstance(data, list):
            for item in data:
                if not isinstance(item, dict):
                    continue
                rid = item.get("id")
                if rid is None:
                    continue
                out[str(rid)] = item
        elif isinstance(data, dict):
            for k, v in data.items():
                if isinstance(v, dict):
                    out[str(k)] = v
        return out

    def _llm_json_query(self, prompt: str) -> Optional[Dict[str, Any]]:
        if self.llm_client is None:
            return None
        msgs = [
            {"role": "system", "content": "You are a precise JSON generator. Reply with JSON object only."},
            {"role": "user", "content": prompt},
        ]
        try:
            resp = self.llm_client.chat(msgs)
            text = self.llm_client.extract_text(resp)
        except Exception:
            return None
        txt = (text or "").strip()
        txt = re.sub(r"^```(?:json)?\s*", "", txt)
        txt = re.sub(r"\s*```$", "", txt)
        try:
            obj = json.loads(txt)
            return obj if isinstance(obj, dict) else None
        except Exception:
            pass
        m = re.search(r"\{[\s\S]*\}", txt)
        if not m:
            return None
        try:
            obj = json.loads(m.group(0))
            return obj if isinstance(obj, dict) else None
        except Exception:
            return None

    def _init_preference_cache(self) -> Dict[str, Any]:
        if self.precomputed_resident_preferences:
            return {
                "resident_preferences": self.precomputed_resident_preferences,
                "source": "resident_preprocess",
                "resident_count": len(self.precomputed_resident_preferences),
                "candidate_types": ["BS", "RC", "PK", "OF", "SC", "HS"],
            }
        return prepare_resident_preference_cache(
            self.graph_index,
            resident_profiles=self.resident_profiles,
            llm_json_fn=self._llm_json_query if self.llm_client else None,
            llm_required=True,
        )

    def _init_road_distance_model(self) -> Optional[WalkingRoadDistance]:
        try:
            return WalkingRoadDistance(str(self.init_pickle_path))
        except Exception:
            return None

    def _init_simulation_assessment(self) -> Dict[str, Any]:
        if self.skip_simulation_assessment:
            return {
                "metrics": {
                    "objective_avg_move_distance": None,
                    "subjective_interview_score": None,
                    "reflection_sentiment_index": None,
                },
                "skipped": True,
                "reason": "skip_simulation_assessment",
            }
        return evaluate_simulation_assessment(
            self.graph_index,
            resident_profiles=self.resident_profiles,
            llm_json_fn=self._llm_json_query if self.llm_client else None,
            llm_required=True,
        )

    def _build_aoi_geometry_and_meta(self):
        geom: Dict[str, Polygon] = {}
        landuse: Dict[str, str] = {}
        names: Dict[str, str] = {}
        for n in self.graph_index.nodes:
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
                    geom[aid] = Polygon(coords)
                except Exception:
                    continue
            landuse[aid] = str(n.get("urban_land_use") or n.get("use") or "")
            names[aid] = str(n.get("name") or "")
        return geom, landuse, names

    @staticmethod
    def _load_landuse_from_pickle(path: Path) -> Dict[str, str]:
        try:
            m = pickle.load(open(path, "rb"))
            aois = m.get("aois") or []
            out = {}
            for a in aois:
                out[str(a.get("id"))] = str(a.get("urban_land_use") or "")
            return out
        except Exception:
            return {}

    def _build_aoi_suffix_index(self) -> Dict[int, List[str]]:
        out: Dict[int, List[str]] = {}
        for aid in self.base_landuse.keys():
            try:
                k = int(aid) % 1000
            except Exception:
                continue
            out.setdefault(k, []).append(aid)
        return out

    @staticmethod
    def _infer_aoi_id_profile(known_aoi_ids: set[str]) -> tuple[str, int]:
        known = sorted(str(aid) for aid in known_aoi_ids if str(aid).isdigit())
        if not known:
            return "", 0
        prefix = known[0]
        for aid in known[1:]:
            i = 0
            max_i = min(len(prefix), len(aid))
            while i < max_i and prefix[i] == aid[i]:
                i += 1
            prefix = prefix[:i]
            if not prefix:
                break
        suffix_width = max(len(aid) - len(prefix) for aid in known)
        return prefix, max(0, suffix_width)

    def _normalize_aoi_id(self, raw: Any) -> str:
        s = str(raw).strip()
        if s in self.base_landuse:
            return s
        digits = "".join(ch for ch in s if ch.isdigit())
        if digits:
            if digits in self.base_landuse:
                return digits
            stripped = digits.lstrip("0") or "0"
            if self.aoi_id_suffix_width > 0 and len(stripped) <= self.aoi_id_suffix_width:
                candidate = f"{self.aoi_id_prefix}{stripped.zfill(self.aoi_id_suffix_width)}"
                if candidate in self.base_landuse:
                    return candidate

            suffix_candidates = {digits}
            if self.aoi_id_suffix_width > 0 and len(stripped) <= self.aoi_id_suffix_width:
                suffix_candidates.add(stripped.zfill(self.aoi_id_suffix_width))
            elif stripped != digits:
                suffix_candidates.add(stripped)
            cand = [
                aid
                for aid in self.base_landuse.keys()
                if any(str(aid).endswith(suffix) for suffix in suffix_candidates)
            ]
            if len(cand) == 1:
                return cand[0]
        return s

    def _normalize_aoi_ids(self, raw_ids: List[Any]) -> List[str]:
        seen = set()
        out = []
        for x in raw_ids:
            aid = self._normalize_aoi_id(x)
            if aid in self.base_landuse and aid not in seen:
                seen.add(aid)
                out.append(aid)
        return out

    def _build_relation_pair_frequency(self) -> Dict[tuple, int]:
        out: Dict[tuple, int] = {}
        for e in self.graph_index.edges_by_type.get("mobility", []):
            pair = parse_relation_pair(e.get("target"))
            if not pair:
                continue
            a0, a1 = str(pair[0]), str(pair[1])
            k = tuple(sorted([a0, a1]))
            out[k] = out.get(k, 0) + 1
        return out

    def _is_negative_comment(self, txt: str) -> bool:
        t = (txt or "").lower()
        bad = [
            "not enough",
            "insufficient",
            "lack",
            "missing",
            "far",
            "inconvenient",
            "bad",
            "poor",
            "no ",
            "need ",
            "should ",
            "problem",
            "crowded",
            "noisy",
            "unsafe",
        ]
        return any(w in t for w in bad)

    def _init_aoi_analysis(self) -> Dict[str, Dict[str, Any]]:
        # Prefer end-to-end LLM parsing/summarization for AOI feedback analysis.
        llm_out = self._init_aoi_analysis_llm()
        if llm_out:
            return llm_out
        # Fallback to local heuristic only if LLM is unavailable or parsing fails.
        return self._init_aoi_analysis_heuristic()

    def _init_aoi_analysis_llm(self) -> Dict[str, Dict[str, Any]]:
        by_aoi: Dict[str, List[str]] = {}
        for e in self.graph_index.edges_by_type.get("obj_reflection", []):
            aid = self._normalize_aoi_id(e.get("target"))
            comments = e.get("comments") or []
            if isinstance(comments, str):
                comments = [comments]
            by_aoi.setdefault(aid, []).extend([str(c) for c in comments if str(c).strip()])

        out: Dict[str, Dict[str, Any]] = {}
        for aid in self.base_landuse.keys():
            out[aid] = {
                "aoi_id": aid,
                "landuse": self.base_landuse.get(aid, ""),
                "modifiable": self.base_landuse.get(aid, "") in MODIFIABLE_LANDUSE,
                "self_total_reflections": 0,
                "self_negative_ratio": 0.0,
                "self_summary": "No targeted land-use-related feedback found.",
                "self_importance": 1,
                "neighbor_total_reflections": 0,
                "neighbor_negative_ratio": 0.0,
                "neighbor_summary": "No targeted surrounding-area feedback found.",
                "neighbor_importance": 1,
            }

        if not self.llm_client:
            return {}
        has_feedback = [aid for aid in self.base_landuse.keys() if len(by_aoi.get(aid, [])) > 0]
        if not has_feedback:
            return out

        # Keep batch small to avoid long-context degradation in one single call.
        batch_size = 12
        batches: List[List[str]] = [has_feedback[i : i + batch_size] for i in range(0, len(has_feedback), batch_size)]
        total_batches = len(batches)
        self._pp_log(
            f"aoi_analysis_llm: feedback_aois={len(has_feedback)}, "
            f"batch_size={batch_size}, total_batches={total_batches}, "
            f"workers={self.aoi_llm_batch_workers}"
        )

        def _run_batch(batch_idx: int, aids: List[str]) -> Dict[str, Any]:
            bt = time.perf_counter()
            payload = []
            for aid in aids:
                payload.append(
                    {
                        "aoi_id": aid,
                        "name": self.aoi_name.get(aid, ""),
                        "landuse": self.base_landuse.get(aid, ""),
                        "modifiable": self.base_landuse.get(aid, "") in MODIFIABLE_LANDUSE,
                        "reflection_count": len(by_aoi.get(aid, [])),
                        "reflections": by_aoi.get(aid, [])[:80],
                    }
                )
            prompt = (
                "You are an urban-planning feedback analyst.\n"
                "For each AOI, read ONLY its own reflection texts and produce structured analysis.\n"
                "Critical definition:\n"
                "- `self_*`: opinions about this AOI itself.\n"
                "- `neighbor_*`: opinions in this AOI's own reflections that talk about surrounding areas/facilities near this AOI.\n"
                "- Do NOT use reflections from other AOIs when producing `neighbor_*`.\n"
                "Output JSON only:\n"
                "{\n"
                "  \"aoi_analysis\": [\n"
                "    {\n"
                "      \"aoi_id\": \"...\",\n"
                "      \"self\": {\n"
                "        \"negative_ratio\": 0-1,\n"
                "        \"summary\": \"one short paragraph (2-5 sentences) that clearly covers consensus and conflicts\"\n"
                "      },\n"
                "      \"neighbor\": {\n"
                "        \"total_reflections\": int,\n"
                "        \"negative_ratio\": 0-1,\n"
                "        \"summary\": \"one short paragraph (2-5 sentences) that clearly covers consensus and conflicts\"\n"
                "      }\n"
                "    }\n"
                "  ]\n"
                "}\n"
                f"Input:\n{json.dumps(payload, ensure_ascii=False)}"
            )
            obj = self._llm_json_query(prompt) or {}
            self._pp_log(
                f"aoi_analysis_llm batch {batch_idx + 1}/{total_batches} done "
                f"(aois={len(aids)}, elapsed={time.perf_counter() - bt:.2f}s)"
            )
            return obj

        batch_results: List[Dict[str, Any]] = [{} for _ in range(total_batches)]
        if self.aoi_llm_batch_workers > 1 and total_batches > 1:
            with ThreadPoolExecutor(max_workers=min(self.aoi_llm_batch_workers, total_batches)) as ex:
                futures = [ex.submit(_run_batch, bi, aids) for bi, aids in enumerate(batches)]
                for bi, fut in enumerate(futures):
                    try:
                        batch_results[bi] = fut.result()
                    except Exception:
                        batch_results[bi] = {}
        else:
            for bi, aids in enumerate(batches):
                batch_results[bi] = _run_batch(bi, aids)

        for bi, obj in enumerate(batch_results):
            rows = obj.get("aoi_analysis") if isinstance(obj, dict) else None
            if not isinstance(rows, list):
                continue
            for item in rows:
                if not isinstance(item, dict):
                    continue
                aid = self._normalize_aoi_id(item.get("aoi_id"))
                if aid not in out:
                    continue
                s = item.get("self") if isinstance(item.get("self"), dict) else {}
                n = item.get("neighbor") if isinstance(item.get("neighbor"), dict) else {}
                out[aid].update(
                    {
                        "self_total_reflections": int(len(by_aoi.get(aid, []))),
                        "self_negative_ratio": round(max(0.0, min(1.0, float(s.get("negative_ratio") or 0.0))), 4),
                        "self_summary": str(s.get("summary") or out[aid]["self_summary"]),
                        "neighbor_total_reflections": max(
                            0,
                            min(
                                int(len(by_aoi.get(aid, []))),
                                int(n.get("total_reflections") or 0),
                            ),
                        ),
                        "neighbor_negative_ratio": round(max(0.0, min(1.0, float(n.get("negative_ratio") or 0.0))), 4),
                        "neighbor_summary": str(n.get("summary") or out[aid]["neighbor_summary"]),
                    }
                )
        # Add importance only after stats/summary are ready.
        # Unified pipeline:
        # 1) LLM scores urgency from (count, ratio, summary),
        # 2) deterministic thresholding filters low-signal AOIs,
        # 3) global ranking converts candidates to level 1/2/3.
        self_raw_score, neighbor_raw_score = self._llm_urgency_scores(out, has_feedback)
        self_levels = self._build_importance_levels_with_thresholds(out, self_raw_score, scope="self")
        nei_levels = self._build_importance_levels_with_thresholds(out, neighbor_raw_score, scope="neighbor")
        for aid in out.keys():
            out[aid]["self_importance"] = int(self_levels.get(aid, 1))
            out[aid]["neighbor_importance"] = int(nei_levels.get(aid, 1))
        return out

    def _build_importance_levels_with_thresholds(
        self,
        analysis_map: Dict[str, Dict[str, Any]],
        llm_scores: Dict[str, float],
        scope: str,
    ) -> Dict[str, int]:
        """Assign urgency levels with fixed global quotas by modifiable-AOI population.

        Rule requested by user:
        - Base population = total modifiable AOIs.
        - Level 3 = top 15% of base population.
        - Level 2 = next 25% of base population.
        - Level 1 = remaining AOIs.
        - For self-scope, immutable AOIs are always level 1.
        """
        if scope not in {"self", "neighbor"}:
            scope = "self"
        levels: Dict[str, int] = {aid: 1 for aid in analysis_map.keys()}

        modifiable_ids = [
            aid for aid, lu in self.base_landuse.items() if str(lu) in MODIFIABLE_LANDUSE and aid in analysis_map
        ]
        mod_n = len(modifiable_ids)
        if mod_n <= 0:
            return levels

        # Fixed quotas from modifiable-AOI population.
        k3 = max(1, int(round(mod_n * float(self.importance_level3_ratio))))
        k2 = max(1, int(round(mod_n * float(self.importance_level2_ratio))))
        k3 = min(k3, mod_n)
        k2 = min(k2, max(mod_n - k3, 0))

        # Rank all modifiable AOIs by current scope score.
        # Deterministic tie-breaker by AOI id.
        ranked = sorted(
            [(aid, float(llm_scores.get(aid, 0.0))) for aid in modifiable_ids],
            key=lambda kv: (kv[1], kv[0]),
            reverse=True,
        )
        ids3 = {aid for aid, _ in ranked[:k3]}
        ids2 = {aid for aid, _ in ranked[k3 : k3 + k2]}

        for aid in ids3:
            levels[aid] = 3
        for aid in ids2:
            levels[aid] = 2

        # Immutable AOIs remain level 1 (already default). This explicitly covers
        # the self-scope requirement that non-modifiable types are level 1.
        return levels

    def _llm_urgency_scores(
        self,
        analysis_map: Dict[str, Dict[str, Any]],
        focus_aoi_ids: List[str],
    ) -> tuple[Dict[str, float], Dict[str, float]]:
        """LLM-based urgency scoring from prepared stats/summaries.

        Returns raw score maps (0-100) for self and neighbor channels.
        """
        self_score: Dict[str, float] = {}
        nei_score: Dict[str, float] = {}
        if not self.llm_client:
            # Fallback to local numeric signal if LLM is unavailable.
            for aid, rec in analysis_map.items():
                st = int(rec.get("self_total_reflections") or 0)
                sr = float(rec.get("self_negative_ratio") or 0.0)
                self_score[aid] = self._urgency_signal(st, int(round(st * sr)), sr)
                nt = int(rec.get("neighbor_total_reflections") or 0)
                nr = float(rec.get("neighbor_negative_ratio") or 0.0)
                nei_score[aid] = self._urgency_signal(nt, int(round(nt * nr)), nr)
            return self_score, nei_score

        batch_size = 18
        total_batches = (len(focus_aoi_ids) + batch_size - 1) // batch_size
        self._pp_log(
            f"urgency_llm: aois={len(focus_aoi_ids)}, batch_size={batch_size}, total_batches={total_batches}"
        )
        for i in range(0, len(focus_aoi_ids), batch_size):
            aids = focus_aoi_ids[i : i + batch_size]
            bidx = (i // batch_size) + 1
            bt = time.perf_counter()
            payload = []
            for aid in aids:
                rec = analysis_map.get(aid, {})
                payload.append(
                    {
                        "aoi_id": aid,
                        "self_total_reflections": int(rec.get("self_total_reflections") or 0),
                        "self_negative_ratio": float(rec.get("self_negative_ratio") or 0.0),
                        "self_summary": str(rec.get("self_summary") or ""),
                        "neighbor_total_reflections": int(rec.get("neighbor_total_reflections") or 0),
                        "neighbor_negative_ratio": float(rec.get("neighbor_negative_ratio") or 0.0),
                        "neighbor_summary": str(rec.get("neighbor_summary") or ""),
                    }
                )
            prompt = (
                "You are scoring AOI renewal urgency for urban planning.\n"
                "For each AOI, score two channels from 0 to 100:\n"
                "- `self_urgency_score`: urgency for updating the AOI itself.\n"
                "- `neighbor_urgency_score`: urgency for updating surrounding area around this AOI.\n"
                "Use only the provided count, negative ratio, and summary text.\n"
                "Scoring standard:\n"
                "- Near-zero score when feedback is mostly positive/neutral and negative_ratio is near 0.\n"
                "- Medium score only when there is clear dissatisfaction signal in count+ratio+text.\n"
                "- High score only when dissatisfaction is both frequent and substantial.\n"
                "Output JSON only:\n"
                "{\n"
                "  \"scores\": [\n"
                "    {\"aoi_id\":\"...\", \"self_urgency_score\":0-100, \"neighbor_urgency_score\":0-100}\n"
                "  ]\n"
                "}\n"
                f"Input:\n{json.dumps(payload, ensure_ascii=False)}"
            )
            obj = self._llm_json_query(prompt) or {}
            self._pp_log(
                f"urgency_llm batch {bidx}/{total_batches} done "
                f"(aois={len(aids)}, elapsed={time.perf_counter() - bt:.2f}s)"
            )
            rows = obj.get("scores") if isinstance(obj, dict) else None
            if not isinstance(rows, list):
                continue
            for item in rows:
                if not isinstance(item, dict):
                    continue
                aid = self._normalize_aoi_id(item.get("aoi_id"))
                if aid not in analysis_map:
                    continue
                try:
                    sv = float(item.get("self_urgency_score"))
                    nv = float(item.get("neighbor_urgency_score"))
                except Exception:
                    continue
                self_score[aid] = max(0.0, min(100.0, sv))
                nei_score[aid] = max(0.0, min(100.0, nv))

        # Fill any missing AOIs with deterministic fallback score.
        for aid, rec in analysis_map.items():
            if aid not in self_score:
                st = int(rec.get("self_total_reflections") or 0)
                sr = float(rec.get("self_negative_ratio") or 0.0)
                self_score[aid] = self._urgency_signal(st, int(round(st * sr)), sr)
            if aid not in nei_score:
                nt = int(rec.get("neighbor_total_reflections") or 0)
                nr = float(rec.get("neighbor_negative_ratio") or 0.0)
                nei_score[aid] = self._urgency_signal(nt, int(round(nt * nr)), nr)
        return self_score, nei_score

    def _init_aoi_analysis_heuristic(self) -> Dict[str, Dict[str, Any]]:
        by_aoi: Dict[str, List[str]] = {}
        for e in self.graph_index.edges_by_type.get("obj_reflection", []):
            aid = self._normalize_aoi_id(e.get("target"))
            comments = e.get("comments") or []
            if isinstance(comments, str):
                comments = [comments]
            by_aoi.setdefault(aid, []).extend([str(c) for c in comments if str(c).strip()])

        out: Dict[str, Dict[str, Any]] = {}
        self_raw_score: Dict[str, float] = {}
        for aid in self.base_landuse.keys():
            comments = by_aoi.get(aid, [])
            negatives = [c for c in comments if self._is_negative_comment(c)]
            total = len(comments)
            neg_ratio = self._smoothed_negative_ratio(total, len(negatives))
            self_raw_score[aid] = self._urgency_signal(total, len(negatives), neg_ratio)
            out[aid] = {
                "aoi_id": aid,
                "landuse": self.base_landuse.get(aid, ""),
                "modifiable": self.base_landuse.get(aid, "") in MODIFIABLE_LANDUSE,
                "self_total_reflections": total,
                "self_negative_ratio": round(neg_ratio, 4),
                "self_summary": self._simple_comment_summary(comments, 3),
                "self_importance": 1,
                # Heuristic fallback cannot reliably isolate neighbor mentions from same-AOI text.
                "neighbor_total_reflections": 0,
                "neighbor_negative_ratio": 0.0,
                "neighbor_summary": "Neighbor-focused analysis requires LLM parsing of this AOI's reflections.",
                "neighbor_importance": 1,
            }
        self_levels = self._rank_to_urgency_levels(self_raw_score)
        for aid, rec in out.items():
            rec["self_importance"] = int(self_levels.get(aid, 1))
        return out

    @staticmethod
    def _urgency_signal(total_reflections: int, negative_reflections: int, negative_ratio: float) -> float:
        """Compute urgency signal from volume + negativity for later global ranking."""
        total = max(0, int(total_reflections))
        neg = max(0, int(negative_reflections))
        ratio = max(0.0, min(float(negative_ratio), 1.0))
        # Weight stronger dissatisfaction more heavily while keeping volume effect.
        return (0.7 * total) + (1.8 * neg) + (6.0 * ratio)

    @staticmethod
    def _smoothed_negative_ratio(total_reflections: int, negative_reflections: int) -> float:
        """Stabilize negative ratio for low-sample AOIs with Beta(1,1) smoothing."""
        total = max(0, int(total_reflections))
        neg = max(0, int(negative_reflections))
        return float((neg + 1.0) / (total + 2.0)) if total >= 0 else 0.0

    @staticmethod
    def _rank_to_urgency_levels(score_map: Dict[str, float]) -> Dict[str, int]:
        """Convert urgency scores to 1/2/3 with strict global ranking ratios.

        Design:
        - Level 3: top ~15% non-zero AOIs.
        - Level 2: next ~25% non-zero AOIs.
        - Level 1: others / zero-signal AOIs.
        """
        nz = [(aid, float(v)) for aid, v in score_map.items() if float(v) > 0.0]
        if not nz:
            return {k: 1 for k in score_map.keys()}
        nz.sort(key=lambda kv: kv[1], reverse=True)
        n = len(nz)
        k3 = max(1, int(round(n * 0.15)))
        k2 = max(1, int(round(n * 0.25)))
        ids3 = {aid for aid, _ in nz[:k3]}
        ids2 = {aid for aid, _ in nz[k3 : k3 + k2]}
        out: Dict[str, int] = {}
        for aid in score_map.keys():
            if aid in ids3:
                out[aid] = 3
            elif aid in ids2:
                out[aid] = 2
            else:
                out[aid] = 1
        return out

    @staticmethod
    def _simple_comment_summary(comments: List[str], top_k: int = 2) -> str:
        if not comments:
            return "No targeted land-use-related feedback found."
        keys = {
            "park": ["park", "green", "open space"],
            "school": ["school", "education"],
            "hospital": ["hospital", "clinic", "medical"],
            "office": ["office", "workplace", "job"],
            "business": ["shop", "mall", "business", "commercial"],
            "recreation": ["sports", "recreation", "leisure", "playground"],
            "industry": ["industrial", "factory", "logistics", "infrastructure"],
            "traffic": ["traffic", "transport", "road", "walk"],
            "noise_safety": ["noisy", "noise", "unsafe", "safety"],
        }
        score = []
        blob = " ".join([c.lower() for c in comments])
        for k, words in keys.items():
            cnt = sum(blob.count(w) for w in words)
            if cnt > 0:
                score.append((k, cnt))
        score.sort(key=lambda x: x[1], reverse=True)
        if not score:
            return "Residents left feedback, but themes are dispersed and no stable shared demand is dominant."
        top = [x[0] for x in score[:top_k]]
        return (
            "Residents commonly emphasize "
            + ", ".join(top)
            + "; these requests appear repeatedly and should be prioritized in local land-use adjustment."
        )

    @staticmethod
    def _simple_conflict_summary(comments: List[str]) -> str:
        if not comments:
            return "No clear conflicting demands."
        blob = " ".join([c.lower() for c in comments])
        has_more_dev = any(w in blob for w in ["more", "add", "increase", "build"])
        has_less_dev = any(w in blob for w in ["reduce", "less", "limit", "avoid"])
        if has_more_dev and has_less_dev:
            return "There is a clear conflict between expansion-oriented and control-oriented preferences, so coordination trade-offs are required."
        return "No strong conflict signal is observed; most opinions are directionally aligned."

    def _llm_text_query(self, prompt: str, image_path: Optional[str] = None) -> str:
        if self.llm_client is None:
            return ""
        content: Any = prompt
        if image_path:
            p = Path(str(image_path))
            if p.exists():
                try:
                    raw = p.read_bytes()
                    mime = mimetypes.guess_type(str(p))[0] or "image/png"
                    data_url = f"data:{mime};base64,{base64.b64encode(raw).decode('ascii')}"
                    content = [{"type": "text", "text": prompt}, {"type": "image_url", "image_url": {"url": data_url}}]
                except Exception:
                    content = prompt
        msgs = [
            {"role": "system", "content": "You are a concise urban planning analyst."},
            {"role": "user", "content": content},
        ]
        try:
            r = self.llm_client.chat(msgs)
            return self.llm_client.extract_text(r).strip()
        except Exception:
            return ""

    def _summarize_aoi_group_text(self, aoi_ids: List[str], focus_aoi: Optional[str], group_name: str, image_path: Optional[str] = None) -> str:
        rows = []
        for aid in aoi_ids[:40]:
            rec = self.aoi_analysis.get(aid, {})
            rows.append(
                {
                    "aoi_id": aid,
                    "landuse": rec.get("landuse"),
                    "self_summary": rec.get("self_summary") or rec.get("self_common_summary"),
                    "self_importance": rec.get("self_importance"),
                    "neighbor_summary": rec.get("neighbor_summary") or rec.get("neighbor_common_summary"),
                    "neighbor_importance": rec.get("neighbor_importance"),
                }
            )
        if not rows:
            return "No AOI evidence available for this region."
        target_ids = [x.get("aoi_id") for x in rows if x.get("aoi_id")]
        prompt = (
            "You are summarizing AOI evidence for planning based on an INITIAL-map image and provided structured AOI evidence. "
            "Output exactly ONE compact paragraph (3-6 sentences). "
            "You must jointly compress both `self_summary` and `neighbor_summary` across AOIs, merge co-referential needs, "
            "and preserve key differences/conflicts that should not be merged. Keep it planning-oriented and spatially grounded. "
            "Do not output bullets, lists, or section headers. "
            "Must explicitly mention 3-12 AOI IDs from the candidates using the format 'AOI <id>'. "
            "Treat the image as initial-plan grounding (not temporary modified plan).\n"
            f"Group: {group_name}. Focus AOI: {focus_aoi or 'N/A'}.\n"
            f"AOI ID candidates: {target_ids}\n"
            f"AOI evidence:\n{json.dumps(rows, ensure_ascii=False)}"
        )
        txt = self._llm_text_query(prompt, image_path=image_path)
        if txt:
            return txt
        # fallback
        top_self = sorted(rows, key=lambda x: int(x.get("self_importance") or 1), reverse=True)[:3]
        top_nei = sorted(rows, key=lambda x: int(x.get("neighbor_importance") or 1), reverse=True)[:3]
        anchor_ids = ", ".join([str(x.get("aoi_id")) for x in rows[:4]])
        return (
            f"AOI {anchor_ids}: region signal centers on {', '.join([x.get('self_summary') or 'mixed needs' for x in top_self])}; "
            f"neighbor pressure suggests {', '.join([x.get('neighbor_summary') or 'mixed pressure' for x in top_nei])}, "
            "with moderate cross-area conflicts."
        )

    def _ensure_geojson_snapshot(self) -> None:
        if self.geojson_snapshot_path.exists():
            return
        ws = MapWorkspace(repo_root=self.repo_root, output_dir=self.run_dir)
        ws.ensure_geojson(pb_path=self.init_pb_path, geojson_cache=self.geojson_snapshot_path)
