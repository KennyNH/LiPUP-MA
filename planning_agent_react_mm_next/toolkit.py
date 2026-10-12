"""Tool registry for multimodal planning agent."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

try:
    from .graph_tools import (
        GraphIndex,
        aoi_landuse_map,
    )
    from .llm_client import OpenAICompatibleClient
    from .map_tools import MapWorkspace
except ImportError:
    from graph_tools import (
        GraphIndex,
        aoi_landuse_map,
    )
    from llm_client import OpenAICompatibleClient
    from map_tools import MapWorkspace
try:
    from .planning_env import PlanningEnvironment
except ImportError:
    from planning_env import PlanningEnvironment


@dataclass
class ToolSpec:
    name: str
    description: str
    input_schema: Dict[str, Any]
    fn: Callable[[Dict[str, Any]], Dict[str, Any]]


class PlanningToolkit:
    def __init__(
        self,
        repo_root: Path,
        output_dir: Path,
        graph_index: GraphIndex,
        map_workspace: MapWorkspace,
        pb_path: Path,
        geojson_cache_path: Path,
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
        self.output_dir = output_dir
        self.graph_index = graph_index
        self.map_workspace = map_workspace
        self.pb_path = pb_path
        self.geojson_cache_path = geojson_cache_path
        self.init_pickle_path = init_pickle_path
        self.latest_map_path: Optional[Path] = None
        self.temp_plan_copy: Dict[str, str] = dict(aoi_landuse_map(graph_index))
        self.env = PlanningEnvironment(
            repo_root=repo_root,
            run_dir=output_dir,
            graph_index=graph_index,
            init_pb_path=pb_path,
            init_pickle_path=init_pickle_path,
            llm_client=llm_client,
            resident_profiles_path=resident_profiles_path,
            resident_preferences=resident_preferences,
            importance_level3_ratio=importance_level3_ratio,
            importance_level2_ratio=importance_level2_ratio,
            service_radius_m=service_radius_m,
            id_service_radius_m=id_service_radius_m,
            ecology_radius_m=ecology_radius_m,
            skip_simulation_assessment=skip_simulation_assessment,
        )
        self.tools = self._build_tools()

    @staticmethod
    def _is_missing_required(value: Any) -> bool:
        if value is None:
            return True
        if isinstance(value, str):
            return not value.strip()
        if isinstance(value, (list, dict, tuple, set)):
            return len(value) == 0
        return False

    def _normalize_tool_args(self, tool_name: str, args: Dict[str, Any]) -> Dict[str, Any]:
        normalized = dict(args or {})

        def alias(target: str, *sources: str) -> None:
            if target in normalized and not self._is_missing_required(normalized.get(target)):
                return
            for source in sources:
                if source in normalized and not self._is_missing_required(normalized.get(source)):
                    normalized[target] = normalized[source]
                    return

        if tool_name == "query_top_degree_aois":
            alias("top_n", "top_k")

        if tool_name in {"render_map", "zoom_in"}:
            alias("aoi_ids", "highlight_aoi_ids")

        if tool_name == "query_aoi_details":
            if "aoi_ids" not in normalized and not self._is_missing_required(normalized.get("aoi_id")):
                normalized["aoi_ids"] = [normalized.get("aoi_id")]

        if tool_name == "query_related_aois":
            if "aoi_id" not in normalized:
                aoi_ids = normalized.get("aoi_ids")
                if isinstance(aoi_ids, list) and aoi_ids:
                    normalized["aoi_id"] = aoi_ids[0]
                elif isinstance(aoi_ids, str) and aoi_ids.strip():
                    normalized["aoi_id"] = aoi_ids

        if tool_name in {"query_aois_in_bbox", "summarize_region_with_vlm", "zoom_in"}:
            bbox = normalized.get("bbox")
            if isinstance(bbox, (list, tuple)) and len(bbox) == 4:
                normalized.setdefault("x1", bbox[0])
                normalized.setdefault("y1", bbox[1])
                normalized.setdefault("x2", bbox[2])
                normalized.setdefault("y2", bbox[3])
            center = normalized.get("center")
            width = normalized.get("width")
            height = normalized.get("height")
            if (
                isinstance(center, dict)
                and "x" in center
                and "y" in center
                and width is not None
                and height is not None
            ):
                cx = float(center["x"])
                cy = float(center["y"])
                half_w = float(width) / 2.0
                half_h = float(height) / 2.0
                normalized.setdefault("x1", cx - half_w)
                normalized.setdefault("y1", cy - half_h)
                normalized.setdefault("x2", cx + half_w)
                normalized.setdefault("y2", cy + half_h)

        if tool_name == "query_aois_by_landuse":
            alias("landuse_types", "landuse", "landuse_type")

        if tool_name == "calculate_aoi_distances":
            if "aoi_id_1" not in normalized:
                source_ids = normalized.get("source_aoi_ids")
                if isinstance(source_ids, list) and source_ids:
                    normalized["aoi_id_1"] = source_ids[0]
                elif isinstance(source_ids, str) and source_ids.strip():
                    normalized["aoi_id_1"] = source_ids
            if "aoi_id_2" not in normalized:
                target_ids = normalized.get("target_aoi_ids")
                if isinstance(target_ids, list) and target_ids:
                    normalized["aoi_id_2"] = target_ids[0]
                elif isinstance(target_ids, str) and target_ids.strip():
                    normalized["aoi_id_2"] = target_ids

        if tool_name == "apply_temp_plan_changes":
            alias("plan", "changes", "update_dict", "edits")

        return normalized

    def _validate_tool_args(self, tool_name: str, args: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        spec = self.tools[tool_name]
        if tool_name in {"query_aois_in_bbox", "summarize_region_with_vlm", "zoom_in"} and "bbox_rel" in args:
            return {
                "error": "coordinate_space_mismatch",
                "tool": tool_name,
                "message": (
                    "`bbox_rel` is only a Stage-1 normalized focus hint. This tool requires absolute "
                    "map coordinates in meters/positions coordinates via x1,y1,x2,y2. Use the "
                    "`bbox_abs` field from stage_requirements when available, or call a map/query "
                    "tool to obtain absolute coordinates."
                ),
                "expected_next_action": (
                    "Retry the same tool with absolute numeric x1,y1,x2,y2, or choose an AOI-list "
                    "tool such as summarize_aoi_list_with_vlm/query_aoi_details."
                ),
                "received_args": args,
            }
        required = list(spec.input_schema.get("required") or [])
        missing = [key for key in required if self._is_missing_required(args.get(key))]
        if missing:
            return {
                "error": "invalid_tool_args",
                "tool": tool_name,
                "message": f"Missing required fields: {missing}",
                "expected_schema": spec.input_schema,
                "received_args": args,
            }
        return None

    @staticmethod
    def _compact_map_result(map_info: Dict[str, Any], map_version: str, aoi_ids: Optional[List[Any]]) -> Dict[str, Any]:
        return {
            "success": bool(map_info.get("success", False)),
            "image_attached": bool(map_info.get("output_path")),
            "_image_path": map_info.get("output_path"),
        }

    def _tool_render_map(self, args: Dict[str, Any]) -> Dict[str, Any]:
        mv = str(args.get("map_version", "initial"))
        aids = args.get("aoi_ids")
        map_info = self.env.render_map_compact(
            map_version=mv,
            aoi_ids=aids,
            output_name=None,
            view_bbox_rel=None,
            highlight_changed=True,
        )
        return self._compact_map_result(map_info, map_version=mv, aoi_ids=aids)

    def _tool_zoom_in(self, args: Dict[str, Any]) -> Dict[str, Any]:
        mv = str(args.get("map_version", "initial"))
        aids = args.get("aoi_ids")
        map_info = self.env.render_map_compact(
            map_version=mv,
            aoi_ids=aids,
            output_name=None,
            view_bbox_rel=[float(args.get("x1")), float(args.get("y1")), float(args.get("x2")), float(args.get("y2"))],
            highlight_changed=True,
        )
        return self._compact_map_result(map_info, map_version=mv, aoi_ids=aids)

    def _build_tools(self) -> Dict[str, ToolSpec]:
        return {
            "query_top_degree_aois": ToolSpec(
                "query_top_degree_aois",
                "Purpose: Query top-N high-traffic AOIs from realized mobility trajectories. "
                "Tip: use `modifiable_only=True` to exclude immutable AOIs (RS and OS), and `False` when you need full city context; try with different top-n. "
                "Input: `top_n` and optional `modifiable_only`. Output: ranked AOIs with endpoint visit counts and land-use.",
                {
                    "type": "object",
                    "properties": {
                        "top_n": {"type": "integer", "default": 10},
                        "modifiable_only": {"type": "boolean", "default": True},
                    },
                    "required": [],
                },
                lambda args: self.env.query_top_degree_aois(
                    top_n=int(args.get("top_n", 10)),
                    modifiable_only=bool(args.get("modifiable_only", True)),
                ),
            ),
            "query_aoi_self_importance": ToolSpec(
                "query_aoi_self_importance",
                "Purpose: retrieve AOIs by self-side land-use renewal urgency. Tip: start from larger number; the tool only provides overview, more detailed queries are needed to support lamd-use modification."
                "Input: exact urgency level (1-3; 3 = most urgent, the ratio is [3: 15%; 2:25%; 1:60%]). Output: AOI IDs whose SELF land-use renewal urgency (summarized from resident feedbacks) equals the input.",
                {
                    "type": "object",
                    "properties": {"importance_level": {"type": "integer"}},
                    "required": ["importance_level"],
                },
                lambda args: self.env.query_importance_ids(importance_level=int(args.get("importance_level", 3)), scope="self"),
            ),
            "query_aoi_neighbor_importance": ToolSpec(
                "query_aoi_neighbor_importance",
                "Purpose: retrieve AOIs by surrounding-area renewal urgency. "
                "Tip: combine with self urgency for conflict/coordination checks. "
                "Input: exact urgency level (1-3; 3 = most urgent, the ratio is [3: 15%; 2:25%; 1:60%]). Output: AOI IDs whose NEIGHBOR zone land-use renewal urgency (summarized from resident feedbacks) equals the input.",
                {
                    "type": "object",
                    "properties": {"importance_level": {"type": "integer"}},
                    "required": ["importance_level"],
                },
                lambda args: self.env.query_importance_ids(importance_level=int(args.get("importance_level", 3)), scope="neighbor"),
            ),
            "query_aois_by_landuse": ToolSpec(
                "query_aois_by_landuse",
                "Purpose: query AOI IDs by land-use type(s). "
                "Tip: supports both land-use code and full name, single or multiple input (comma-separated or list).",
                {
                    "type": "object",
                    "properties": {
                        "landuse_types": {
                            "oneOf": [
                                {"type": "string"},
                                {"type": "array", "items": {"type": "string"}},
                            ]
                        }
                    },
                    "required": ["landuse_types"],
                },
                lambda args: self.env.query_aois_by_landuse(args.get("landuse_types")),
            ),
            "query_aois_in_bbox": ToolSpec(
                "query_aois_in_bbox",
                "Purpose: spatially select AOIs by rectangular area. Tip: pair with `zoom_in`/`summarize_region_with_vlm` for regional diagnosis, DO NOT query too large area. "
                "Input: absolute `[x1,y1,x2,y2]` in positions coordinates, not Stage-1 `bbox_rel`. Output: AOI IDs inside box and count.",
                {
                    "type": "object",
                    "properties": {
                        "x1": {"type": "number"},
                        "y1": {"type": "number"},
                        "x2": {"type": "number"},
                        "y2": {"type": "number"},
                    },
                    "required": ["x1", "y1", "x2", "y2"],
                },
                lambda args: self.env.query_aois_in_bbox(
                    x1=float(args.get("x1")),
                    y1=float(args.get("y1")),
                    x2=float(args.get("x2")),
                    y2=float(args.get("y2")),
                ),
            ),
            "query_aoi_details": ToolSpec(
                "query_aoi_details",
                "Purpose: inspect structured AOI evidence for planning decisions. Tip: use after candidate AOIs are selected. DO NOT query too many AOIs. "
                "Input: AOI IDs as numeric strings. Full IDs are preferred; `AOI_97`-style text is auto-cleaned only as a repair fallback. "
                "Output per AOI: name, land-use state, centroid, immutability flag, stats/summary/renewal urgency of neighbor-area and self-area (if modifiable); "
                ,
                {
                    "type": "object",
                    "properties": {"aoi_ids": {"type": "array", "items": {"type": "string"}}},
                    "required": ["aoi_ids"],
                },
                lambda args: self.env.query_aoi_details(aoi_ids=args.get("aoi_ids") or []),
            ),
            "summarize_region_with_vlm": ToolSpec(
                "summarize_region_with_vlm",
                "Purpose: compress and aggregate planning evidence for one region. "
                "Tip: call this after bbox selection to merge common needs and conflicts. "
                "Input: absolute region bbox `[x1,y1,x2,y2]` in positions coordinates, not Stage-1 `bbox_rel` (smaller zone is preferred). "
                "Output: AOI IDs within region and one compact regional summary paragraph that jointly aggregates self+neighbor evidence. "
                "Tool internally renders a temporary INITIAL-map zoom image for VLM-assisted spatial reasoning.",
                {
                    "type": "object",
                    "properties": {
                        "x1": {"type": "number"},
                        "y1": {"type": "number"},
                        "x2": {"type": "number"},
                        "y2": {"type": "number"},
                    },
                    "required": ["x1", "y1", "x2", "y2"],
                },
                lambda args: self.env.summarize_region(
                    x1=float(args.get("x1")),
                    y1=float(args.get("y1")),
                    x2=float(args.get("x2")),
                    y2=float(args.get("y2")),
                    output_name=None,
                ),
            ),
            "summarize_aoi_list_with_vlm": ToolSpec(
                "summarize_aoi_list_with_vlm",
                "Purpose: compress and aggregate planning evidence for a specified AOI set. Tip: use for cross-AOI synthesis before editing, DO NOT query too many AOIs (at most 30). "
                "Input: numeric AOI ID list, preferably full IDs from tools/stage requirements. Output: one compact summary paragraph preserving both common patterns and key differences, with merged self+neighbor evidence. "
                "Tool internally renders a temporary INITIAL-map image for VLM-assisted spatial reasoning.",
                {
                    "type": "object",
                    "properties": {
                        "aoi_ids": {"type": "array", "items": {"type": "string"}},
                    },
                    "required": ["aoi_ids"],
                },
                lambda args: self.env.summarize_aoi_list(
                    aoi_ids=args.get("aoi_ids") or [],
                    output_name=None,
                ),
            ),
            "query_related_aois": ToolSpec(
                "query_related_aois",
                "Purpose: find AOIs related to a focus AOI by space proximity and mobility trajectory, then aggregate and summarize evidence. Tip: compare nearby needs vs long-range mobility coupling before editing. "
                "Input: numeric `aoi_id`, optional `top_n`, `radius_m`; full ID is preferred. Output: nearby AOIs, mobility-linked top-N AOIs, and compact paragraph summaries for both groups. "
                "Summaries are grounded on INITIAL-map rendering and compress both self+neighbor evidence.",
                {
                    "type": "object",
                    "properties": {
                        "aoi_id": {"type": "string"},
                        "top_n": {"type": "integer", "default": 3},
                        "radius_m": {"type": "number", "default": 50.0},
                    },
                    "required": ["aoi_id"],
                },
                lambda args: self.env.query_related_aois(
                    aoi_id=args.get("aoi_id"),
                    top_n=int(args.get("top_n", 3)),
                    radius_m=float(args.get("radius_m", 50.0)),
                ),
            ),
            "calculate_aoi_distances": ToolSpec(
                "calculate_aoi_distances",
                "Purpose: compare AOI-to-AOI distance under different definitions. "
                "Tip: use `road` for accessibility decisions and `straight_line` for geometric proximity checks. To support quantitative analysis"
                "Input: `aoi_id_1`, `aoi_id_2`, optional `distance_type` (`road`|`straight_line`|`both`, alias `boundary`). "
                "Output requested distance(s) and road calculation mode.",
                {
                    "type": "object",
                    "properties": {
                        "aoi_id_1": {"type": "string"},
                        "aoi_id_2": {"type": "string"},
                        "distance_type": {"type": "string", "default": "both"},
                    },
                    "required": ["aoi_id_1", "aoi_id_2"],
                },
                lambda args: self.env.query_aoi_distances(
                    args.get("aoi_id_1"),
                    args.get("aoi_id_2"),
                    distance_type=str(args.get("distance_type", "both")),
                ),
            ),
            "render_map": ToolSpec(
                "render_map",
                "Purpose: render map for grounding before decisions, initial or modified version, and AOI ID labels can be specified. Tip: call this early and frequently, pair with query-related tools for area spatial grounding."
                "You can choose initial or current-modified land-use version. Prefer the exact full AOI IDs returned by tools for label visualization. "
                "Default map version is `initial`; use `current` only when you need to inspect temporary modifications. "
                "Input: `map_version` (`initial` or `current`) and optional `aoi_ids` (full/int/3-digit suffix supported, used for label targets only). "
                "The map itself keeps full AOI spatial context. Output: image attachment flag.",
                {
                    "type": "object",
                    "properties": {
                        "map_version": {"type": "string", "default": "initial"},
                        "aoi_ids": {"type": "array", "items": {"type": "string"}},
                    },
                    "required": [],
                },
                lambda args: self._tool_render_map(args),
            ),
            "zoom_in": ToolSpec(
                "zoom_in",
                "Purpose: inspect local spatial structure in detail based on rendered map. Tip: call after identifying candidate AOIs or bbox. Pair with query-related tools,"
                "or render_map tools for more focused spacial understanding. "
                "Default map version is `initial`; use `current` only when you need to inspect temporary modifications. "
                "Input: bbox `[x1,y1,x2,y2]` (relatively smaller zone is preferred), optional `map_version`(`initial` or `current`), optional `aoi_ids` (label targets, full/int/3-digit suffix supported). "
                "Map still keeps full AOI context and then zooms by bbox. Output: zoomed image attachment flag.",
                {
                    "type": "object",
                    "properties": {
                        "map_version": {"type": "string", "default": "initial"},
                        "x1": {"type": "number"},
                        "y1": {"type": "number"},
                        "x2": {"type": "number"},
                        "y2": {"type": "number"},
                        "aoi_ids": {"type": "array", "items": {"type": "string"}},
                    },
                    "required": ["x1", "y1", "x2", "y2"],
                },
                lambda args: self._tool_zoom_in(args),
            ),
            "apply_temp_plan_changes": ToolSpec(
                "apply_temp_plan_changes",
                "Purpose: apply and evaluate a land-use update batch, to modify theland-use geo file."
                "Tip: **Better verify modification constraints with query/visual tools before calling**, and iterate small batches. Pay attention to optimize the returned matrics."
                "You can modify the land-use gradually with think (preferred), or change a lot after think."
                "Rollbacks (undo to the original land-use if unsatisfied) and multiple version comparison are encouraged."
                "Input format: {\"aoi_id\":\"TARGET_LANDUSE\"}. "
                "**NOTE**: Each call's modification is cumulative based on the previous modification (not reset to initial version). Finally the modified file will be the final plan."
                "Performs validity/constraint checks, ignores invalid/no-change entries, and returns: step changes, all changes-vs-original, invalid input details, rendered map (all modified AOIs will be highlighted), and service/ecology/satisfaction metrics.",
                {
                    "type": "object",
                    "properties": {
                        "plan": {"type": "object", "additionalProperties": {"type": "string"}},
                    },
                    "required": ["plan"],
                },
                lambda args: self.env.apply_plan_and_evaluate(
                    plan=args.get("plan") or {},
                    render_output_name=None,
                ),
            ),
            "query_all_modifications": ToolSpec(
                "query_all_modifications",
                "Purpose: inspect all cumulative edits against the initial plan. Tip: call before finalizing to avoid accidental drift. Output: all modified AOIs with from/to types.",
                {"type": "object", "properties": {}, "required": []},
                lambda args: self.env.get_all_modifications(),
            ),
        }

    def execute(self, tool_name: str, args: Dict[str, Any]) -> Dict[str, Any]:
        if tool_name not in self.tools:
            return {"error": f"Unknown tool: {tool_name}"}
        normalized_args = self._normalize_tool_args(tool_name, args)
        validation_error = self._validate_tool_args(tool_name, normalized_args)
        if validation_error is not None:
            return validation_error
        try:
            return self.tools[tool_name].fn(normalized_args)
        except Exception as e:
            return {"error": str(e), "tool": tool_name}

    def tool_schemas(self) -> List[Dict[str, Any]]:
        return [{"name": t.name, "description": t.description, "parameters": t.input_schema} for t in self.tools.values()]

    def tool_schemas_text(self) -> str:
        return json.dumps(self.tool_schemas(), ensure_ascii=False, indent=2)
