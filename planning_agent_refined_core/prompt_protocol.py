"""Shared prompt helpers for refined planning prompt builders."""

from __future__ import annotations

import json
from typing import Any, Dict, List


TOOL_ARG_REMINDERS: Dict[str, str] = {
    "query_top_degree_aois": "- `query_top_degree_aois`: use `{\"tool\":\"query_top_degree_aois\",\"args\":{\"top_n\":12,\"modifiable_only\":true}}`. The key is `top_n`, not `top_k`.",
    "query_aoi_self_importance": "- `query_aoi_self_importance`: use only `importance_level` in `args`. This tool does not accept AOI IDs.",
    "query_aoi_neighbor_importance": "- `query_aoi_neighbor_importance`: use only `importance_level` in `args`. This tool does not accept AOI IDs.",
    "query_aois_by_landuse": "- `query_aois_by_landuse`: use `landuse_types`, either a string or a string array.",
    "query_aois_in_bbox": "- `query_aois_in_bbox`: use absolute map/positions coordinates: `x1`, `y1`, `x2`, `y2`. Never pass Stage-1 `bbox_rel` or any 0-1 normalized box to this tool.",
    "query_aoi_details": "- `query_aoi_details`: pass `aoi_ids` as an array of numeric AOI ID strings. Do not write `AOI_97`, `ID:97`, or suffix-only labels when full IDs are available.",
    "summarize_region_with_vlm": "- `summarize_region_with_vlm`: use absolute map/positions coordinates: `x1`, `y1`, `x2`, `y2`. Never pass Stage-1 `bbox_rel`; use `bbox_abs` from stage requirements when available.",
    "summarize_aoi_list_with_vlm": "- `summarize_aoi_list_with_vlm`: pass `aoi_ids` as an array with a focused shortlist.",
    "query_related_aois": "- `query_related_aois`: this tool is single-focus. Use one numeric full AOI ID returned by prior tools or stage requirements, e.g. `{\"tool\":\"query_related_aois\",\"args\":{\"aoi_id\":\"123456789\",\"top_n\":3,\"radius_m\":50}}`.",
    "calculate_aoi_distances": "- `calculate_aoi_distances`: use a single pair each time via `aoi_id_1`, `aoi_id_2`, and optional `distance_type`.",
    "render_map": "- `render_map`: use `map_version` plus optional numeric full AOI IDs in `aoi_ids`. Do not use `highlight_aoi_ids`.",
    "zoom_in": "- `zoom_in`: use absolute map/positions coordinates via `x1`, `y1`, `x2`, `y2`, plus optional numeric full AOI IDs in `aoi_ids`. Never pass Stage-1 `bbox_rel`.",
    "query_all_modifications": "- `query_all_modifications`: use empty `args` `{}`.",
    "apply_temp_plan_changes": "- `apply_temp_plan_changes`: use numeric full AOI IDs, e.g. `{\"tool\":\"apply_temp_plan_changes\",\"args\":{\"plan\":{\"123456789\":\"PK\"}}}`. The required key is `plan`, not `changes`; one call may include one or multiple AOIs. In Phase 3, use multiple calls to test candidate schemes, compare returned metric deltas, and undo a rejected AOI by applying its original land-use before trying another alternative.",
    "tool_evolution": "- `tool_evolution`: use it to create a new helper tool or update an already-evolved tool from this run/bootstrap state. Always provide `tool_name` and `python_body`; when creating a new tool also provide `description` and `input_schema`. The tool is saved into the current run folder and then appears in later tool schemas.",
}


STAGE_EXAMPLES: Dict[str, List[str]] = {
    "phase1": [
        '<tool>{"tool":"query_top_degree_aois","args":{"top_n":12,"modifiable_only":true}}</tool>',
        '<tool>{"tool":"render_map","args":{"map_version":"initial","aoi_ids":["123456789","123456790"]}}</tool>',
    ],
    "phase2": [
        '<tool>{"tool":"query_aoi_details","args":{"aoi_ids":["123456789","123456790","123456791"]}}</tool>',
        '<tool>{"tool":"query_related_aois","args":{"aoi_id":"123456789","top_n":3,"radius_m":50}}</tool>',
        '<tool>{"tool":"summarize_region_with_vlm","args":{"x1":0,"y1":-40,"x2":210,"y2":180}}</tool>',
    ],
    "phase3": [
        '<tool>{"tool":"query_all_modifications","args":{}}</tool>',
        '<tool>{"tool":"apply_temp_plan_changes","args":{"plan":{"123456789":"PK"}}}</tool>',
        '<tool>{"tool":"apply_temp_plan_changes","args":{"plan":{"123456789":"BS","123456790":"HS"}}}</tool>',
    ],
}


def build_allowed_tool_schema_json(context: Dict[str, Any], allowed_tools: List[str]) -> str:
    raw = context.get("available_tool_schemas") or []
    allowed = set(allowed_tools)
    filtered: List[Dict[str, Any]] = []
    if isinstance(raw, list):
        for item in raw:
            if not isinstance(item, dict):
                continue
            if item.get("name") in allowed:
                filtered.append(item)
    if not filtered:
        filtered = [{"name": name} for name in allowed_tools]
    return json.dumps(filtered, ensure_ascii=False, indent=2)


def build_tool_protocol_block(stage_name: str, allowed_tools: List[str]) -> str:
    lines = [
        "## Tool Call Contract",
        '- Always emit `<tool>{"tool":"TOOL_NAME","args":{...}}</tool>`.',
        "- Tool calls and final results are mutually exclusive: one assistant response must contain exactly one `<tool>` block OR exactly one `<result>` block, never both.",
        "- The only valid order is `<think>...</think>` immediately followed by the single `<tool>...</tool>` or `<result>...</result>` block.",
        "- Do not use OpenAI-style `name` / `arguments` function-call format in the final output block.",
        "- Every tool call must satisfy the exact schema below; do not rely on implicit defaults for required fields.",
        "- If a tool returns `invalid_tool_args`, `output_protocol_error`, or any schema-related error, correct the arguments immediately instead of changing the planning goal.",
        "- Format errors and tool-call errors must be fixed immediately in the same counted step; these retries do not consume a new counted step.",
        "- If you are warned that this is the last counted step, finish the remaining necessary tool work now instead of deferring it.",
        "- Any evolved tools already loaded for this run, including bootstrapped ones, are listed in the Available Tool Schemas block below.",
        "- If you evolve a tool successfully, that tool becomes available later in the current run. Use the returned tool name and schema exactly.",
    ]
    for tool_name in allowed_tools:
        reminder = TOOL_ARG_REMINDERS.get(tool_name)
        if reminder:
            lines.append(reminder)
    examples = STAGE_EXAMPLES.get(stage_name) or []
    if examples:
        lines.append("Stage-local valid tool call examples:")
        lines.extend([f"- `{example}`" for example in examples])
    return "\n".join(lines) + "\n"
