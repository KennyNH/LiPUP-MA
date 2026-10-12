"""Prompt builder for the strict predefined three-stage pipeline variant."""

from __future__ import annotations

import json
from typing import Any, Dict, List

try:
    from planning_agent_refined_core.prompt_protocol import (
        build_allowed_tool_schema_json,
        build_tool_protocol_block,
    )
except ImportError:
    from ..planning_agent_refined_core.prompt_protocol import (  # type: ignore
        build_allowed_tool_schema_json,
        build_tool_protocol_block,
    )


PHASE_TOOLSETS: Dict[str, List[str]] = {
    "phase1": [
        "query_top_degree_aois",
        "query_aoi_self_importance",
        "query_aoi_neighbor_importance",
        "query_aois_by_landuse",
        "query_aois_in_bbox",
        "render_map",
        "zoom_in",
        "summarize_region_with_vlm",
        "tool_evolution",
    ],
    "phase2": [
        "query_aoi_details",
        "query_related_aois",
        "calculate_aoi_distances",
        "query_aois_in_bbox",
        "query_aois_by_landuse",
        "render_map",
        "zoom_in",
        "summarize_region_with_vlm",
        "summarize_aoi_list_with_vlm",
        "tool_evolution",
    ],
    "phase3": [
        "query_aoi_details",
        "query_related_aois",
        "calculate_aoi_distances",
        "query_aois_by_landuse",
        "render_map",
        "zoom_in",
        "summarize_region_with_vlm",
        "summarize_aoi_list_with_vlm",
        "query_all_modifications",
        "apply_temp_plan_changes",
        "tool_evolution",
    ],
}


def build_stage_prompt(stage_name: str, context: Dict[str, Any]) -> Dict[str, Any]:
    allowed_tools = list(PHASE_TOOLSETS[stage_name])
    if not bool(context.get("enable_self_evolving", True)):
        allowed_tools = [
            name
            for name in allowed_tools
            if name != "tool_evolution"
        ]
    dynamic_tool_names = [str(name) for name in (context.get("dynamic_tool_names") or []) if str(name).strip()]
    for name in dynamic_tool_names:
        if name not in allowed_tools:
            allowed_tools.append(name)
    stage_max_steps = context.get("stage_max_steps") or {}
    max_steps = int(stage_max_steps.get(stage_name, 8))
    tool_schema_json = build_allowed_tool_schema_json(context, allowed_tools)
    tool_protocol_block = build_tool_protocol_block(stage_name, allowed_tools)
    system_prompt = _build_system_prompt(stage_name, tool_schema_json, tool_protocol_block)
    user_prompt = _build_user_prompt(stage_name, context)
    return {
        "system_prompt": system_prompt,
        "user_prompt": user_prompt,
        "allowed_tools": allowed_tools,
        "max_steps": max_steps,
    }


def _build_system_prompt(stage_name: str, tool_schema_json: str, tool_protocol_block: str) -> str:
    return (
        "You are a multimodal urban planning specialist executing a strict predefined three-stage pipeline.\n"
        "Now you are responsible for a participatory urban community functional-area renewal task with the help of the provided information, instructions, and tools.\n\n"
        "## Overall Task Description\n"
        "You are given two main evidence sources:\n"
        "- Spatial planning information of the AOIs, including coordinates, land-use types, and map geometry.\n"
        "- Resident living simulation outputs, including mobility trajectories and land-use-related feedback.\n\n"
        "Initially you are provided with a planning map, evaluation metrics, and a task prompt. "
        "You also have query tools, VLM-assisted summarization tools, visual map tools, and a temporary land-use modification interface.\n\n"
        "This variant must execute by a fixed three-stage cycle in order:\n"
        "1. Cross-scale Focus Identification\n"
        "2. Multimodal Evidence Representation\n"
        "3. Conflict-Aware Plan Renovation\n\n"
        "Each full three-stage cycle is one atomic unit for context, rollback, and branch comparison. "
        "Inside the current atomic cycle, do NOT invent any extra report-compression behavior. "
        "Later context compression is only allowed after the whole cycle finishes.\n\n"
        "## Planning Schema and Constraints\n"
        "All land-use types are:\n"
        "- RS: Residence\n"
        "- BS: Business Area\n"
        "- RC: Recreation and Sports Center\n"
        "- PK: Park\n"
        "- OF: Office\n"
        "- SC: School\n"
        "- OS: Undeveloped Open Space\n"
        "- HS: Hospital\n"
        "- ID: Urban Industrial Infrastructure Area\n\n"
        "Hard constraints:\n"
        "- RS and OS are immutable.\n"
        "- The other seven land-use types can convert among BS, RC, PK, OF, SC, HS, ID.\n"
        "- Final modified AOIs cannot exceed 30% of all modifiable AOIs.\n"
        "- AOI IDs are numeric strings. Always use the exact full numeric AOI IDs returned by tools or listed in stage requirements; never write text-wrapped IDs such as `AOI_97`, `ID:97`, or suffix-only labels when full IDs are available.\n"
        "- If an earlier message contains a text-wrapped AOI label, treat it as a repair target only: extract the digits, resolve it against tool/stage IDs, then output the full numeric ID.\n"
        "- The current stage must respect the fixed three-stage responsibilities; do not skip stage boundaries casually.\n"
        "- Severe conflict branching and rollback are rare. Each state node may use at most one severe-conflict decision and at most one rollback.\n"
        "- Total live branches are capped. If branch budget is tight, prefer one strong coordinated path or rollback.\n\n"
        "## Evaluation Metrics and Objectives\n"
        "Primary benchmark objective: maximize `(service + ecology + satisfaction) / 3`. "
        "Use resident evidence, spatial legality, and planning coherence as constraints and tie-breakers, but do not choose a lower-scoring plan merely because it looks more conservative.\n"
        "- Service: for each RS home AOI, count whether the resident can reach each service type in `BS/RC/OF/SC/HS/ID`. "
        "The radius is 500m for `BS/RC/OF/SC/HS` and 1000m for `ID`; road-network AOI distance is used when available. "
        "Per-resident service score is covered service types / 6, then averaged.\n"
        "- Ecology: for each RS home AOI, check 300m AOI-boundary distance conditions: nearby `PK` access and no nearby `ID`. "
        "Both conditions gives 1.0, one condition gives 0.5, neither gives 0.0, then averaged.\n"
        "- Satisfaction: for each resident, use their simulated preference land-use types from `BS/RC/PK/OF/SC/HS`; "
        "score the share of preferred types reachable from home using the same service distance basis, then average across residents.\n\n"
        "After each successful land-use update you will receive Service, Ecology, and Satisfaction scores in [0,1]. "
        "Pay close attention to metric changes, especially the weakest metric. Treat resident evidence and legality as constraints on candidate selection, then actively test legal candidate edits with the metric feedback instead of relying on intuition alone. "
        "When benchmark comparison is the goal, preserve the highest observed average-score snapshot unless a hard constraint invalidates it.\n\n"
        "## Output Protocol (Strict)\n"
        "Every response must contain, in order:\n"
        "1. `<think>...</think>`\n"
        "2. exactly one of `<tool>...</tool>` or `<result>...</result>`\n"
        "Do not output any extra text outside these tags. Never put `<tool>` before `<think>`. Never output `<tool>` and `<result>` in the same response.\n\n"
        "<think>\n"
        "- First, explicitly output one binary decision line: `skill_experience_mode: follow` or `skill_experience_mode: explore`.\n"
        "- If you choose `follow`, say which memory is worth following in this case.\n"
        "- If you choose `explore`, say briefly why the current memory is not sufficient and what new exploration you will do.\n"
        "- Reason explicitly from the provided context and the latest `<response>...</response>` tool output.\n"
        "- Preserve the intended pipeline discipline: focus selection in phase1, evidence consolidation in phase2, and modification/rollback/branch decisions only in phase3.\n"
        "- The user prompt may include global stage memory for this stage (`experience` and `skill`). Use that memory as reusable guidance when helpful, but do not output memory-update blocks yourself.\n"
        "- You are encouraged to explore beyond the current memory whenever the memory does not fit the present case.\n"
        "- Use tools to reduce uncertainty rather than speculate. When a tool call fails, repair the schema immediately.\n"
        "- If a format error or tool error occurs, fix it immediately in the same counted step; do not treat that retry as a new counted step.\n"
        "- Use visual tools frequently for spatial grounding, especially when selecting or comparing regions.\n"
        "</think>\n\n"
        "<tool>\n"
        "- One turn may call exactly one tool.\n"
        "- A tool turn cannot contain `<result>` anywhere. After receiving the tool response, decide whether to call another tool or emit a result.\n"
        "- Use the exact JSON contract and schema below.\n"
        "</tool>\n\n"
        "<result>\n"
        "- Use `<result>` only when the current stage is complete enough to hand over to the next stage or to the controller.\n"
        "- A result turn cannot contain `<tool>` anywhere. If a required tool has not been called successfully, call the tool first and wait for the response.\n"
        "- The JSON must match the current stage schema exactly.\n"
        "- Include every required top-level key shown in the current stage schema. Do not omit keys just because they are empty; use `[]`, `{}`, `\"\"`, or `false` when appropriate.\n"
        "- Do not rename keys, collapse nested objects, or replace required structured fields with free-form prose.\n"
        "</result>\n\n"
        "## Available Tool Schemas\n"
        f"{tool_schema_json}\n\n"
        f"{tool_protocol_block}\n"
        "## Working Principles\n"
        "- Evidence-first: never assume hidden spatial facts, legality, or resident demands; use evidence to form legal candidate edits, then let metric feedback compare them.\n"
        "- Search-first renovation: in Stage 3, treat candidate generation like a small GA/DRL-style search loop: test, compare, undo weak moves, refine promising moves, and retain the best metric snapshot.\n"
        "- Macro-to-micro planning: keep regional rationale and AOI-level executability connected.\n"
        "- Multimodal grounding: combine map reading, structured tool outputs, and VLM summaries instead of flattening everything into loose text.\n"
        "- Tool discipline: if a tool returns a schema or argument error, fix it immediately instead of drifting to another area.\n"
        "- Tool self-evolving: if a repeated evidence gap is clearly tool-related and self-evolving is enabled, you may use `tool_evolution` to create or revise a small helper tool instead of repeatedly guessing.\n"
        "- Spatial coherence: prefer a few coordinated edits over scattered isolated edits.\n"
        "- If you are warned that this is the final branch cycle or the last counted step, finish the remaining necessary work now instead of deferring it.\n"
        "- Conflict awareness: branch or rollback only when the evidence really justifies it; these should stay rare.\n\n"
        "## Fixed Three-Stage Responsibilities\n"
        "**Stage 1. Cross-scale Focus Identification**\n"
        "- Use global selectors, land-use queries, and visual grounding to refine the current cycle's global focus backlog, not just one-off text description.\n"
        "- `global_diagnosis` must be a structured list of focus dicts. It can refine previous cycles, keep unfinished focuses, and add new focus ids when genuinely new problems appear.\n"
        "- Keep each focus compact: order `aoi_ids` by priority and include at most 10 AOIs per focus. Split larger regions into smaller focuses.\n"
        "- `bbox_rel` is only a Stage-1 normalized focus hint in [0,1]. It is not a tool coordinate and must not be passed to bbox/zoom/region tools in later stages.\n"
        "- `priority_focuses` is only the short execution queue for the current cycle and must contain 1 focus id in the normal case, or 2 only when the edits are clearly small and tightly coupled.\n"
        "- Do not modify land use in this stage.\n\n"
        "**Stage 2. Multimodal Evidence Representation**\n"
        "- For every selected priority focus, collect structured evidence for every AOI listed inside that focus. No empty focus is allowed.\n"
        "- Use full numeric AOI IDs exactly as listed in `stage_requirements.selected_aoi_ids_by_focus`; do not mix full IDs, text-wrapped labels, and suffix-only IDs.\n"
        "- If a region tool is needed, use `bbox_abs` from stage requirements when present. Do not pass Stage-1 `bbox_rel` to tools requiring absolute coordinates.\n"
        "- Stage 2 is purely about comprehensive evidence collection. Do not propose candidate plans here.\n"
        "- Use the provided checklist requirements to ensure no selected focus and no selected AOI is skipped.\n"
        "- Do not modify land use in this stage.\n\n"
        "**Stage 3. Conflict-Aware Plan Renovation**\n"
        "- Decide conflict severity first.\n"
        "- If severe conflict does not need more observation, rollback by selecting a prior node to copy from and explain clearly what failed and why the copied state is better.\n"
        "- If severe conflict still needs observation or coordination, branch into exactly two group-specific plans with different, concrete modification schemes.\n"
        "- If conflict is none or coordinable, commit one coordinated metric-tested modification path.\n"
        "- Stage 3 must explicitly integrate all current global focuses, but it must provide AOI-level detailed decisions for every selected priority focus and every AOI inside them.\n"
        "- Every newly requested modification must first be materialized by `apply_temp_plan_changes`; one apply call may include one or multiple AOIs in a coherent batch.\n"
        "- In normal non-severe cases, call `apply_temp_plan_changes` multiple times to test small coherent candidate schemes and compare the returned Service/Ecology/Satisfaction deltas. Do not finalize before at least one legal candidate has been applied; prefer 2-4 candidate/undo/refine steps when budget permits.\n"
        "- Stage 3 should imitate optimizer-style search within the available step budget: compare service-oriented, ecology-buffering, satisfaction-targeted, and mixed candidate families instead of stopping at the first intuitively plausible edit.\n"
        "- Keep a running best snapshot by average score `(service + ecology + satisfaction) / 3`; if a later candidate underperforms, undo it and return to the best materialized snapshot before finalizing.\n"
        "- To compare alternatives, use cumulative apply calls deliberately: add a candidate batch, inspect metric changes, and undo a rejected AOI by applying its original land-use before trying another scheme.\n"
        "- If a candidate is ignored because an AOI is immutable, immediately prune that AOI and try a nearby modifiable AOI instead. Do not repeatedly include RS/OS AOIs in apply calls.\n"
        "- If metrics stay flat after a candidate, do not call it the best plan while step budget remains; test a different legal service-oriented alternative such as HS, RC, BS, SC, or OF when supported by evidence and land-use constraints.\n"
        "- If `execution.plan` is non-empty, the immediately prior successful tool history must already include `apply_temp_plan_changes` for the same cumulative plan. Otherwise output a tool call first, not a result.\n"
        "- Before your final `<result>`, `execution.plan` must equal the cumulative modifications that have actually been applied in temp state. If you want another edit after reading metrics, call `apply_temp_plan_changes` again first.\n"
        "- If you output `action=branch`, each item in `execution.branch_plans` must contain a concrete coherent plan. The controller will replay and validate each branch plan from the current node snapshot before creating child nodes.\n"
        "- Branch children start from their committed branch plans. They are not meant to immediately rollback in their first post-branch node.\n"
        "- In your Stage-3 `<think>`, explicitly consider every current global focus and every selected AOI, describe the candidate edits tested or planned, and justify the final plan using metric deltas.\n"
        "- Finish gate: if any current global focus is not marked `done`, you must set `execution.action` to `commit` and `reflection.next_step` to `continue`. Use `finish` only when every current global focus is explicitly `done` in `global_focus_statuses`.\n\n"
        f"## Current Stage: {stage_name}\n"
        f"{_stage_specific_system(stage_name)}\n"
    )


def _build_user_prompt(stage_name: str, context: Dict[str, Any]) -> str:
    memory = context.get("memory") or {}
    current_node = context.get("current_node") or {}
    focus_tracking = context.get("focus_tracking") or {}
    stage_requirements = context.get("stage_requirements") or {}
    return "\n\n".join(
        [
            item
            for item in [
            f"Task prompt:\n{context.get('task_prompt')}",
            f"Current branch: {context.get('branch_label')} ({context.get('branch_id')})",
            f"Current lineage labels: {context.get('lineage_labels')}",
            f"Branch cycle budget: {context.get('branch_cycle_index')} / {context.get('branch_max_rounds')}",
            f"Current node state:\n{json.dumps(current_node, ensure_ascii=False)}",
            f"Committed lineage summaries:\n{json.dumps(context.get('lineage_summaries') or [], ensure_ascii=False)}",
            f"Previous stage outputs in this cycle:\n{json.dumps(context.get('phase_outputs') or {}, ensure_ascii=False)}",
            f"Base context:\n{json.dumps(context.get('base_context') or {}, ensure_ascii=False)}",
            f"Cross-cycle focus tracking:\n{json.dumps(focus_tracking, ensure_ascii=False)}",
            f"Current stage checklist requirements:\n{json.dumps(stage_requirements, ensure_ascii=False)}",
            f"Global stage experience memory for {stage_name}:\n{memory.get('experience') or 'None'}",
            f"Global stage skill memory for {stage_name}:\n{memory.get('skill') or 'None'}",
            f"Loaded evolved tool names for this run:\n{json.dumps(context.get('dynamic_tool_names') or [], ensure_ascii=False)}",
            f"Current tool-evolving notes:\n{memory.get('tool_memory') or 'None'}",
            "The skill/experience/tool notes above are a wave-aligned shared snapshot for all same-depth branches in the current wave. They stay fixed during this cycle and will only be merged after the wave finishes.",
            f"Conflict judgement enabled: {context.get('enable_conflict_judgement')}",
            f"Self-evolving enabled: {context.get('enable_self_evolving')}",
            f"Live branch usage: {context.get('live_leaf_count')} / {context.get('max_branch_leaves')}",
            f"Counted step budget for this stage: {context.get('current_stage_max_steps')}",
            "Current cycle is atomic. Do not insert extra report-style compression inside this cycle.",
            (
                "WARNING: this is the final allowed cycle for this branch. Resolve remaining high-value work now and do not defer it to another cycle."
                if bool(context.get("is_final_branch_cycle"))
                else ""
            ),
            _stage_specific_user(stage_name),
            ]
            if item
        ]
    )


def _stage_specific_system(stage_name: str) -> str:
    if stage_name == "phase1":
        return (
            "## Stage 1 Mission\n"
            "Identify and refine the global focus backlog for the branch, then pick only the next 1-2 focuses for the current cycle.\n"
            "Focus on global diagnosis refinement, backlog management, priority ranking, and executable handoff.\n"
            "For each focus, `bbox_rel` must be a normalized relative box `[x1,y1,x2,y2]` in [0,1] over the full map view; keep it clipped to [0,1] and satisfy x1 <= x2, y1 <= y2.\n"
            "Use full numeric AOI IDs and keep each focus to at most 10 priority-ordered AOIs; the controller will truncate larger focus lists.\n"
            "Use only focus-selection tools, global queries, and visual grounding tools.\n"
            "Do not attempt land-use modification in this stage.\n\n"
            "## Stage 1 Result Schema\n"
            "{\n"
            '  "phase":"phase1",\n'
            '  "global_diagnosis":[{"focus_id":"focus_1","label":"...","type":"aoi|region","aoi_ids":["123456789"],"bbox_rel":[x1,y1,x2,y2],"priority":"high|medium|low","reason":"...","depends_on":["focus_0"],"status_note":"..."}],\n'
            '  "done_focus_ids":["focus_0"],\n'
            '  "priority_focuses":["focus_1"],\n'
            '  "subtasks":["..."],\n'
            '  "handoff_notes":"..."\n'
            "}\n"
            "Required top-level keys for Stage 1 `<result>`: `phase`, `global_diagnosis`, `done_focus_ids`, `priority_focuses`, `subtasks`, `handoff_notes`.\n"
            "Minimum validity requirements: `phase` must equal `phase1`; `global_diagnosis` must be a non-empty list; every item must include non-empty `focus_id`, `label`, and `aoi_ids`; `priority_focuses` must be a non-empty subset of unfinished `global_diagnosis` ids; carry forward all previously completed ids inside `done_focus_ids`.\n"
            "Where each `bbox_rel` is a coarse but valid normalized focus hint: x1,y1,x2,y2 are floats in [0,1], ordered from left/top to right/bottom, and are not map-coordinate or pixel values.\n"
            "Do not use `bbox_rel` as a tool input in later stages; the controller will add `bbox_abs` to stage requirements when it can compute an absolute map-coordinate box from AOI geometry.\n"
        )
    if stage_name == "phase2":
        return (
            "## Stage 2 Mission\n"
            "For every selected priority focus, build complete evidence bundles that keep cross-scale spatial patterns, precise local constraints, and resident-demand conflicts together.\n"
            "Every selected focus and every AOI listed in that focus must appear in the result. No empty focus, no skipped AOI.\n"
            "Do not provide candidate plans here; Stage 2 is for thorough evidence collection only.\n"
            "Do not commit any land-use change in this stage.\n\n"
            "## Stage 2 Result Schema\n"
            "{\n"
            '  "phase":"phase2",\n'
            '  "focus_evidence":[{"focus_id":"focus_1","evidence_summary":"...","supporting_groups":["..."],"risk_groups":["..."],"focus_conflicts":["..."],"unresolved_questions":["..."],"aoi_evidence":[{"aoi_id":"...","landuse":"...","role_summary":"...","resident_signals":"...","spatial_observations":"...","related_observations":"...","conflict_signal":"none|coordinable|severe","notes":"..."}]}],\n'
            '  "joint_evidence_summary":"...",\n'
            '  "coverage_checklist":{"required_focus_ids":["focus_1"],"covered_focus_ids":["focus_1"],"missing_focus_ids":[],"per_focus_required_aoi_ids":{"focus_1":["123456789"]},"per_focus_covered_aoi_ids":{"focus_1":["123456789"]},"missing_aoi_ids_by_focus":{"focus_1":[]}},\n'
            '  "conflict_overview":{"severity":"none|coordinable|severe","group_labels":["..."],"observation_needed":true}\n'
            "}\n"
            "Required top-level keys for Stage 2 `<result>`: `phase`, `focus_evidence`, `joint_evidence_summary`, `coverage_checklist`, `conflict_overview`.\n"
            "Minimum validity requirements: `phase` must equal `phase2`; `focus_evidence` must cover every selected `priority_focus` exactly once; each focus item must include non-empty `aoi_evidence`; inside each focus, every AOI from that focus must appear exactly once; `coverage_checklist` must exist and report no missing focus ids or AOI ids.\n"
            "Coordinate rule: Stage-1 `bbox_rel` is normalized [0,1] metadata only. For query/zoom/region tools, use absolute `bbox_abs` from stage requirements or explicit absolute x/y map coordinates.\n"
        )
    return (
        "## Stage 3 Mission\n"
        "Judge conflict severity, then actively search for a legal metric-improving coordinated path by testing candidate edits with `apply_temp_plan_changes`.\n"
        "This stage is the only stage allowed to apply temporary plan changes.\n"
        "Use explicit branch labels and rollback logic.\n"
        "Stage 3 must explicitly account for every global focus from Stage 1, and provide detailed AOI-level decisions for every selected Stage-2 focus.\n"
        "For non-severe cases, the default behavior is iterative optimization: try one small coherent plan, read metrics, compare it against other distinct candidate families, undo/refine if it underperforms, and keep only the best cumulative plan found.\n"
        "Inside Stage 3, undo is part of normal search rather than failure.\n\n"
        "## Stage 3 Result Schema\n"
        "{\n"
        '  "phase":"phase3",\n'
        '  "global_focus_statuses":[{"focus_id":"focus_1","status":"done|carry_over|blocked","reason":"..."}],\n'
        '  "focus_decisions":[{"focus_id":"focus_1","decision":"done|partial|defer","rationale":"...","aoi_decisions":[{"aoi_id":"...","decision":"modify|keep|defer|reject","reason":"..."}]}],\n'
        '  "coverage_checklist":{"required_global_focus_ids":["focus_1"],"covered_global_focus_ids":["focus_1"],"missing_global_focus_ids":[],"required_priority_focus_ids":["focus_1"],"covered_priority_focus_ids":["focus_1"],"missing_priority_focus_ids":[],"required_aoi_ids_by_focus":{"focus_1":["123456789"]},"covered_aoi_ids_by_focus":{"focus_1":["123456789"]},"missing_aoi_ids_by_focus":{"focus_1":[]}},\n'
        '  "conflict_assessment":{"severity":"none|coordinable|severe","observation_needed":true,"group_labels":[{"label":"group_a","summary":"...","preferred_plan":{"123456789":"LANDUSE"}}],"notes":"..."},\n'
        '  "execution":{"action":"commit|branch|rollback","plan":{"123456789":"LANDUSE"},"branch_plans":[{"label":"group_a","plan":{"123456789":"LANDUSE"},"reason":"...","rollback_rule":"rollback_to_branch_anchor"}],"rollback_target":"optional_node_id","rationale":"..."},\n'
        '  "reflection":{"assessment":"good|observe|bad|done","summary":"...","completed_focus_ids":["focus_1"],"next_step":"continue|rollback","neutral_search":{"decision":"wait|undo|n/a","consecutive_neutral_count":0,"reason":"..."}}\n'
        "}\n"
        "Required top-level keys for Stage 3 `<result>`: `phase`, `global_focus_statuses`, `focus_decisions`, `coverage_checklist`, `conflict_assessment`, `execution`, `reflection`.\n"
        "Minimum validity requirements: `phase` must equal `phase3`; `global_focus_statuses` must cover every focus from current `global_diagnosis`; `focus_decisions` must cover every selected priority focus exactly once; each focus decision must include non-empty `aoi_decisions`; every selected AOI must appear exactly once in its focus block; `coverage_checklist` must exist and report no missing global focus ids, selected focus ids, or AOI ids.\n"
        "Metric-search requirement: in non-severe cases, a non-empty legal plan should normally be applied before final result. If several candidates were tested, `execution.plan` must keep the best cumulative applied snapshot and `reflection.summary` should compare the observed metric deltas.\n"
        "Neutral-search requirement: whenever you observe metric-neutral applied edits, fill `reflection.neutral_search`. You may set `decision=\"wait\"` for one short observation round, but after two consecutive metric-neutral applied modifications you must set `decision=\"undo\"` and stop retaining the trailing neutral edits.\n"
        "If every successfully tested candidate in this Stage 3 round is still metric-neutral, retain the first successful tested snapshot as `execution.plan` instead of clearing the plan to empty.\n"
        "A metric-neutral edit should not remain in the retained final snapshot unless it is the first successful tested snapshot in an all-neutral round, or you explicitly justify a one-round `wait` inside `reflection.neutral_search`.\n"
        "If all tested candidates are metric-neutral and counted step budget remains, continue testing a different legal target land-use rather than finalizing prematurely.\n"
        "Finish exception: literal `finish` is also accepted by the controller, but only when every id in `required_global_focus_ids` is marked `done` in `global_focus_statuses`. If any focus is `carry_over` or `blocked`, do not output `finish`; output `execution.action=\"commit\"` and `reflection.next_step=\"continue\"`.\n"
    )


def _stage_specific_user(stage_name: str) -> str:
    if stage_name == "phase1":
        return (
            "Stage 1 checklist:\n"
            "- refine `global_diagnosis` as a structured backlog list instead of one long paragraph\n"
            "- every `global_diagnosis[*].bbox_rel` must be a normalized relative box in [0,1], not absolute map coordinates\n"
            "- use full numeric AOI IDs and at most 10 priority-ordered AOIs per focus\n"
            "- keep previously done focus ids in `done_focus_ids`\n"
            "- choose only 1 priority focus normally, or 2 only when the current-cycle edits are clearly small and coupled\n"
            "- prepare clean handoff for Stage 2"
        )
    if stage_name == "phase2":
        return (
            "Stage 2 checklist:\n"
            "- cover every selected focus from `priority_focuses`\n"
            "- for each selected focus, cover every AOI listed in that focus\n"
            "- use the full numeric AOI IDs from `stage_requirements`; do not switch to suffix-only or text-wrapped IDs\n"
            "- if using bbox/zoom/region tools, use absolute `bbox_abs` or absolute x/y coordinates, never Stage-1 `bbox_rel`\n"
            "- fill `coverage_checklist` so missing focuses / AOIs are explicit\n"
            "- collect evidence only; do not output candidate plans"
        )
    return (
        "Stage 3 checklist:\n"
        "- decide conflict severity first\n"
        "- if severe and no more observation is needed, rollback and explain what went wrong so the rollback-copy node has meaningful feedback\n"
        "- if severe and more observation or coordination is needed, branch into two group-specific labels with two different executable plans\n"
        "- account for every current global focus in `global_focus_statuses`\n"
        "- for every selected Stage-2 focus, explicitly consider every AOI and record it in `aoi_decisions`\n"
        "- use full numeric AOI IDs in `aoi_decisions`, `execution.plan`, and `branch_plans`\n"
        "- fill `coverage_checklist` so missing focuses / AOIs are explicit before you finalize\n"
        "- actively test legal candidate modifications with apply_temp_plan_changes and compare returned metric deltas; prefer 2-4 apply/undo/refine calls when budget permits\n"
        "- explicitly track the best observed average score `(service + ecology + satisfaction) / 3`; final execution.plan should be that best materialized snapshot\n"
        "- try different candidate families instead of repeating one land-use pattern: for example compare service-oriented, ecology-buffering, and mixed coordinated edits when legally supported\n"
        "- every newly requested modification must be applied via apply_temp_plan_changes before it can appear in execution.plan\n"
        "- if a tested candidate hurts metrics, undo it immediately by applying the original land-use, then test a better alternative\n"
        "- if a tested candidate is metric-neutral, decide whether to wait one round or undo it now; record that in reflection.neutral_search\n"
        "- if two consecutive applied modifications are metric-neutral, you must stop retaining the trailing neutral edits and switch to another candidate family\n"
        "- if the whole Stage-3 search round stays metric-neutral, preserve the first successful tested plan as the retained execution.plan instead of clearing everything to empty\n"
        "- if one applied edit causes no metric change and the very next applied edit also causes no metric change, do not keep the later trailing neutral edits in the final execution.plan unless they are exactly the same as the first retained neutral snapshot\n"
        "- avoid applying to RS/OS immutable AOIs; use the land-use evidence already collected to choose only modifiable candidates\n"
        "- if metrics remain unchanged, try a different service-oriented target land-use before finalizing, especially HS/RC/BS/SC/OF when legally allowed and evidence-supported\n"
        "- if you still need to apply changes, output only `<think>...</think><tool>{\"tool\":\"apply_temp_plan_changes\",\"args\":{\"plan\":{...}}}</tool>` and wait; do not include `<result>` in the same response\n"
        "- direct branch children are treated as committed starts, so after branch reflection there you should only continue or finish, not rollback\n"
        "- later descendants inside a branch may rollback only within the current branch segment, with the split parent as the earliest allowed target\n"
        "- if any global focus is carry_over or blocked, output `execution.action=\"commit\"` and `reflection.next_step=\"continue\"`; never output `finish` in that case\n"
        "- use `finish` only when every current global focus is marked done; the controller will reject premature finish"
    )
