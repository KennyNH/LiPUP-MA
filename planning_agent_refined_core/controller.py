"""Controller for three-stage planning with explicit state tree and branching."""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path
import re
from threading import Lock
from typing import Any, Callable, Dict, List, Optional, Tuple

from .evolution import EvolutionNotebook
from .models import BranchRuntime, CycleExecutionOutput, CycleRecord, StageRunResult, StateNode
from .stage_agent import StageAgentRuntimeError, StageReActAgent

try:
    from planning_agent_react_mm_next.utils import save_json
except ImportError:
    from ..planning_agent_react_mm_next.utils import save_json  # type: ignore


PHASES = ("phase1", "phase2", "phase3")
MAX_AOIS_PER_FOCUS = 10


class RefinedPlanningController:
    NEUTRAL_METRIC_TOL = 1e-9

    def __init__(
        self,
        repo_root: Path,
        output_dir: Path,
        task_prompt: str,
        client_factory: Callable[[], Any],
        toolkit_factory: Callable[[Path], Any],
        prompt_builder: Callable[[str, Dict[str, Any]], Dict[str, Any]],
        evolution: EvolutionNotebook,
        variant_name: str,
        max_rounds: int = 8,
        stage_max_steps: Optional[Dict[str, int]] = None,
        max_branch_leaves: int = 8,
        concurrent_branch_workers: int = 2,
        enable_conflict_judgement: bool = True,
        enable_self_evolving: bool = True,
        verbose: bool = False,
    ):
        self.repo_root = repo_root
        self.output_dir = output_dir
        self.task_prompt = task_prompt
        self.client_factory = client_factory
        self.toolkit_factory = toolkit_factory
        self.prompt_builder = prompt_builder
        self.evolution = evolution
        self.variant_name = variant_name
        self.max_rounds = max(1, int(max_rounds))
        raw_stage_max_steps = stage_max_steps or {}
        self.stage_max_steps = {
            stage_name: max(1, int(raw_stage_max_steps.get(stage_name, 8)))
            for stage_name in PHASES
        }
        self.max_branch_leaves = max(1, int(max_branch_leaves))
        self.concurrent_branch_workers = max(1, int(concurrent_branch_workers))
        self.enable_conflict_judgement = bool(enable_conflict_judgement)
        self.enable_self_evolving = bool(enable_self_evolving)
        self.verbose = verbose
        self.nodes: Dict[str, StateNode] = {}
        self.cycles: List[CycleRecord] = []
        self._node_seq = 0
        self._branch_seq = 0
        self._cycle_seq = 0
        self._seq_lock = Lock()
        self._known_aoi_ids: set[str] = set()
        self._known_aoi_bounds: Dict[str, Tuple[float, float, float, float]] = {}

    def run(self) -> Dict[str, Any]:
        active: List[BranchRuntime] = [self._bootstrap_root_branch()]
        terminal: List[BranchRuntime] = []

        while active:
            ready: List[BranchRuntime] = []
            for runtime in active:
                if runtime.cycle_count >= self.max_rounds:
                    node = self.nodes.get(runtime.current_node_id)
                    if node is not None and not node.finished:
                        node.finished = True
                        node.terminated_reason = node.terminated_reason or "max_rounds_reached"
                    terminal.append(runtime)
                else:
                    ready.append(runtime)
            if not ready:
                break
            live_leaf_count = len(ready) + len(terminal)
            results, ordered_runtimes, wave_memory_snapshot, wave_dynamic_tool_names = self._execute_wave(ready, live_leaf_count=live_leaf_count)
            self._consolidate_self_evolving_wave(results, ordered_runtimes, wave_memory_snapshot, wave_dynamic_tool_names)
            next_active: List[BranchRuntime] = []
            for runtime in ordered_runtimes:
                output = results[runtime.branch_id]
                node = self.nodes[output.input_node_id]
                cycle = CycleRecord(
                    cycle_index=output.cycle_index,
                    branch_id=output.branch_id,
                    input_node_id=output.input_node_id,
                    stage_results=output.stage_results,
                    stage_updates=output.stage_updates,
                    action=output.action,
                    summary=output.summary,
                    metrics_after=output.current_metrics,
                    metadata=output.metadata,
                )
                self.cycles.append(cycle)

                if output.action == "rollback":
                    node.rollback_used = True
                    rollback_target = self.nodes.get(output.rollback_target_id or "")
                    rollback_child = self._create_rollback_copy_node(
                        parent=node,
                        target=rollback_target or node,
                        branch_id=runtime.branch_id,
                        branch_label=runtime.branch_label,
                        cycle_index=output.cycle_index,
                        stage_summary=output.summary,
                        branch_group_labels=list(node.branch_group_labels),
                        current_node=node,
                        metadata=output.metadata,
                    )
                    runtime.current_node_id = rollback_child.node_id
                    next_active.append(runtime)
                    continue

                if output.action == "branch":
                    node.severe_conflict_used = True
                    spawned = self._spawn_branch_children(runtime, node, output)
                    next_active.extend(spawned)
                    continue

                finished = output.action == "finish"
                terminated_reason = "completed_branch" if finished else None
                child = self._create_child_node(
                    parent=node,
                    branch_id=runtime.branch_id,
                    plan=output.current_plan,
                    metrics=output.current_metrics,
                    cycle_index=output.cycle_index,
                    stage_summary=output.summary,
                    branch_label=runtime.branch_label,
                    branch_group_labels=list(node.branch_group_labels),
                    rollback_anchor_id=node.rollback_anchor_id,
                    rollback_rule=node.rollback_rule,
                    finished=finished,
                    terminated_reason=terminated_reason,
                    node_origin="finish" if finished else "commit",
                    metadata=output.metadata,
                )
                runtime.current_node_id = child.node_id
                if finished:
                    terminal.append(runtime)
                else:
                    next_active.append(runtime)
            active = next_active

        finalists = terminal if terminal else active
        terminal_candidates = self._build_terminal_candidates(finalists)
        metric_best_candidates = [item for item in terminal_candidates if item.get("is_metric_best")]
        representative_best = metric_best_candidates[0] if metric_best_candidates else (terminal_candidates[0] if terminal_candidates else None)
        best_runtime, best_node = self._select_best_leaf(finalists)
        best_plan = dict(best_node.plan) if best_node else {}
        best_metrics = dict(best_node.metrics) if best_node else {}
        if best_node is not None:
            self._materialize_best_plan(best_plan)
        metric_best_branch_ids = [str(item.get("branch_id")) for item in metric_best_candidates]
        metric_best_node_ids = [str(item.get("node_id")) for item in metric_best_candidates]
        metric_best_score = representative_best.get("metric_score") if isinstance(representative_best, dict) else None
        root_id = self._root_node_id()

        self.evolution.export(self.output_dir / "evolution_notebook.json")
        save_json(
            self.output_dir / "state_tree.json",
            {
                "variant": self.variant_name,
                "root_id": root_id,
                "nodes": {node_id: asdict(node) for node_id, node in self.nodes.items()},
                "cycles": [asdict(cycle) for cycle in self.cycles],
            },
        )
        save_json(
            self.output_dir / "terminal_branches.json",
            {
                "metric_best_branch_ids": metric_best_branch_ids,
                "metric_best_node_ids": metric_best_node_ids,
                "metric_best_score": metric_best_score,
                "branches": terminal_candidates,
            },
        )
        save_json(
            self.output_dir / "summary.json",
            {
                "variant": self.variant_name,
                "output_dir": str(self.output_dir),
                "output_root": str(self.output_dir.parent),
                "best_branch_id": best_runtime.branch_id if best_runtime else None,
                "best_node_id": best_node.node_id if best_node else None,
                "best_plan": best_plan,
                "best_metrics": best_metrics,
                "metric_best_branch_ids": metric_best_branch_ids,
                "metric_best_node_ids": metric_best_node_ids,
                "metric_best_score": metric_best_score,
                "cycle_count": len(self.cycles),
                "node_count": len(self.nodes),
                "terminal_branch_count": len(finalists),
                "terminal_plans": terminal_candidates,
            },
        )
        return {
            "best_branch_id": best_runtime.branch_id if best_runtime else None,
            "best_node_id": best_node.node_id if best_node else None,
            "best_plan": best_plan,
            "best_metrics": best_metrics,
            "metric_best_branch_ids": metric_best_branch_ids,
            "metric_best_node_ids": metric_best_node_ids,
            "metric_best_score": metric_best_score,
            "terminal_plans": terminal_candidates,
            "cycle_count": len(self.cycles),
            "node_count": len(self.nodes),
        }

    def _bootstrap_root_branch(self) -> BranchRuntime:
        branch = self._make_branch_runtime(branch_label="main")
        self._refresh_known_aoi_ids(branch)
        base_metrics = branch.toolkit.env.evaluate_base().get("metrics", {})
        root = self._new_node(
            parent_id=None,
            branch_id=branch.branch_id,
            branch_label="main",
            plan={},
            metrics=base_metrics,
            cycle_index=0,
            root_id="",
            depth=0,
            stage_summary="Root state from the initial plan.",
            branch_group_labels=[],
            rollback_anchor_id=None,
            rollback_rule="rollback_to_parent",
            finished=False,
            terminated_reason=None,
            node_origin="root",
            metadata={"root": True},
        )
        branch.current_node_id = root.node_id
        branch.lineage_labels = ["main"]
        return branch

    def _refresh_known_aoi_ids(self, runtime: BranchRuntime) -> None:
        try:
            base_landuse = getattr(runtime.toolkit.env, "base_landuse", {}) or {}
        except Exception:
            base_landuse = {}
        if isinstance(base_landuse, dict) and base_landuse:
            self._known_aoi_ids.update(str(aid) for aid in base_landuse.keys())
        try:
            aoi_geom = getattr(runtime.toolkit.env, "aoi_geom", {}) or {}
        except Exception:
            aoi_geom = {}
        if isinstance(aoi_geom, dict):
            for aid, geom in aoi_geom.items():
                try:
                    minx, miny, maxx, maxy = geom.bounds
                except Exception:
                    continue
                self._known_aoi_bounds[str(aid)] = (float(minx), float(miny), float(maxx), float(maxy))

    def _execute_wave(
        self,
        runtimes: List[BranchRuntime],
        live_leaf_count: int,
    ) -> Tuple[Dict[str, CycleExecutionOutput], List[BranchRuntime], Dict[str, Any], List[str]]:
        ordered = list(runtimes)
        wave_memory_snapshot = self.evolution.snapshot() if self.enable_self_evolving else {"stage_books": {}, "tool_events": []}
        wave_dynamic_tool_names = ordered[0].toolkit.dynamic_tool_names() if ordered and hasattr(ordered[0].toolkit, "dynamic_tool_names") else []
        for runtime in ordered:
            if hasattr(runtime.toolkit, "freeze_tool_registry"):
                runtime.toolkit.freeze_tool_registry()
        if len(ordered) == 1 or self.concurrent_branch_workers <= 1:
            try:
                results = {
                    runtime.branch_id: self._execute_single_cycle(
                        runtime,
                        self.nodes[runtime.current_node_id],
                        live_leaf_count,
                        wave_memory_snapshot,
                        wave_dynamic_tool_names,
                    )
                    for runtime in ordered
                }
            finally:
                for runtime in ordered:
                    if hasattr(runtime.toolkit, "unfreeze_tool_registry"):
                        runtime.toolkit.unfreeze_tool_registry()
            return results, ordered, wave_memory_snapshot, list(wave_dynamic_tool_names)

        results: Dict[str, CycleExecutionOutput] = {}
        try:
            with ThreadPoolExecutor(max_workers=min(self.concurrent_branch_workers, len(ordered))) as pool:
                future_map = {
                    pool.submit(
                        self._execute_single_cycle,
                        runtime,
                        self.nodes[runtime.current_node_id],
                        live_leaf_count,
                        wave_memory_snapshot,
                        wave_dynamic_tool_names,
                    ): runtime
                    for runtime in ordered
                }
                for future in as_completed(future_map):
                    runtime = future_map[future]
                    results[runtime.branch_id] = future.result()
        finally:
            for runtime in ordered:
                if hasattr(runtime.toolkit, "unfreeze_tool_registry"):
                    runtime.toolkit.unfreeze_tool_registry()
        return results, ordered, wave_memory_snapshot, list(wave_dynamic_tool_names)

    @staticmethod
    def _dedupe_str_list(raw: Any) -> List[str]:
        out: List[str] = []
        seen = set()
        if not isinstance(raw, list):
            return out
        for item in raw:
            value = str(item).strip()
            if not value or value in seen:
                continue
            seen.add(value)
            out.append(value)
        return out

    @staticmethod
    def _infer_aoi_id_profile(known_aoi_ids: Optional[set[str]]) -> Tuple[str, int]:
        known = sorted(str(aid) for aid in (known_aoi_ids or set()) if str(aid).isdigit())
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

    @classmethod
    def _canonicalize_aoi_id(cls, raw: Any, known_aoi_ids: Optional[set[str]] = None) -> str:
        text = str(raw or "").strip()
        if not text:
            return ""
        known = known_aoi_ids or set()
        if text in known:
            return text
        digits = "".join(ch for ch in text if ch.isdigit())
        if not digits:
            return text
        if known:
            prefix, suffix_width = cls._infer_aoi_id_profile(known)
            stripped = digits.lstrip("0") or "0"
            if suffix_width > 0 and len(stripped) <= suffix_width:
                candidate = f"{prefix}{stripped.zfill(suffix_width)}"
                if candidate in known:
                    return candidate

            suffix_candidates = {digits}
            if suffix_width > 0 and len(stripped) <= suffix_width:
                suffix_candidates.add(stripped.zfill(suffix_width))
            elif stripped != digits:
                suffix_candidates.add(stripped)
            matches = [
                aid
                for aid in known
                if any(str(aid).endswith(suffix) for suffix in suffix_candidates)
            ]
            if len(matches) == 1:
                return matches[0]
            return text
        return text

    def _focus_bbox_abs(self, aoi_ids: Any) -> List[float]:
        ids = self._canonicalize_aoi_list(aoi_ids, self._known_aoi_ids, max_items=None)
        bounds = [self._known_aoi_bounds[aid] for aid in ids if aid in self._known_aoi_bounds]
        if not bounds:
            return []
        minx = min(item[0] for item in bounds)
        miny = min(item[1] for item in bounds)
        maxx = max(item[2] for item in bounds)
        maxy = max(item[3] for item in bounds)
        return [round(minx, 3), round(miny, 3), round(maxx, 3), round(maxy, 3)]

    def _enrich_focus_entry(self, focus: Dict[str, Any]) -> Dict[str, Any]:
        out = dict(focus)
        bbox_abs = self._focus_bbox_abs(out.get("aoi_ids") or [])
        if bbox_abs:
            out["bbox_abs"] = bbox_abs
        return out

    @classmethod
    def _canonicalize_aoi_list(
        cls,
        raw: Any,
        known_aoi_ids: Optional[set[str]] = None,
        max_items: Optional[int] = None,
    ) -> List[str]:
        out: List[str] = []
        seen = set()
        if not isinstance(raw, list):
            return out
        for item in raw:
            value = cls._canonicalize_aoi_id(item, known_aoi_ids)
            if not value or value in seen:
                continue
            seen.add(value)
            out.append(value)
            if max_items is not None and len(out) >= int(max_items):
                break
        return out

    @classmethod
    def _canonicalize_plan(cls, raw: Any, known_aoi_ids: Optional[set[str]] = None) -> Dict[str, str]:
        if not isinstance(raw, dict):
            return {}
        out: Dict[str, str] = {}
        for aid, landuse in raw.items():
            canonical = cls._canonicalize_aoi_id(aid, known_aoi_ids)
            if canonical:
                out[canonical] = str(landuse).strip().upper()
        return out

    @staticmethod
    def _parse_focus_index(focus_id: str) -> int:
        text = str(focus_id).strip()
        if not text:
            return 0
        parts = text.rsplit("_", 1)
        if len(parts) == 2 and parts[1].isdigit():
            return int(parts[1])
        digits = "".join(ch for ch in text if ch.isdigit())
        return int(digits) if digits else 0

    @classmethod
    def _normalize_focus_entry(
        cls,
        raw: Any,
        known_aoi_ids: Optional[set[str]] = None,
        max_aoi_ids: Optional[int] = MAX_AOIS_PER_FOCUS,
    ) -> Optional[Dict[str, Any]]:
        if not isinstance(raw, dict):
            return None
        focus_id = str(raw.get("focus_id") or "").strip()
        label = str(raw.get("label") or "").strip()
        if not focus_id or not label:
            return None
        raw_aoi_ids = raw.get("aoi_ids") or []
        aoi_ids = cls._canonicalize_aoi_list(raw_aoi_ids, known_aoi_ids, max_items=max_aoi_ids)
        if not aoi_ids:
            return None
        bbox_raw = raw.get("bbox_rel")
        bbox_rel: List[float] = []
        if isinstance(bbox_raw, list) and len(bbox_raw) == 4:
            for value in bbox_raw:
                try:
                    bbox_rel.append(float(value))
                except Exception:
                    bbox_rel = []
                    break
            if bbox_rel:
                x1, y1, x2, y2 = bbox_rel
                if (
                    min(bbox_rel) < 0.0
                    or max(bbox_rel) > 1.0
                    or x1 > x2
                    or y1 > y2
                ):
                    bbox_rel = []
        return {
            "focus_id": focus_id,
            "label": label,
            "type": str(raw.get("type") or "region"),
            "aoi_ids": aoi_ids,
            "truncated_aoi_ids": (
                len(raw_aoi_ids) > len(aoi_ids)
                if isinstance(raw_aoi_ids, list)
                else False
            ),
            "bbox_rel": bbox_rel,
            "priority": str(raw.get("priority") or "medium"),
            "reason": str(raw.get("reason") or ""),
            "depends_on": cls._dedupe_str_list(raw.get("depends_on") or []),
            "status_note": str(raw.get("status_note") or raw.get("note") or ""),
        }

    def _collect_focus_tracking(self, node_id: str) -> Dict[str, Any]:
        lineage = set(self._lineage_node_ids(node_id))
        registry: Dict[str, Dict[str, Any]] = {}
        completed = set()
        stage1_history: List[Dict[str, Any]] = []
        latest_priority_ids: List[str] = []
        latest_done_ids: List[str] = []

        for cycle in self.cycles:
            if cycle.input_node_id not in lineage:
                continue
            phase1 = cycle.stage_results.get("phase1") if isinstance(cycle.stage_results.get("phase1"), dict) else {}
            phase3 = cycle.stage_results.get("phase3") if isinstance(cycle.stage_results.get("phase3"), dict) else {}
            raw_global = phase1.get("global_diagnosis") if isinstance(phase1, dict) else []
            normalized_global = []
            if isinstance(raw_global, list):
                for item in raw_global:
                    normalized = self._normalize_focus_entry(item, self._known_aoi_ids)
                    if not normalized:
                        continue
                    focus_id = normalized["focus_id"]
                    merged = dict(registry.get(focus_id, {}))
                    merged.update(normalized)
                    merged["last_seen_cycle_index"] = cycle.cycle_index
                    registry[focus_id] = merged
                    normalized_global.append(dict(merged))
            latest_priority_ids = self._dedupe_str_list(phase1.get("priority_focuses") or [])
            latest_done_ids = self._dedupe_str_list(phase1.get("done_focus_ids") or [])
            completed.update(latest_done_ids)

            focus_decisions = phase3.get("focus_decisions") if isinstance(phase3, dict) else []
            if isinstance(focus_decisions, list):
                for item in focus_decisions:
                    if not isinstance(item, dict):
                        continue
                    focus_id = str(item.get("focus_id") or "").strip()
                    decision = str(item.get("decision") or "").strip().lower()
                    if focus_id and decision == "done":
                        completed.add(focus_id)
            reflection = phase3.get("reflection") if isinstance(phase3, dict) else {}
            completed.update(self._dedupe_str_list(reflection.get("completed_focus_ids") or []))

            stage1_history.append(
                {
                    "cycle_index": cycle.cycle_index,
                    "global_diagnosis": normalized_global,
                    "priority_focuses": list(latest_priority_ids),
                    "done_focus_ids": list(latest_done_ids),
                }
            )

        for focus_id, entry in registry.items():
            entry["is_done"] = focus_id in completed

        all_focus_ids = sorted(registry.keys(), key=lambda item: (self._parse_focus_index(item), item))
        done_focus_ids = [focus_id for focus_id in all_focus_ids if focus_id in completed]
        active_focus_ids = [focus_id for focus_id in all_focus_ids if focus_id not in completed]
        next_focus_index = max([self._parse_focus_index(item) for item in all_focus_ids] + [0]) + 1
        return {
            "focus_registry": {focus_id: dict(registry[focus_id]) for focus_id in all_focus_ids},
            "all_focus_ids": all_focus_ids,
            "done_focus_ids": done_focus_ids,
            "active_focus_ids": active_focus_ids,
            "latest_priority_focus_ids": latest_priority_ids,
            "latest_done_focus_ids": latest_done_ids,
            "stage1_history": stage1_history[-6:],
            "next_focus_id_hint": f"focus_{next_focus_index}",
        }

    def _build_stage_requirements(
        self,
        focus_tracking: Dict[str, Any],
        stage_results: Dict[str, Dict[str, Any]],
    ) -> Dict[str, Any]:
        phase1 = stage_results.get("phase1") if isinstance(stage_results.get("phase1"), dict) else {}
        raw_global = phase1.get("global_diagnosis") if isinstance(phase1, dict) else []
        current_global_focuses: List[Dict[str, Any]] = []
        focus_map: Dict[str, Dict[str, Any]] = {}
        if isinstance(raw_global, list):
            for item in raw_global:
                normalized = self._normalize_focus_entry(item, self._known_aoi_ids)
                if not normalized:
                    continue
                normalized = self._enrich_focus_entry(normalized)
                current_global_focuses.append(normalized)
                focus_map[normalized["focus_id"]] = normalized
        done_focus_ids = self._dedupe_str_list(phase1.get("done_focus_ids") or [])
        priority_focus_ids = [
            focus_id
            for focus_id in self._dedupe_str_list(phase1.get("priority_focuses") or [])
            if focus_id in focus_map and focus_id not in done_focus_ids
        ]
        selected_focuses = [focus_map[focus_id] for focus_id in priority_focus_ids if focus_id in focus_map]
        selected_aoi_ids_by_focus = {
            item["focus_id"]: list(item.get("aoi_ids") or [])
            for item in selected_focuses
        }
        return {
            "current_global_focuses": current_global_focuses,
            "current_global_focus_ids": [item["focus_id"] for item in current_global_focuses],
            "done_focus_ids": done_focus_ids,
            "priority_focus_ids": priority_focus_ids,
            "selected_focuses": selected_focuses,
            "selected_aoi_ids_by_focus": selected_aoi_ids_by_focus,
            "pending_global_focus_ids": [
                item["focus_id"]
                for item in current_global_focuses
                if item["focus_id"] not in done_focus_ids
            ],
            "focus_tracking": focus_tracking,
        }

    def _make_stage_result_validator(
        self,
        stage_name: str,
        focus_tracking: Dict[str, Any],
        stage_results: Dict[str, Dict[str, Any]],
        toolkit: Optional[Any] = None,
        apply_history_start: int = 0,
    ) -> Callable[[Dict[str, Any]], Optional[str]]:
        requirements = self._build_stage_requirements(focus_tracking, stage_results)

        def validator(result_obj: Dict[str, Any]) -> Optional[str]:
            if stage_name == "phase1":
                return self._validate_phase1_result(result_obj, focus_tracking)
            if stage_name == "phase2":
                return self._validate_phase2_result(result_obj, requirements)
            if stage_name == "phase3":
                return self._validate_phase3_result(
                    result_obj,
                    requirements,
                    toolkit=toolkit,
                    apply_history_start=apply_history_start,
                )
            return None

        return validator

    def _normalize_stage_result(self, stage_name: str, result_obj: Dict[str, Any]) -> Dict[str, Any]:
        result = json.loads(json.dumps(result_obj, ensure_ascii=False))
        if stage_name == "phase1":
            raw_global = result.get("global_diagnosis")
            normalized_global = []
            if isinstance(raw_global, list):
                for item in raw_global:
                    normalized = self._normalize_focus_entry(item, self._known_aoi_ids)
                    if normalized:
                        normalized_global.append(self._enrich_focus_entry(normalized))
            result["global_diagnosis"] = normalized_global
            return result
        if stage_name == "phase2":
            for item in result.get("focus_evidence") or []:
                if not isinstance(item, dict):
                    continue
                for aoi_item in item.get("aoi_evidence") or []:
                    if isinstance(aoi_item, dict):
                        aoi_item["aoi_id"] = self._canonicalize_aoi_id(aoi_item.get("aoi_id"), self._known_aoi_ids)
            checklist = result.get("coverage_checklist")
            if isinstance(checklist, dict):
                for key in ("per_focus_required_aoi_ids", "per_focus_covered_aoi_ids", "missing_aoi_ids_by_focus"):
                    values = checklist.get(key)
                    if isinstance(values, dict):
                        checklist[key] = {
                            str(focus_id): self._canonicalize_aoi_list(aoi_ids, self._known_aoi_ids, max_items=None)
                            for focus_id, aoi_ids in values.items()
                        }
            return result
        if stage_name == "phase3":
            for item in result.get("focus_decisions") or []:
                if not isinstance(item, dict):
                    continue
                for aoi_item in item.get("aoi_decisions") or []:
                    if isinstance(aoi_item, dict):
                        aoi_item["aoi_id"] = self._canonicalize_aoi_id(aoi_item.get("aoi_id"), self._known_aoi_ids)
            checklist = result.get("coverage_checklist")
            if isinstance(checklist, dict):
                for key in ("required_aoi_ids_by_focus", "covered_aoi_ids_by_focus", "missing_aoi_ids_by_focus"):
                    values = checklist.get(key)
                    if isinstance(values, dict):
                        checklist[key] = {
                            str(focus_id): self._canonicalize_aoi_list(aoi_ids, self._known_aoi_ids, max_items=None)
                            for focus_id, aoi_ids in values.items()
                        }
            execution = result.get("execution")
            if isinstance(execution, dict):
                execution["plan"] = self._canonicalize_plan(execution.get("plan"), self._known_aoi_ids)
                branch_plans = execution.get("branch_plans")
                if isinstance(branch_plans, list):
                    for spec in branch_plans:
                        if isinstance(spec, dict):
                            spec["plan"] = self._canonicalize_plan(spec.get("plan"), self._known_aoi_ids)
            return result
        return result

    def _validate_phase1_result(self, result_obj: Dict[str, Any], focus_tracking: Dict[str, Any]) -> Optional[str]:
        if str(result_obj.get("phase") or "").strip().lower() != "phase1":
            return "Stage 1 result must set `phase` to `phase1`."
        raw_global = result_obj.get("global_diagnosis")
        if not isinstance(raw_global, list) or not raw_global:
            return "Stage 1 `global_diagnosis` must be a non-empty list of focus dicts."
        global_focuses: List[Dict[str, Any]] = []
        seen_ids = set()
        for item in raw_global:
            normalized = self._normalize_focus_entry(item, self._known_aoi_ids)
            if not normalized:
                return "Each Stage 1 `global_diagnosis` item must have non-empty `focus_id`, `label`, and `aoi_ids`."
            if len(normalized.get("bbox_rel") or []) != 4:
                return (
                    "Each Stage 1 `global_diagnosis` item must include `bbox_rel` as "
                    "[x1, y1, x2, y2] normalized to [0,1] with x1 <= x2 and y1 <= y2."
                )
            focus_id = normalized["focus_id"]
            if focus_id in seen_ids:
                return "Stage 1 `global_diagnosis` cannot repeat focus ids."
            seen_ids.add(focus_id)
            global_focuses.append(normalized)
        done_focus_ids = self._dedupe_str_list(result_obj.get("done_focus_ids") or [])
        previously_done = set(focus_tracking.get("done_focus_ids") or [])
        if not previously_done.issubset(set(done_focus_ids)):
            return "Stage 1 must carry forward all previously completed focus ids in `done_focus_ids`."
        if set(done_focus_ids) & seen_ids:
            return "Stage 1 active `global_diagnosis` focus ids cannot overlap with `done_focus_ids`."
        existing_focus_ids = set(focus_tracking.get("all_focus_ids") or [])
        next_hint = str(focus_tracking.get("next_focus_id_hint") or "focus_1")
        next_index = self._parse_focus_index(next_hint)
        for item in global_focuses:
            focus_id = item["focus_id"]
            if focus_id in previously_done:
                return "Stage 1 cannot reuse an already completed focus id inside active `global_diagnosis`."
            if focus_id not in existing_focus_ids and self._parse_focus_index(focus_id) < next_index:
                return "New Stage 1 focus ids must keep increasing instead of reusing old smaller ids."
        priority_focus_ids = self._dedupe_str_list(result_obj.get("priority_focuses") or [])
        if not priority_focus_ids:
            return "Stage 1 must select at least one priority focus id."
        if len(priority_focus_ids) > 2:
            return "Stage 1 `priority_focuses` may contain at most 2 focus ids."
        active_focus_ids = seen_ids - set(done_focus_ids)
        if not set(priority_focus_ids).issubset(active_focus_ids):
            return "Stage 1 `priority_focuses` must be ids chosen from unfinished `global_diagnosis` entries only."
        return None

    def _validate_phase2_result(self, result_obj: Dict[str, Any], requirements: Dict[str, Any]) -> Optional[str]:
        if str(result_obj.get("phase") or "").strip().lower() != "phase2":
            return "Stage 2 result must set `phase` to `phase2`."
        required_focus_ids = list(requirements.get("priority_focus_ids") or [])
        required_aoi_ids_by_focus = {
            str(k): list(v)
            for k, v in (requirements.get("selected_aoi_ids_by_focus") or {}).items()
        }
        raw_focus_evidence = result_obj.get("focus_evidence")
        if not isinstance(raw_focus_evidence, list) or not raw_focus_evidence:
            return "Stage 2 `focus_evidence` must be a non-empty list."
        evidence_map: Dict[str, Dict[str, Any]] = {}
        for item in raw_focus_evidence:
            if not isinstance(item, dict):
                return "Each Stage 2 `focus_evidence` item must be a dict."
            focus_id = str(item.get("focus_id") or "").strip()
            if not focus_id:
                return "Each Stage 2 `focus_evidence` item must include `focus_id`."
            if focus_id in evidence_map:
                return "Stage 2 cannot repeat focus ids inside `focus_evidence`."
            aoi_evidence = item.get("aoi_evidence")
            if not isinstance(aoi_evidence, list) or not aoi_evidence:
                return "Each Stage 2 focus must include non-empty `aoi_evidence`."
            covered_aoi_ids = []
            seen_aoi_ids = set()
            for aoi_item in aoi_evidence:
                if not isinstance(aoi_item, dict):
                    return "Each Stage 2 `aoi_evidence` item must be a dict."
                aoi_id = self._canonicalize_aoi_id(aoi_item.get("aoi_id"), self._known_aoi_ids)
                if not aoi_id:
                    return "Each Stage 2 `aoi_evidence` item must include `aoi_id`."
                if aoi_id in seen_aoi_ids:
                    return "Stage 2 cannot repeat AOI ids inside one focus evidence block."
                seen_aoi_ids.add(aoi_id)
                covered_aoi_ids.append(aoi_id)
            evidence_map[focus_id] = {"covered_aoi_ids": covered_aoi_ids}
        if set(evidence_map.keys()) != set(required_focus_ids):
            return "Stage 2 must cover every selected priority focus exactly once."
        for focus_id in required_focus_ids:
            required_aoi_ids = required_aoi_ids_by_focus.get(focus_id, [])
            covered_aoi_ids = evidence_map.get(focus_id, {}).get("covered_aoi_ids", [])
            covered_set = {self._canonicalize_aoi_id(x, self._known_aoi_ids) for x in covered_aoi_ids}
            required_set = {self._canonicalize_aoi_id(x, self._known_aoi_ids) for x in required_aoi_ids}
            if covered_set != required_set:
                missing = sorted(required_set - covered_set)
                extra = sorted(covered_set - required_set)
                return (
                    f"Stage 2 AOI coverage mismatch for {focus_id}. "
                    f"Use full AOI IDs and cover exactly the selected list. missing={missing}; extra={extra}; required={sorted(required_set)}"
                )
        checklist = result_obj.get("coverage_checklist")
        if not isinstance(checklist, dict):
            return "Stage 2 must include `coverage_checklist`."
        missing_focus_ids = self._dedupe_str_list(checklist.get("missing_focus_ids") or [])
        if missing_focus_ids:
            return "Stage 2 `coverage_checklist` still reports missing focus ids."
        missing_aoi_ids_by_focus = checklist.get("missing_aoi_ids_by_focus")
        if not isinstance(missing_aoi_ids_by_focus, dict):
            return "Stage 2 `coverage_checklist.missing_aoi_ids_by_focus` must be a dict."
        for focus_id in required_focus_ids:
            if self._dedupe_str_list(missing_aoi_ids_by_focus.get(focus_id) or []):
                return f"Stage 2 checklist still reports missing AOIs for {focus_id}."
        return None

    def _validate_phase3_result(
        self,
        result_obj: Dict[str, Any],
        requirements: Dict[str, Any],
        toolkit: Optional[Any] = None,
        apply_history_start: int = 0,
    ) -> Optional[str]:
        if str(result_obj.get("phase") or "").strip().lower() != "phase3":
            return "Stage 3 result must set `phase` to `phase3`."
        required_global_focus_ids = list(requirements.get("current_global_focus_ids") or [])
        required_priority_focus_ids = list(requirements.get("priority_focus_ids") or [])
        required_aoi_ids_by_focus = {
            str(k): list(v)
            for k, v in (requirements.get("selected_aoi_ids_by_focus") or {}).items()
        }
        raw_global_statuses = result_obj.get("global_focus_statuses")
        if not isinstance(raw_global_statuses, list) or not raw_global_statuses:
            return "Stage 3 must include `global_focus_statuses` for every current global focus."
        status_ids = []
        for item in raw_global_statuses:
            if not isinstance(item, dict):
                return "Each Stage 3 `global_focus_statuses` item must be a dict."
            focus_id = str(item.get("focus_id") or "").strip()
            if not focus_id:
                return "Each Stage 3 `global_focus_statuses` item must include `focus_id`."
            status_ids.append(focus_id)
        if set(status_ids) != set(required_global_focus_ids):
            return "Stage 3 `global_focus_statuses` must cover every focus from the current `global_diagnosis`."

        raw_focus_decisions = result_obj.get("focus_decisions")
        if not isinstance(raw_focus_decisions, list) or not raw_focus_decisions:
            return "Stage 3 must include non-empty `focus_decisions`."
        decision_map: Dict[str, Dict[str, Any]] = {}
        for item in raw_focus_decisions:
            if not isinstance(item, dict):
                return "Each Stage 3 `focus_decisions` item must be a dict."
            focus_id = str(item.get("focus_id") or "").strip()
            if not focus_id:
                return "Each Stage 3 `focus_decisions` item must include `focus_id`."
            if focus_id in decision_map:
                return "Stage 3 cannot repeat focus ids inside `focus_decisions`."
            aoi_decisions = item.get("aoi_decisions")
            if not isinstance(aoi_decisions, list) or not aoi_decisions:
                return "Each Stage 3 focus decision must include non-empty `aoi_decisions`."
            seen_aoi_ids = set()
            covered_aoi_ids = []
            for aoi_item in aoi_decisions:
                if not isinstance(aoi_item, dict):
                    return "Each Stage 3 `aoi_decisions` item must be a dict."
                aoi_id = self._canonicalize_aoi_id(aoi_item.get("aoi_id"), self._known_aoi_ids)
                if not aoi_id:
                    return "Each Stage 3 `aoi_decisions` item must include `aoi_id`."
                if aoi_id in seen_aoi_ids:
                    return "Stage 3 cannot repeat AOI ids inside one focus decision block."
                seen_aoi_ids.add(aoi_id)
                covered_aoi_ids.append(aoi_id)
            decision_map[focus_id] = {"covered_aoi_ids": covered_aoi_ids}
        if set(decision_map.keys()) != set(required_priority_focus_ids):
            return "Stage 3 `focus_decisions` must cover every selected priority focus exactly once."
        for focus_id in required_priority_focus_ids:
            covered_set = {
                self._canonicalize_aoi_id(x, self._known_aoi_ids)
                for x in decision_map.get(focus_id, {}).get("covered_aoi_ids", [])
            }
            required_set = {
                self._canonicalize_aoi_id(x, self._known_aoi_ids)
                for x in required_aoi_ids_by_focus.get(focus_id, [])
            }
            if covered_set != required_set:
                missing = sorted(required_set - covered_set)
                extra = sorted(covered_set - required_set)
                return (
                    f"Stage 3 AOI decision coverage mismatch for {focus_id}. "
                    f"Use full AOI IDs and cover exactly the selected list. missing={missing}; extra={extra}; required={sorted(required_set)}"
                )

        checklist = result_obj.get("coverage_checklist")
        if not isinstance(checklist, dict):
            return "Stage 3 must include `coverage_checklist`."
        if self._dedupe_str_list(checklist.get("missing_global_focus_ids") or []):
            return "Stage 3 checklist still reports missing global focuses."
        missing_priority_focus_ids = self._dedupe_str_list(checklist.get("missing_priority_focus_ids") or [])
        if missing_priority_focus_ids:
            return "Stage 3 checklist still reports missing selected focuses."
        missing_aoi_ids_by_focus = checklist.get("missing_aoi_ids_by_focus")
        if not isinstance(missing_aoi_ids_by_focus, dict):
            return "Stage 3 `coverage_checklist.missing_aoi_ids_by_focus` must be a dict."
        for focus_id in required_priority_focus_ids:
            if self._dedupe_str_list(missing_aoi_ids_by_focus.get(focus_id) or []):
                return f"Stage 3 checklist still reports missing AOIs for {focus_id}."
        status_map = {
            str(item.get("focus_id") or "").strip(): str(item.get("status") or "").strip().lower()
            for item in raw_global_statuses
            if isinstance(item, dict)
        }
        reflection = result_obj.get("reflection") if isinstance(result_obj.get("reflection"), dict) else {}
        completed_focus_ids = self._dedupe_str_list(reflection.get("completed_focus_ids") or [])
        completed_without_done = [
            focus_id
            for focus_id in completed_focus_ids
            if status_map.get(focus_id) != "done"
        ]
        if completed_without_done:
            return (
                "Stage 3 `reflection.completed_focus_ids` may only include focuses whose "
                f"`global_focus_statuses.status` is `done`. inconsistent={completed_without_done}"
            )
        execution = result_obj.get("execution") if isinstance(result_obj.get("execution"), dict) else {}
        action = str(execution.get("action") or "").strip().lower()
        next_step = str(reflection.get("next_step") or "").strip().lower()
        neutral_search = reflection.get("neutral_search") if isinstance(reflection.get("neutral_search"), dict) else {}
        if action == "finish" or next_step == "finish":
            unresolved = [focus_id for focus_id in required_global_focus_ids if status_map.get(focus_id) != "done"]
            if unresolved:
                if isinstance(execution, dict) and action in {"commit", ""} and isinstance(reflection, dict):
                    execution["action"] = "commit"
                    reflection["next_step"] = "continue"
                    note = (
                        "Auto-corrected invalid finish request to commit/continue because unresolved global "
                        f"focuses remain: {unresolved}."
                    )
                    rationale = str(execution.get("rationale") or "").strip()
                    execution["rationale"] = f"{rationale} {note}".strip()
                    summary = str(reflection.get("summary") or "").strip()
                    reflection["summary"] = f"{summary} {note}".strip()
                    action = "commit"
                    next_step = "continue"
                else:
                    return (
                        "Stage 3 cannot request finish until every current global focus is marked done. "
                        f"Use action=commit and next_step=continue when carrying focuses over. unresolved={unresolved}"
                    )
            if action == "finish" and unresolved:
                return (
                    "Stage 3 cannot request finish until every current global focus is marked done. "
                    f"Use action=commit and next_step=continue when carrying focuses over. unresolved={unresolved}"
                )
        if action in {"commit", "finish"}:
            requested_plan = self._canonicalize_plan(execution.get("plan"), self._known_aoi_ids)
            actual_plan: Dict[str, str] = {}
            if toolkit is not None:
                try:
                    actual_plan = self._canonicalize_plan(toolkit.env.snapshot_plan_mapping(), self._known_aoi_ids)
                except Exception:
                    actual_plan = {}
            if requested_plan != actual_plan:
                return (
                    "Stage 3 `execution.plan` must equal the cumulative temp-plan snapshot already materialized by "
                    "`apply_temp_plan_changes`. If you want a modification, call the tool before final result. "
                    f"requested={requested_plan}; applied_snapshot={actual_plan}"
                )
            neutral_issue = self._validate_phase3_neutral_search(
                toolkit=toolkit,
                actual_plan=actual_plan,
                neutral_search=neutral_search,
                action=action,
                next_step=next_step,
                apply_history_start=apply_history_start,
            )
            if neutral_issue:
                return neutral_issue
        return None

    @classmethod
    def _has_positive_metric_delta(cls, delta: Dict[str, Any]) -> bool:
        for value in (delta or {}).values():
            try:
                if float(value) > cls.NEUTRAL_METRIC_TOL:
                    return True
            except Exception:
                continue
        return False

    @classmethod
    def _has_any_metric_delta(cls, delta: Dict[str, Any]) -> bool:
        for value in (delta or {}).values():
            try:
                if abs(float(value)) > cls.NEUTRAL_METRIC_TOL:
                    return True
            except Exception:
                continue
        return False

    def _validate_phase3_neutral_search(
        self,
        toolkit: Optional[Any],
        actual_plan: Dict[str, str],
        neutral_search: Dict[str, Any],
        action: str,
        next_step: str,
        apply_history_start: int = 0,
    ) -> Optional[str]:
        if toolkit is None or not hasattr(toolkit, "apply_history"):
            return None
        raw_history = getattr(toolkit, "apply_history", None)
        if not isinstance(raw_history, list) or not raw_history:
            return None
        sliced_history = raw_history[int(apply_history_start):] if int(apply_history_start) > 0 else raw_history
        apply_history = [item for item in sliced_history if isinstance(item, dict) and item.get("success")]
        if not apply_history:
            return None

        consecutive_neutral = 0
        trailing_neutral_entries: List[Dict[str, Any]] = []
        for item in reversed(apply_history):
            step_changes = item.get("step_changes") if isinstance(item.get("step_changes"), list) else []
            if not step_changes:
                continue
            metric_delta = item.get("metric_delta") if isinstance(item.get("metric_delta"), dict) else {}
            if self._has_any_metric_delta(metric_delta):
                break
            consecutive_neutral += 1
            trailing_neutral_entries.append(item)

        if not trailing_neutral_entries:
            return None

        raw_wait = str(neutral_search.get("decision") or "").strip().lower()
        declared_streak = neutral_search.get("consecutive_neutral_count")
        declared_reason = str(neutral_search.get("reason") or "").strip()
        all_attempts_neutral = all(
            not self._has_any_metric_delta(item.get("metric_delta") if isinstance(item.get("metric_delta"), dict) else {})
            for item in apply_history
            if isinstance(item.get("step_changes"), list) and item.get("step_changes")
        )
        first_snapshot = {}
        if apply_history:
            first_snapshot = self._canonicalize_plan(
                apply_history[0].get("snapshot_plan_after"),
                self._known_aoi_ids,
            )
        if action in {"commit", "finish"} and len(apply_history) >= 2 and consecutive_neutral >= 2 and actual_plan:
            if all_attempts_neutral and first_snapshot and actual_plan == first_snapshot:
                if raw_wait not in {"wait", "undo", "n/a"}:
                    return "Stage 3 `reflection.neutral_search.decision` must be one of `wait`, `undo`, or `n/a`."
                if raw_wait in {"wait", "undo"} and not declared_reason:
                    return "Stage 3 `reflection.neutral_search.reason` is required when decision is `wait` or `undo`."
                if declared_streak is not None:
                    try:
                        declared_value = int(declared_streak)
                    except Exception:
                        return "Stage 3 `reflection.neutral_search.consecutive_neutral_count` must be an integer."
                    if declared_value != consecutive_neutral:
                        return (
                            "Stage 3 `reflection.neutral_search.consecutive_neutral_count` must match the observed "
                            f"trailing neutral apply count. observed={consecutive_neutral}; declared={declared_value}"
                        )
                return None
            changed_ids = []
            for item in trailing_neutral_entries:
                for aoi_id in item.get("changed_aoi_ids") or []:
                    if aoi_id not in changed_ids:
                        changed_ids.append(aoi_id)
            return (
                "Stage 3 ended with two or more consecutive metric-neutral apply_temp_plan_changes while still "
                "keeping a non-empty temp snapshot. If every tested candidate in this stage was metric-neutral, "
                "restore the first successfully tested snapshot and retain that first attempt as execution.plan; "
                "otherwise undo the trailing neutral edits by re-applying original land-use on those AOIs, or try a "
                f"different candidate first. trailing_neutral_count={consecutive_neutral}; "
                f"trailing_changed_aoi_ids={changed_ids}; current_snapshot={actual_plan}; first_snapshot={first_snapshot}. "
                "In `reflection.neutral_search`, explicitly report decision=`undo` or decision=`wait` with reason."
            )

        if raw_wait:
            if raw_wait not in {"wait", "undo", "n/a"}:
                return "Stage 3 `reflection.neutral_search.decision` must be one of `wait`, `undo`, or `n/a`."
            if raw_wait == "wait" and consecutive_neutral >= 2 and actual_plan:
                return (
                    "Stage 3 cannot keep decision=`wait` after two consecutive metric-neutral applied modifications "
                    "while a temp snapshot is still active. Undo the neutral edits and try another candidate."
                )
            if raw_wait in {"wait", "undo"} and not declared_reason:
                return "Stage 3 `reflection.neutral_search.reason` is required when decision is `wait` or `undo`."
            if declared_streak is not None:
                try:
                    declared_value = int(declared_streak)
                except Exception:
                    return "Stage 3 `reflection.neutral_search.consecutive_neutral_count` must be an integer."
                if declared_value != consecutive_neutral:
                    return (
                        "Stage 3 `reflection.neutral_search.consecutive_neutral_count` must match the observed "
                        f"trailing neutral apply count. observed={consecutive_neutral}; declared={declared_value}"
                    )
        elif consecutive_neutral > 0:
            return (
                "Stage 3 observed metric-neutral apply_temp_plan_changes. Add `reflection.neutral_search` with "
                "`decision`, `consecutive_neutral_count`, and `reason` so the controller can tell whether you are "
                "waiting one round or undoing the neutral edits."
            )
        return None

    @classmethod
    def _evaluate_finish_gate(cls, stage_results: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        phase1 = stage_results.get("phase1") if isinstance(stage_results.get("phase1"), dict) else {}
        phase3 = stage_results.get("phase3") if isinstance(stage_results.get("phase3"), dict) else {}
        raw_global = phase1.get("global_diagnosis") if isinstance(phase1, dict) else []
        current_focus_ids = []
        if isinstance(raw_global, list):
            for item in raw_global:
                normalized = cls._normalize_focus_entry(item)
                if normalized:
                    current_focus_ids.append(normalized["focus_id"])
        current_focus_ids = cls._dedupe_str_list(current_focus_ids)
        already_done = set(cls._dedupe_str_list(phase1.get("done_focus_ids") or []))
        global_statuses = phase3.get("global_focus_statuses") if isinstance(phase3, dict) else []
        completed_now = set()
        remaining = []
        if isinstance(global_statuses, list):
            for item in global_statuses:
                if not isinstance(item, dict):
                    continue
                focus_id = str(item.get("focus_id") or "").strip()
                status = str(item.get("status") or "").strip().lower()
                if not focus_id:
                    continue
                if status == "done":
                    completed_now.add(focus_id)
                else:
                    remaining.append(focus_id)
        unresolved = [focus_id for focus_id in current_focus_ids if focus_id not in already_done and focus_id not in completed_now]
        return {
            "allow_finish": not unresolved,
            "current_focus_ids": current_focus_ids,
            "already_done_focus_ids": sorted(already_done),
            "completed_now_focus_ids": sorted(completed_now),
            "unresolved_focus_ids": unresolved,
            "reported_remaining_focus_ids": cls._dedupe_str_list(remaining),
        }

    @staticmethod
    def _validate_skill_experience_think(think_text: str) -> Optional[str]:
        text = str(think_text or "").strip()
        match = re.search(r"skill_experience_mode\s*:\s*(follow|explore)\b", text, flags=re.IGNORECASE)
        if not match:
            return (
                "Inside <think>, explicitly output one binary decision line: "
                "`skill_experience_mode: follow` or `skill_experience_mode: explore`."
            )
        return None

    def _consolidate_self_evolving_wave(
        self,
        results: Dict[str, CycleExecutionOutput],
        ordered_runtimes: List[BranchRuntime],
        wave_memory_snapshot: Dict[str, Any],
        wave_dynamic_tool_names: List[str],
    ) -> None:
        if not self.enable_self_evolving:
            for output in results.values():
                for stage_name in PHASES:
                    output.stage_updates[stage_name] = EvolutionNotebook.stage_block_from_snapshot(wave_memory_snapshot, stage_name)
                output.metadata["wave_self_evolving_alignment"] = {
                    "used_shared_snapshot": False,
                    "wave_dynamic_tool_names": list(wave_dynamic_tool_names),
                }
            return
        stage_payloads: Dict[str, List[Dict[str, Any]]] = {stage_name: [] for stage_name in PHASES}
        merged_tool_events: List[Dict[str, Any]] = []
        seen_tool_keys = set()
        for runtime in ordered_runtimes:
            output = results.get(runtime.branch_id)
            if output is None:
                continue
            payloads = output.stage_consolidation_payloads or {}
            for stage_name in PHASES:
                item = payloads.get(stage_name)
                if isinstance(item, dict):
                    stage_payloads[stage_name].append(item)
                    for event in item.get("tool_events") or []:
                        if not isinstance(event, dict):
                            continue
                        tool_key = (
                            str(event.get("proposal_id") or ""),
                            str(event.get("tool_name") or ""),
                            str(event.get("operation") or ""),
                        )
                        if tool_key in seen_tool_keys:
                            continue
                        seen_tool_keys.add(tool_key)
                        merged_tool_events.append(dict(event))
        self.evolution.record_tool_events(merged_tool_events)
        merged_stage_updates: Dict[str, Dict[str, str]] = {}
        for stage_name in PHASES:
            merged = self.evolution.consolidate_wave_stage(
                stage_name,
                stage_payloads.get(stage_name) or [],
                metadata={"branch_count": len(stage_payloads.get(stage_name) or []), "dynamic_tool_names": list(wave_dynamic_tool_names)},
            )
            merged_stage_updates[stage_name] = {
                "experience": merged.get("experience", ""),
                "skill": merged.get("skill", ""),
                "tool_memory": self.evolution.tool_block(),
            }
        for output in results.values():
            output.stage_updates = {stage_name: dict(merged_stage_updates[stage_name]) for stage_name in PHASES}
            output.metadata["wave_self_evolving_alignment"] = {
                "used_shared_snapshot": True,
                "wave_dynamic_tool_names": list(wave_dynamic_tool_names),
                "merged_tool_event_count": len(merged_tool_events),
                "merged_stage_updates": {stage_name: dict(merged_stage_updates[stage_name]) for stage_name in PHASES},
            }

    def _execute_single_cycle(
        self,
        runtime: BranchRuntime,
        node: StateNode,
        live_leaf_count: int,
        wave_memory_snapshot: Dict[str, Any],
        wave_dynamic_tool_names: List[str],
    ) -> CycleExecutionOutput:
        runtime.cycle_count += 1
        cycle_index = self._reserve_cycle_index()
        restore = runtime.toolkit.env.restore_plan_snapshot(node.plan, name=f"{runtime.branch_id}_node_{node.node_id}_cycle_{cycle_index}")
        current_metrics = restore.get("metrics") or runtime.toolkit.env.evaluate_base().get("metrics", {})
        lineage_summaries = self._collect_lineage_summaries(node.node_id, limit=6)
        focus_tracking = self._collect_focus_tracking(node.node_id)
        stage_results: Dict[str, Dict[str, Any]] = {}
        stage_updates: Dict[str, Dict[str, str]] = {}
        stage_trace_payload: Dict[str, Any] = {}
        stage_full_trace_payload: Dict[str, Any] = {}
        stage_consolidation_payloads: Dict[str, Dict[str, Any]] = {}
        stage_apply_history_starts: Dict[str, int] = {}
        trace_path = runtime.output_dir / f"cycle_{cycle_index:03d}.json"
        full_trace_path = runtime.output_dir / f"cycle_{cycle_index:03d}_full.json"

        for stage_name in PHASES:
            if self.enable_self_evolving:
                memory = EvolutionNotebook.stage_block_from_snapshot(wave_memory_snapshot, stage_name)
            else:
                memory = {"experience": "", "skill": "", "tool_memory": "Self-evolving is disabled for this run."}
            allowed_tools: List[str] = []
            stage_requirements = self._build_stage_requirements(focus_tracking, stage_results)
            stage_context = {
                "variant_name": self.variant_name,
                "task_prompt": self.task_prompt,
                "branch_id": runtime.branch_id,
                "branch_label": runtime.branch_label,
                "lineage_labels": runtime.lineage_labels,
                "cycle_index": cycle_index,
                "current_node": {
                    "node_id": node.node_id,
                    "plan": node.plan,
                    "metrics": current_metrics,
                    "node_origin": node.node_origin,
                    "copied_from_node_id": node.copied_from_node_id,
                    "rollback_rule": node.rollback_rule,
                    "rollback_anchor_id": node.rollback_anchor_id,
                    "branch_group_labels": node.branch_group_labels,
                    "severe_conflict_used": node.severe_conflict_used,
                    "rollback_used": node.rollback_used,
                },
                "lineage_summaries": lineage_summaries,
                "base_context": runtime.toolkit.base_context(),
                "available_tool_schemas": runtime.toolkit.tool_schemas(),
                "phase_outputs": dict(stage_results),
                "focus_tracking": focus_tracking,
                "stage_requirements": stage_requirements,
                "memory": memory,
                "enable_conflict_judgement": self.enable_conflict_judgement,
                "enable_self_evolving": self.enable_self_evolving,
                "stage_max_steps": self.stage_max_steps,
                "current_stage_max_steps": self.stage_max_steps.get(stage_name, 8),
                "max_branch_leaves": self.max_branch_leaves,
                "live_leaf_count": live_leaf_count,
                "branch_cycle_index": runtime.cycle_count,
                "branch_max_rounds": self.max_rounds,
                "remaining_branch_cycles_after_current": max(0, self.max_rounds - runtime.cycle_count),
                "is_final_branch_cycle": runtime.cycle_count >= self.max_rounds,
                "dynamic_tool_names": list(wave_dynamic_tool_names),
            }
            prompt_bundle = self.prompt_builder(stage_name, stage_context)
            allowed_tools = list(prompt_bundle.get("allowed_tools") or [])
            initial_image_path = self._maybe_prepare_initial_image(runtime, stage_name, cycle_index)
            tool_event_start = len(runtime.toolkit.tool_events)
            apply_history_start = len(getattr(runtime.toolkit, "apply_history", []) or [])
            stage_apply_history_starts[stage_name] = apply_history_start
            agent = StageReActAgent(
                stage_name=stage_name,
                client=runtime.client,
                toolkit=runtime.toolkit,
                system_prompt=str(prompt_bundle.get("system_prompt") or ""),
                user_prompt=str(prompt_bundle.get("user_prompt") or ""),
                allowed_tools=allowed_tools,
                max_steps=int(prompt_bundle.get("max_steps") or self.stage_max_steps.get(stage_name, 8)),
                initial_image_path=initial_image_path,
                result_validator=self._make_stage_result_validator(
                    stage_name,
                    focus_tracking,
                    dict(stage_results),
                    toolkit=runtime.toolkit,
                    apply_history_start=apply_history_start,
                ),
                thought_validator=self._validate_skill_experience_think,
                min_tool_calls_before_result=1 if stage_name == "phase2" else 0,
                verbose=self.verbose,
            )
            try:
                stage_run = agent.run()
            except StageAgentRuntimeError as exc:
                stage_trace_payload[stage_name] = {
                    "trace": [asdict(x) for x in exc.trace],
                    "result": {},
                    "error": str(exc),
                    "counted_steps_used": exc.counted_steps_used,
                    "total_attempts": exc.total_attempts,
                    "non_counted_attempts": max(0, int(exc.total_attempts) - int(exc.counted_steps_used)),
                }
                stage_full_trace_payload[stage_name] = {
                    "full_trace": list(exc.full_trace),
                    "result": {},
                    "error": str(exc),
                    "raw_last_response": exc.raw_last_response,
                    "counted_steps_used": exc.counted_steps_used,
                    "total_attempts": exc.total_attempts,
                    "non_counted_attempts": max(0, int(exc.total_attempts) - int(exc.counted_steps_used)),
                }
                save_json(trace_path, stage_trace_payload)
                save_json(full_trace_path, stage_full_trace_payload)
                raise
            normalized_result = self._normalize_stage_result(stage_name, stage_run.structured_result)
            stage_results[stage_name] = normalized_result
            stage_trace_payload[stage_name] = {
                "trace": [asdict(x) for x in stage_run.trace],
                "result": normalized_result,
                "counted_steps_used": stage_run.counted_steps_used,
                "total_attempts": stage_run.total_attempts,
                "non_counted_attempts": max(0, int(stage_run.total_attempts) - int(stage_run.counted_steps_used)),
            }
            stage_full_trace_payload[stage_name] = {
                "full_trace": list(stage_run.full_trace),
                "result": normalized_result,
                "counted_steps_used": stage_run.counted_steps_used,
                "total_attempts": stage_run.total_attempts,
                "non_counted_attempts": max(0, int(stage_run.total_attempts) - int(stage_run.counted_steps_used)),
            }
            tool_events = runtime.toolkit.consume_tool_events(tool_event_start)
            stage_updates[stage_name] = dict(memory)
            stage_consolidation_payloads[stage_name] = {
                "branch_id": runtime.branch_id,
                "branch_label": runtime.branch_label,
                "cycle_index": cycle_index,
                "stage_name": stage_name,
                "node_id": node.node_id,
                "trace": list(stage_run.full_trace),
                "result": normalized_result,
                "tool_events": tool_events,
            }

        action, rollback_target_id, branch_specs = self._resolve_cycle_action(
            node=node,
            runtime=runtime,
            stage_results=stage_results,
            live_leaf_count=live_leaf_count,
        )

        current_plan = runtime.toolkit.env.snapshot_plan_mapping()
        if action in {"commit", "finish"}:
            current_plan = self._normalize_committed_plan(
                runtime=runtime,
                node=node,
                cycle_index=cycle_index,
                current_plan=current_plan,
                stage_results=stage_results,
                phase3_apply_history_start=stage_apply_history_starts.get("phase3", 0),
            )

        if action == "branch":
            current_plan = dict(node.plan)
            current_metrics = dict(node.metrics)
        elif action == "rollback":
            rollback_target = self.nodes.get(rollback_target_id or "")
            current_plan = dict(rollback_target.plan) if rollback_target is not None else dict(node.plan)
            current_metrics = dict(rollback_target.metrics) if rollback_target is not None else dict(node.metrics)
        else:
            current_metrics = (
                runtime.toolkit.env.evaluate_temp().get("metrics", {})
                if runtime.toolkit.env.temp_state
                else runtime.toolkit.env.evaluate_base().get("metrics", {})
            )

        summary = self._compose_cycle_summary(
            branch_label=runtime.branch_label,
            cycle_index=cycle_index,
            stage_results=stage_results,
            action=action,
            metrics=current_metrics,
        )
        cycle_metadata = self._build_cycle_metadata(
            stage_results=stage_results,
            trace_path=trace_path,
            full_trace_path=full_trace_path,
            action=action,
            rollback_target_id=rollback_target_id,
            branch_specs=branch_specs,
        )
        save_json(trace_path, stage_trace_payload)
        save_json(full_trace_path, stage_full_trace_payload)
        return CycleExecutionOutput(
            branch_id=runtime.branch_id,
            branch_label=runtime.branch_label,
            input_node_id=node.node_id,
            cycle_index=cycle_index,
            stage_results=stage_results,
            stage_updates=stage_updates,
            action=action,
            summary=summary,
            current_plan=current_plan,
            current_metrics=current_metrics,
            branch_specs=branch_specs,
            rollback_target_id=rollback_target_id,
            metadata=cycle_metadata,
            stage_consolidation_payloads=stage_consolidation_payloads,
        )

    def _materialize_branch_specs(
        self,
        runtime: BranchRuntime,
        node: StateNode,
        branch_specs: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        validated: List[Dict[str, Any]] = []
        seen_plan_keys = set()
        for idx, spec in enumerate(branch_specs[:2]):
            plan = spec.get("plan") if isinstance(spec.get("plan"), dict) else {}
            if not plan:
                continue
            runtime.toolkit.env.restore_plan_snapshot(node.plan, name=f"{runtime.branch_id}_branch_validate_{idx + 1}")
            applied = runtime.toolkit.env.apply_plan(plan, export_pb=False)
            if not applied.get("success"):
                continue
            validated_plan = runtime.toolkit.env.snapshot_plan_mapping()
            if not validated_plan:
                continue
            plan_key = tuple(sorted((str(k), str(v)) for k, v in validated_plan.items()))
            if plan_key in seen_plan_keys:
                continue
            seen_plan_keys.add(plan_key)
            validated_metrics = (
                runtime.toolkit.env.evaluate_temp().get("metrics", {})
                if runtime.toolkit.env.temp_state
                else runtime.toolkit.env.evaluate_base().get("metrics", {})
            )
            validated.append(
                {
                    **spec,
                    "validated_plan": dict(validated_plan),
                    "validated_metrics": dict(validated_metrics),
                    "validation_apply_result": applied,
                }
            )
        runtime.toolkit.env.restore_plan_snapshot(node.plan, name=f"{runtime.branch_id}_branch_validate_restore")
        return validated

    def _resolve_cycle_action(
        self,
        node: StateNode,
        runtime: BranchRuntime,
        stage_results: Dict[str, Dict[str, Any]],
        live_leaf_count: int,
    ) -> Tuple[str, Optional[str], List[Dict[str, Any]]]:
        phase3 = stage_results.get("phase3") or {}
        execution = phase3.get("execution") if isinstance(phase3.get("execution"), dict) else {}
        reflection = phase3.get("reflection") if isinstance(phase3.get("reflection"), dict) else {}
        conflict = phase3.get("conflict_assessment") if isinstance(phase3.get("conflict_assessment"), dict) else {}
        action = str(execution.get("action") or "commit").strip().lower()
        if not self.enable_conflict_judgement:
            action = "finish" if str(reflection.get("next_step") or "").lower() == "finish" else "commit"
        else:
            severity = str(conflict.get("severity") or "").lower()
            if severity.startswith("severe") and action not in {"branch", "rollback"}:
                action = "branch" if bool(conflict.get("observation_needed")) else "rollback"
            if str(reflection.get("next_step") or "").lower() == "rollback":
                action = "rollback"
            if str(reflection.get("next_step") or "").lower() == "finish":
                action = "finish"

        finish_gate = self._evaluate_finish_gate(stage_results)
        if action == "finish" and not bool(finish_gate.get("allow_finish")):
            execution["controller_action_override"] = "commit_due_to_unresolved_global_focuses"
            reflection["next_step"] = "continue"
            reflection["finish_gate_note"] = (
                "Code-level finish gate blocked termination because unresolved global focuses remain: "
                + ", ".join(finish_gate.get("unresolved_focus_ids") or [])
            )
            action = "commit"
        elif action == "commit" and bool(finish_gate.get("allow_finish")):
            execution["controller_action_override"] = "finish_all_current_focuses_done"
            reflection["next_step"] = "finish"
            reflection["finish_gate_note"] = (
                "Code-level finish gate terminated this branch because every current global focus "
                "is already marked done; starting another Stage 1 cycle would force a completed "
                "focus back into the active backlog."
            )
            action = "finish"

        rollback_target = None
        branch_specs: List[Dict[str, Any]] = []
        if action == "rollback" and node.node_origin == "branch_child":
            action = "finish" if str(reflection.get("next_step") or "").lower() == "finish" else "commit"
        if action == "rollback":
            rollback_target = self._resolve_rollback_target(node, execution.get("rollback_target"))
        if action == "branch":
            raw_branch_specs = self._normalize_branch_specs(execution.get("branch_plans"))
            branch_specs = self._materialize_branch_specs(runtime, node, raw_branch_specs)
            if node.severe_conflict_used or live_leaf_count - 1 + len(branch_specs) > self.max_branch_leaves:
                if not node.rollback_used and node.node_origin != "branch_child":
                    return "rollback", self._resolve_rollback_target(node, execution.get("rollback_target")), []
                fallback_plan = branch_specs[0].get("validated_plan") if branch_specs else {}
                if isinstance(fallback_plan, dict) and fallback_plan:
                    runtime.toolkit.env.restore_plan_snapshot(node.plan, name=f"{runtime.branch_id}_fallback_branch")
                    runtime.toolkit.env.apply_plan(fallback_plan, export_pb=False)
                    return "commit", None, []
                return "commit", None, []
            if len(branch_specs) < 2:
                if not node.rollback_used and node.node_origin != "branch_child":
                    return "rollback", self._resolve_rollback_target(node, execution.get("rollback_target")), []
                fallback_plan = branch_specs[0].get("validated_plan") if branch_specs else {}
                if isinstance(fallback_plan, dict) and fallback_plan:
                    runtime.toolkit.env.restore_plan_snapshot(node.plan, name=f"{runtime.branch_id}_fallback_branch")
                    runtime.toolkit.env.apply_plan(fallback_plan, export_pb=False)
                return "commit", None, []
        return action, rollback_target, branch_specs

    def _spawn_branch_children(
        self,
        runtime: BranchRuntime,
        node: StateNode,
        output: CycleExecutionOutput,
    ) -> List[BranchRuntime]:
        specs = list(output.branch_specs[:2])
        group_profiles = output.metadata.get("group_profiles") if isinstance(output.metadata, dict) else []
        group_profile_map = {
            str(item.get("label")): item
            for item in group_profiles
            if isinstance(item, dict) and str(item.get("label") or "").strip()
        }
        new_runtimes: List[BranchRuntime] = []
        for idx, spec in enumerate(specs):
            child_runtime = runtime if idx == 0 else self._make_branch_runtime(spec.get("label") or f"branch_{idx + 1}")
            label = str(spec.get("label") or child_runtime.branch_label or f"branch_{idx + 1}")
            child_runtime.branch_label = label
            child_runtime.lineage_labels = list(runtime.lineage_labels) + [label]
            child_runtime.cycle_count = runtime.cycle_count
            child_plan = spec.get("validated_plan") if isinstance(spec.get("validated_plan"), dict) else {}
            child_metrics = spec.get("validated_metrics") if isinstance(spec.get("validated_metrics"), dict) else {}
            if not child_plan:
                continue
            child = self._create_child_node(
                parent=node,
                branch_id=child_runtime.branch_id,
                plan=child_plan,
                metrics=child_metrics,
                cycle_index=output.cycle_index,
                stage_summary=output.summary,
                branch_label=label,
                branch_group_labels=list(node.branch_group_labels) + [label],
                rollback_anchor_id=node.node_id,
                rollback_rule=str(spec.get("rollback_rule") or "rollback_to_branch_anchor"),
                finished=False,
                terminated_reason=None,
                node_origin="branch_child",
                metadata={
                    "branch_reason": spec.get("reason"),
                    "group_label": label,
                    "group_profile": group_profile_map.get(label, {}),
                    "preferred_plan": group_profile_map.get(label, {}).get("preferred_plan"),
                    "validated_metrics": child_metrics,
                    "validated_plan": child_plan,
                    "source_cycle": output.cycle_index,
                },
            )
            child_runtime.current_node_id = child.node_id
            new_runtimes.append(child_runtime)
        return new_runtimes

    def _make_branch_runtime(self, branch_label: str) -> BranchRuntime:
        with self._seq_lock:
            self._branch_seq += 1
            branch_id = f"branch_{self._branch_seq:03d}"
        branch_dir = self.output_dir / "branches" / branch_id
        branch_dir.mkdir(parents=True, exist_ok=True)
        client = self.client_factory()
        toolkit = self.toolkit_factory(branch_dir)
        return BranchRuntime(
            branch_id=branch_id,
            branch_label=branch_label,
            current_node_id="",
            output_dir=branch_dir,
            toolkit=toolkit,
            client=client,
            lineage_labels=[branch_label],
        )

    def _new_node(
        self,
        parent_id: Optional[str],
        branch_id: str,
        branch_label: str,
        plan: Dict[str, str],
        metrics: Dict[str, Any],
        cycle_index: int,
        root_id: str,
        depth: int,
        stage_summary: str,
        branch_group_labels: List[str],
        rollback_anchor_id: Optional[str],
        rollback_rule: str,
        finished: bool,
        terminated_reason: Optional[str],
        node_origin: str,
        metadata: Optional[Dict[str, Any]] = None,
        copied_from_node_id: Optional[str] = None,
        severe_conflict_used: bool = False,
        rollback_used: bool = False,
    ) -> StateNode:
        with self._seq_lock:
            self._node_seq += 1
            node_id = f"node_{self._node_seq:04d}"
        node = StateNode(
            node_id=node_id,
            parent_id=parent_id,
            branch_id=branch_id,
            branch_label=branch_label,
            plan=dict(plan),
            metrics=dict(metrics),
            created_cycle_index=cycle_index,
            root_id=root_id,
            depth=int(depth),
            stage_summary=stage_summary,
            branch_group_labels=list(branch_group_labels),
            severe_conflict_used=bool(severe_conflict_used),
            rollback_used=bool(rollback_used),
            rollback_anchor_id=rollback_anchor_id,
            rollback_rule=rollback_rule,
            finished=finished,
            terminated_reason=terminated_reason,
            node_origin=node_origin,
            copied_from_node_id=copied_from_node_id,
            metadata=dict(metadata or {}),
        )
        if parent_id is None and not node.root_id:
            node.root_id = node.node_id
        self.nodes[node_id] = node
        return node

    def _create_child_node(
        self,
        parent: StateNode,
        branch_id: str,
        plan: Dict[str, str],
        metrics: Dict[str, Any],
        cycle_index: int,
        stage_summary: str,
        branch_label: str,
        branch_group_labels: List[str],
        rollback_anchor_id: Optional[str],
        rollback_rule: str,
        finished: bool,
        terminated_reason: Optional[str],
        node_origin: str,
        metadata: Optional[Dict[str, Any]] = None,
        copied_from_node_id: Optional[str] = None,
        severe_conflict_used: bool = False,
        rollback_used: bool = False,
    ) -> StateNode:
        node = self._new_node(
            parent_id=parent.node_id,
            branch_id=branch_id,
            branch_label=branch_label,
            plan=plan,
            metrics=metrics,
            cycle_index=cycle_index,
            root_id=parent.root_id or parent.node_id,
            depth=parent.depth + 1,
            stage_summary=stage_summary,
            branch_group_labels=branch_group_labels,
            rollback_anchor_id=rollback_anchor_id,
            rollback_rule=rollback_rule,
            finished=finished,
            terminated_reason=terminated_reason,
            node_origin=node_origin,
            copied_from_node_id=copied_from_node_id,
            severe_conflict_used=severe_conflict_used,
            rollback_used=rollback_used,
            metadata=metadata,
        )
        parent.children.append(node.node_id)
        return node

    def _create_rollback_copy_node(
        self,
        parent: StateNode,
        target: StateNode,
        branch_id: str,
        branch_label: str,
        cycle_index: int,
        stage_summary: str,
        branch_group_labels: List[str],
        current_node: StateNode,
        metadata: Dict[str, Any],
    ) -> StateNode:
        rollback_anchor_id = current_node.rollback_anchor_id if current_node.rollback_anchor_id is not None else target.rollback_anchor_id
        rollback_rule = current_node.rollback_rule if current_node.rollback_anchor_id is not None else target.rollback_rule
        rollback_metadata = dict(metadata)
        reflection = rollback_metadata.get("phase3_reflection") if isinstance(rollback_metadata.get("phase3_reflection"), dict) else {}
        execution = rollback_metadata.get("phase3_execution") if isinstance(rollback_metadata.get("phase3_execution"), dict) else {}
        rollback_metadata.update(
            {
                "rollback_copy": True,
                "rollback_from_node_id": current_node.node_id,
                "rollback_target_id": target.node_id,
                "rollback_target_created_cycle_index": target.created_cycle_index,
                "rollback_reason_summary": str(reflection.get("summary") or execution.get("rationale") or ""),
                "copied_plan": dict(target.plan),
                "copied_metrics": dict(target.metrics),
            }
        )
        return self._create_child_node(
            parent=parent,
            branch_id=branch_id,
            plan=dict(target.plan),
            metrics=dict(target.metrics),
            cycle_index=cycle_index,
            stage_summary=stage_summary,
            branch_label=branch_label,
            branch_group_labels=branch_group_labels,
            rollback_anchor_id=rollback_anchor_id,
            rollback_rule=rollback_rule,
            finished=False,
            terminated_reason=None,
            node_origin="rollback_copy",
            copied_from_node_id=target.node_id,
            severe_conflict_used=target.severe_conflict_used,
            rollback_used=target.rollback_used,
            metadata=rollback_metadata,
        )

    def _resolve_rollback_target(self, node: StateNode, requested: Optional[Any]) -> Optional[str]:
        lineage = self._eligible_rollback_lineage(node)
        requested_id = str(requested) if requested is not None else ""
        if requested_id and requested_id in lineage:
            return requested_id
        if node.rollback_rule == "rollback_to_branch_anchor" and node.rollback_anchor_id in lineage:
            return node.rollback_anchor_id
        if node.parent_id in lineage:
            return node.parent_id
        return lineage[0] if lineage else node.node_id

    def _eligible_rollback_lineage(self, node: StateNode) -> List[str]:
        lineage = self._lineage_node_ids(node.node_id)
        if node.rollback_rule != "rollback_to_branch_anchor" or not node.rollback_anchor_id:
            return lineage
        if node.rollback_anchor_id not in lineage:
            return lineage
        anchor_index = lineage.index(node.rollback_anchor_id)
        return lineage[: anchor_index + 1]

    def _lineage_node_ids(self, node_id: str) -> List[str]:
        out = []
        current = self.nodes.get(node_id)
        while current is not None:
            out.append(current.node_id)
            current = self.nodes.get(current.parent_id) if current.parent_id else None
        return out

    def _collect_lineage_summaries(self, node_id: str, limit: int = 6) -> List[Dict[str, Any]]:
        summaries: List[Dict[str, Any]] = []
        lineage = set(self._lineage_node_ids(node_id))
        for cycle in self.cycles:
            if cycle.input_node_id in lineage:
                summaries.append(
                    {
                        "cycle_index": cycle.cycle_index,
                        "branch_id": cycle.branch_id,
                        "summary": cycle.summary,
                        "action": cycle.action,
                    }
                )
        return summaries[-limit:]

    def _root_node_id(self) -> Optional[str]:
        for node in self.nodes.values():
            if node.parent_id is None:
                return node.node_id
        return None

    def _build_terminal_candidates(self, runtimes: List[BranchRuntime]) -> List[Dict[str, Any]]:
        candidates: List[Dict[str, Any]] = []
        for runtime in runtimes:
            node = self.nodes.get(runtime.current_node_id)
            if node is None:
                continue
            metadata = dict(node.metadata)
            group_profiles = metadata.get("group_profiles") if isinstance(metadata.get("group_profiles"), list) else []
            metric_score = self._metric_score(node.metrics)
            candidates.append(
                {
                    "branch_id": runtime.branch_id,
                    "branch_label": runtime.branch_label,
                    "node_id": node.node_id,
                    "root_id": node.root_id,
                    "depth": node.depth,
                    "plan": dict(node.plan),
                    "metrics": dict(node.metrics),
                    "metric_score": metric_score,
                    "is_metric_best": False,
                    "branch_group_labels": list(node.branch_group_labels),
                    "group_profiles": group_profiles,
                    "group_descriptions": [
                        {
                            "label": str(item.get("label") or ""),
                            "summary": str(item.get("summary") or ""),
                            "preferred_plan": item.get("preferred_plan") if isinstance(item.get("preferred_plan"), dict) else {},
                        }
                        for item in group_profiles
                        if isinstance(item, dict)
                    ],
                    "finished": bool(node.finished),
                    "terminated_reason": node.terminated_reason,
                    "trace_path": metadata.get("trace_path"),
                    "cycle_action": metadata.get("cycle_action"),
                    "metadata": metadata,
                }
            )
        candidates.sort(
            key=lambda item: (
                -float(item.get("metric_score", float("-inf"))),
                str(item.get("branch_id")),
                str(item.get("node_id")),
            )
        )
        if candidates:
            best_score = max(float(item.get("metric_score", float("-inf"))) for item in candidates)
            for item in candidates:
                item["is_metric_best"] = abs(float(item.get("metric_score", float("-inf"))) - best_score) <= 1e-12
        return candidates

    def _normalize_committed_plan(
        self,
        runtime: BranchRuntime,
        node: StateNode,
        cycle_index: int,
        current_plan: Dict[str, str],
        stage_results: Dict[str, Dict[str, Any]],
        phase3_apply_history_start: int = 0,
    ) -> Dict[str, str]:
        phase3 = stage_results.get("phase3") if isinstance(stage_results.get("phase3"), dict) else {}
        execution = phase3.get("execution") if isinstance(phase3.get("execution"), dict) else {}
        reflection = phase3.get("reflection") if isinstance(phase3.get("reflection"), dict) else {}
        neutral_search = reflection.get("neutral_search") if isinstance(reflection.get("neutral_search"), dict) else {}
        requested_plan = self._canonicalize_plan(execution.get("plan"), self._known_aoi_ids)
        current_plan = self._canonicalize_plan(current_plan, self._known_aoi_ids)
        if (
            str(neutral_search.get("decision") or "").strip().lower() == "undo"
            and runtime.toolkit is not None
            and hasattr(runtime.toolkit, "apply_history")
        ):
            raw_history = getattr(runtime.toolkit, "apply_history", None)
            if isinstance(raw_history, list) and raw_history:
                sliced_history = raw_history[int(phase3_apply_history_start):] if int(phase3_apply_history_start) > 0 else raw_history
                stage_history = [item for item in sliced_history if isinstance(item, dict) and item.get("success")]
                candidate_rows = [
                    item
                    for item in stage_history
                    if isinstance(item.get("step_changes"), list) and item.get("step_changes")
                ]
                if candidate_rows:
                    all_neutral = all(
                        not self._has_any_metric_delta(item.get("metric_delta") if isinstance(item.get("metric_delta"), dict) else {})
                        for item in candidate_rows
                    )
                    first_snapshot = self._canonicalize_plan(
                        candidate_rows[0].get("snapshot_plan_after"),
                        self._known_aoi_ids,
                    )
                    if all_neutral and first_snapshot and current_plan != first_snapshot:
                        runtime.toolkit.env.restore_plan_snapshot(node.plan, name=f"{runtime.branch_id}_cycle_{cycle_index}_retain_first_neutral")
                        runtime.toolkit.env.apply_plan(first_snapshot, export_pb=False)
                        current_plan = self._canonicalize_plan(
                            runtime.toolkit.env.snapshot_plan_mapping(),
                            self._known_aoi_ids,
                        )
                        execution["neutral_fallback_note"] = (
                            "All successful Stage-3 apply attempts were metric-neutral, so the committed temp snapshot "
                            "was restored to the first successful tested plan instead of the last neutral state."
                        )
        if execution:
            if requested_plan and requested_plan != current_plan:
                execution["requested_plan"] = dict(requested_plan)
                execution["plan_consistency_note"] = (
                    "execution.plan was overwritten with the cumulative applied temp-plan snapshot because "
                    "every committed modification must be materialized via apply_temp_plan_changes first."
                )
            execution["plan"] = dict(current_plan)
        return current_plan

    def _select_best_leaf(self, runtimes: List[BranchRuntime]) -> Tuple[Optional[BranchRuntime], Optional[StateNode]]:
        best_runtime: Optional[BranchRuntime] = None
        best_node: Optional[StateNode] = None
        best_score = float("-inf")
        for runtime in runtimes:
            node = self.nodes.get(runtime.current_node_id)
            if node is None:
                continue
            score = self._metric_score(node.metrics)
            if score > best_score:
                best_runtime = runtime
                best_node = node
                best_score = score
        return best_runtime, best_node

    def _materialize_best_plan(self, plan: Dict[str, str]) -> None:
        materialize_dir = self.output_dir / "best_plan_materialized"
        materialize_dir.mkdir(parents=True, exist_ok=True)
        toolkit = self.toolkit_factory(materialize_dir)
        toolkit.env.restore_plan_snapshot(plan, name="best_plan")
        metrics = (
            toolkit.env.evaluate_temp().get("metrics", {})
            if toolkit.env.temp_state
            else toolkit.env.evaluate_base().get("metrics", {})
        )
        render = toolkit.env.render_map_compact(
            map_version="temp" if toolkit.env.temp_state else "initial",
            aoi_ids=list(plan.keys()),
            output_name="best_plan_map.png",
            view_bbox_rel=None,
            highlight_changed=True,
        )
        save_json(
            materialize_dir / "best_plan.json",
            {"final_plan": plan, "metrics": metrics, "render": render},
        )

    def _maybe_prepare_initial_image(self, runtime: BranchRuntime, stage_name: str, cycle_index: int) -> Optional[Path]:
        if stage_name != "phase1":
            return None
        try:
            rendered = runtime.toolkit.env.render_map_compact(
                map_version="temp" if runtime.toolkit.env.temp_state else "initial",
                aoi_ids=None,
                output_name=f"cycle_{cycle_index:03d}_{stage_name}_initial.png",
                view_bbox_rel=None,
                highlight_changed=True,
            )
        except Exception:
            return None
        path_str = rendered.get("output_path")
        if isinstance(path_str, str):
            path = Path(path_str)
            if path.exists():
                return path
        return None

    def _compose_cycle_summary(
        self,
        branch_label: str,
        cycle_index: int,
        stage_results: Dict[str, Dict[str, Any]],
        action: str,
        metrics: Dict[str, Any],
    ) -> str:
        phase1 = stage_results.get("phase1") or {}
        phase2 = stage_results.get("phase2") or {}
        phase3 = stage_results.get("phase3") or {}
        global_focuses = phase1.get("global_diagnosis") or []
        focuses = phase1.get("priority_focuses") or []
        focus_labels = []
        focus_map = {}
        if isinstance(global_focuses, list):
            for item in global_focuses:
                normalized = self._normalize_focus_entry(item, self._known_aoi_ids)
                if normalized:
                    focus_map[normalized["focus_id"]] = normalized
        for item in focuses[:3]:
            focus_id = str(item).strip()
            if focus_id in focus_map:
                focus_labels.append(str(focus_map[focus_id].get("label") or focus_id))
            elif focus_id:
                focus_labels.append(focus_id)
        evidence = str(
            phase2.get("joint_evidence_summary")
            or phase2.get("evidence_summary")
            or ((phase2.get("focus_evidence") or [{}])[0].get("evidence_summary") if isinstance(phase2.get("focus_evidence"), list) and phase2.get("focus_evidence") else "")
            or ""
        ).strip()
        reflection = phase3.get("reflection") if isinstance(phase3.get("reflection"), dict) else {}
        reflection_text = str(reflection.get("summary") or reflection.get("assessment") or "").strip()
        metric_str = json.dumps(metrics, ensure_ascii=False)
        summary = (
            f"Cycle {cycle_index} on branch `{branch_label}`: focuses={focus_labels or ['none']}; "
            f"evidence={evidence[:200] or 'none'}; action={action}; "
            f"reflection={reflection_text[:160] or 'none'}; metrics={metric_str}."
        )
        return summary[:900]

    @staticmethod
    def _build_cycle_metadata(
        stage_results: Dict[str, Dict[str, Any]],
        trace_path: Path,
        full_trace_path: Path,
        action: str,
        rollback_target_id: Optional[str],
        branch_specs: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        phase1 = stage_results.get("phase1") if isinstance(stage_results.get("phase1"), dict) else {}
        phase2 = stage_results.get("phase2") if isinstance(stage_results.get("phase2"), dict) else {}
        phase3 = stage_results.get("phase3") if isinstance(stage_results.get("phase3"), dict) else {}
        conflict_overview = phase2.get("conflict_overview") if isinstance(phase2.get("conflict_overview"), dict) else {}
        conflict_assessment = phase3.get("conflict_assessment") if isinstance(phase3.get("conflict_assessment"), dict) else {}
        execution = phase3.get("execution") if isinstance(phase3.get("execution"), dict) else {}
        reflection = phase3.get("reflection") if isinstance(phase3.get("reflection"), dict) else {}
        finish_gate = RefinedPlanningController._evaluate_finish_gate(stage_results)
        return {
            "trace_path": str(trace_path),
            "full_trace_path": str(full_trace_path),
            "cycle_action": action,
            "rollback_target_id": rollback_target_id,
            "phase1_global_diagnosis": phase1.get("global_diagnosis") if isinstance(phase1.get("global_diagnosis"), list) else [],
            "phase1_priority_focuses": phase1.get("priority_focuses") if isinstance(phase1.get("priority_focuses"), list) else [],
            "phase1_done_focus_ids": phase1.get("done_focus_ids") if isinstance(phase1.get("done_focus_ids"), list) else [],
            "phase2_conflict_overview": conflict_overview,
            "phase3_conflict_assessment": conflict_assessment,
            "phase3_execution": execution,
            "phase3_reflection": reflection,
            "phase3_global_focus_statuses": phase3.get("global_focus_statuses") if isinstance(phase3.get("global_focus_statuses"), list) else [],
            "phase3_focus_decisions": phase3.get("focus_decisions") if isinstance(phase3.get("focus_decisions"), list) else [],
            "finish_gate": finish_gate,
            "branch_specs": branch_specs,
            "group_profiles": RefinedPlanningController._collect_group_profiles(conflict_overview, conflict_assessment),
        }

    @staticmethod
    def _collect_group_profiles(
        conflict_overview: Dict[str, Any],
        conflict_assessment: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        profiles: Dict[str, Dict[str, Any]] = {}
        raw_overview_groups = conflict_overview.get("group_labels") if isinstance(conflict_overview, dict) else []
        if isinstance(raw_overview_groups, list):
            for item in raw_overview_groups:
                if isinstance(item, str) and item.strip():
                    profiles[item.strip()] = {
                        "label": item.strip(),
                        "summary": item.strip(),
                        "source": "phase2_conflict_overview",
                    }
                elif isinstance(item, dict):
                    label = str(item.get("label") or "").strip()
                    if label:
                        profiles[label] = {
                            "label": label,
                            "summary": str(item.get("summary") or item.get("description") or label),
                            "source": "phase2_conflict_overview",
                            "preferred_plan": item.get("preferred_plan") if isinstance(item.get("preferred_plan"), dict) else {},
                        }
        raw_assessment_groups = conflict_assessment.get("group_labels") if isinstance(conflict_assessment, dict) else []
        if isinstance(raw_assessment_groups, list):
            for item in raw_assessment_groups:
                if not isinstance(item, dict):
                    continue
                label = str(item.get("label") or "").strip()
                if not label:
                    continue
                merged = profiles.get(label, {"label": label})
                merged.update(
                    {
                        "label": label,
                        "summary": str(item.get("summary") or merged.get("summary") or label),
                        "preferred_plan": item.get("preferred_plan") if isinstance(item.get("preferred_plan"), dict) else {},
                        "source": "phase3_conflict_assessment",
                    }
                )
                profiles[label] = merged
        return list(profiles.values())

    @staticmethod
    def _normalize_branch_specs(raw_specs: Any) -> List[Dict[str, Any]]:
        specs: List[Dict[str, Any]] = []
        if not isinstance(raw_specs, list):
            return specs
        for idx, item in enumerate(raw_specs[:2]):
            if not isinstance(item, dict):
                continue
            plan = item.get("plan")
            if not isinstance(plan, dict):
                plan = {}
            specs.append(
                {
                    "label": str(item.get("label") or f"group_branch_{idx + 1}"),
                    "plan": RefinedPlanningController._canonicalize_plan(plan),
                    "reason": str(item.get("reason") or ""),
                    "rollback_rule": str(item.get("rollback_rule") or "rollback_to_branch_anchor"),
                }
            )
        return specs

    @staticmethod
    def _metric_score(metrics: Dict[str, Any]) -> float:
        values = []
        for key in ("service", "ecology", "satisfaction"):
            value = metrics.get(key)
            if value is None:
                continue
            try:
                values.append(float(value))
            except Exception:
                continue
        if not values:
            return float("-inf")
        return sum(values) / len(values)

    def _reserve_cycle_index(self) -> int:
        with self._seq_lock:
            self._cycle_seq += 1
            return self._cycle_seq
