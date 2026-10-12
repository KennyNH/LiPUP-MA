"""Refined toolkit/environment with explicit snapshot restore and tool evolution hooks."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import re
import shutil
from threading import RLock
from typing import Any, Dict, List, Optional

try:
    from planning_agent_react_mm_next.graph_tools import GraphIndex, aoi_landuse_map
    from planning_agent_react_mm_next.llm_client import OpenAICompatibleClient
    from planning_agent_react_mm_next.map_tools import MapWorkspace
    from planning_agent_react_mm_next.planning_env import PlanningEnvironment
    from planning_agent_react_mm_next.toolkit import PlanningToolkit, ToolSpec
    from planning_agent_react_mm_next.utils import save_json
except ImportError:
    from ..planning_agent_react_mm_next.graph_tools import GraphIndex, aoi_landuse_map  # type: ignore
    from ..planning_agent_react_mm_next.llm_client import OpenAICompatibleClient  # type: ignore
    from ..planning_agent_react_mm_next.map_tools import MapWorkspace  # type: ignore
    from ..planning_agent_react_mm_next.planning_env import PlanningEnvironment  # type: ignore
    from ..planning_agent_react_mm_next.toolkit import PlanningToolkit, ToolSpec  # type: ignore
    from ..planning_agent_react_mm_next.utils import save_json  # type: ignore


class RefinedPlanningEnvironment(PlanningEnvironment):
    def restore_plan_snapshot(self, plan: Dict[str, str], name: str = "temp_plan") -> Dict[str, Any]:
        restored = self.init_temp_copy(name=name)
        if not plan:
            return {
                "success": True,
                "restore": restored,
                "current_plan": {},
                "metrics": self.evaluate_temp().get("metrics", {}),
            }
        applied = self.apply_plan(plan, export_pb=False)
        if not applied.get("success"):
            return {"success": False, "restore": restored, "apply": applied}
        return {
            "success": True,
            "restore": restored,
            "apply": applied,
            "current_plan": self.snapshot_plan_mapping(),
            "metrics": self.evaluate_temp().get("metrics", {}),
        }

    def snapshot_plan_mapping(self) -> Dict[str, str]:
        if not self.temp_state:
            return {}
        rows = self.get_all_modifications().get("changed_items", [])
        out: Dict[str, str] = {}
        for item in rows:
            aid = item.get("aoi_id")
            target = item.get("to")
            if aid and target:
                out[str(aid)] = str(target)
        return out


class RefinedPlanningToolkit(PlanningToolkit):
    _shared_tool_lock = RLock()

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
        shared_tool_root: Optional[Path] = None,
        bootstrap_from: Optional[Path] = None,
    ):
        self.shared_tool_root = (shared_tool_root or output_dir).resolve()
        self.self_evolving_dir = self.shared_tool_root / "self_evolving"
        self.tool_evolution_dir = self.self_evolving_dir / "tool_evolution"
        self.dynamic_tool_dir = self.self_evolving_dir / "dynamic_tools"
        self.self_evolving_dir.mkdir(parents=True, exist_ok=True)
        self.tool_evolution_dir.mkdir(parents=True, exist_ok=True)
        self.dynamic_tool_dir.mkdir(parents=True, exist_ok=True)
        self._bootstrap_self_evolving_assets(bootstrap_from)
        self.tool_events: List[Dict[str, Any]] = self._load_existing_tool_events()
        self.apply_history: List[Dict[str, Any]] = []
        self._apply_seq = 0
        self._frozen_tools: Optional[Dict[str, ToolSpec]] = None
        super().__init__(
            repo_root=repo_root,
            output_dir=output_dir,
            graph_index=graph_index,
            map_workspace=map_workspace,
            pb_path=pb_path,
            geojson_cache_path=geojson_cache_path,
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
        self.env = RefinedPlanningEnvironment(
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

    def _normalize_tool_args(self, tool_name: str, args: Dict[str, Any]) -> Dict[str, Any]:
        normalized = super()._normalize_tool_args(tool_name, args)
        if tool_name == "tool_evolution":
            if "tool_name" not in normalized and isinstance(normalized.get("name"), str):
                normalized["tool_name"] = normalized.get("name")
            if "input_schema" not in normalized:
                for alias in ("schema", "parameters"):
                    candidate = normalized.get(alias)
                    if isinstance(candidate, dict):
                        normalized["input_schema"] = candidate
                        break
            if "python_body" not in normalized:
                for alias in ("python_code", "code", "implementation", "body"):
                    candidate = normalized.get(alias)
                    if isinstance(candidate, str) and candidate.strip():
                        normalized["python_body"] = candidate
                        break
            if "description" not in normalized and isinstance(normalized.get("summary"), str):
                normalized["description"] = normalized.get("summary")
        return normalized

    def _build_static_tools(self) -> Dict[str, ToolSpec]:
        tools = super()._build_tools()
        tools["tool_evolution"] = ToolSpec(
            "tool_evolution",
            "Purpose: write Python code to create a new reusable helper tool for this run, or update an existing evolved tool from this run/bootstrap state. "
            "Provide the tool name, Python body for `run(toolkit, args)`, and when creating a new tool also provide a concise description plus exact JSON input schema. "
            "Evolved tools are saved in the current run folder and become available to later stages and branches in this run.",
            {
                "type": "object",
                "properties": {
                    "stage": {"type": "string"},
                    "tool_name": {"type": "string"},
                    "description": {"type": "string"},
                    "input_schema": {"type": "object"},
                    "python_body": {"type": "string"},
                    "gap": {"type": "string"},
                    "expected_value": {"type": "string"},
                },
                "required": ["stage", "tool_name", "python_body"],
            },
            lambda args: self._evolve_dynamic_tool(args),
        )
        return tools

    def _build_tools(self) -> Dict[str, ToolSpec]:
        tools = self._build_static_tools()
        tools.update(self._load_dynamic_tools())
        return tools

    def _record_tool_event(self, kind: str, args: Dict[str, Any]) -> Dict[str, Any]:
        proposal_id = f"{kind}_{len(self.tool_events) + 1:03d}"
        item = {
            "proposal_id": proposal_id,
            "kind": kind,
            "stage": str(args.get("stage") or ""),
            "tool_name": str(args.get("tool_name") or ""),
            "title": str(args.get("title") or args.get("tool_name") or proposal_id),
            "gap": str(args.get("gap") or ""),
            "description": str(args.get("description") or ""),
            "expected_value": str(args.get("expected_value") or ""),
            "input_schema": args.get("input_schema") if isinstance(args.get("input_schema"), dict) else None,
            "note": "Recorded for self-evolving review.",
        }
        self.tool_events.append(item)
        save_json(self.tool_evolution_dir / f"{proposal_id}.json", item)
        return {"success": True, **item}

    @staticmethod
    def _resolve_asset_dir(root: Optional[Path], leaf_name: str) -> Optional[Path]:
        if root is None:
            return None
        candidate = root.resolve()
        if candidate.name == leaf_name and candidate.exists():
            return candidate
        nested = candidate / "self_evolving" / leaf_name
        if nested.exists():
            return nested
        direct = candidate / leaf_name
        if direct.exists():
            return direct
        return None

    @staticmethod
    def _copy_tree_contents(src: Path, dst: Path) -> None:
        if not src.exists():
            return
        dst.mkdir(parents=True, exist_ok=True)
        for item in sorted(src.iterdir()):
            target = dst / item.name
            if item.is_dir():
                RefinedPlanningToolkit._copy_tree_contents(item, target)
            elif item.is_file():
                shutil.copy2(item, target)

    def _bootstrap_self_evolving_assets(self, bootstrap_from: Optional[Path]) -> None:
        source_root = Path(bootstrap_from).resolve() if bootstrap_from else None
        if source_root is None:
            return
        if not source_root.exists():
            raise FileNotFoundError(f"Bootstrap self-evolving path does not exist: {source_root}")
        src_dynamic = self._resolve_asset_dir(source_root, "dynamic_tools")
        src_events = self._resolve_asset_dir(source_root, "tool_evolution")
        with self._shared_tool_lock:
            if src_dynamic is not None:
                self._copy_tree_contents(src_dynamic, self.dynamic_tool_dir)
            if src_events is not None:
                self._copy_tree_contents(src_events, self.tool_evolution_dir)

    def _load_existing_tool_events(self) -> List[Dict[str, Any]]:
        events: List[Dict[str, Any]] = []
        for event_path in sorted(self.tool_evolution_dir.glob("*.json")):
            try:
                payload = json.loads(event_path.read_text(encoding="utf-8"))
            except Exception:
                continue
            if isinstance(payload, dict):
                events.append(payload)
        return events

    @staticmethod
    def _validate_tool_name(tool_name: str) -> Optional[str]:
        name = str(tool_name or "").strip()
        if not name:
            return "tool_name is empty"
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name):
            return "tool_name must be a valid Python-style identifier"
        return None

    @staticmethod
    def _wrap_python_body(python_body: str) -> str:
        code = str(python_body or "").strip()
        if not code:
            code = 'return {"error": "empty_python_body"}'
        if "def run(" in code:
            return "from __future__ import annotations\n\n" + code.strip() + "\n"
        indented = "\n".join(f"    {line}" if line.strip() else "" for line in code.splitlines())
        return "from __future__ import annotations\n\n\ndef run(toolkit, args):\n" + indented + "\n"

    def _load_tool_manifest(self, tool_name: str) -> Optional[Dict[str, Any]]:
        manifest_path = self.dynamic_tool_dir / f"{tool_name}.json"
        if not manifest_path.exists():
            return None
        try:
            with open(manifest_path, "r", encoding="utf-8") as f:
                payload = json.load(f)
        except Exception:
            return None
        return payload if isinstance(payload, dict) else None

    def _build_dynamic_tool_spec(self, manifest: Dict[str, Any]) -> Optional[ToolSpec]:
        tool_name = str(manifest.get("tool_name") or "").strip()
        description = str(manifest.get("description") or "").strip()
        input_schema = manifest.get("input_schema")
        module_path = Path(str(manifest.get("module_path") or "")).resolve()
        if not tool_name or not description or not isinstance(input_schema, dict) or not module_path.exists():
            return None
        module_name = f"dynamic_tool_{tool_name}_{abs(hash(str(module_path)))}"
        try:
            spec = importlib.util.spec_from_file_location(module_name, str(module_path))
            if spec is None or spec.loader is None:
                return None
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            run_fn = getattr(module, "run", None)
        except Exception as exc:
            return ToolSpec(
                name=tool_name,
                description=description,
                input_schema=input_schema,
                fn=lambda args, err=str(exc), name=tool_name: {"error": "dynamic_tool_load_error", "tool": name, "message": err},
            )
        if not callable(run_fn):
            return ToolSpec(
                name=tool_name,
                description=description,
                input_schema=input_schema,
                fn=lambda args, name=tool_name: {"error": "dynamic_tool_missing_run", "tool": name},
            )
        return ToolSpec(
            name=tool_name,
            description=description,
            input_schema=input_schema,
            fn=lambda args, fn=run_fn: fn(self, args),
        )

    def _load_dynamic_tools(self) -> Dict[str, ToolSpec]:
        dynamic_tools: Dict[str, ToolSpec] = {}
        if not self.dynamic_tool_dir.exists():
            return dynamic_tools
        for manifest_path in sorted(self.dynamic_tool_dir.glob("*.json")):
            try:
                with open(manifest_path, "r", encoding="utf-8") as f:
                    manifest = json.load(f)
            except Exception:
                continue
            if not isinstance(manifest, dict):
                continue
            spec = self._build_dynamic_tool_spec(manifest)
            if spec is not None:
                dynamic_tools[spec.name] = spec
        return dynamic_tools

    def refresh_dynamic_tools(self) -> None:
        if self._frozen_tools is not None:
            return
        with self._shared_tool_lock:
            self.tools = self._build_tools()

    def freeze_tool_registry(self) -> None:
        with self._shared_tool_lock:
            self.tools = self._build_tools()
            self._frozen_tools = dict(self.tools)

    def unfreeze_tool_registry(self) -> None:
        with self._shared_tool_lock:
            self._frozen_tools = None
            self.tools = self._build_tools()

    def dynamic_tool_names(self) -> List[str]:
        if self._frozen_tools is not None:
            static_names = set(self._build_static_tools().keys())
            return sorted(name for name in self._frozen_tools.keys() if name not in static_names)
        self.refresh_dynamic_tools()
        static_names = set(self._build_static_tools().keys())
        return sorted(name for name in self.tools.keys() if name not in static_names)

    def _evolve_dynamic_tool(self, args: Dict[str, Any]) -> Dict[str, Any]:
        tool_name = str(args.get("tool_name") or "").strip()
        name_error = self._validate_tool_name(tool_name)
        if name_error:
            return {"error": "invalid_tool_name", "message": name_error}
        existing_manifest = self._load_tool_manifest(tool_name)
        existing_spec = self.tools.get(tool_name)
        is_existing_dynamic = existing_manifest is not None
        is_builtin_only = existing_spec is not None and existing_manifest is None
        if is_builtin_only:
            return {
                "error": "builtin_tool_update_not_allowed",
                "tool_name": tool_name,
                "message": "tool_evolution may create new helper tools or update previously evolved tools, but it does not override built-in tools.",
            }
        operation = "update" if is_existing_dynamic else "create"
        description = str(
            args.get("description")
            or (existing_manifest or {}).get("description")
            or (existing_spec.description if existing_spec is not None else "")
            or ""
        ).strip()
        if not description:
            return {"error": "missing_tool_description", "tool_name": tool_name, "message": "description is required when creating a new evolved tool"}
        input_schema = args.get("input_schema")
        if input_schema is None:
            input_schema = (existing_manifest or {}).get("input_schema")
        if input_schema is None and existing_spec is not None:
            input_schema = existing_spec.input_schema
        if not isinstance(input_schema, dict):
            return {"error": "invalid_input_schema", "tool_name": tool_name, "message": "input_schema must be a JSON object"}
        python_body = str(args.get("python_body") or "").strip()
        if not python_body:
            return {"error": "missing_python_body", "tool_name": tool_name}

        module_code = self._wrap_python_body(python_body)
        module_path = self.dynamic_tool_dir / f"{tool_name}.py"
        manifest_path = self.dynamic_tool_dir / f"{tool_name}.json"
        try:
            compile(module_code, str(module_path), "exec")
        except Exception as exc:
            return {"error": "dynamic_tool_compile_error", "tool_name": tool_name, "message": str(exc)}

        payload = {
            "tool_name": tool_name,
            "description": description,
            "input_schema": input_schema,
            "module_path": str(module_path),
            "stage": str(args.get("stage") or ""),
            "gap": str(args.get("gap") or ""),
            "expected_value": str(args.get("expected_value") or ""),
            "kind": "tool_evolution",
            "operation": operation,
        }
        with self._shared_tool_lock:
            module_path.write_text(module_code, encoding="utf-8")
            save_json(manifest_path, payload)
            self.tools = self._build_tools()
        event = {
            "stage": str(args.get("stage") or ""),
            "tool_name": tool_name,
            "title": tool_name,
            "gap": str(args.get("gap") or ""),
            "description": description,
            "expected_value": str(args.get("expected_value") or ""),
            "input_schema": input_schema,
            "operation": operation,
            "module_path": str(module_path),
        }
        recorded = self._record_tool_event("tool_evolution", event)
        return {
            "success": True,
            "tool_name": tool_name,
            "kind": "tool_evolution",
            "operation": operation,
            "description": description,
            "input_schema": input_schema,
            "module_path": str(module_path),
            "tool_schema": {"name": tool_name, "description": description, "parameters": input_schema},
            "message": f"Evolved tool {operation}d and registered for the current run.",
            "event": recorded,
        }

    def consume_tool_events(self, start_index: int) -> List[Dict[str, Any]]:
        return list(self.tool_events[start_index:])

    def consume_apply_history(self, start_index: int) -> List[Dict[str, Any]]:
        return list(self.apply_history[start_index:])

    @staticmethod
    def _metric_delta(before: Dict[str, Any], after: Dict[str, Any]) -> Dict[str, float]:
        out: Dict[str, float] = {}
        keys = set(before.keys()) | set(after.keys())
        for key in keys:
            try:
                before_value = float(before.get(key, 0.0))
                after_value = float(after.get(key, 0.0))
            except Exception:
                continue
            out[str(key)] = round(after_value - before_value, 10)
        return out

    @staticmethod
    def _is_neutral_metric_delta(delta: Dict[str, float], tol: float = 1e-9) -> bool:
        if not delta:
            return True
        return all(abs(float(value)) <= tol for value in delta.values())

    def execute(self, tool_name: str, args: Dict[str, Any]) -> Dict[str, Any]:
        if self._frozen_tools is None:
            self.refresh_dynamic_tools()
        else:
            self.tools = dict(self._frozen_tools)
        result = super().execute(tool_name, args)
        if tool_name == "apply_temp_plan_changes" and isinstance(result, dict):
            self._apply_seq += 1
            before_metrics = result.get("before_metrics") if isinstance(result.get("before_metrics"), dict) else {}
            after_metrics = result.get("after_metrics") if isinstance(result.get("after_metrics"), dict) else {}
            step_changes = result.get("step_changes") if isinstance(result.get("step_changes"), list) else []
            metric_delta = self._metric_delta(before_metrics, after_metrics)
            snapshot_after = {}
            try:
                snapshot_after = self.env.snapshot_plan_mapping()
            except Exception:
                snapshot_after = {}
            self.apply_history.append(
                {
                    "seq": self._apply_seq,
                    "tool_name": tool_name,
                    "requested_plan": dict(args or {}),
                    "step_changes": step_changes,
                    "changed_aoi_ids": [
                        str(item.get("aoi_id"))
                        for item in step_changes
                        if isinstance(item, dict) and item.get("aoi_id") is not None
                    ],
                    "before_metrics": before_metrics,
                    "after_metrics": after_metrics,
                    "metric_delta": metric_delta,
                    "is_neutral": self._is_neutral_metric_delta(metric_delta),
                    "snapshot_plan_after": snapshot_after,
                    "success": bool(result.get("success")),
                }
            )
        return result

    def tool_schemas(self, allowed_names: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        if self._frozen_tools is None:
            self.refresh_dynamic_tools()
            active_tools = self.tools
        else:
            active_tools = self._frozen_tools
        if not allowed_names:
            return [
                {"name": t.name, "description": t.description, "parameters": t.input_schema}
                for t in active_tools.values()
            ]
        allowed = set(allowed_names)
        return [
            {"name": t.name, "description": t.description, "parameters": t.input_schema}
            for t in active_tools.values()
            if t.name in allowed
        ]

    def tool_schemas_text(self, allowed_names: Optional[List[str]] = None) -> str:
        return json.dumps(self.tool_schemas(allowed_names), ensure_ascii=False, indent=2)

    def base_context(self) -> Dict[str, Any]:
        return {
            "total_aoi_count": len(self.env.base_landuse),
            "modifiable_aoi_count": self.env.modifiable_base_count,
            "max_allowed_aoi_change_count": self.env.max_changed_allowed,
            "base_metrics": self.env.evaluate_base().get("metrics", {}),
            "initial_landuse_distribution": self._landuse_distribution(),
            "simulation_assessment": self.env.evaluate_simulation_metrics().get("metrics", {}),
        }

    def _landuse_distribution(self) -> Dict[str, int]:
        dist: Dict[str, int] = {}
        for lu in aoi_landuse_map(self.graph_index).values():
            dist[str(lu)] = dist.get(str(lu), 0) + 1
        return dist
