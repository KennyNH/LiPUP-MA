#!/usr/bin/env python3
"""Entry point for the refined predefined-pipeline planning agent."""

from __future__ import annotations

import argparse
from datetime import datetime
import json
import os
import pickle
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import yaml

_THIS = Path(__file__).resolve()
_REPO_ROOT = _THIS.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

try:
    from planning_agent_react_mm_next.graph_tools import aoi_landuse_map, build_graph_index
    from planning_agent_react_mm_next.landuse import MODIFIABLE_LANDUSE
    from planning_agent_react_mm_next.llm_client import LLMConfig, OpenAICompatibleClient
    from planning_agent_react_mm_next.map_tools import MapWorkspace
    from planning_agent_react_mm_next.utils import load_json, save_json
    from planning_agent_refined_core.controller import RefinedPlanningController
    from planning_agent_refined_core.evolution import EvolutionNotebook
    from planning_agent_refined_core.mock_client import SequentialMockClient
    from planning_agent_refined_core.toolkit import RefinedPlanningToolkit
    from planning_agent_pipeline_mm_refined.prompts import build_stage_prompt
except ImportError:
    from ..planning_agent_react_mm_next.graph_tools import aoi_landuse_map, build_graph_index  # type: ignore
    from ..planning_agent_react_mm_next.landuse import MODIFIABLE_LANDUSE  # type: ignore
    from ..planning_agent_react_mm_next.llm_client import LLMConfig, OpenAICompatibleClient  # type: ignore
    from ..planning_agent_react_mm_next.map_tools import MapWorkspace  # type: ignore
    from ..planning_agent_react_mm_next.utils import load_json, save_json  # type: ignore
    from ..planning_agent_refined_core.controller import RefinedPlanningController  # type: ignore
    from ..planning_agent_refined_core.evolution import EvolutionNotebook  # type: ignore
    from ..planning_agent_refined_core.mock_client import SequentialMockClient  # type: ignore
    from ..planning_agent_refined_core.toolkit import RefinedPlanningToolkit  # type: ignore
    from .prompts import build_stage_prompt


def load_config(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def read_task_prompt(task_prompt_path: Path) -> str:
    with open(task_prompt_path, "r", encoding="utf-8") as f:
        return f.read().strip()


def choose_mock_edit(base_landuse: Dict[str, str]) -> Tuple[str, str, str]:
    for aid, lu in sorted(base_landuse.items()):
        if lu not in MODIFIABLE_LANDUSE:
            continue
        for target in sorted(MODIFIABLE_LANDUSE):
            if target != lu:
                return aid, lu, target
    raise RuntimeError("No valid mock editable AOI found")


def build_mock_outputs(aoi_id: str, current_lu: str, target_lu: str) -> List[str]:
    def tool_block(tool_name: str, args: Dict[str, Any]) -> str:
        return f'<tool>{json.dumps({"tool": tool_name, "args": args}, ensure_ascii=False)}</tool>'

    def result_block(payload: Dict[str, Any]) -> str:
        return f"<result>{json.dumps(payload, ensure_ascii=False)}</result>"

    return [
        "<think>skill_experience_mode: follow\nI will follow the pipeline and collect one global focus signal first.</think>"
        + tool_block("query_top_degree_aois", {"top_n": 5, "modifiable_only": True}),
        f"<think>skill_experience_mode: follow\nThe first stage has enough information to hand over one focus.</think>"
        + result_block(
            {
                "phase": "phase1",
                "global_diagnosis": [
                    {
                        "focus_id": "focus_1",
                        "label": f"focus_{aoi_id}",
                        "type": "aoi",
                        "aoi_ids": [aoi_id],
                        "bbox_rel": [0, 0, 1, 1],
                        "priority": "high",
                        "reason": "Chosen as a legal editable AOI for deterministic verification.",
                        "depends_on": [],
                        "status_note": "One deterministic smoke-test focus.",
                    }
                ],
                "done_focus_ids": [],
                "priority_focuses": ["focus_1"],
                "subtasks": [f"Inspect AOI {aoi_id}"],
                "handoff_notes": "Use detailed AOI evidence next.",
            }
        ),
        "<think>skill_experience_mode: follow\nI will collect one local evidence bundle for the selected focus.</think>"
        + tool_block("query_aoi_details", {"aoi_ids": [aoi_id]}),
        "<think>skill_experience_mode: follow\nThe evidence stage is complete and can hand over a single safe action.</think>"
        + result_block(
            {
                "phase": "phase2",
                "focus_evidence": [
                    {
                        "focus_id": "focus_1",
                        "evidence_summary": f"AOI {aoi_id} can be updated from {current_lu} to {target_lu} for smoke-test verification.",
                        "supporting_groups": ["smoke_test_group"],
                        "risk_groups": [],
                        "focus_conflicts": [],
                        "unresolved_questions": [],
                        "aoi_evidence": [
                            {
                                "aoi_id": aoi_id,
                                "landuse": current_lu,
                                "role_summary": "A deterministic editable AOI for smoke-test verification.",
                                "resident_signals": "Smoke test uses a synthetic strong update signal.",
                                "spatial_observations": "The AOI is legally modifiable in the smoke-test environment.",
                                "related_observations": "No additional related-AOI evidence is needed for the smoke test.",
                                "conflict_signal": "none",
                                "notes": f"Safe to update from {current_lu} to {target_lu}.",
                            }
                        ],
                    }
                ],
                "joint_evidence_summary": "Evidence is sufficient for one deterministic edit.",
                "coverage_checklist": {
                    "required_focus_ids": ["focus_1"],
                    "covered_focus_ids": ["focus_1"],
                    "missing_focus_ids": [],
                    "per_focus_required_aoi_ids": {"focus_1": [aoi_id]},
                    "per_focus_covered_aoi_ids": {"focus_1": [aoi_id]},
                    "missing_aoi_ids_by_focus": {"focus_1": []},
                },
                "conflict_overview": {
                    "severity": "none",
                    "group_labels": [],
                    "observation_needed": False,
                },
            }
        ),
        "<think>skill_experience_mode: follow\nI will execute the only candidate edit and then finish the pipeline.</think>"
        + tool_block("apply_temp_plan_changes", {"plan": {aoi_id: target_lu}}),
        "<think>skill_experience_mode: follow\nThe edit has been applied, so the pipeline can terminate cleanly.</think>"
        + result_block(
            {
                "phase": "phase3",
                "global_focus_statuses": [
                    {
                        "focus_id": "focus_1",
                        "status": "done",
                        "reason": "The only smoke-test focus was fully executed.",
                    }
                ],
                "focus_decisions": [
                    {
                        "focus_id": "focus_1",
                        "decision": "done",
                        "rationale": "The only required focus is fully handled by one legal edit.",
                        "aoi_decisions": [
                            {
                                "aoi_id": aoi_id,
                                "decision": "modify",
                                "reason": f"Update {current_lu} to {target_lu} for deterministic end-to-end verification.",
                            }
                        ],
                    }
                ],
                "coverage_checklist": {
                    "required_global_focus_ids": ["focus_1"],
                    "covered_global_focus_ids": ["focus_1"],
                    "missing_global_focus_ids": [],
                    "required_priority_focus_ids": ["focus_1"],
                    "covered_priority_focus_ids": ["focus_1"],
                    "missing_priority_focus_ids": [],
                    "required_aoi_ids_by_focus": {"focus_1": [aoi_id]},
                    "covered_aoi_ids_by_focus": {"focus_1": [aoi_id]},
                    "missing_aoi_ids_by_focus": {"focus_1": []},
                },
                "conflict_assessment": {
                    "severity": "none",
                    "observation_needed": False,
                    "group_labels": [],
                    "notes": "No branching is needed in the smoke test.",
                },
                "execution": {
                    "action": "finish",
                    "plan": {aoi_id: target_lu},
                    "branch_plans": [],
                    "rollback_target": "",
                    "rationale": "Finish after one successful deterministic edit.",
                },
                "reflection": {
                    "neutral_search": {"decision": "n/a", "reason": "Single deterministic smoke edit; no quality claim."},
                    "assessment": "done",
                    "summary": "The pipeline completed one legal edit end-to-end.",
                    "completed_focus_ids": ["focus_1"],
                    "next_step": "finish",
                },
            }
        ),
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run refined predefined-pipeline planning agent")
    parser.add_argument("--config", type=str, default="planning_agent_pipeline_mm_refined/config.yaml")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--mock-smoke", action="store_true", help="Run a deterministic local smoke test without external LLM calls.")
    parser.add_argument("--disable-conflict-judgement", action="store_true", help="Disable explicit conflict judgement, rollback, and branching.")
    parser.add_argument("--disable-self-evolving", action="store_true", help="Disable experience/skill/tool self-evolving.")
    parser.add_argument(
        "--init-self-evolving-from",
        type=str,
        default="",
        help="Optional path to another agent run folder (or its self_evolving subfolder) whose stage memory and evolved tools should be bootstrapped into this run.",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent.parent
    cfg = load_config((repo_root / args.config).resolve())
    paths = cfg.get("paths", {})
    llm_cfg = cfg.get("llm", {})
    run_cfg = cfg.get("run", {})

    pb_path = (repo_root / paths.get("pb_map", "data/example/berlin_0.pb")).resolve()
    pickle_path = (repo_root / paths.get("pickle_map", "data/example/berlin_0.pickle")).resolve()
    graph_path = (repo_root / paths.get("graph_json", "outputs/graph.json")).resolve()
    resident_profiles_path = (repo_root / paths.get("resident_profiles", "outputs/preprocess/berlin_profiles.json")).resolve()
    resident_preferences_path = (repo_root / str(paths.get("resident_preferences", ""))).resolve() if paths.get("resident_preferences") else None
    task_prompt_path = (repo_root / paths.get("task_prompt", "planning_agent_pipeline_mm_refined/task_prompt.txt")).resolve()
    out_root = (repo_root / paths.get("outputs", "outputs/planning")).resolve()
    run_dir = out_root / datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str((run_dir / ".mplconfig").resolve()))
    graph_index = build_graph_index(load_json(graph_path))
    base_landuse = aoi_landuse_map(graph_index)
    resident_preferences = {}
    if resident_preferences_path is not None:
        pref_obj = load_json(resident_preferences_path)
        pref_map = pref_obj.get("resident_preferences") if isinstance(pref_obj, dict) else None
        if not isinstance(pref_map, dict):
            raise ValueError(f"resident preference file invalid: {resident_preferences_path}")
        resident_preferences = {
            str(k): [str(x) for x in v]
            for k, v in pref_map.items()
            if isinstance(v, list)
        }

    provider = str(llm_cfg.get("provider", "openai")).lower()
    hf_home = str(llm_cfg.get("hf_home", ""))
    if provider == "vllm" and hf_home:
        os.environ["HF_HOME"] = hf_home
    api_key = llm_cfg.get("api_key") or os.getenv(str(llm_cfg.get("api_key_env", "OPENAI_API_KEY")))
    if provider != "vllm" and not api_key and not args.mock_smoke:
        raise ValueError("Missing API key. Set llm.api_key or environment variable in llm.api_key_env")
    if provider == "vllm" and not api_key:
        api_key = "EMPTY"

    llm_config = LLMConfig(
        provider=provider,
        api_key=str(api_key or ""),
        model=str(llm_cfg.get("model", "gpt-4o-mini")),
        base_url=llm_cfg.get("base_url"),
        temperature=float(llm_cfg.get("temperature", 0.2)),
        max_tokens=int(llm_cfg.get("max_tokens", 2600)),
        timeout=int(llm_cfg.get("timeout", 120)),
        dump_api_io=bool(llm_cfg.get("dump_api_io", False)),
    )
    OpenAICompatibleClient.reset_global_usage()
    stage_max_steps = {
        "phase1": int(run_cfg.get("phase1_max_steps", 8)),
        "phase2": int(run_cfg.get("phase2_max_steps", 16)),
        "phase3": int(run_cfg.get("phase3_max_steps", 16)),
    }
    bootstrap_self_evolving_from = str(args.init_self_evolving_from or run_cfg.get("init_self_evolving_from") or "").strip()
    bootstrap_path = Path(bootstrap_self_evolving_from).resolve() if bootstrap_self_evolving_from else None
    enable_conflict_judgement = bool(run_cfg.get("enable_conflict_judgement", True)) and not args.disable_conflict_judgement
    enable_self_evolving = bool(run_cfg.get("enable_self_evolving", True)) and not args.disable_self_evolving

    if args.mock_smoke:
        mock_aoi_id, mock_from, mock_to = choose_mock_edit(base_landuse)
        client_factory = lambda: SequentialMockClient(build_mock_outputs(mock_aoi_id, mock_from, mock_to))
        env_client_factory = lambda: None
        evolution_client = None
    else:
        client_factory = lambda: OpenAICompatibleClient(llm_config)
        env_client_factory = (
            (lambda: OpenAICompatibleClient(llm_config))
            if bool(run_cfg.get("enable_environment_llm", True))
            else (lambda: None)
        )
        evolution_client = OpenAICompatibleClient(llm_config) if enable_self_evolving else None

    def toolkit_factory(branch_dir: Path):
        branch_dir.mkdir(parents=True, exist_ok=True)
        map_workspace = MapWorkspace(repo_root=repo_root, output_dir=branch_dir)
        map_workspace.load_map(pb_path)
        return RefinedPlanningToolkit(
            repo_root=repo_root,
            output_dir=branch_dir,
            graph_index=graph_index,
            map_workspace=map_workspace,
            pb_path=pb_path,
            geojson_cache_path=branch_dir / "map_snapshot.geojson",
            init_pickle_path=pickle_path,
            llm_client=env_client_factory(),
            resident_profiles_path=resident_profiles_path,
            resident_preferences=resident_preferences,
            importance_level3_ratio=float(run_cfg.get("importance_level3_ratio", 0.15)),
            importance_level2_ratio=float(run_cfg.get("importance_level2_ratio", 0.25)),
            service_radius_m=float(run_cfg.get("service_radius_m", 500.0)),
            id_service_radius_m=float(run_cfg.get("id_service_radius_m", 1000.0)),
            ecology_radius_m=float(run_cfg.get("ecology_radius_m", 300.0)),
            skip_simulation_assessment=bool(run_cfg.get("skip_simulation_assessment", False)),
            shared_tool_root=run_dir,
            bootstrap_from=bootstrap_path,
        )

    controller = RefinedPlanningController(
        repo_root=repo_root,
        output_dir=run_dir,
        task_prompt=read_task_prompt(task_prompt_path),
        client_factory=client_factory,
        toolkit_factory=toolkit_factory,
        prompt_builder=(
            lambda stage_name, context: build_stage_prompt(
                stage_name,
                {
                    **context,
                },
            )
        ),
        evolution=EvolutionNotebook(run_dir, client=evolution_client, enabled=enable_self_evolving, bootstrap_from=bootstrap_path),
        variant_name="pipeline_refined",
        max_rounds=1 if args.mock_smoke else int(run_cfg.get("max_rounds", 8)),
        stage_max_steps=stage_max_steps,
        max_branch_leaves=int(run_cfg.get("max_branch_leaves", 8)),
        concurrent_branch_workers=int(run_cfg.get("concurrent_branch_workers", 2)),
        enable_conflict_judgement=enable_conflict_judgement,
        enable_self_evolving=enable_self_evolving,
        verbose=not args.quiet,
    )
    result = controller.run()
    usage = OpenAICompatibleClient.get_global_usage()
    save_json(run_dir / "llm_usage.json", usage)
    summary_path = run_dir / "summary.json"
    if summary_path.exists():
        summary_payload = load_json(summary_path)
        summary_payload["llm_usage"] = usage
        save_json(summary_path, summary_payload)
    print(json.dumps({"run_dir": str(run_dir), "llm_usage": usage, **result}, ensure_ascii=False))


if __name__ == "__main__":
    main()
