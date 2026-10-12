import asyncio
import copy
import ast
import json
import os
import random
import shutil
import sys
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple

import numpy as np
import yaml
from networkx.readwrite import json_graph

from agentsociety.cityagent import default
from agentsociety.configs import (
    AgentsConfig,
    Config,
    EnvConfig,
    ExpConfig,
    LLMConfig,
    MapConfig,
)
from agentsociety.configs.agent import AgentConfig
from agentsociety.configs.exp import WorkflowStepConfig, WorkflowType
from agentsociety.environment import EnvironmentConfig
from agentsociety.simulation import AgentSociety
from agentsociety.storage import DatabaseConfig

from map_utils import dict2pb_path, modify_aois
from planning_agent_react_mm_next.graph_tools import build_graph_index
from planning_agent_react_mm_next.llm_client import LLMConfig as EvalLLMConfig
from planning_agent_react_mm_next.llm_client import OpenAICompatibleClient as EvalLLMClient
from planning_agent_react_mm_next.metrics_tools import (
    WalkingRoadDistance,
    evaluate_accessibility_proxy,
    evaluate_simulation_assessment,
)
from planning_agent_react_mm_next.utils import load_json, save_json
from resident_preprocess import load_and_validate_preprocess


def load_cfg(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _deep_get(d: Dict[str, Any], keys: list[str], default: Any = None) -> Any:
    cur: Any = d
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


def _safe_eval_numeric_expr(v: Any, default: int) -> int:
    """Parse int/float or a simple arithmetic string like '5*60' safely."""
    if isinstance(v, (int, float)):
        return int(v)
    s = str(v or "").strip()
    if not s:
        return int(default)
    try:
        return int(float(s))
    except Exception:
        pass
    try:
        node = ast.parse(s, mode="eval")
    except Exception:
        return int(default)

    def _eval(n):
        if isinstance(n, ast.Expression):
            return _eval(n.body)
        if isinstance(n, ast.Constant) and isinstance(n.value, (int, float)):
            return float(n.value)
        if isinstance(n, ast.Num):
            return float(n.n)
        if isinstance(n, ast.UnaryOp) and isinstance(n.op, (ast.UAdd, ast.USub)):
            val = _eval(n.operand)
            return val if isinstance(n.op, ast.UAdd) else -val
        if isinstance(n, ast.BinOp) and isinstance(n.op, (ast.Add, ast.Sub, ast.Mult, ast.Div, ast.FloorDiv)):
            l = _eval(n.left)
            r = _eval(n.right)
            if isinstance(n.op, ast.Add):
                return l + r
            if isinstance(n.op, ast.Sub):
                return l - r
            if isinstance(n.op, ast.Mult):
                return l * r
            if isinstance(n.op, ast.Div):
                return l / r
            if isinstance(n.op, ast.FloorDiv):
                return l // r
        raise ValueError("unsupported_expr")

    try:
        return int(_eval(node))
    except Exception:
        return int(default)


def add_map_config(map_path: str, sim_cfg: Dict[str, Any]):
    seed = int(_deep_get(sim_cfg, ["runtime", "seed"], 42))
    random.seed(seed)
    np.random.seed(seed)

    llm_cfg = sim_cfg.get("llm_config") or {}
    agents_cfg = sim_cfg.get("agents_config") or {}
    exp_cfg = sim_cfg.get("exp_config") or {}
    profile_path = str(_deep_get(sim_cfg, ["paths", "resident_profiles_path"], "") or "").strip()
    if not profile_path:
        raise RuntimeError("paths.resident_profiles_path is missing; load resident_preprocess_path before simulation.")

    llm_config = LLMConfig(
        provider=str(llm_cfg.get("provider", "deepseek")),
        api_key=_resolve_api_key(llm_cfg),
        model=str(llm_cfg.get("model", "deepseek-chat")),
        base_url=llm_cfg.get("base_url"),
        concurrency=int(llm_cfg.get("concurrency", 500)),
        timeout=int(llm_cfg.get("timeout", 50)),
    )

    env_config = EnvConfig(
        db=DatabaseConfig(
            enabled=True,
            db_type="sqlite",
        ),
        home_dir="./outputs/simulator",
    )

    agents_config = AgentsConfig(
        citizens=[
            AgentConfig(
                agent_class="resident",
                number=int(agents_cfg.get("number", 20)),
                memory_from_file=profile_path,
            )
        ],
    )

    exp_config = ExpConfig(
        name="test_1",
        workflow=[
            WorkflowStepConfig(
                type=WorkflowType.RUN,
                days=float(exp_cfg.get("days", 1)),
                ticks_per_step=_safe_eval_numeric_expr(exp_cfg.get("ticks_per_step", 5 * 60), 5 * 60),
            )
        ],
        environment=EnvironmentConfig(
            start_tick=_safe_eval_numeric_expr(exp_cfg.get("start_tick", 6 * 60 * 60), 6 * 60 * 60),
        ),
    )

    map_config = MapConfig(file_path=map_path)
    config = Config(
        llm=[llm_config],
        env=env_config,
        map=map_config,
        agents=agents_config,
        exp=exp_config,
    )

    return default(config)


async def main(config):
    society = AgentSociety.create(config)
    stream_memory = None

    try:
        await society.init()
        logs, memory_graph = await society.run()
        stream_memory = await society.gather("stream_memory", keep_id=True)
    finally:
        await society.close()

    input_tokens = 0
    output_tokens = 0
    # `logs` is a Logs object (not iterable). Aggregate directly from logs.llm_log.
    for llm_log in (logs.llm_log or []):
        input_tokens += int(llm_log.get("input_tokens", 0))
        output_tokens += int(llm_log.get("output_tokens", 0))

    token_usage = {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": input_tokens + output_tokens,
    }

    return memory_graph, token_usage, stream_memory


def _ensure_iter_suffix(path: Path, iter_idx: int) -> Path:
    stem = path.stem
    suffix = path.suffix
    if stem.endswith(f"_{iter_idx}"):
        return path
    if "_" in stem and stem.rsplit("_", 1)[-1].isdigit():
        base = stem.rsplit("_", 1)[0]
    else:
        base = stem
    return path.with_name(f"{base}_{iter_idx}{suffix}")


def prepare_run_workspace(origin_pb: Path, origin_pickle: Path, out_root: Path) -> Dict[str, Path]:
    run_dir = out_root / datetime.now().strftime("%Y%m%d_%H%M%S")
    maps_dir = run_dir / "maps"
    graphs_dir = run_dir / "graphs"
    planning_dir = run_dir / "planning"
    metrics_dir = run_dir / "metrics"

    for d in [run_dir, maps_dir, graphs_dir, planning_dir, metrics_dir]:
        d.mkdir(parents=True, exist_ok=True)

    init_pb = _ensure_iter_suffix(maps_dir / origin_pb.name, 0)
    init_pickle = _ensure_iter_suffix(maps_dir / origin_pickle.name, 0)
    shutil.copy2(origin_pb, init_pb)
    shutil.copy2(origin_pickle, init_pickle)

    origin_geojson = origin_pb.with_suffix(".geojson")
    if origin_geojson.exists():
        shutil.copy2(origin_geojson, maps_dir / origin_geojson.name)

    return {
        "run_dir": run_dir,
        "maps_dir": maps_dir,
        "graphs_dir": graphs_dir,
        "planning_dir": planning_dir,
        "metrics_dir": metrics_dir,
        "map_pb": init_pb,
        "map_pickle": init_pickle,
    }


def prepare_existing_run_workspace(
    source_run_dir: Path,
    out_root: Path,
    copy_existing: bool = True,
) -> Dict[str, Path]:
    source_run_dir = source_run_dir.resolve()
    if not source_run_dir.exists() or not source_run_dir.is_dir():
        raise RuntimeError(f"simulation_only.resume_run_dir not found: {source_run_dir}")

    if copy_existing:
        out_root.mkdir(parents=True, exist_ok=True)
        run_dir = out_root / datetime.now().strftime("%Y%m%d_%H%M%S")
        shutil.copytree(source_run_dir, run_dir)
    else:
        run_dir = source_run_dir

    maps_dir = run_dir / "maps"
    graphs_dir = run_dir / "graphs"
    planning_dir = run_dir / "planning"
    metrics_dir = run_dir / "metrics"
    for d in [maps_dir, graphs_dir, planning_dir, metrics_dir]:
        d.mkdir(parents=True, exist_ok=True)

    pickles = sorted(maps_dir.glob("*.pickle"))
    pbs = sorted(maps_dir.glob("*.pb"))
    if not pickles:
        raise RuntimeError(f"no map pickle found under existing run dir: {maps_dir}")
    map_pickle = pickles[0]
    map_pb = map_pickle.with_suffix(".pb")
    if not map_pb.exists():
        if not pbs:
            raise RuntimeError(f"no map pb found under existing run dir: {maps_dir}")
        map_pb = pbs[0]

    return {
        "run_dir": run_dir,
        "maps_dir": maps_dir,
        "graphs_dir": graphs_dir,
        "planning_dir": planning_dir,
        "metrics_dir": metrics_dir,
        "map_pb": map_pb,
        "map_pickle": map_pickle,
        "source_run_dir": source_run_dir,
    }


def _find_existing_graph_for_sim_eval(run_dir: Path) -> Path:
    candidates = [
        run_dir / "graphs" / "graph_0.json",
        run_dir / "graphs" / "graph_final_eval.json",
    ]
    candidates.extend(sorted((run_dir / "graphs").glob("graph_*.json")))
    for p in candidates:
        if p.exists():
            return p
    raise RuntimeError(f"no graph json found under existing run dir: {run_dir / 'graphs'}")


def _find_existing_map_pickle_for_sim_eval(run_dir: Path, summary: Dict[str, Any]) -> Path:
    for key in ["initial_map_pickle", "current_map_pickle", "final_map_pickle"]:
        raw = str(summary.get(key) or "").strip()
        if raw and Path(raw).exists():
            return Path(raw)
    pickles = sorted((run_dir / "maps").glob("*.pickle"))
    if pickles:
        return pickles[0]
    raise RuntimeError(f"no map pickle found under existing run dir: {run_dir / 'maps'}")


def _non_overwrite_path(path: Path) -> Path:
    parent = path.parent
    parent.mkdir(parents=True, exist_ok=True)

    def _try_reserve(candidate: Path) -> bool:
        try:
            fd = os.open(candidate, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.close(fd)
            return True
        except FileExistsError:
            return False

    if _try_reserve(path):
        return path
    stem = path.stem
    suffix = path.suffix
    for i in range(1, 10000):
        candidate = parent / f"{stem}_{i:03d}{suffix}"
        if _try_reserve(candidate):
            return candidate
    raise RuntimeError(f"could not find non-overwrite filename for: {path}")


def _format_dist_cut_for_filename(dist_cut: float) -> str:
    text = f"{float(dist_cut):.6f}".rstrip("0").rstrip(".")
    if not text:
        text = "0"
    return text.replace("-", "neg").replace(".", "p")


def _append_simulation_only_metrics(
    run_dir: Path,
    graph_json_path: Path,
    map_pickle_path: Path,
    resident_profiles: Dict[str, Dict[str, Any]],
    resident_preferences: Dict[str, list[str]],
    eval_runtime: Dict[str, Any],
    shared_cache_dir: Path,
    eval_retry: int,
    token_usage: Optional[Dict[str, int]] = None,
    source_run_dir: Optional[Path] = None,
    resumed: bool = False,
    simulation_llm_required: bool = False,
    compare_with_last_sim: bool = False,
    update_run_summary: bool = True,
    disable_overwrite: bool = False,
    dist_cut: float = 0.0,
    dist_cut_eval: bool = False,
    eval_output_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    metric_cache: Dict[str, Any] = {}
    planning_metrics = _safe_evaluate_planning_metrics(
        graph_json_path=graph_json_path,
        pickle_map_path=map_pickle_path,
        resident_preferences=resident_preferences,
        cache=metric_cache,
        retries=eval_retry,
    )
    simulation_metrics = _safe_evaluate_simulation_metrics(
        graph_json_path=graph_json_path,
        resident_profiles=resident_profiles,
        cache=metric_cache,
        eval_runtime=eval_runtime,
        retries=eval_retry,
        simulation_llm_required=simulation_llm_required,
        dist_cut=dist_cut,
    )
    metrics_full = {
        "planning_metrics": planning_metrics,
        "simulation_metrics": simulation_metrics,
    }
    record = {
        "mode": "simulation_only",
        "resumed": resumed,
        "source_run_dir": str(source_run_dir) if source_run_dir else "",
        "compare_with_last_sim": compare_with_last_sim,
        "last_graph_json_path": str(eval_runtime.get("last_graph_json_path") or ""),
        "graph_json": str(graph_json_path),
        "map_pickle": str(map_pickle_path),
        "simulation_token_usage": token_usage or {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0},
        "metrics_full": metrics_full,
        "simulation_metrics": simulation_metrics,
        "five_metrics": _extract_five_metrics(metrics_full),
        "simulation_llm_required": simulation_llm_required,
    }
    if dist_cut_eval:
        record["dist_cut_eval"] = True
        record["dist_cut"] = dist_cut
        eval_filename = f"simulation_only_eval_dist_cut_{_format_dist_cut_for_filename(dist_cut)}.json"
    else:
        eval_filename = "simulation_only_eval_compare_with_last_sim.json" if compare_with_last_sim else "simulation_only_eval.json"
    target_eval_dir = eval_output_dir or (run_dir / "metrics")
    target_eval_dir.mkdir(parents=True, exist_ok=True)
    eval_path = target_eval_dir / eval_filename
    if disable_overwrite and not dist_cut_eval:
        eval_path = _non_overwrite_path(eval_path)
        record["disable_overwrite"] = True
    save_json(eval_path, record)

    if not update_run_summary:
        return record

    summary_path = run_dir / "run_summary.json"
    if summary_path.exists():
        try:
            run_summary = load_json(summary_path)
            if not isinstance(run_summary, dict):
                run_summary = {}
        except Exception:
            run_summary = {}
    else:
        run_summary = {}
    run_summary.update(
        {
            "run_dir": str(run_dir),
            "mode": "simulation_only",
            "simulation_only": True,
            "simulation_only_resumed": resumed,
            "simulation_only_source_run_dir": str(source_run_dir) if source_run_dir else "",
            "simulation_only_evaluation": record,
            "simulation_only_evaluation_json": str(eval_path),
            "status": "finished",
        }
    )
    save_json(summary_path, run_summary)
    return record


def _load_resident_profiles(path: Path) -> Dict[str, Dict[str, Any]]:
    if not path.exists():
        return {}
    try:
        data = load_json(path)
    except Exception:
        return {}

    out: Dict[str, Dict[str, Any]] = {}
    if isinstance(data, list):
        for item in data:
            if isinstance(item, dict) and item.get("id") is not None:
                out[str(item.get("id"))] = item
    elif isinstance(data, dict):
        for k, v in data.items():
            if isinstance(v, dict):
                out[str(k)] = v
    return out


def _load_and_validate_preprocess_with_sidecar_fallback(
    repo_root: Path,
    preprocess_path: Path,
    map_pickle_path: Path,
    map_pb_path: Path,
    resident_count: int,
    allow_mismatch: bool = False,
):
    bundle_pickle_path = map_pickle_path
    bundle_pb_path = map_pb_path
    if allow_mismatch:
        bundle = load_json(preprocess_path)
        if not isinstance(bundle, dict):
            raise RuntimeError(f"resident preprocess bundle must be JSON object: {preprocess_path}")
        meta = bundle.get("metadata") if isinstance(bundle.get("metadata"), dict) else {}
        bundle_pickle_raw = str(meta.get("map_pickle_path") or meta.get("map_pickle_abs") or "").strip()
        if bundle_pickle_raw:
            bundle_pickle_path = Path(bundle_pickle_raw)
        bundle_pb_raw = str(meta.get("map_pb_path") or meta.get("map_pb_abs") or "").strip()
        bundle_pb_path = Path(bundle_pb_raw) if bundle_pb_raw else None
    try:
        return load_and_validate_preprocess(
            preprocess_path,
            map_pickle_path=bundle_pickle_path,
            map_pb_path=None if allow_mismatch else bundle_pb_path,
            resident_count=resident_count,
        )
    except FileNotFoundError as e:
        msg = str(e)
        if "resident preprocess" not in msg or "sidecar missing" not in msg:
            raise

    bundle = load_json(preprocess_path)
    if not isinstance(bundle, dict):
        raise RuntimeError(f"resident preprocess bundle must be JSON object: {preprocess_path}")
    stem = preprocess_path.stem
    local_profiles = preprocess_path.with_name(f"{stem}_profiles.json")
    local_preferences = preprocess_path.with_name(f"{stem}_preferences.json")
    if not local_profiles.exists() or not local_preferences.exists():
        raise FileNotFoundError(
            f"resident preprocess sidecar fallback missing: {local_profiles}, {local_preferences}"
        )

    patched = copy.deepcopy(bundle)
    patched.setdefault("metadata", {})
    patched["metadata"]["profiles_path"] = str(local_profiles.resolve())
    patched["metadata"]["preferences_path"] = str(local_preferences.resolve())
    patch_dir = repo_root / "outputs" / "shared_cache" / "resident_preprocess_bundle_fallback"
    patch_dir.mkdir(parents=True, exist_ok=True)
    patched_path = patch_dir / preprocess_path.name
    save_json(patched_path, patched)
    return load_and_validate_preprocess(
        patched_path,
        map_pickle_path=bundle_pickle_path,
        map_pb_path=None if allow_mismatch else bundle_pb_path,
        resident_count=resident_count,
    )


def _resolve_api_key(llm_cfg: Dict[str, Any]) -> str:
    api_key = str(llm_cfg.get("api_key") or "").strip()
    if api_key:
        return api_key
    env_name = str(llm_cfg.get("api_key_env") or "").strip()
    if env_name:
        return str(os.getenv(env_name, "")).strip()
    return ""


def _build_eval_runtime(cfg: Dict[str, Any], require_api_key: bool = True) -> Dict[str, Any]:
    llm_cfg = cfg.get("llm") or {}
    run_cfg = cfg.get("run") or {}
    eval_cfg = run_cfg.get("evaluation") or {}
    provider = str(eval_cfg.get("provider") or llm_cfg.get("provider") or "openai")
    base_url = eval_cfg.get("base_url") or llm_cfg.get("base_url")
    api_key = str(eval_cfg.get("api_key") or "").strip() or _resolve_api_key(eval_cfg) or _resolve_api_key(llm_cfg)
    temperature = float(eval_cfg.get("temperature", llm_cfg.get("temperature", 0.0)))
    max_tokens = int(eval_cfg.get("max_tokens", llm_cfg.get("max_tokens", 4096)))
    timeout = int(eval_cfg.get("timeout", llm_cfg.get("timeout", 120)))
    api_retries = int(run_cfg.get("evaluation_retry_per_call", 10))

    model = str(eval_cfg.get("model") or llm_cfg.get("model") or "gpt-4o-mini")
    enable_thinking_raw = eval_cfg.get("enable_thinking", None)
    enable_thinking: Optional[bool]
    if enable_thinking_raw is None or str(enable_thinking_raw).strip() == "":
        enable_thinking = None
    else:
        enable_thinking = bool(enable_thinking_raw)
    reasoning_effort = str(eval_cfg.get("reasoning_effort") or "").strip()
    sentiment_scale = int(eval_cfg.get("sentiment_scale", 5))
    if sentiment_scale not in (3, 5):
        sentiment_scale = 5
    if require_api_key and provider != "vllm" and not api_key:
        raise RuntimeError(
            "evaluation LLM api key is empty. Set evaluation.api_key in sim runtime yaml "
            "or planner config llm.api_key."
        )

    return {
        "provider": provider,
        "base_url": base_url,
        "api_key": api_key,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "timeout": timeout,
        "api_retries": max(1, api_retries),
        "model": model,
        "enable_thinking": enable_thinking,
        "reasoning_effort": reasoning_effort,
        "sentiment_scale": sentiment_scale,
    }


def _maybe_load_last_graph_index(cache: Dict[str, Any], eval_runtime: Dict[str, Any]) -> Optional[Any]:
    if not bool(eval_runtime.get("compare_with_last_sim", False)):
        return None
    raw = str(eval_runtime.get("last_graph_json_path") or "").strip()
    if not raw:
        raise RuntimeError("evaluation.compare_with_last_sim=true requires evaluation.last_graph_json_path")
    path = Path(raw).expanduser()
    if not path.is_absolute():
        path = Path(__file__).resolve().parent / path
    path = path.resolve()
    if not path.exists():
        raise RuntimeError(f"evaluation.last_graph_json_path not found: {path}")
    graph_key = str(path) + f"::{path.stat().st_mtime_ns}::{path.stat().st_size}"
    if graph_key not in cache.get("graph_index", {}):
        cache.setdefault("graph_index", {})[graph_key] = build_graph_index(load_json(path))
    return cache["graph_index"][graph_key]


def _load_sim_runtime_cfg(repo_root: Path, cfg_path: Optional[Path]) -> Dict[str, Any]:
    default_cfg = repo_root / "sim_script_runtime.yaml"
    final_cfg_path = cfg_path or default_cfg
    if not final_cfg_path.exists():
        raise RuntimeError(f"runtime config not found: {final_cfg_path}")
    cfg = load_cfg(final_cfg_path) or {}
    if not isinstance(cfg, dict):
        raise RuntimeError(f"runtime config must be dict: {final_cfg_path}")
    cfg["_cfg_path"] = str(final_cfg_path)
    return cfg


def _get_eval_llm_json_fn(
    cache: Dict[str, Any],
    eval_runtime: Dict[str, Any],
    model_name: str,
) -> Callable[[str], Optional[Dict[str, Any]]]:
    clients = cache.setdefault("eval_llm_clients", {})
    key = json.dumps(
        {
            "provider": eval_runtime.get("provider"),
            "base_url": eval_runtime.get("base_url"),
            "model": model_name,
            "temperature": eval_runtime.get("temperature"),
            "max_tokens": eval_runtime.get("max_tokens"),
            "timeout": eval_runtime.get("timeout"),
            "api_retries": eval_runtime.get("api_retries"),
            "enable_thinking": eval_runtime.get("enable_thinking"),
            "reasoning_effort": eval_runtime.get("reasoning_effort"),
        },
        ensure_ascii=False,
        sort_keys=True,
    )
    if key not in clients:
        clients[key] = EvalLLMClient(
            EvalLLMConfig(
                provider=str(eval_runtime.get("provider")),
                api_key=str(eval_runtime.get("api_key") or ""),
                model=str(model_name),
                base_url=eval_runtime.get("base_url"),
                temperature=float(eval_runtime.get("temperature", 0.0)),
                max_tokens=int(eval_runtime.get("max_tokens", 2400)),
                timeout=int(eval_runtime.get("timeout", 120)),
                dump_api_io=False,
                enable_thinking=eval_runtime.get("enable_thinking"),
                reasoning_effort=str(eval_runtime.get("reasoning_effort") or "").strip() or None,
            )
        )
    cli: EvalLLMClient = clients[key]
    api_retries = max(1, int(eval_runtime.get("api_retries", 1)))

    def _call(prompt: str) -> Optional[Dict[str, Any]]:
        msgs = [
            {"role": "system", "content": "You are a precise JSON generator. Reply with JSON object only."},
            {"role": "user", "content": prompt},
        ]
        for _ in range(api_retries):
            try:
                resp = cli.chat(msgs)
                text = cli.extract_text(resp)
            except Exception:
                continue
            txt = (text or "").strip()
            txt = txt.removeprefix("```json").removeprefix("```").removesuffix("```").strip()
            try:
                obj = json.loads(txt)
                if isinstance(obj, dict):
                    return obj
            except Exception:
                pass
            import re
            m = re.search(r"\{[\s\S]*\}", txt)
            if not m:
                continue
            try:
                obj = json.loads(m.group(0))
                if isinstance(obj, dict):
                    return obj
            except Exception:
                continue
        return None

    return _call


def evaluate_before_after(
    graph_json_path: Path,
    pickle_map_path: Path,
    resident_profiles: Dict[str, Dict[str, Any]],
    resident_preferences: Dict[str, list[str]],
    cache: Dict[str, Any],
    eval_runtime: Dict[str, Any],
    persistent_cache_dir: Optional[Path] = None,
    include_simulation_metrics: bool = True,
    simulation_llm_required: bool = True,
) -> Dict[str, Any]:
    graph_key = str(graph_json_path.resolve()) + f"::{graph_json_path.stat().st_mtime_ns}::{graph_json_path.stat().st_size}"
    map_key = str(pickle_map_path.resolve()) + f"::{pickle_map_path.stat().st_mtime_ns}::{pickle_map_path.stat().st_size}"

    if graph_key not in cache.get("graph_index", {}):
        cache.setdefault("graph_index", {})[graph_key] = build_graph_index(load_json(graph_json_path))
    idx = cache["graph_index"][graph_key]

    eval_model = str(eval_runtime.get("model") or "gpt-4o-mini")
    pref_cache = {"resident_preferences": resident_preferences, "source": "resident_preprocess"}

    if map_key not in cache.get("road", {}):
        try:
            cache.setdefault("road", {})[map_key] = WalkingRoadDistance(str(pickle_map_path))
        except Exception:
            cache.setdefault("road", {})[map_key] = None
    road_model = cache["road"][map_key]

    import pickle as pkl
    m = pkl.load(open(pickle_map_path, "rb"))
    curr_landuse = {str(a.get("id")): str(a.get("urban_land_use") or "") for a in (m.get("aois") or [])}

    planning_metrics = evaluate_accessibility_proxy(
        idx,
        service_radius=500.0,
        id_service_radius=1000.0,
        ecology_radius=300.0,
        landuse_override=curr_landuse,
        resident_preferences=(pref_cache or {}).get("resident_preferences"),
        road_distance_model=road_model,
    )
    if not include_simulation_metrics:
        return {
            "planning_metrics": planning_metrics,
            "simulation_metrics": {
                "metrics": {
                    "objective_avg_move_distance": None,
                    "subjective_interview_score": None,
                    "reflection_sentiment_index": None,
                }
            },
        }
    last_idx = _maybe_load_last_graph_index(cache, eval_runtime)
    sentiment_scale = int(eval_runtime.get("sentiment_scale", 5))
    interview_llm_fn = None
    sentiment_llm_fn = None
    if simulation_llm_required:
        interview_llm_fn = _get_eval_llm_json_fn(cache, eval_runtime, eval_model)
        sentiment_llm_fn = _get_eval_llm_json_fn(cache, eval_runtime, eval_model)

    simulation_metrics = evaluate_simulation_assessment(
        idx,
        resident_profiles=resident_profiles,
        last_idx=last_idx,
        interview_llm_json_fn=interview_llm_fn,
        sentiment_llm_json_fn=sentiment_llm_fn,
        sentiment_scale=sentiment_scale,
        llm_required=simulation_llm_required,
    )

    return {
        "planning_metrics": planning_metrics,
        "simulation_metrics": simulation_metrics,
    }


def _safe_evaluate_before_after(
    graph_json_path: Path,
    pickle_map_path: Path,
    resident_profiles: Dict[str, Dict[str, Any]],
    resident_preferences: Dict[str, list[str]],
    cache: Dict[str, Any],
    eval_runtime: Dict[str, Any],
    persistent_cache_dir: Optional[Path] = None,
    retries: int = 10,
    include_simulation_metrics: bool = True,
    simulation_llm_required: bool = True,
) -> Dict[str, Any]:
    last_err: Optional[Exception] = None
    for i in range(max(1, retries)):
        try:
            return evaluate_before_after(
                graph_json_path=graph_json_path,
                pickle_map_path=pickle_map_path,
                resident_profiles=resident_profiles,
                resident_preferences=resident_preferences,
                cache=cache,
                eval_runtime=eval_runtime,
                persistent_cache_dir=persistent_cache_dir,
                include_simulation_metrics=include_simulation_metrics,
                simulation_llm_required=simulation_llm_required,
            )
        except Exception as e:
            last_err = e
            print(
                json.dumps(
                    {
                        "stage": "evaluate_before_after",
                        "retry": f"{i + 1}/{retries}",
                        "error_type": type(e).__name__,
                        "error": str(e),
                    },
                    ensure_ascii=False,
                )
            )
    return {
        "error": f"evaluation_failed: {type(last_err).__name__ if last_err else 'unknown'}: {last_err}",
        "planning_metrics": {"metrics": {"service": None, "ecology": None, "satisfaction": None}},
        "simulation_metrics": {
            "metrics": {
                "objective_avg_move_distance": None,
                "subjective_interview_score": None,
                "reflection_sentiment_index": None,
            }
        },
    }


def evaluate_planning_metrics_only(
    graph_json_path: Path,
    pickle_map_path: Path,
    resident_preferences: Dict[str, list[str]],
    cache: Dict[str, Any],
) -> Dict[str, Any]:
    graph_key = str(graph_json_path.resolve()) + f"::{graph_json_path.stat().st_mtime_ns}::{graph_json_path.stat().st_size}"
    map_key = str(pickle_map_path.resolve()) + f"::{pickle_map_path.stat().st_mtime_ns}::{pickle_map_path.stat().st_size}"

    if graph_key not in cache.get("graph_index", {}):
        cache.setdefault("graph_index", {})[graph_key] = build_graph_index(load_json(graph_json_path))
    idx = cache["graph_index"][graph_key]

    if map_key not in cache.get("road", {}):
        try:
            cache.setdefault("road", {})[map_key] = WalkingRoadDistance(str(pickle_map_path))
        except Exception:
            cache.setdefault("road", {})[map_key] = None
    road_model = cache["road"][map_key]

    import pickle as pkl
    m = pkl.load(open(pickle_map_path, "rb"))
    curr_landuse = {str(a.get("id")): str(a.get("urban_land_use") or "") for a in (m.get("aois") or [])}

    return evaluate_accessibility_proxy(
        idx,
        service_radius=500.0,
        id_service_radius=1000.0,
        ecology_radius=300.0,
        landuse_override=curr_landuse,
        resident_preferences=resident_preferences,
        road_distance_model=road_model,
    )


def _safe_evaluate_planning_metrics(
    graph_json_path: Path,
    pickle_map_path: Path,
    resident_preferences: Dict[str, list[str]],
    cache: Dict[str, Any],
    retries: int = 10,
) -> Dict[str, Any]:
    last_err: Optional[Exception] = None
    for i in range(max(1, retries)):
        try:
            return evaluate_planning_metrics_only(
                graph_json_path=graph_json_path,
                pickle_map_path=pickle_map_path,
                resident_preferences=resident_preferences,
                cache=cache,
            )
        except Exception as e:
            last_err = e
            print(
                json.dumps(
                    {
                        "stage": "evaluate_planning_metrics_only",
                        "retry": f"{i + 1}/{retries}",
                        "error_type": type(e).__name__,
                        "error": str(e),
                    },
                    ensure_ascii=False,
                )
            )
    return {
        "error": f"planning_evaluation_failed: {type(last_err).__name__ if last_err else 'unknown'}: {last_err}",
        "metrics": {"service": None, "ecology": None, "satisfaction": None},
    }


def evaluate_simulation_metrics_only(
    graph_json_path: Path,
    resident_profiles: Dict[str, Dict[str, Any]],
    cache: Dict[str, Any],
    eval_runtime: Dict[str, Any],
    simulation_llm_required: bool = False,
    dist_cut: float = 0.0,
) -> Dict[str, Any]:
    graph_key = str(graph_json_path.resolve()) + f"::{graph_json_path.stat().st_mtime_ns}::{graph_json_path.stat().st_size}"
    if graph_key not in cache.get("graph_index", {}):
        cache.setdefault("graph_index", {})[graph_key] = build_graph_index(load_json(graph_json_path))
    idx = cache["graph_index"][graph_key]
    last_idx = _maybe_load_last_graph_index(cache, eval_runtime)

    eval_model = str(eval_runtime.get("model") or "gpt-4o-mini")
    sentiment_scale = int(eval_runtime.get("sentiment_scale", 5))
    interview_llm_fn = None
    sentiment_llm_fn = None
    if simulation_llm_required:
        interview_llm_fn = _get_eval_llm_json_fn(cache, eval_runtime, eval_model)
        sentiment_llm_fn = _get_eval_llm_json_fn(cache, eval_runtime, eval_model)

    return evaluate_simulation_assessment(
        idx,
        resident_profiles=resident_profiles,
        last_idx=last_idx,
        interview_llm_json_fn=interview_llm_fn,
        sentiment_llm_json_fn=sentiment_llm_fn,
        sentiment_scale=sentiment_scale,
        llm_required=simulation_llm_required,
        dist_cut=dist_cut,
    )


def _safe_evaluate_simulation_metrics(
    graph_json_path: Path,
    resident_profiles: Dict[str, Dict[str, Any]],
    cache: Dict[str, Any],
    eval_runtime: Dict[str, Any],
    retries: int = 10,
    simulation_llm_required: bool = False,
    dist_cut: float = 0.0,
) -> Dict[str, Any]:
    if simulation_llm_required:
        retries = 1
    last_err: Optional[Exception] = None
    for i in range(max(1, retries)):
        try:
            return evaluate_simulation_metrics_only(
                graph_json_path=graph_json_path,
                resident_profiles=resident_profiles,
                cache=cache,
                eval_runtime=eval_runtime,
                simulation_llm_required=simulation_llm_required,
                dist_cut=dist_cut,
            )
        except Exception as e:
            last_err = e
            print(
                json.dumps(
                    {
                        "stage": "evaluate_simulation_metrics_only",
                        "retry": f"{i + 1}/{retries}",
                        "error_type": type(e).__name__,
                        "error": str(e),
                    },
                    ensure_ascii=False,
                )
            )
    return {
        "error": f"simulation_evaluation_failed: {type(last_err).__name__ if last_err else 'unknown'}: {last_err}",
        "metrics": {
            "objective_avg_move_distance": None,
            "subjective_interview_score": None,
            "reflection_sentiment_index": None,
        },
    }


def _extract_five_metrics(metrics_obj: Dict[str, Any]) -> Dict[str, Any]:
    pm = (metrics_obj or {}).get("planning_metrics", {}).get("metrics", {})
    sm = (metrics_obj or {}).get("simulation_metrics", {}).get("metrics", {})
    return {
        "service": pm.get("service"),
        "ecology": pm.get("ecology"),
        "satisfaction": pm.get("satisfaction"),
        "objective_avg_move_distance": sm.get("objective_avg_move_distance"),
        "subjective_interview_score": sm.get("subjective_interview_score"),
        "reflection_sentiment_index": sm.get("reflection_sentiment_index"),
    }


def _apply_planner_runtime_overrides(cfg: Dict[str, Any], planning_runtime_cfg: Dict[str, Any]) -> None:
    """Apply planner settings owned by sim_script_runtime*.yaml."""
    if not isinstance(planning_runtime_cfg, dict):
        return
    llm_override = planning_runtime_cfg.get("llm")
    if isinstance(llm_override, dict):
        cfg.setdefault("llm", {})
        for k, v in llm_override.items():
            if v is not None:
                cfg["llm"][k] = v
    run_override = planning_runtime_cfg.get("run")
    if isinstance(run_override, dict):
        cfg.setdefault("run", {})
        for k, v in run_override.items():
            cfg["run"][k] = v
    if bool(planning_runtime_cfg.get("planning_only", False)):
        cfg.setdefault("run", {})
        cfg["run"]["skip_simulation_assessment"] = True


def run_plan_agent(
    repo_root: Path,
    planner_cfg_template: Path,
    planner_task_prompt: Path,
    pb_path: Path,
    pickle_path: Path,
    graph_json_path: Path,
    resident_profiles_path: Path,
    resident_preferences_path: Path,
    iter_plan_dir: Path,
    planner_overrides: Optional[Dict[str, Any]] = None,
) -> Tuple[Dict[str, str], Optional[Path], Dict[str, Any]]:
    """
    调用 planning_agent_pipeline_mm_refined。
    返回：modify_dict, planner_materialized_pickle(可选), planner_summary
    """
    iter_plan_dir.mkdir(parents=True, exist_ok=True)
    planner_outputs_root = iter_plan_dir / "outputs"
    planner_outputs_root.mkdir(parents=True, exist_ok=True)

    cfg = load_cfg(planner_cfg_template)
    cfg.setdefault("paths", {})
    cfg["paths"]["pb_map"] = str(pb_path.relative_to(repo_root))
    cfg["paths"]["pickle_map"] = str(pickle_path.relative_to(repo_root))
    cfg["paths"]["graph_json"] = str(graph_json_path.relative_to(repo_root))
    cfg["paths"]["resident_profiles"] = str(resident_profiles_path.relative_to(repo_root))
    cfg["paths"]["resident_preferences"] = str(resident_preferences_path.relative_to(repo_root))
    cfg["paths"]["task_prompt"] = str(planner_task_prompt.relative_to(repo_root))
    cfg["paths"]["outputs"] = str(planner_outputs_root.relative_to(repo_root))
    _apply_planner_runtime_overrides(cfg, planner_overrides or {})

    cfg_path = iter_plan_dir / "planner_config.yaml"
    with open(cfg_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, allow_unicode=True, sort_keys=False)

    # Direct in-process planner invocation (no subprocess/cmd).
    # Keep planner console output behavior unchanged.
    try:
        from planning_agent_pipeline_mm_refined import run_planning as planner_entry
    except Exception as e:
        return {}, None, {"success": False, "error": f"planner_import_error: {e}"}

    old_cwd = Path.cwd()
    old_argv = list(sys.argv)
    result_box: Dict[str, str] = {"error": ""}

    def _planner_worker():
        try:
            os.chdir(str(repo_root))
            sys.argv = [
                "planning_agent_pipeline_mm_refined.run_planning",
                "--config",
                str(cfg_path.relative_to(repo_root)),
            ]
            planner_entry.main()
        except Exception as e:
            result_box["error"] = str(e)
        finally:
            sys.argv = old_argv
            os.chdir(str(old_cwd))

    t = threading.Thread(target=_planner_worker, daemon=True)
    t.start()
    t0 = time.time()
    heartbeat_sec = int(os.getenv("SIM_PLAN_HEARTBEAT_SEC", "15"))
    while t.is_alive():
        t.join(timeout=heartbeat_sec)
        if t.is_alive():
            elapsed = int(time.time() - t0)
            print(f"[planner-progress] running... elapsed={elapsed}s")

    planner_err = result_box.get("error", "")

    # Keep compatibility files for debugging.
    (iter_plan_dir / "planner_stdout.log").write_text(
        "Direct-call mode: planner outputs are printed directly to console.\n",
        encoding="utf-8",
    )
    (iter_plan_dir / "planner_stderr.log").write_text(planner_err, encoding="utf-8")

    if planner_err:
        return {}, None, {"success": False, "error": f"planner_runtime_error: {planner_err}"}

    run_dirs = sorted([p for p in planner_outputs_root.iterdir() if p.is_dir()])
    if not run_dirs:
        return {}, None, {"success": False, "error": "planner_no_output_dir"}

    planner_run_dir = run_dirs[-1]
    summary_path = planner_run_dir / "summary.json"
    if not summary_path.exists():
        return {}, None, {"success": False, "error": "planner_summary_missing", "planner_run_dir": str(planner_run_dir)}

    summary = load_json(summary_path)
    llm_usage = summary.get("llm_usage") if isinstance(summary.get("llm_usage"), dict) else None
    if llm_usage is None:
        llm_usage_path = planner_run_dir / "llm_usage.json"
        if llm_usage_path.exists():
            try:
                llm_usage = load_json(llm_usage_path)
            except Exception:
                llm_usage = None
    if llm_usage is None:
        llm_usage = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
    best_plan = summary.get("best_plan") if isinstance(summary.get("best_plan"), dict) else {}
    modify_dict = {str(k): str(v) for k, v in best_plan.items()}

    materialized_pickle = planner_run_dir / "best_plan_materialized" / "best_plan.pickle"
    if not materialized_pickle.exists():
        materialized_pickle = None

    summary["success"] = True
    summary["planner_run_dir"] = str(planner_run_dir)
    summary["llm_usage"] = llm_usage
    return modify_dict, materialized_pickle, summary


def _classify_planner_error(err: str) -> str:
    e = str(err or "").lower()
    if "unsupported parameter" in e and "max_tokens" in e:
        return "api_unsupported_parameter_max_tokens"
    if "api key" in e or "authentication" in e or "unauthorized" in e:
        return "api_auth_error"
    if "timeout" in e:
        return "api_timeout"
    if "rate limit" in e or "429" in e:
        return "api_rate_limit"
    if "planner_summary_missing" in e:
        return "planner_output_missing"
    if "planner_no_output_dir" in e:
        return "planner_output_dir_missing"
    if "planner_import_error" in e:
        return "planner_import_error"
    if "planner_runtime_error" in e:
        return "planner_runtime_error"
    return "planner_unknown_error"


def run_simulation_with_retry(
    map_path: Path,
    sim_cfg: Dict[str, Any],
    max_retries: int = 10,
) -> Tuple[Optional[Any], Dict[str, int], Optional[Dict[int, Any]], Optional[str]]:
    last_err: Optional[Exception] = None
    for i in range(max(1, max_retries)):
        try:
            config = add_map_config(map_path=str(map_path), sim_cfg=sim_cfg)
            graph, token_usage, stream_memory = asyncio.run(main(config=config))
            return graph, token_usage, stream_memory, None
        except Exception as e:
            last_err = e
            print(
                json.dumps(
                    {
                        "stage": "simulation",
                        "retry": f"{i + 1}/{max_retries}",
                        "error_type": type(e).__name__,
                        "error": str(e),
                    },
                    ensure_ascii=False,
                )
            )
            time.sleep(1.0)
    return None, {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}, None, (
        f"{type(last_err).__name__}: {last_err}" if last_err else "unknown"
    )


if __name__ == "__main__":
    repo_root = Path(__file__).resolve().parent
    cfg_arg = Path(sys.argv[1]) if len(sys.argv) > 1 else None
    sim_runtime_cfg = _load_sim_runtime_cfg(repo_root, cfg_arg)

    runtime_cfg = sim_runtime_cfg.get("runtime") or {}
    paths_cfg = sim_runtime_cfg.get("paths") or {}
    eval_cfg = sim_runtime_cfg.get("evaluation") or {}
    planning_cfg = sim_runtime_cfg.get("planning") or {}
    simulation_only_cfg = sim_runtime_cfg.get("simulation_only") or {}
    planning_only = bool(planning_cfg.get("planning_only", False))
    simulation_only = bool(simulation_only_cfg.get("enabled", False))
    planning_only_graph_json_cfg = str(planning_cfg.get("planning_only_graph_json") or "").strip()

    seed = int(runtime_cfg.get("seed", 42))
    random.seed(seed)
    np.random.seed(seed)

    proxy = str(runtime_cfg.get("all_proxy") or "").strip()
    if proxy:
        os.environ["ALL_PROXY"] = proxy

    origin_map_pb_raw = str(paths_cfg.get("origin_map_pb") or "").strip()
    if not origin_map_pb_raw and not (simulation_only and str(simulation_only_cfg.get("resume_run_dir") or "").strip()):
        origin_map_pb_raw = "data/example/berlin_0.pb"
    origin_map_path = repo_root / origin_map_pb_raw if origin_map_pb_raw else Path()
    origin_map_path_pickle = origin_map_path.with_suffix(".pickle") if origin_map_pb_raw else Path()
    planner_cfg_template = repo_root / str(
        planning_cfg.get("cfg_template", "planning_agent_pipeline_mm_refined/config.yaml")
    )
    planner_task_prompt = repo_root / str(
        planning_cfg.get("task_prompt", "planning_agent_pipeline_mm_refined/task_prompt.txt")
    )
    resident_preprocess_raw = str(paths_cfg.get("resident_preprocess_path") or "").strip()
    allow_mismatch = bool(paths_cfg.get("allow_mismatch", False))
    if not resident_preprocess_raw:
        raise RuntimeError(
            "paths.resident_preprocess_path is required. Run resident_preprocess/resident_preprocess.py first."
        )
    resident_preprocess_path = (repo_root / resident_preprocess_raw).resolve()

    planner_cfg_obj = load_cfg(planner_cfg_template)
    if simulation_only:
        planner_cfg_for_eval = {"llm": copy.deepcopy(sim_runtime_cfg.get("llm_config") or {}), "run": {"evaluation": {}}}
    else:
        planner_cfg_for_eval = copy.deepcopy(planner_cfg_obj)
    if isinstance(eval_cfg, dict):
        planner_cfg_for_eval.setdefault("run", {})
        planner_cfg_for_eval["run"].setdefault("evaluation", {})
        # evaluation fields come from sim runtime yaml; no api_key_env / timeout field here.
        for k in ["provider", "api_key", "base_url", "model", "temperature", "max_tokens", "sentiment_scale"]:
            v = eval_cfg.get(k)
            if v is not None:
                planner_cfg_for_eval["run"]["evaluation"][k] = v
    if isinstance(planning_cfg, dict) and not simulation_only:
        _apply_planner_runtime_overrides(planner_cfg_for_eval, planning_cfg)
    simulation_llm_required = bool(simulation_only_cfg.get("simulation_llm_required", False)) if simulation_only else True
    eval_runtime = _build_eval_runtime(planner_cfg_for_eval, require_api_key=simulation_llm_required or not simulation_only)
    eval_runtime["last_graph_json_path"] = (
        str(eval_cfg.get("last_graph_json_path") or "").strip() if simulation_only and isinstance(eval_cfg, dict) else ""
    )
    eval_runtime["compare_with_last_sim"] = (
        bool(eval_cfg.get("compare_with_last_sim", False)) if simulation_only and isinstance(eval_cfg, dict) else False
    )

    out_root = repo_root / str(paths_cfg.get("out_root", "sim_plan_runs"))
    workspace = None
    resume_run_dir_cfg = str(simulation_only_cfg.get("resume_run_dir") or "").strip()
    if simulation_only and resume_run_dir_cfg:
        resume_run_dir = (repo_root / resume_run_dir_cfg).resolve()
        copy_existing = bool(simulation_only_cfg.get("copy_existing_run", True))
        workspace = prepare_existing_run_workspace(resume_run_dir, out_root, copy_existing=copy_existing)
    else:
        workspace = prepare_run_workspace(origin_map_path, origin_map_path_pickle, out_root)
    run_dir = workspace["run_dir"]
    shared_cache_dir = repo_root / "outputs" / "shared_cache"
    shared_cache_dir.mkdir(parents=True, exist_ok=True)

    resident_preprocess = _load_and_validate_preprocess_with_sidecar_fallback(
        repo_root,
        resident_preprocess_path,
        map_pickle_path=workspace["map_pickle"],
        map_pb_path=workspace["map_pb"],
        resident_count=int((sim_runtime_cfg.get("agents_config") or {}).get("number", 0)),
        allow_mismatch=allow_mismatch,
    )
    sim_runtime_cfg.setdefault("paths", {})
    sim_runtime_cfg["paths"]["resident_profiles_path"] = str(resident_preprocess.profiles_path)
    resident_profiles_path = resident_preprocess.profiles_path
    resident_preferences_path = resident_preprocess.preferences_path
    resident_preferences = resident_preprocess.resident_preferences

    cfg_src = Path(str(sim_runtime_cfg.get("_cfg_path") or "")).resolve() if sim_runtime_cfg.get("_cfg_path") else None
    cfg_copy_in_run = None
    direct_resume_eval = bool(
        simulation_only
        and resume_run_dir_cfg
        and not bool(simulation_only_cfg.get("copy_existing_run", True))
    )
    direct_compare_eval = bool(
        simulation_only
        and direct_resume_eval
        and bool(eval_runtime.get("compare_with_last_sim", False))
    )
    add_new_label = bool(simulation_only_cfg.get("add_new_label", False))
    simulation_only_eval_dir = (
        run_dir / "metric_new"
        if simulation_only and resume_run_dir_cfg and add_new_label
        else run_dir / "metrics"
    )
    dist_cut_cfg = float(simulation_only_cfg.get("dist_cut", 0.07))
    direct_dist_cut_eval = bool(
        simulation_only
        and direct_resume_eval
        and dist_cut_cfg > 0
    )
    if cfg_src is not None and cfg_src.exists() and not direct_resume_eval:
        cfg_copy_in_run = run_dir / "sim_script_runtime.yaml"
        try:
            with open(cfg_copy_in_run, "w", encoding="utf-8") as f:
                yaml.safe_dump(sim_runtime_cfg, f, allow_unicode=True, sort_keys=False)
        except Exception:
            cfg_copy_in_run = None

    map_path = workspace["map_pb"]
    map_path_pickle = workspace["map_pickle"]

    total_iters = 1 if planning_only else int(runtime_cfg.get("total_iters", 3))
    sim_retry = int(runtime_cfg.get("simulation_retry_per_iter", 10))
    eval_retry = int(runtime_cfg.get("evaluation_retry_per_call", 10))

    resident_profiles = _load_resident_profiles(resident_profiles_path)
    metric_cache: Dict[str, Any] = {}

    if simulation_only:
        eval_retry = int(runtime_cfg.get("evaluation_retry_per_call", 10))
        if resume_run_dir_cfg:
            existing_summary = {}
            summary_path = run_dir / "run_summary.json"
            if summary_path.exists():
                try:
                    existing_summary = load_json(summary_path)
                except Exception:
                    existing_summary = {}
            graph_json_path = _find_existing_graph_for_sim_eval(run_dir)
            map_path_pickle = _find_existing_map_pickle_for_sim_eval(run_dir, existing_summary if isinstance(existing_summary, dict) else {})
            record = _append_simulation_only_metrics(
                run_dir=run_dir,
                graph_json_path=graph_json_path,
                map_pickle_path=map_path_pickle,
                resident_profiles=resident_profiles,
                resident_preferences=resident_preferences,
                eval_runtime=eval_runtime,
                shared_cache_dir=shared_cache_dir,
                eval_retry=eval_retry,
                source_run_dir=workspace.get("source_run_dir"),
                resumed=True,
                simulation_llm_required=False if direct_dist_cut_eval else simulation_llm_required,
                compare_with_last_sim=bool(eval_runtime.get("compare_with_last_sim", False)),
                update_run_summary=not direct_resume_eval,
                disable_overwrite=bool(simulation_only_cfg.get("disable_overwrite", False)),
                dist_cut=dist_cut_cfg if direct_dist_cut_eval else 0.0,
                dist_cut_eval=direct_dist_cut_eval,
                eval_output_dir=simulation_only_eval_dir,
            )
            print(json.dumps({"run_dir": str(run_dir), "simulation_only_metrics": record["five_metrics"]}, ensure_ascii=False))
            sys.exit(0)

        sim_retry = int(runtime_cfg.get("simulation_retry_per_iter", 10))
        map_path = workspace["map_pb"]
        map_path_pickle = workspace["map_pickle"]
        graph, token_usage, stream_memory, sim_err = run_simulation_with_retry(
            map_path=map_path,
            sim_cfg=sim_runtime_cfg,
            max_retries=sim_retry,
        )
        if graph is None:
            run_summary = {
                "run_dir": str(run_dir),
                "mode": "simulation_only",
                "simulation_only": True,
                "status": "simulation_failed",
                "simulation_error": sim_err,
                "simulation_token_usage": token_usage,
                "initial_map_pb": str(map_path),
                "initial_map_pickle": str(map_path_pickle),
                "runtime_cfg_path": sim_runtime_cfg.get("_cfg_path"),
            }
            save_json(run_dir / "run_summary.json", run_summary)
            print(json.dumps({"run_dir": str(run_dir), "simulation_error": sim_err}, ensure_ascii=False))
            sys.exit(0)

        graph_json_path = workspace["graphs_dir"] / "graph_0.json"
        data = json_graph.node_link_data(graph, edges="links")
        with open(graph_json_path, "w", encoding="utf-8") as outfile:
            json.dump(data, outfile, ensure_ascii=False)
        stream_memory_path = workspace["graphs_dir"] / "stream_memory_0.json"
        with open(stream_memory_path, "w", encoding="utf-8") as outfile:
            json.dump(stream_memory or {}, outfile, ensure_ascii=False)

        cfg_src = Path(str(sim_runtime_cfg.get("_cfg_path") or "")).resolve() if sim_runtime_cfg.get("_cfg_path") else None
        if cfg_src is not None and cfg_src.exists():
            with open(run_dir / "sim_script_runtime.yaml", "w", encoding="utf-8") as f:
                yaml.safe_dump(sim_runtime_cfg, f, allow_unicode=True, sort_keys=False)

        record = _append_simulation_only_metrics(
            run_dir=run_dir,
            graph_json_path=graph_json_path,
            map_pickle_path=map_path_pickle,
            resident_profiles=resident_profiles,
            resident_preferences=resident_preferences,
            eval_runtime=eval_runtime,
            shared_cache_dir=shared_cache_dir,
            eval_retry=eval_retry,
            token_usage=token_usage,
            resumed=False,
            simulation_llm_required=simulation_llm_required,
            compare_with_last_sim=bool(eval_runtime.get("compare_with_last_sim", False)),
            disable_overwrite=bool(simulation_only_cfg.get("disable_overwrite", False)),
            eval_output_dir=simulation_only_eval_dir,
        )
        run_summary = load_json(run_dir / "run_summary.json")
        run_summary.update(
            {
                "initial_map_pb": str(map_path),
                "initial_map_pickle": str(map_path_pickle),
                "current_map_pb": str(map_path),
                "current_map_pickle": str(map_path_pickle),
                "runtime_cfg_path": sim_runtime_cfg.get("_cfg_path"),
                "resident_preprocess_path": str(resident_preprocess.bundle_path),
                "resident_profiles_path": str(resident_profiles_path),
                "resident_preferences_path": str(resident_preferences_path),
                "stream_memory_json": str(stream_memory_path),
                "shared_cache_dir": str(shared_cache_dir),
            }
        )
        save_json(run_dir / "run_summary.json", run_summary)
        print(json.dumps({"run_dir": str(run_dir), "simulation_only_metrics": record["five_metrics"]}, ensure_ascii=False))
        sys.exit(0)

    run_summary: Dict[str, Any] = {
        "run_dir": str(run_dir),
        "initial_map_pb": str(map_path),
        "initial_map_pickle": str(map_path_pickle),
        "current_map_pb": str(map_path),
        "current_map_pickle": str(map_path_pickle),
        "evaluation_runtime": {
            "provider": eval_runtime.get("provider"),
            "model": eval_runtime.get("model"),
            "sentiment_scale": eval_runtime.get("sentiment_scale"),
        },
        "planning_only": planning_only,
        "planning_only_graph_json": "",
        "resident_preprocess_path": str(resident_preprocess.bundle_path),
        "resident_profiles_path": str(resident_profiles_path),
        "resident_preferences_path": str(resident_preferences_path),
        "shared_cache_dir": str(shared_cache_dir),
        "runtime_cfg_path": sim_runtime_cfg.get("_cfg_path"),
        "runtime_cfg_in_run": str(cfg_copy_in_run) if cfg_copy_in_run else "",
        "iterations": [],
        "status": "running",
    }
    # Persist immediately so partial runs still have a discoverable summary file.
    save_json(run_dir / "run_summary.json", run_summary)
    # Track the latest successfully applied planning state for robust fallback auditing.
    last_success_state: Dict[str, Any] = {
        "iter": -1,
        "map_pb": str(map_path),
        "map_pickle": str(map_path_pickle),
        "planner_run_dir": "",
    }

    print(
        json.dumps(
            {
                "run_dir": str(run_dir),
                "iters": total_iters,
                "planning_only": planning_only,
                "evaluation_runtime": run_summary["evaluation_runtime"],
            },
            ensure_ascii=False,
        )
    )

    for iter in range(total_iters):
        print(f"\n===== iter {iter} =====")

        token_usage = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
        if planning_only:
            if not planning_only_graph_json_cfg:
                raise RuntimeError(
                    "planning_only=true requires planning.planning_only_graph_json to point to an existing graph json"
                )
            graph_json_path = (repo_root / planning_only_graph_json_cfg).resolve()
            if not graph_json_path.exists():
                raise RuntimeError(f"planning_only graph json not found: {graph_json_path}")
            run_summary["planning_only_graph_json"] = str(graph_json_path)
            print(json.dumps({"iter": iter, "planning_only_graph_json": str(graph_json_path)}, ensure_ascii=False))
        else:
            '''
            1. 居民 simulation 和 graph 构建
            '''
            graph, token_usage, stream_memory, sim_err = run_simulation_with_retry(
                map_path=map_path,
                sim_cfg=sim_runtime_cfg,
                max_retries=sim_retry,
            )
            print("token_usage", token_usage)

            if graph is None:
                # hard fallback: keep previous map unchanged and continue next iter
                map_path_new = Path(str(map_path).replace(f"_{iter}.pb", f"_{iter + 1}.pb"))
                map_path_new_pickle = map_path_new.with_suffix(".pickle")
                shutil.copy2(map_path, map_path_new)
                shutil.copy2(map_path_pickle, map_path_new_pickle)
                iter_record = {
                    "iter": iter,
                    "simulation_failed": True,
                    "simulation_error": sim_err,
                    "token_usage": token_usage,
                    "map_before": {"pb": str(map_path), "pickle": str(map_path_pickle)},
                    "map_after": {"pb": str(map_path_new), "pickle": str(map_path_new_pickle)},
                }
                save_json(workspace["metrics_dir"] / f"iter_{iter:03d}.json", iter_record)
                run_summary["iterations"].append(iter_record)
                run_summary["current_map_pb"] = str(map_path_new)
                run_summary["current_map_pickle"] = str(map_path_new_pickle)
                save_json(run_dir / "run_summary.json", run_summary)
                map_path, map_path_pickle = map_path_new, map_path_new_pickle
                continue

            graph_json_path = workspace["graphs_dir"] / f"graph_{iter}.json"
            data = json_graph.node_link_data(graph, edges="links")
            with open(graph_json_path, "w", encoding="utf-8") as outfile:
                json.dump(data, outfile, ensure_ascii=False)
            stream_memory_path = workspace["graphs_dir"] / f"stream_memory_{iter}.json"
            with open(stream_memory_path, "w", encoding="utf-8") as outfile:
                json.dump(stream_memory or {}, outfile, ensure_ascii=False)

        '''
        2. 规划前指标评估
        '''
        metrics_before = _safe_evaluate_before_after(
            graph_json_path=graph_json_path,
            pickle_map_path=map_path_pickle,
            resident_profiles=resident_profiles,
            resident_preferences=resident_preferences,
            cache=metric_cache,
            eval_runtime=eval_runtime,
            persistent_cache_dir=shared_cache_dir,
            retries=eval_retry,
            include_simulation_metrics=not planning_only,
        )

        '''
        3. plan 和地图文件修改
        用 refined planner 替代原 plan() + 写死 modify_dict
        '''
        iter_plan_dir = workspace["planning_dir"] / f"iter_{iter:03d}"
        modify_dict, planner_pickle, planner_info = run_plan_agent(
            repo_root=repo_root,
            planner_cfg_template=planner_cfg_template,
            planner_task_prompt=planner_task_prompt,
            pb_path=map_path,
            pickle_path=map_path_pickle,
            graph_json_path=graph_json_path,
            resident_profiles_path=resident_profiles_path,
            resident_preferences_path=resident_preferences_path,
            iter_plan_dir=iter_plan_dir,
            planner_overrides=planning_cfg,
        )

        # 保持你原来的命名更新逻辑
        map_path_new = Path(str(map_path).replace(f"_{iter}.pb", f"_{iter + 1}.pb"))
        map_path_new_pickle = map_path_new.with_suffix(".pickle")

        apply_mode = ""
        if planner_pickle is not None:
            # planner 已经产出完整地图，优先直接复用
            shutil.copy2(planner_pickle, map_path_new_pickle)
            apply_mode = "use_planner_materialized_pickle"
        elif modify_dict:
            # 否则走你原来的 modify_aois 逻辑
            modify_aois(modify_dict, str(map_path_pickle), str(map_path_new_pickle))
            apply_mode = "apply_modify_dict"
        else:
            # planner 失败或无修改，保底复制上一轮
            shutil.copy2(map_path_pickle, map_path_new_pickle)
            apply_mode = "no_change_copy_previous"

        dict2pb_path(str(map_path_new_pickle), str(map_path_new))

        planner_success = bool(planner_info.get("success", False))
        planner_error = str(planner_info.get("error", "")) if not planner_success else ""
        planner_error_type = _classify_planner_error(planner_error) if planner_error else ""
        if not planner_success:
            print(
                json.dumps(
                    {
                        "iter": iter,
                        "planner_error_type": planner_error_type,
                        "planner_error": planner_error,
                        "fallback": "keep_latest_success_state_and_continue",
                        "latest_success_state": last_success_state,
                    },
                    ensure_ascii=False,
                )
            )
        else:
            last_success_state = {
                "iter": iter,
                "map_pb": str(map_path_new),
                "map_pickle": str(map_path_new_pickle),
                "planner_run_dir": str(planner_info.get("planner_run_dir", "")),
            }

        '''
        4. 规划后评估
        '''
        metrics_after = _safe_evaluate_before_after(
            graph_json_path=graph_json_path,
            pickle_map_path=map_path_new_pickle,
            resident_profiles=resident_profiles,
            resident_preferences=resident_preferences,
            cache=metric_cache,
            eval_runtime=eval_runtime,
            persistent_cache_dir=shared_cache_dir,
            retries=eval_retry,
            include_simulation_metrics=not planning_only,
        )

        # Important: simulation metrics depend on simulation graph.
        # In this iteration, graph is fixed (already simulated before planning),
        # so only planning metrics are meaningful for before/after plan comparison.
        planning_before = (metrics_before or {}).get("planning_metrics", {})
        planning_after = (metrics_after or {}).get("planning_metrics", {})
        simulation_from_current_graph = (metrics_before or {}).get("simulation_metrics", {})

        iter_record = {
            "iter": iter,
            "graph_json": str(graph_json_path),
            "planning_only": planning_only,
            "token_usage": token_usage,
            "planner": {
                "success": planner_success,
                "apply_mode": apply_mode,
                "modify_dict": modify_dict,
                "token_usage": planner_info.get("llm_usage", {}),
                "error_type": planner_error_type,
                "error": planner_error,
                "fallback_latest_success_state": last_success_state if not planner_success else {},
                "info": planner_info,
            },
            "map_before": {
                "pb": str(map_path),
                "pickle": str(map_path_pickle),
            },
            "map_after": {
                "pb": str(map_path_new),
                "pickle": str(map_path_new_pickle),
            },
            "metrics_before_plan": metrics_before,
            "metrics_after_plan": metrics_after,
            "planning_metrics_before_plan": planning_before,
            "planning_metrics_after_plan": planning_after,
            "simulation_metrics_from_current_simulation": simulation_from_current_graph,
            "five_metrics_snapshot_current_graph": {
                "before_plan": _extract_five_metrics(metrics_before),
                "after_plan": {
                    "service": (planning_after.get("metrics") or {}).get("service"),
                    "ecology": (planning_after.get("metrics") or {}).get("ecology"),
                    "satisfaction": (planning_after.get("metrics") or {}).get("satisfaction"),
                    "objective_avg_move_distance": (simulation_from_current_graph.get("metrics") or {}).get("objective_avg_move_distance"),
                    "subjective_interview_score": (simulation_from_current_graph.get("metrics") or {}).get("subjective_interview_score"),
                    "reflection_sentiment_index": (simulation_from_current_graph.get("metrics") or {}).get("reflection_sentiment_index"),
                },
            },
        }
        save_json(workspace["metrics_dir"] / f"iter_{iter:03d}.json", iter_record)
        run_summary["iterations"].append(iter_record)
        run_summary["current_map_pb"] = str(map_path_new)
        run_summary["current_map_pickle"] = str(map_path_new_pickle)
        save_json(run_dir / "run_summary.json", run_summary)

        print(
            json.dumps(
                {
                    "iter": iter,
                    "planner_success": bool(planner_info.get("success", False)),
                    "apply_mode": apply_mode,
                    "planner_token_usage": planner_info.get("llm_usage", {}),
                    "planning_before": planning_before.get("metrics", {}),
                    "planning_after": planning_after.get("metrics", {}),
                    "simulation_metrics_current_graph": simulation_from_current_graph.get("metrics", {}),
                },
                ensure_ascii=False,
            )
        )

        # 继续你原有迭代方式
        map_path = map_path_new
        map_path_pickle = map_path_new_pickle

    run_summary["final_map_pb"] = str(map_path)
    run_summary["final_map_pickle"] = str(map_path_pickle)

    if planning_only:
        run_summary["final_evaluation"] = {
            "skipped": True,
            "reason": "planning_only_mode",
        }
        run_summary["status"] = "finished"
    else:
        # Final evaluation simulation (after the last planning result is applied).
        # This gives simulation-side metrics on the latest plan state.
        print("\n===== final evaluation simulation =====")
        final_eval_graph_path = workspace["graphs_dir"] / "graph_final_eval.json"
        final_graph, final_sim_token_usage, final_stream_memory, final_sim_err = run_simulation_with_retry(
            map_path=map_path,
            sim_cfg=sim_runtime_cfg,
            max_retries=sim_retry,
        )
        if final_graph is None:
            run_summary["final_evaluation"] = {
                "error": f"final_simulation_failed: {final_sim_err}",
                "simulation_token_usage": final_sim_token_usage,
            }
            run_summary["status"] = "finished_with_final_eval_error"
            save_json(run_dir / "run_summary.json", run_summary)
            print(json.dumps({"run_dir": str(run_dir), "final_map_pb": str(map_path)}, ensure_ascii=False))
            sys.exit(0)
        final_data = json_graph.node_link_data(final_graph, edges="links")
        with open(final_eval_graph_path, "w", encoding="utf-8") as outfile:
            json.dump(final_data, outfile, ensure_ascii=False)
        final_stream_memory_path = workspace["graphs_dir"] / "stream_memory_final_eval.json"
        with open(final_stream_memory_path, "w", encoding="utf-8") as outfile:
            json.dump(final_stream_memory or {}, outfile, ensure_ascii=False)

        final_eval_metrics = _safe_evaluate_before_after(
            graph_json_path=final_eval_graph_path,
            pickle_map_path=map_path_pickle,
            resident_profiles=resident_profiles,
            resident_preferences=resident_preferences,
            cache=metric_cache,
            eval_runtime=eval_runtime,
            persistent_cache_dir=shared_cache_dir,
            retries=eval_retry,
        )
        final_five = _extract_five_metrics(final_eval_metrics)
        run_summary["final_evaluation"] = {
            "graph_json": str(final_eval_graph_path),
            "stream_memory_json": str(final_stream_memory_path),
            "simulation_token_usage": final_sim_token_usage,
            "metrics_full": final_eval_metrics,
            "five_metrics": final_five,
        }
        run_summary["status"] = "finished"
        print(
            json.dumps(
                {
                    "final_evaluation_five_metrics": final_five,
                    "final_evaluation_sim_token_usage": final_sim_token_usage,
                },
                ensure_ascii=False,
            )
        )

    save_json(run_dir / "run_summary.json", run_summary)
    print(json.dumps({"run_dir": str(run_dir), "final_map_pb": str(map_path)}, ensure_ascii=False))
