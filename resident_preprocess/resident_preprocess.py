from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import random
import re
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

_THIS = Path(__file__).resolve()
_REPO_ROOT = _THIS.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

SATISFACTION_CANDIDATE_TYPES = ("BS", "RC", "PK", "OF", "SC", "HS")
WORK_TYPE_WEIGHTS = {
    "OF": 1.0,
    "SC": 0.5,
    "BS": 0.8,
    "ID": 0.5,
    "HS": 0.5,
    "RC": 0.2,
    "PK": 0.1,
}


@dataclass(frozen=True)
class ResidentPreprocess:
    bundle_path: Path
    profiles_path: Path
    preferences_path: Path
    resident_profiles: Dict[str, Dict[str, Any]]
    resident_preferences: Dict[str, List[str]]
    bundle: Dict[str, Any]


def _load_json(path: Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _load_yaml(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _save_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def _sha1_file(path: Path) -> str:
    h = hashlib.sha1()
    with open(path, "rb") as f:
        while True:
            b = f.read(1024 * 1024)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def _sha1_json_obj(obj: Any) -> str:
    return hashlib.sha1(json.dumps(obj, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()


def _resolve_path(raw: str | Path, base_dir: Path) -> Path:
    p = Path(str(raw))
    if p.is_absolute():
        return p.resolve()
    return (base_dir / p).resolve()


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _rel_to_repo(path: Path) -> str:
    p = path.resolve()
    try:
        return str(p.relative_to(_repo_root()))
    except ValueError:
        return str(p)


def _resolve_metadata_path(raw: str, bundle_path: Path) -> Path:
    p = Path(str(raw or ""))
    if p.is_absolute():
        return p.resolve()
    candidates = [
        (_repo_root() / p).resolve(),
        (bundle_path.parent / p).resolve(),
    ]
    for cand in candidates:
        if cand.exists():
            return cand
    return candidates[0]


def _deep_get(d: Dict[str, Any], keys: List[str], default: Any = None) -> Any:
    cur: Any = d
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


def _resolve_api_key(llm_cfg: Dict[str, Any]) -> str:
    api_key = str(llm_cfg.get("api_key") or "").strip()
    if api_key:
        return api_key
    env_name = str(llm_cfg.get("api_key_env") or "").strip()
    if env_name:
        import os

        return str(os.getenv(env_name, "")).strip()
    return ""


def _get_llm_json_fn(llm_cfg: Dict[str, Any]):
    from planning_agent_react_mm_next.llm_client import LLMConfig, OpenAICompatibleClient

    provider = str(llm_cfg.get("provider") or "openai")
    api_key = _resolve_api_key(llm_cfg)
    if provider != "vllm" and not api_key:
        raise RuntimeError("resident preprocess LLM api key is empty. Set llm.api_key or llm.api_key_env.")
    cli = OpenAICompatibleClient(
        LLMConfig(
            provider=provider,
            api_key=api_key if api_key else "EMPTY",
            model=str(llm_cfg.get("model") or "gpt-5.4-mini"),
            base_url=llm_cfg.get("base_url"),
            temperature=float(llm_cfg.get("temperature", 0.0)),
            max_tokens=int(llm_cfg.get("max_tokens", 4096)),
            timeout=int(llm_cfg.get("timeout", 120)),
            dump_api_io=bool(llm_cfg.get("dump_api_io", False)),
        )
    )

    def _call(prompt: str) -> Optional[Dict[str, Any]]:
        msgs = [
            {"role": "system", "content": "You are a precise JSON generator. Reply with JSON object only."},
            {"role": "user", "content": prompt},
        ]
        try:
            resp = cli.chat(msgs)
            text = cli.extract_text(resp)
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

    return _call


def _load_profile_records(path: Path) -> List[Dict[str, Any]]:
    data = _load_json(path)
    rows: List[Dict[str, Any]] = []
    if isinstance(data, list):
        for item in data:
            if isinstance(item, dict):
                rows.append(dict(item))
    elif isinstance(data, dict) and all(isinstance(v, dict) for v in data.values()):
        for k, v in data.items():
            row = dict(v)
            row.setdefault("id", k)
            rows.append(row)
    if not rows:
        raise RuntimeError(f"resident profile file empty/invalid: {path}")
    return rows


def _extract_aoi_id(row: Dict[str, Any], key: str) -> Optional[int]:
    v = row.get(key)
    if isinstance(v, int):
        return int(v)
    if isinstance(v, str) and v.strip().isdigit():
        return int(v.strip())
    if isinstance(v, dict):
        p = v.get("aoi_position")
        if isinstance(p, dict):
            aoi_id = p.get("aoi_id")
            if isinstance(aoi_id, int):
                return int(aoi_id)
            if isinstance(aoi_id, str) and aoi_id.strip().isdigit():
                return int(aoi_id.strip())
    return None


def _load_map(map_pickle_path: Path) -> Dict[str, Any]:
    # Some map pickles were written by NumPy 2.x and reference numpy._core.
    # NumPy 1.x exposes the same modules under numpy.core.
    try:
        import numpy.core as np_core  # type: ignore
        import numpy.core.multiarray as np_multiarray  # type: ignore
        import numpy.core.numeric as np_numeric  # type: ignore
        import sys

        sys.modules.setdefault("numpy._core", np_core)
        sys.modules.setdefault("numpy._core.multiarray", np_multiarray)
        sys.modules.setdefault("numpy._core.numeric", np_numeric)
    except Exception:
        pass
    with open(map_pickle_path, "rb") as f:
        obj = pickle.load(f)
    if not isinstance(obj, dict):
        raise RuntimeError(f"map pickle must contain dict: {map_pickle_path}")
    return obj


def _choose_locations(rows: List[Dict[str, Any]], map_obj: Dict[str, Any], seed: int) -> List[Dict[str, Any]]:
    aois = map_obj.get("aois") or []
    home_ids: List[int] = []
    home_weights: List[float] = []
    work_ids: List[int] = []
    work_weights: List[float] = []

    for aoi in aois:
        aid = aoi.get("id")
        if not isinstance(aid, int):
            continue
        lu = str(aoi.get("urban_land_use") or "")
        if lu == "RS":
            area = float(aoi.get("area") or 1.0)
            home_ids.append(aid)
            home_weights.append(area if area > 0 else 1.0)
        elif lu in WORK_TYPE_WEIGHTS:
            work_ids.append(aid)
            work_weights.append(float(WORK_TYPE_WEIGHTS[lu]))

    if not home_ids:
        raise RuntimeError("No RS AOI found for resident home assignment.")
    if not work_ids:
        raise RuntimeError("No valid work AOI found for resident work assignment.")

    rng = random.Random(int(seed))
    out: List[Dict[str, Any]] = []
    for row in rows:
        r = dict(row)
        hid = _extract_aoi_id(r, "home")
        wid = _extract_aoi_id(r, "work")
        if hid is None:
            hid = int(rng.choices(home_ids, weights=home_weights, k=1)[0])
        if wid is None:
            wid = int(rng.choices(work_ids, weights=work_weights, k=1)[0])
        r["home"] = {"aoi_position": {"aoi_id": int(hid)}}
        r["work"] = {"aoi_position": {"aoi_id": int(wid)}}
        out.append(r)
    return out


def _profiles_to_dict(rows: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for row in rows:
        rid = row.get("id")
        if rid is None:
            continue
        out[str(rid)] = row
    return out


def _aoi_landuse_map(map_obj: Dict[str, Any]) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for aoi in map_obj.get("aois") or []:
        aid = aoi.get("id")
        if aid is not None:
            out[str(aid)] = str(aoi.get("urban_land_use") or "")
    return out


def _build_preference_cache_with_llm(
    resident_profiles: Dict[str, Dict[str, Any]],
    map_obj: Dict[str, Any],
    llm_cfg: Dict[str, Any],
) -> Dict[str, Any]:
    llm_json_fn = _get_llm_json_fn(llm_cfg)
    batch_size = max(1, int(llm_cfg.get("preference_batch_size", 20)))
    aoi_landuse = _aoi_landuse_map(map_obj)
    payload = []
    for rid in sorted(resident_profiles.keys(), key=lambda x: int(x) if x.isdigit() else x):
        profile = resident_profiles[rid]
        home_id = _extract_aoi_id(profile, "home")
        work_id = _extract_aoi_id(profile, "work")
        payload.append(
            {
                "resident_id": rid,
                "home_aoi_id": home_id,
                "home_landuse": aoi_landuse.get(str(home_id), "") if home_id is not None else "",
                "work_aoi_id": work_id,
                "work_landuse": aoi_landuse.get(str(work_id), "") if work_id is not None else "",
                "background": profile,
            }
        )

    landuse_full = {
        "BS": "Business Area",
        "RC": "Recreation and Sports Center",
        "PK": "Park",
        "OF": "Office",
        "SC": "School",
        "HS": "Hospital",
    }
    prefs: Dict[str, List[str]] = {}
    for start in range(0, len(payload), batch_size):
        batch = payload[start : start + batch_size]
        batch_ids = [str(item["resident_id"]) for item in batch]
        prompt = (
            "You are an urban planning analysis assistant. For each resident, infer the most needed service land-use types "
            "from the resident profile, fixed home/work locations, and the home/work land-use context.\n"
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
            f"Required resident_ids: {batch_ids}\n"
            f"Input data:\n{json.dumps(batch, ensure_ascii=False)}"
        )
        obj = llm_json_fn(prompt) or {}
        llm_map = obj.get("resident_preferences") if isinstance(obj, dict) else None
        if not isinstance(llm_map, dict):
            raise RuntimeError(f"LLM did not return resident_preferences for batch resident_ids={batch_ids}")

        for rid in batch_ids:
            arr = llm_map.get(rid)
            if arr is None and str(rid).isdigit():
                arr = llm_map.get(int(rid))
            if not isinstance(arr, list):
                raise RuntimeError(f"LLM preferences missing resident_id={rid}")
            cleaned: List[str] = []
            for v in arr:
                lu = str(v)
                if lu in SATISFACTION_CANDIDATE_TYPES and lu not in cleaned:
                    cleaned.append(lu)
                if len(cleaned) >= 6:
                    break
            if not cleaned:
                raise RuntimeError(f"LLM preferences empty/invalid for resident_id={rid}: {arr}")
            prefs[str(rid)] = cleaned

    return {
        "resident_preferences": prefs,
        "source": "resident_preprocess_llm_profile_location",
        "resident_count": len(prefs),
        "candidate_types": list(SATISFACTION_CANDIDATE_TYPES),
        "model": str(llm_cfg.get("model") or "gpt-5.4-mini"),
        "batch_size": batch_size,
    }


def _normalize_rows_for_count(rows: List[Dict[str, Any]], resident_count: int) -> List[Dict[str, Any]]:
    if resident_count <= 0:
        raise ValueError("resident_count must be positive")
    if len(rows) < resident_count:
        raise ValueError(
            f"raw profile count {len(rows)} is smaller than requested resident_count {resident_count}"
        )
    selected = [dict(row) for row in rows[:resident_count]]
    seen = set()
    for i, row in enumerate(selected, start=1):
        rid = row.get("id")
        if rid is None or str(rid) in seen:
            row["id"] = i
            rid = i
        seen.add(str(rid))
    return selected


def build_preprocess(
    raw_profiles_path: Path,
    map_pickle_path: Path,
    resident_count: int,
    output_dir: Path,
    seed: int = 42,
    map_pb_path: Optional[Path] = None,
    llm_cfg: Optional[Dict[str, Any]] = None,
    name: str = "",
) -> ResidentPreprocess:
    raw_profiles_path = raw_profiles_path.resolve()
    map_pickle_path = map_pickle_path.resolve()
    map_pb_path = map_pb_path.resolve() if map_pb_path else map_pickle_path.with_suffix(".pb").resolve()
    if not raw_profiles_path.exists():
        raise FileNotFoundError(f"raw_profiles_path missing: {raw_profiles_path}")
    if not map_pickle_path.exists():
        raise FileNotFoundError(f"map_pickle_path missing: {map_pickle_path}")

    rows = _normalize_rows_for_count(_load_profile_records(raw_profiles_path), int(resident_count))
    map_obj = _load_map(map_pickle_path)
    profiles = _choose_locations(rows, map_obj, int(seed))
    profile_map = _profiles_to_dict(profiles)
    pref_obj = _build_preference_cache_with_llm(profile_map, map_obj, llm_cfg or {})
    preferences = pref_obj.get("resident_preferences") if isinstance(pref_obj, dict) else None
    if not isinstance(preferences, dict):
        raise RuntimeError("resident preference preprocessing did not return resident_preferences")

    output_dir.mkdir(parents=True, exist_ok=True)
    short = hashlib.sha1(
        json.dumps(
            {
                "raw_profiles": str(raw_profiles_path),
                "map_pickle": str(map_pickle_path),
                "resident_count": int(resident_count),
                "seed": int(seed),
                "profile_hash": _sha1_json_obj(profiles),
            },
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()[:12]
    stem = name.strip() or f"resident_preprocess_{short}"
    bundle_path = output_dir / f"{stem}.json"
    profiles_path = output_dir / f"{stem}_profiles.json"
    preferences_path = output_dir / f"{stem}_preferences.json"

    metadata = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "seed": int(seed),
        "resident_count": int(resident_count),
        "raw_profiles_path": _rel_to_repo(raw_profiles_path),
        "raw_profiles_sha1": _sha1_file(raw_profiles_path),
        "map_pickle_path": _rel_to_repo(map_pickle_path),
        "map_pickle_abs": _rel_to_repo(map_pickle_path),
        "map_pickle_sha1": _sha1_file(map_pickle_path),
        "map_pb_path": _rel_to_repo(map_pb_path) if map_pb_path.exists() else "",
        "map_pb_abs": _rel_to_repo(map_pb_path) if map_pb_path.exists() else "",
        "map_pb_sha1": _sha1_file(map_pb_path) if map_pb_path.exists() else "",
        "preference_source": str(pref_obj.get("source") or ""),
        "preference_model": str((llm_cfg or {}).get("model") or "gpt-5.4-mini"),
        "profiles_path": _rel_to_repo(profiles_path),
        "preferences_path": _rel_to_repo(preferences_path),
        "profiles_sha1": _sha1_json_obj(profiles),
        "preferences_sha1": _sha1_json_obj(pref_obj),
    }
    bundle = {
        "metadata": metadata,
        "resident_profiles": profiles,
        "resident_preferences": preferences,
        "preference_cache": pref_obj,
    }
    _save_json(profiles_path, profiles)
    _save_json(preferences_path, pref_obj)
    _save_json(bundle_path, bundle)
    return ResidentPreprocess(
        bundle_path=bundle_path.resolve(),
        profiles_path=profiles_path.resolve(),
        preferences_path=preferences_path.resolve(),
        resident_profiles=profile_map,
        resident_preferences=preferences,
        bundle=bundle,
    )


def _load_bundle(bundle_path: Path) -> ResidentPreprocess:
    bundle_path = bundle_path.resolve()
    bundle = _load_json(bundle_path)
    if not isinstance(bundle, dict):
        raise ValueError(f"resident preprocess bundle must be JSON object: {bundle_path}")
    meta = bundle.get("metadata") if isinstance(bundle.get("metadata"), dict) else {}
    profiles_path = _resolve_metadata_path(str(meta.get("profiles_path") or ""), bundle_path)
    preferences_path = _resolve_metadata_path(str(meta.get("preferences_path") or ""), bundle_path)
    if not profiles_path.exists():
        raise FileNotFoundError(f"resident preprocess profiles sidecar missing: {profiles_path}")
    if not preferences_path.exists():
        raise FileNotFoundError(f"resident preprocess preferences sidecar missing: {preferences_path}")
    profile_rows = _load_json(profiles_path)
    pref_obj = _load_json(preferences_path)
    if not isinstance(profile_rows, list):
        raise ValueError(f"resident preprocess profiles sidecar must be list: {profiles_path}")
    if not isinstance(pref_obj, dict) or not isinstance(pref_obj.get("resident_preferences"), dict):
        raise ValueError(f"resident preprocess preferences sidecar invalid: {preferences_path}")
    profile_map = _profiles_to_dict([dict(x) for x in profile_rows if isinstance(x, dict)])
    preferences = {
        str(k): [str(x) for x in v]
        for k, v in (pref_obj.get("resident_preferences") or {}).items()
        if isinstance(v, list)
    }
    return ResidentPreprocess(
        bundle_path=bundle_path,
        profiles_path=profiles_path,
        preferences_path=preferences_path,
        resident_profiles=profile_map,
        resident_preferences=preferences,
        bundle=bundle,
    )


def load_and_validate_preprocess(
    preprocess_path: Path,
    map_pickle_path: Path,
    resident_count: int,
    map_pb_path: Optional[Path] = None,
) -> ResidentPreprocess:
    if not preprocess_path:
        raise ValueError(
            "resident_preprocess_path is required. Run resident preprocessing first and set paths.resident_preprocess_path."
        )
    rp = _load_bundle(preprocess_path)
    meta = rp.bundle.get("metadata") if isinstance(rp.bundle.get("metadata"), dict) else {}
    expected_count = int(meta.get("resident_count", -1))
    if expected_count != int(resident_count):
        raise ValueError(
            f"resident preprocess mismatch: resident_count={expected_count}, runtime resident_count={resident_count}. "
            "Run resident_preprocess/resident_preprocess.py for this resident count."
        )
    cfg_pickle = map_pickle_path.resolve()
    bundle_pickle = _resolve_metadata_path(str(meta.get("map_pickle_path") or meta.get("map_pickle_abs") or ""), rp.bundle_path)
    bundle_pickle_sha1 = str(meta.get("map_pickle_sha1") or "").strip()
    cfg_pickle_sha1 = _sha1_file(cfg_pickle) if cfg_pickle.exists() else ""
    if cfg_pickle != bundle_pickle and (not bundle_pickle_sha1 or cfg_pickle_sha1 != bundle_pickle_sha1):
        raise ValueError(
            f"resident preprocess mismatch: map_pickle={bundle_pickle}, runtime map_pickle={cfg_pickle}. "
            "Run resident preprocessing for this map."
        )
    if map_pb_path is not None:
        bundle_pb_raw = str(meta.get("map_pb_path") or meta.get("map_pb_abs") or "").strip()
        if bundle_pb_raw:
            bundle_pb = _resolve_metadata_path(bundle_pb_raw, rp.bundle_path)
            cfg_pb = map_pb_path.resolve()
            bundle_pb_sha1 = str(meta.get("map_pb_sha1") or "").strip()
            cfg_pb_sha1 = _sha1_file(cfg_pb) if cfg_pb.exists() else ""
            if cfg_pb != bundle_pb and (not bundle_pb_sha1 or cfg_pb_sha1 != bundle_pb_sha1):
                raise ValueError(
                    f"resident preprocess mismatch: map_pb={bundle_pb}, runtime map_pb={cfg_pb}. "
                    "Run resident preprocessing for this map."
                )
    if len(rp.resident_profiles) != int(resident_count):
        raise ValueError(
            f"resident preprocess sidecar contains {len(rp.resident_profiles)} residents, expected {resident_count}"
        )
    missing_pref = sorted(set(rp.resident_profiles.keys()) - set(rp.resident_preferences.keys()))
    if missing_pref:
        raise ValueError(
            f"resident preprocess preferences missing {len(missing_pref)} resident ids, first={missing_pref[:5]}"
        )
    return rp


def main() -> None:
    parser = argparse.ArgumentParser(description="Build unified resident profile/location/preference preprocess bundle.")
    parser.add_argument("--config", default="", help="YAML config. CLI args override config values.")
    parser.add_argument("--profiles", default="", help="Raw resident profile JSON.")
    parser.add_argument("--map-pickle", default="", help="Map pickle used for location assignment.")
    parser.add_argument("--map-pb", default="", help="Optional map pb path for strict runtime matching.")
    parser.add_argument("--resident-count", default=0, type=int, help="Number of residents to keep/preprocess.")
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--out-dir", default="resident_preprocess")
    parser.add_argument("--name", default="", help="Optional output stem without .json.")
    args = parser.parse_args()

    repo_root = Path.cwd()
    cfg = _load_yaml(_resolve_path(args.config, repo_root)) if args.config else {}

    def cfg_or_arg(arg_name: str, *keys: str, default: Any = "") -> Any:
        v = getattr(args, arg_name)
        if arg_name == "resident_count" and int(v or 0) <= 0:
            return _deep_get(cfg, list(keys), default)
        if v not in (None, ""):
            return v
        return _deep_get(cfg, list(keys), default)

    llm_cfg = dict(cfg.get("llm") or {})
    result = build_preprocess(
        raw_profiles_path=_resolve_path(cfg_or_arg("profiles", "paths", "profiles"), repo_root),
        map_pickle_path=_resolve_path(cfg_or_arg("map_pickle", "paths", "map_pickle"), repo_root),
        map_pb_path=_resolve_path(cfg_or_arg("map_pb", "paths", "map_pb"), repo_root) if cfg_or_arg("map_pb", "paths", "map_pb") else None,
        resident_count=int(cfg_or_arg("resident_count", "resident", "count")),
        output_dir=_resolve_path(cfg_or_arg("out_dir", "paths", "out_dir", default="resident_preprocess"), repo_root),
        seed=int(cfg_or_arg("seed", "resident", "seed", default=42)),
        llm_cfg=llm_cfg,
        name=str(cfg_or_arg("name", "output", "name", default="") or ""),
    )
    print(
        json.dumps(
            {
                "ok": True,
                "bundle_path": str(result.bundle_path),
                "profiles_path": str(result.profiles_path),
                "preferences_path": str(result.preferences_path),
                "resident_count": len(result.resident_profiles),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
