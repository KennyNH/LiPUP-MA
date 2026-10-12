"""Offline integration check; temporary synthetic feedback is not research data."""
from pathlib import Path
import json
import os
import pickle
import shutil
import subprocess
import sys
import tempfile
from unittest.mock import patch

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    from resident_preprocess.resident_preprocess import build_preprocess, load_and_validate_preprocess
    from map_utils import dict2pb_path
    from pycityproto.city.map.v2.map_pb2 import Map
    import sim_script

    with tempfile.TemporaryDirectory(prefix="lipup-smoke-") as tmp:
        work = Path(tmp)
        for ext in ("pb", "pickle"):
            shutil.copyfile(ROOT / f"data/example/berlin_0.{ext}", work / f"berlin_0.{ext}")
        pb, pkl = work / "berlin_0.pb", work / "berlin_0.pickle"
        with pkl.open("rb") as f:
            map_data = pickle.load(f)
        dict2pb_path(pkl, work / "roundtrip.pb")
        assert Map.FromString(pb.read_bytes()) == Map.FromString((work / "roundtrip.pb").read_bytes())
        # Mock only preference inference; use real location assignment and bundle validation.
        def preferences(profiles, map_obj, llm_cfg):
            return {"resident_preferences": {rid: ["PK", "SC", "HS"] for rid in profiles}, "source": "smoke_mock"}
        with patch("resident_preprocess.resident_preprocess._build_preference_cache_with_llm", preferences):
            bundle = build_preprocess(ROOT / "data/example/profiles.json", pkl, 10, work, map_pb_path=pb, name="berlin")
        load_and_validate_preprocess(bundle.bundle_path, map_pickle_path=pkl, map_pb_path=pb, resident_count=10)
        cfg = yaml.safe_load((ROOT / "sim_script_runtime.yaml").read_text())
        cfg["paths"]["resident_profiles_path"] = str(bundle.profiles_path)
        with patch.dict(os.environ, {"OPENAI_API_KEY": "smoke-placeholder"}):
            config = sim_script.add_map_config(str(pb), cfg)
            assert config.llm[0].api_key == "smoke-placeholder"
        graph = {"directed": True, "multigraph": False, "graph": {}, "nodes": [
            {"id": a["id"], "node_type": "aoi_node", "urban_land_use": a.get("urban_land_use", "")}
            for a in map_data["aois"]], "links": []}
        (work / "graph.json").write_text(json.dumps(graph))
        planner = yaml.safe_load((ROOT / "planning_agent_pipeline_mm_refined/config.yaml").read_text())
        planner["paths"].update(pb_map=str(pb), pickle_map=str(pkl), graph_json=str(work / "graph.json"),
                                resident_profiles=str(bundle.profiles_path), resident_preferences=str(bundle.preferences_path), outputs=str(work / "planning"))
        planner["run"]["skip_simulation_assessment"] = True
        (work / "planner.yaml").write_text(yaml.safe_dump(planner))
        env = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1", "MPLCONFIGDIR": str(work / "mpl"), "LIPUP_CACHE_DIR": str(work / "cache")}
        subprocess.run([sys.executable, "-B", "-m", "planning_agent_pipeline_mm_refined.run_planning",
                        "--config", str(work / "planner.yaml"), "--mock-smoke", "--quiet"], cwd=ROOT, env=env, check=True)
        summaries = list((work / "planning").glob("*/summary.json"))
        assert len(summaries) == 1
        summary = json.loads(summaries[0].read_text())
        assert summary["best_plan"], "Mock planner did not commit a map edit"
        materialized = summaries[0].parent / "best_plan_materialized/best_plan.pickle"
        with materialized.open("rb") as f:
            changed = {str(a["id"]): a["urban_land_use"] for a in pickle.load(f)["aois"]}
        assert all(changed[str(aid)] == lu for aid, lu in summary["best_plan"].items())
    print("PASS: map roundtrip, resident preprocessing, simulation config, three-stage planner and materialized map. No live LLM/simulation was run.")


if __name__ == "__main__":
    main()
