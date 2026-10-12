# LiPUP-MA

Minimal code release for **LiPUP-MA: A Residential Experience-centric Multi-Agent Framework for Living-in-the-loop Participatory Urban Planning**.

The pipeline alternates resident simulation, a plan-centric graph-based experience bank, and a three-stage spatial planner with conflict-aware branching, reflection and skill/experience evolution. A final simulation evaluates the revised map.

## Quick start

Use Python 3.11 on Linux x86-64 (the simulator also provides a macOS ARM64 binary). Run all commands from this directory:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
export OPENAI_API_KEY="your-key"
python resident_preprocess/resident_preprocess.py --config resident_preprocess/config.yaml
python sim_script.py sim_script_runtime.yaml
```

The first command assigns home/work locations and infers resident preferences. The second runs the simulation → planning → evaluation loop. Internet access is needed for the configured LLM, the simulator binary (automatically downloaded on first use), and the FastEmbed model. No GPU-specific package is required by this release.

The example uses **10 residents, 0.05 simulated days per simulation, one planning iteration, and up to two planning cycles**. These are small demonstration settings, not the paper's experimental settings. Change resident counts in both `resident_preprocess/config.yaml` and `sim_script_runtime.yaml`, then regenerate preprocessing. The bundled profile file contains only 10 records.

Configure the simulation model in `sim_script_runtime.yaml`, the planner model in `planning_agent_pipeline_mm_refined/config.yaml`, and preference inference in `resident_preprocess/config.yaml`. The default is an OpenAI-compatible `gpt-4o-mini` setup; set `base_url` and `model` for another compatible service. All credentials are read from `OPENAI_API_KEY`. Evaluation inherits the planner's LLM configuration.

## Offline check

```bash
python -B scripts/smoke_test.py
```

This checks protobuf/pickle consistency, resident location assignment and bundle validation, simulation configuration, and the three-stage planner through a committed, materialized map edit. Preferences and planner responses are mocked, with a temporary map-only graph. It makes no live LLM calls, does not run resident simulation, and deletes its temporary outputs. It is a wiring check, not an evaluation of planning quality.

## Files

- `sim_script.py`: simulation/planning loop and evaluation.
- `agentsociety/`: the project-modified simulation runtime and its required dependencies, with the upstream license retained.
- `resident_preprocess/`: home/work assignment and preference inference.
- `planning_agent_pipeline_mm_refined/`: planner entry point, prompts and configuration.
- `planning_agent_refined_core/`: stages, state tree, branching and self-evolution.
- `planning_agent_react_mm_next/`: shared map, experience graph, tool and metric implementations; this is a dependency of the refined planner.
- `planning_agent/sim4pup_agent/`: only the two map conversion/loading helpers required by the planner.
- `data/example/`: one initial Berlin Adlershof map and 10 example resident profiles.

All generated bundles, logs, simulator downloads and runs go under `outputs/`, which is ignored by Git. Generated GeoJSON next to the example map is also ignored. The experience graph is produced by simulation; no historical graphs, plans or experiment outputs are bundled. For a standalone planning run, set `paths.graph_json`, `resident_profiles` and `resident_preferences` in the planner configuration to your generated inputs, then run `python -m planning_agent_pipeline_mm_refined.run_planning`.

## Data and third-party attribution

The initial Berlin Adlershof example map contains: approximately 68 KiB protobuf plus 117 KiB pickle. Both formats are retained because simulation and planning consume different representations. The 10 resident profiles use pseudonymous identifiers and are demonstration inputs, not a representative Berlin population.

The simulation runtime derives from AgentSociety 1.5.6 and contains project-specific resident behavior and experience graph changes. The release also removes the import-time endpoint probe and ensures the simulator download directory exists. Its Apache-2.0 license is preserved in `agentsociety/LICENSE`. The original map's source/license was not documented in the provided map directory; this release does not assign a new license to it. No license is assigned here to the project's original code or profile data.

Third-party notices identify the upstream dependency authors, not the authors of this submission.
