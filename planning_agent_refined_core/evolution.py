"""Simple stage-wise self-evolving memory notebook."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from threading import RLock
from typing import Any, Dict, List, Optional

try:
    from planning_agent_react_mm_next.utils import save_json
except ImportError:
    from ..planning_agent_react_mm_next.utils import save_json  # type: ignore


@dataclass
class StageNotebook:
    experience: str = ""
    skill: str = ""


class EvolutionNotebook:
    _CONSERVATIVE_MEMORY_PATTERNS = (
        r"\bno[-\s]?change\b",
        r"\bdefer(?:red|ring)?\b",
        r"\bstatus\s+quo\b",
        r"\bpreserve\s+(?:the\s+)?(?:current\s+)?(?:land[-\s]?use|mix|state|configuration)\b",
        r"\bkeep\s+(?:the\s+)?(?:current\s+)?(?:land[-\s]?use|mix|state|configuration|unchanged)\b",
        r"\bavoid\s+(?:unsupported|speculative|forcing)\b",
        r"\bdo\s+not\s+force\b",
        r"\bprefer\s+(?:caution|no[-\s]?change|preserv)",
        r"\bcaution\b",
        r"\bevidence\s+(?:is\s+)?(?:sparse|weak|insufficient)\b",
        r"\binsufficient\s+evidence\b",
        r"\bresident[-\s]?backed\s+conversion\b",
        r"\blegal(?:ity)?\s+or\s+metric\s+shifts\s+clearly\s+justify\b",
        r"证据不足",
        r"保守",
        r"不改",
        r"不做修改",
        r"延后",
        r"维持现状",
        r"保持现状",
    )
    _RESULT_BIAS_MEMORY_PATTERNS = (
        r"\bpreserv(?:e|ing)?\s+(?:the\s+)?(?:service|business|commercial|recreation|recreational)[-\s]?(?:bearing|support(?:ing)?|oriented)?\s+(?:bs|rc|pk|hs|sc|of|id)(?:\s*(?:/|,|and|or)\s*(?:bs|rc|pk|hs|sc|of|id))*\s*(?:aoi|aois|parcel|parcels|node|nodes|anchor|anchors|fabric|strip|uses?)\b",
        r"\bkeep\s+(?:the\s+)?(?:service|business|commercial|recreation|recreational)[-\s]?(?:bearing|support(?:ing)?|oriented)?\s+(?:bs|rc|pk|hs|sc|of|id)(?:\s*(?:/|,|and|or)\s*(?:bs|rc|pk|hs|sc|of|id))*\s*(?:aoi|aois|parcel|parcels|node|nodes|anchor|anchors|fabric|strip|uses?)\s+(?:intact|unchanged|stable)\b",
        r"\bpreserv(?:e|ing)?\s+(?:the\s+)?(?:service|business|commercial|recreation|recreational|bs|rc|pk|hs|sc|of|id)[-\s]?(?:bearing|support(?:ing)?|oriented)?\s*(?:aoi|aois|parcel|parcels|node|nodes|anchor|anchors|fabric|strip|uses?)\b",
        r"\bkeep\s+(?:the\s+)?(?:service|business|commercial|recreation|recreational|bs|rc|pk|hs|sc|of|id)[-\s]?(?:bearing|support(?:ing)?|oriented)?\s*(?:aoi|aois|parcel|parcels|node|nodes|anchor|anchors|fabric|strip|uses?)\s+(?:intact|unchanged|stable)\b",
        r"\bprotect\s+(?:the\s+)?(?:service|business|commercial|recreation|recreational|bs|rc|pk|hs|sc|of|id)[-\s]?(?:bearing|support(?:ing)?|oriented)?\s*(?:aoi|aois|parcel|parcels|node|nodes|anchor|anchors|fabric|strip|uses?)\b",
        r"\b(?:bs|rc|pk|hs|sc|of|id)\s+(?:should|must)\s+(?:remain|stay|be\s+preserved|be\s+kept)\b",
        r"\b(?:convert|conversion|target|prioriti[sz]e|prefer)\s+(?:[^.]{0,80}\s+)?(?:to|toward|as)\s+(?:bs|rc|pk|hs|sc|of|id)\b",
        r"\b(?:bs|rc|pk|hs|sc|of|id)[-\s]?(?:first|heavy|oriented|priority|bias)\b",
        r"保留.*(?:服务|商业|休闲|公园|学校|医院|办公|工业|节点|锚点|地块)",
        r"保护.*(?:服务|商业|休闲|公园|学校|医院|办公|工业|节点|锚点|地块)",
        r"优先.*(?:改为|转为).*(?:BS|RC|PK|HS|SC|OF|ID|公园|商业|医院|学校|办公|工业)",
    )

    def __init__(
        self,
        output_dir: Path,
        client: Optional[Any],
        enabled: bool = True,
        bootstrap_from: Optional[Path] = None,
    ):
        self.output_dir = output_dir
        self.client = client
        self.enabled = bool(enabled)
        self.self_evolving_dir = self.output_dir / "self_evolving"
        self.self_evolving_dir.mkdir(parents=True, exist_ok=True)
        self.stage_memory_path = self.self_evolving_dir / "stage_memory.json"
        self.tool_events_path = self.self_evolving_dir / "tool_events.json"
        self.consolidation_log_path = self.self_evolving_dir / "consolidation_llm_calls.jsonl"
        self.stage_books: Dict[str, StageNotebook] = self._default_stage_books()
        self.tool_events: List[Dict[str, Any]] = []
        self._lock = RLock()
        self._bootstrap_from(bootstrap_from)
        self._sanitize_all_stage_books()
        self._persist_stage_books()
        self._persist_tool_events()

    @staticmethod
    def _default_stage_books() -> Dict[str, StageNotebook]:
        return {
            "phase1": StageNotebook(
                experience="When the first scan is still uncertain, compare global hotspots, urgency levels, and at least one map view before narrowing the focus list.",
                skill="Workflow: scan globally, name 2-4 priority focuses, explain why each focus matters at both city scale and local scale.",
            ),
            "phase2": StageNotebook(
                experience="When evidence is fragmented, combine structured AOI queries, distance checks, and one compact regional summary before proposing edits.",
                skill="Workflow: inspect focus AOIs, gather nearby and mobility-linked evidence, then compress support, risks, and unresolved conflicts into one evidence bundle.",
            ),
            "phase3": StageNotebook(
                experience="When edits look promising, modify in small coherent batches, read metrics immediately, and treat rollback as a rare but explicit decision.",
                skill="Workflow: judge conflicts first, execute one branch-safe renovation action, reflect on metrics and side effects, then commit/continue while any global focus remains unresolved; use finish only when every current focus is done.",
            ),
        }

    def stage_block(self, stage_name: str) -> Dict[str, str]:
        book = self.stage_books.get(stage_name, StageNotebook())
        return {
            "experience": (book.experience or "").strip(),
            "skill": (book.skill or "").strip(),
            "tool_memory": self.tool_block(),
        }

    def snapshot(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "stage_books": {
                    name: {"experience": book.experience, "skill": book.skill}
                    for name, book in self.stage_books.items()
                },
                "tool_events": [dict(item) for item in self.tool_events],
            }

    @classmethod
    def stage_block_from_snapshot(cls, snapshot: Optional[Dict[str, Any]], stage_name: str) -> Dict[str, str]:
        stage_books = snapshot.get("stage_books") if isinstance(snapshot, dict) else {}
        entry = stage_books.get(stage_name) if isinstance(stage_books, dict) else {}
        experience = str(entry.get("experience") or "").strip() if isinstance(entry, dict) else ""
        skill = str(entry.get("skill") or "").strip() if isinstance(entry, dict) else ""
        tool_events = snapshot.get("tool_events") if isinstance(snapshot, dict) else []
        return {
            "experience": experience,
            "skill": skill,
            "tool_memory": cls.tool_block_from_events(tool_events if isinstance(tool_events, list) else []),
        }

    def tool_block(self) -> str:
        return self.tool_block_from_events(self.tool_events)

    @staticmethod
    def tool_block_from_events(tool_events: List[Dict[str, Any]]) -> str:
        if not tool_events:
            return "No evolved-tool notes recorded yet."
        recent = tool_events[-4:]
        lines = []
        for item in recent:
            operation = str(item.get("operation") or "").strip()
            op_prefix = f"{operation} " if operation else ""
            lines.append(
                f"- [{item.get('kind')}] {op_prefix}{item.get('title') or item.get('gap') or item.get('proposal_id')}: "
                f"{item.get('expected_value') or item.get('description') or item.get('note') or ''}".strip()
            )
        return "\n".join(lines[:4])

    def record_tool_events(self, events: List[Dict[str, Any]]) -> None:
        if not events:
            return
        with self._lock:
            self.tool_events.extend(events)
            self._persist_tool_events()

    def maybe_refresh(
        self,
        stage_name: str,
        stage_trace: List[Dict[str, Any]],
        stage_result: Dict[str, Any],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, str]:
        """Refresh the global stage memory once after a stage finishes.

        Input:
        - the finished stage trajectory
        - the final structured stage result
        - the existing global memory for this stage

        Output:
        - updated global `experience` and `skill` for this stage only

        If no LLM client is available, this method simply returns the current memories.
        """
        current = self.stage_books.get(stage_name, StageNotebook())
        if not self.enabled or self.client is None:
            return {"experience": current.experience, "skill": current.skill}
        condensed_trace = stage_trace[-20:]
        system_prompt = "You are a precise JSON generator. Reply with JSON only."
        prompt = (
            "You are maintaining the global memory for one planning stage in a multi-stage planning agent.\n"
            "This refresh happens exactly once after the stage is complete.\n"
            "Use the finished stage trajectory to decide whether the global stage memory should change.\n"
            "Definitions:\n"
            "- experience: concise reusable decision rules, failure-avoidance tips, or execution heuristics for this stage.\n"
            "- skill: concise reusable SOP or workflow structure for this stage.\n"
            "Requirements:\n"
            "- Keep both concise, practical, and more general than the current case.\n"
            "- Preserve useful prior memory unless the new stage trajectory gives a better replacement.\n"
            "- Memory may only describe process skills: tool-use order, evidence checks, metric-comparison workflow, schema repair, retry handling, and search discipline.\n"
            "- Do not record land-use outcome preferences or AOI-type protection rules. Never teach that any land-use family, service node, parcel type, or anchor should generally be preserved, protected, prioritized, or converted.\n"
            "- Memory must help the agent discover metric-sensitive, executable next actions without biasing which land-use result to choose.\n"
            "- Never write conservative rules such as preferring no-change, deferring edits, preserving the current mix, or avoiding edits because evidence is sparse/weak/insufficient.\n"
            "- If the stage made no useful metric-sensitive change, extract an exploration/search lesson instead of a no-change lesson, or keep the previous memory unchanged.\n"
            "- Do not copy case-specific AOI ids, exact coordinates, or one-off facts unless they clearly generalize.\n"
            "- If the new stage adds no reusable knowledge, keep the previous memory unchanged.\n"
            "- experience <= 90 words; skill <= 120 words.\n"
            "Return JSON only:\n"
            "{\"experience\":\"...\",\"skill\":\"...\"}\n"
            f"Stage: {stage_name}\n"
            f"Current experience: {current.experience}\n"
            f"Current skill: {current.skill}\n"
            f"Finished stage trajectory: {json.dumps(condensed_trace, ensure_ascii=False)}\n"
            f"Stage result: {json.dumps(stage_result, ensure_ascii=False)}"
        )
        text = ""
        try:
            resp = self.client.chat(
                [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt},
                ]
            )
            text = self.client.extract_text(resp).strip()
            updated = self._extract_json_object(text)
        except Exception:
            updated = None
        self._append_consolidation_log(
            {
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "stage_name": stage_name,
                "metadata": metadata or {},
                "request": {
                    "system": system_prompt,
                    "user": prompt,
                },
                "response_text": text,
                "parsed_response": updated if isinstance(updated, dict) else None,
                "success": isinstance(updated, dict),
            }
        )
        if not isinstance(updated, dict):
            return {"experience": current.experience, "skill": current.skill}
        exp, skill = self._sanitize_stage_update(
            stage_name=stage_name,
            experience=updated.get("experience") or current.experience,
            skill=updated.get("skill") or current.skill,
            current=current,
        )
        with self._lock:
            self.stage_books[stage_name] = StageNotebook(experience=exp, skill=skill)
            self._persist_stage_books()
        return {"experience": exp, "skill": skill}

    def consolidate_wave_stage(
        self,
        stage_name: str,
        stage_payloads: List[Dict[str, Any]],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, str]:
        current = self.stage_books.get(stage_name, StageNotebook())
        if not stage_payloads:
            return {"experience": current.experience, "skill": current.skill}
        if not self.enabled or self.client is None:
            return {"experience": current.experience, "skill": current.skill}
        condensed_items = []
        for item in stage_payloads:
            if not isinstance(item, dict):
                continue
            condensed_items.append(
                {
                    "branch_id": item.get("branch_id"),
                    "branch_label": item.get("branch_label"),
                    "cycle_index": item.get("cycle_index"),
                    "trace": list(item.get("trace") or [])[-12:],
                    "result": item.get("result") if isinstance(item.get("result"), dict) else {},
                }
            )
        system_prompt = "You are a precise JSON generator. Reply with JSON only."
        prompt = (
            "You are maintaining the global memory for one planning stage in a multi-branch planning agent.\n"
            "All listed branch runs come from the same branch-depth wave, so they must share one aligned global memory after consolidation.\n"
            "Use the branch outcomes together to update one shared stage memory.\n"
            "Definitions:\n"
            "- experience: concise reusable decision rules, failure-avoidance tips, or heuristics for this stage.\n"
            "- skill: concise reusable SOP or workflow structure for this stage.\n"
            "Requirements:\n"
            "- Keep both concise and reusable.\n"
            "- Preserve useful prior memory unless the wave gives a better shared replacement.\n"
            "- Shared memory may only describe process skills: tool-use order, evidence checks, metric-comparison workflow, schema repair, retry handling, and search discipline.\n"
            "- Do not record land-use outcome preferences or AOI-type protection rules. Never teach that any land-use family, service node, parcel type, or anchor should generally be preserved, protected, prioritized, or converted.\n"
            "- Shared memory must help the agent discover metric-sensitive, executable next actions without biasing which land-use result to choose.\n"
            "- Never write conservative rules such as preferring no-change, deferring edits, preserving the current mix, or avoiding edits because evidence is sparse/weak/insufficient.\n"
            "- If branches produced no useful metric-sensitive change, extract an exploration/search lesson instead of a no-change lesson, or keep the previous memory unchanged.\n"
            "- Integrate all branches rather than following only one branch.\n"
            "- If branches disagree, keep the shared memory broad enough to cover both, or stay with the previous memory.\n"
            "- Do not copy one-off AOI ids or coordinates unless clearly generalizable.\n"
            "- experience <= 90 words; skill <= 120 words.\n"
            "Return JSON only:\n"
            "{\"experience\":\"...\",\"skill\":\"...\"}\n"
            f"Stage: {stage_name}\n"
            f"Current experience: {current.experience}\n"
            f"Current skill: {current.skill}\n"
            f"Wave branch payloads: {json.dumps(condensed_items, ensure_ascii=False)}"
        )
        text = ""
        try:
            resp = self.client.chat(
                [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt},
                ]
            )
            text = self.client.extract_text(resp).strip()
            updated = self._extract_json_object(text)
        except Exception:
            updated = None
        self._append_consolidation_log(
            {
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "stage_name": stage_name,
                "metadata": {"kind": "wave_consolidation", **(metadata or {})},
                "request": {"system": system_prompt, "user": prompt},
                "response_text": text,
                "parsed_response": updated if isinstance(updated, dict) else None,
                "success": isinstance(updated, dict),
            }
        )
        if not isinstance(updated, dict):
            return {"experience": current.experience, "skill": current.skill}
        exp, skill = self._sanitize_stage_update(
            stage_name=stage_name,
            experience=updated.get("experience") or current.experience,
            skill=updated.get("skill") or current.skill,
            current=current,
        )
        with self._lock:
            self.stage_books[stage_name] = StageNotebook(experience=exp, skill=skill)
            self._persist_stage_books()
        return {"experience": exp, "skill": skill}

    def export(self, path: Path) -> None:
        save_json(
            path,
            {
                "enabled": self.enabled,
                "self_evolving_dir": str(self.self_evolving_dir),
                "stage_memory_path": str(self.stage_memory_path),
                "tool_events_path": str(self.tool_events_path),
                "consolidation_log_path": str(self.consolidation_log_path),
                "stage_books": {
                    name: {"experience": book.experience, "skill": book.skill}
                    for name, book in self.stage_books.items()
                },
                "tool_events": self.tool_events,
            },
        )

    @staticmethod
    def _resolve_self_evolving_root(root: Optional[Path]) -> Optional[Path]:
        if root is None:
            return None
        candidate = root.resolve()
        if (candidate / "self_evolving").exists():
            return candidate / "self_evolving"
        if candidate.name == "self_evolving" and candidate.exists():
            return candidate
        return candidate if candidate.exists() else None

    def _bootstrap_from(self, bootstrap_from: Optional[Path]) -> None:
        source_root = self._resolve_self_evolving_root(bootstrap_from)
        if source_root is None:
            return
        if not source_root.exists():
            raise FileNotFoundError(f"Bootstrap self-evolving path does not exist: {source_root}")
        loaded_books = self._load_stage_books_from(source_root)
        if loaded_books:
            self.stage_books.update(loaded_books)
        loaded_events = self._load_tool_events_from(source_root)
        if loaded_events:
            self.tool_events = loaded_events

    def _load_stage_books_from(self, source_root: Path) -> Dict[str, StageNotebook]:
        stage_memory_path = source_root / "stage_memory.json"
        if stage_memory_path.exists():
            try:
                payload = json.loads(stage_memory_path.read_text(encoding="utf-8"))
            except Exception:
                payload = None
            books = self._parse_stage_books_payload(payload)
            if books:
                return books
        notebook_path = source_root.parent / "evolution_notebook.json" if source_root.name == "self_evolving" else source_root / "evolution_notebook.json"
        if notebook_path.exists():
            try:
                payload = json.loads(notebook_path.read_text(encoding="utf-8"))
            except Exception:
                payload = None
            books = self._parse_stage_books_payload(payload)
            if books:
                return books
        return {}

    def _load_tool_events_from(self, source_root: Path) -> List[Dict[str, Any]]:
        tool_events_path = source_root / "tool_events.json"
        if tool_events_path.exists():
            try:
                payload = json.loads(tool_events_path.read_text(encoding="utf-8"))
            except Exception:
                payload = None
            if isinstance(payload, list):
                return [item for item in payload if isinstance(item, dict)]
        notebook_path = source_root.parent / "evolution_notebook.json" if source_root.name == "self_evolving" else source_root / "evolution_notebook.json"
        if notebook_path.exists():
            try:
                payload = json.loads(notebook_path.read_text(encoding="utf-8"))
            except Exception:
                payload = None
            if isinstance(payload, dict):
                events = payload.get("tool_events")
                if isinstance(events, list):
                    return [item for item in events if isinstance(item, dict)]
        return []

    @staticmethod
    def _parse_stage_books_payload(payload: Any) -> Dict[str, StageNotebook]:
        if not isinstance(payload, dict):
            return {}
        raw_books = payload.get("stage_books", payload)
        if not isinstance(raw_books, dict):
            return {}
        books: Dict[str, StageNotebook] = {}
        for stage_name, entry in raw_books.items():
            if not isinstance(entry, dict):
                continue
            books[str(stage_name)] = StageNotebook(
                experience=str(entry.get("experience") or "").strip(),
                skill=str(entry.get("skill") or "").strip(),
            )
        return books

    def _persist_stage_books(self) -> None:
        save_json(
            self.stage_memory_path,
            {
                "stage_books": {
                    name: {"experience": book.experience, "skill": book.skill}
                    for name, book in self.stage_books.items()
                }
            },
        )

    def _persist_tool_events(self) -> None:
        save_json(self.tool_events_path, list(self.tool_events))

    def _append_consolidation_log(self, entry: Dict[str, Any]) -> None:
        with open(self.consolidation_log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    def _sanitize_all_stage_books(self) -> None:
        defaults = self._default_stage_books()
        for stage_name, current in list(self.stage_books.items()):
            fallback = defaults.get(stage_name, StageNotebook())
            exp, skill = self._sanitize_stage_update(
                stage_name=stage_name,
                experience=current.experience,
                skill=current.skill,
                current=fallback,
            )
            self.stage_books[stage_name] = StageNotebook(experience=exp, skill=skill)

    def _sanitize_stage_update(
        self,
        stage_name: str,
        experience: Any,
        skill: Any,
        current: StageNotebook,
    ) -> tuple[str, str]:
        default = self._default_stage_books().get(stage_name, StageNotebook())
        current_experience = current.experience or default.experience
        current_skill = current.skill or default.skill
        exp = self._compact(experience or current_experience, 600)
        skill_text = self._compact(skill or current_skill, 800)
        if self._contains_conservative_memory(exp):
            exp = self._compact(
                current_experience if not self._contains_conservative_memory(current_experience) else default.experience,
                600,
            )
        if self._contains_conservative_memory(skill_text):
            skill_text = self._compact(
                current_skill if not self._contains_conservative_memory(current_skill) else default.skill,
                800,
            )
        return exp, skill_text

    @classmethod
    def _contains_conservative_memory(cls, text: Any) -> bool:
        normalized = " ".join(str(text or "").lower().split())
        if not normalized:
            return False
        patterns = cls._CONSERVATIVE_MEMORY_PATTERNS + cls._RESULT_BIAS_MEMORY_PATTERNS
        return any(re.search(pattern, normalized, flags=re.IGNORECASE) for pattern in patterns)

    @staticmethod
    def _compact(text: Any, limit: int) -> str:
        compact = " ".join(str(text or "").split())
        return compact[:limit]

    @staticmethod
    def _extract_json_object(text: str) -> Optional[Dict[str, Any]]:
        cleaned = re.sub(r"^```(?:json)?\s*", "", text.strip())
        cleaned = re.sub(r"\s*```$", "", cleaned)
        try:
            obj = json.loads(cleaned)
            return obj if isinstance(obj, dict) else None
        except Exception:
            pass
        match = re.search(r"\{[\s\S]*\}", cleaned)
        if not match:
            return None
        try:
            obj = json.loads(match.group(0))
            return obj if isinstance(obj, dict) else None
        except Exception:
            return None
