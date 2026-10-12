"""Dataclasses shared by refined planning controller and agents."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class PhaseTrace:
    step: int
    kind: str
    payload: Dict[str, Any]


@dataclass
class StageRunResult:
    stage_name: str
    structured_result: Dict[str, Any]
    trace: List[PhaseTrace] = field(default_factory=list)
    full_trace: List[Dict[str, Any]] = field(default_factory=list)
    raw_last_response: str = ""
    counted_steps_used: int = 0
    total_attempts: int = 0


@dataclass
class CycleRecord:
    cycle_index: int
    branch_id: str
    input_node_id: str
    stage_results: Dict[str, Dict[str, Any]]
    stage_updates: Dict[str, Dict[str, str]]
    action: str
    summary: str
    metrics_after: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class StateNode:
    node_id: str
    parent_id: Optional[str]
    branch_id: str
    branch_label: str
    plan: Dict[str, str]
    metrics: Dict[str, Any]
    created_cycle_index: int
    root_id: str
    depth: int
    stage_summary: str = ""
    branch_group_labels: List[str] = field(default_factory=list)
    severe_conflict_used: bool = False
    rollback_used: bool = False
    rollback_anchor_id: Optional[str] = None
    rollback_rule: str = "rollback_to_parent"
    finished: bool = False
    terminated_reason: Optional[str] = None
    children: List[str] = field(default_factory=list)
    node_origin: str = "normal"
    copied_from_node_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class BranchRuntime:
    branch_id: str
    branch_label: str
    current_node_id: str
    output_dir: Path
    toolkit: Any
    client: Any
    lineage_labels: List[str] = field(default_factory=list)
    cycle_count: int = 0


@dataclass
class CycleExecutionOutput:
    branch_id: str
    branch_label: str
    input_node_id: str
    cycle_index: int
    stage_results: Dict[str, Dict[str, Any]]
    stage_updates: Dict[str, Dict[str, str]]
    action: str
    summary: str
    current_plan: Dict[str, str] = field(default_factory=dict)
    current_metrics: Dict[str, Any] = field(default_factory=dict)
    branch_specs: List[Dict[str, Any]] = field(default_factory=list)
    rollback_target_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    stage_consolidation_payloads: Dict[str, Dict[str, Any]] = field(default_factory=dict)
