"""Land-use constants and validation helpers for participatory planning."""

from __future__ import annotations

from typing import Dict, Iterable, Set

LANDUSE_LABELS: Dict[str, str] = {
    "RS": "Residence",
    "BS": "Business Area",
    "RC": "Recreation and Sports Center",
    "PK": "Park",
    "OF": "Office",
    "SC": "School",
    "OS": "Open Space",
    "HS": "Hospital",
    "ID": "Urban Industrial and Infrastructure Area",
}

IMMUTABLE_LANDUSE: Set[str] = {"RS", "OS"}
MODIFIABLE_LANDUSE: Set[str] = set(LANDUSE_LABELS.keys()) - IMMUTABLE_LANDUSE

LANDUSE_COLORS: Dict[str, str] = {
    "RS": "#E69F00",
    "BS": "#D55E00",
    "RC": "#CC79A7",
    "PK": "#009E73",
    "OF": "#0072B2",
    "SC": "#F0E442",
    "OS": "#999999",
    "HS": "#56B4E9",
    "ID": "#6B3F2A",
    "unknown": "#9E9E9E",
}


def is_valid_landuse(code: str) -> bool:
    return code in LANDUSE_LABELS


def can_change_from(code: str) -> bool:
    return code in MODIFIABLE_LANDUSE


def can_change_to(code: str) -> bool:
    return code in MODIFIABLE_LANDUSE


def validate_plan_mapping(plan: Dict[str, str], aoi_landuse: Dict[str, str]) -> Dict[str, str]:
    """Keep only legal updates and normalize AOI ids to string keys."""
    clean: Dict[str, str] = {}
    for raw_id, target in plan.items():
        aoi_id = str(raw_id)
        old = aoi_landuse.get(aoi_id)
        if not old:
            continue
        if not is_valid_landuse(target):
            continue
        if not can_change_from(old):
            continue
        if not can_change_to(target):
            continue
        if old == target:
            continue
        clean[aoi_id] = target
    return clean


def modifiable_types_text(types: Iterable[str] | None = None) -> str:
    values = sorted(types or MODIFIABLE_LANDUSE)
    return ", ".join(values)
