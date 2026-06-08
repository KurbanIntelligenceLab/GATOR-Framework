"""Gate engine: configurable deterministic screening gates.

Loads a YAML config, applies a chain of physics gates to descriptor data,
and produces PhysicsProfile objects with physics-labeled verdicts.

All thresholds, labels, and screening mappings are configurable — no magic
numbers in this module.
"""

from __future__ import annotations

import csv
import operator as op
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

__all__ = [
    "GateDefinition",
    "GateVerdict",
    "PhysicsProfile",
    "ScreeningConfig",
    "load_config",
    "load_records_from_csv",
    "run_gates",
]

# ---------------------------------------------------------------------------
# Data structures (frozen for immutability)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class GateVerdict:
    """Result of applying a single gate to one system."""

    gate_name: str
    value: Any
    physics_label: str
    screening_verdict: str
    threshold_description: str


@dataclass(frozen=True)
class PhysicsProfile:
    """Aggregate gate results for one system."""

    system: str
    verdicts: tuple[GateVerdict, ...]
    overall: str  # pass / borderline / fail
    pass_count: int
    flag_count: int
    fail_count: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "system": self.system,
            "overall": self.overall,
            "pass_count": self.pass_count,
            "flag_count": self.flag_count,
            "fail_count": self.fail_count,
            "verdicts": [
                {
                    "gate": v.gate_name,
                    "value": v.value,
                    "physics_label": v.physics_label,
                    "screening_verdict": v.screening_verdict,
                    "threshold": v.threshold_description,
                }
                for v in self.verdicts
            ],
        }


@dataclass(frozen=True)
class GateDefinition:
    """Parsed gate definition from YAML config."""

    name: str
    description: str
    column: str
    gate_type: str
    thresholds: list[float]
    physics_labels: Any  # list or dict depending on gate type
    screening_map: dict[str, str]
    use_magnitude: bool = False
    doe_bounds: list[float] = field(default_factory=list)
    operator_str: str = "<"
    threshold_value: float = 0.0


@dataclass(frozen=True)
class ScreeningConfig:
    """Typed, validated config loaded from YAML."""

    gates: tuple[GateDefinition, ...]
    aggregation_rules: dict[str, str]
    rag_top_k: int = 5
    rag_embedding_model: str = "all-MiniLM-L6-v2"
    rag_literature_dir: str = "literature"


# ---------------------------------------------------------------------------
# Config loading with fail-fast validation
# ---------------------------------------------------------------------------


def load_config(path: str | Path) -> ScreeningConfig:
    """Load and validate a screening config from YAML. Raises on bad config."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with path.open() as f:
        raw = yaml.safe_load(f)

    if not isinstance(raw, dict):
        raise ValueError(f"Config must be a YAML mapping, got {type(raw).__name__}")

    gates_raw = raw.get("gates")
    if not isinstance(gates_raw, list) or len(gates_raw) == 0:
        raise ValueError("Config must define at least one gate under 'gates'")

    gates: list[GateDefinition] = []
    for i, g in enumerate(gates_raw):
        gates.append(_parse_gate(g, index=i))

    aggregation = raw.get("aggregation", {})
    if not isinstance(aggregation, dict):
        raise ValueError("'aggregation' must be a mapping")

    rag = raw.get("rag", {})

    return ScreeningConfig(
        gates=tuple(gates),
        aggregation_rules=aggregation,
        rag_top_k=int(rag.get("top_k", 5)),
        rag_embedding_model=str(rag.get("embedding_model", "all-MiniLM-L6-v2")),
        rag_literature_dir=str(rag.get("literature_dir", "literature")),
    )


def _parse_gate(g: dict[str, Any], index: int) -> GateDefinition:
    """Parse and validate a single gate definition."""
    name = g.get("name")
    if not name:
        raise ValueError(f"Gate at index {index} missing 'name'")

    gate_type = g.get("type")
    if gate_type not in ("range_multi", "threshold", "doe_window", "entropy_corrected_doe"):
        raise ValueError(
            f"Gate '{name}': unknown type '{gate_type}'. "
            "Expected: range_multi, threshold, doe_window, entropy_corrected_doe"
        )

    column = g.get("column")
    if not column:
        raise ValueError(f"Gate '{name}': missing 'column'")

    screening_map = g.get("screening_map")
    if not isinstance(screening_map, dict):
        raise ValueError(f"Gate '{name}': 'screening_map' must be a mapping")

    # Validate screening_map values
    for label, verdict in screening_map.items():
        if verdict not in ("pass", "flag", "fail"):
            raise ValueError(
                f"Gate '{name}': screening_map['{label}'] = '{verdict}' "
                "is invalid. Must be: pass, flag, or fail"
            )

    return GateDefinition(
        name=name,
        description=str(g.get("description", "")),
        column=column,
        gate_type=gate_type,
        thresholds=list(g.get("thresholds", [])),
        physics_labels=g.get("physics_labels", []),
        screening_map=screening_map,
        use_magnitude=bool(g.get("use_magnitude", False)),
        doe_bounds=list(g.get("doe_bounds", [])),
        operator_str=str(g.get("operator", "<")),
        threshold_value=float(g.get("value", 0.0)),
    )


# ---------------------------------------------------------------------------
# Gate execution
# ---------------------------------------------------------------------------

_OPERATORS = {
    "<": op.lt,
    "<=": op.le,
    ">": op.gt,
    ">=": op.ge,
    "==": op.eq,
    "!=": op.ne,
}


def _execute_gate(gate: GateDefinition, row: dict[str, Any]) -> GateVerdict:
    """Apply a single gate to one system's data row."""
    raw_value = row.get(gate.column)
    if raw_value is None:
        return GateVerdict(
            gate_name=gate.name,
            value=None,
            physics_label="missing_data",
            screening_verdict="fail",
            threshold_description=f"column '{gate.column}' not found",
        )

    value = _to_float(raw_value)

    if gate.gate_type == "range_multi":
        return _gate_range_multi(gate, value, raw_value)
    if gate.gate_type == "threshold":
        return _gate_threshold(gate, value, raw_value)
    if gate.gate_type == "doe_window":
        return _gate_doe_window(gate, value, raw_value)
    if gate.gate_type == "entropy_corrected_doe":
        return _gate_entropy_corrected_doe(gate, value, raw_value)
    raise ValueError(f"Unknown gate type: {gate.gate_type}")


def _gate_range_multi(gate: GateDefinition, value: float | None, raw: object) -> GateVerdict:
    """Gate with multiple thresholds producing N+1 bins."""
    if value is None:
        return _missing_verdict(gate, raw)

    compare_value = abs(value) if gate.use_magnitude else value
    labels = gate.physics_labels

    if not isinstance(labels, list) or len(labels) != len(gate.thresholds) + 1:
        raise ValueError(
            f"Gate '{gate.name}': expected {len(gate.thresholds) + 1} physics_labels "
            f"for {len(gate.thresholds)} thresholds, got {len(labels)}"
        )

    # Find which bin the value falls into
    bin_index = 0
    for t in gate.thresholds:
        if compare_value >= t:
            bin_index += 1
        else:
            break

    physics_label = labels[bin_index]
    screening_verdict = gate.screening_map.get(physics_label, "fail")
    magnitude_note = " (magnitude)" if gate.use_magnitude else ""
    thresholds_str = ", ".join(str(t) for t in gate.thresholds)

    return GateVerdict(
        gate_name=gate.name,
        value=value,
        physics_label=physics_label,
        screening_verdict=screening_verdict,
        threshold_description=f"thresholds: [{thresholds_str}]{magnitude_note}",
    )


def _gate_threshold(gate: GateDefinition, value: float | None, raw: object) -> GateVerdict:
    """Gate with a single threshold and comparison operator."""
    if value is None:
        return _missing_verdict(gate, raw)

    compare_fn = _OPERATORS.get(gate.operator_str)
    if compare_fn is None:
        raise ValueError(f"Gate '{gate.name}': unknown operator '{gate.operator_str}'")

    result = compare_fn(value, gate.threshold_value)
    labels = gate.physics_labels

    if isinstance(labels, dict):
        physics_label = labels.get(str(result).lower(), str(result))
    else:
        physics_label = str(result)

    screening_verdict = gate.screening_map.get(physics_label, "fail")

    return GateVerdict(
        gate_name=gate.name,
        value=value,
        physics_label=physics_label,
        screening_verdict=screening_verdict,
        threshold_description=f"value {gate.operator_str} {gate.threshold_value}",
    )


def _gate_doe_window(gate: GateDefinition, value: float | None, raw: object) -> GateVerdict:
    """DOE window check: is |value| within [low, high]?"""
    if value is None:
        return _missing_verdict(gate, raw)

    compare_value = abs(value) if gate.use_magnitude else value
    low, high = gate.doe_bounds[0], gate.doe_bounds[1]
    labels = gate.physics_labels

    if isinstance(labels, dict):
        if low <= compare_value <= high:
            physics_label = labels.get("inside", "inside")
        else:
            physics_label = labels.get("outside", "outside")
    else:
        physics_label = "inside" if low <= compare_value <= high else "outside"

    screening_verdict = gate.screening_map.get(physics_label, "fail")
    magnitude_note = " (magnitude)" if gate.use_magnitude else ""

    return GateVerdict(
        gate_name=gate.name,
        value=value,
        physics_label=physics_label,
        screening_verdict=screening_verdict,
        threshold_description=f"DOE window [{low}, {high}]{magnitude_note}",
    )


def _gate_entropy_corrected_doe(
    gate: GateDefinition, value: float | None, raw: object
) -> GateVerdict:
    """Entropy-corrected DOE check matching classify_doe_label() logic.

    Logic:
        value > 0         → Endothermic
        |value| < low     → Outside--weak
        low ≤ |value| ≤ high → Inside
        |value| > high    → Outside--strong
    """
    if value is None:
        return _missing_verdict(gate, raw)

    labels = gate.physics_labels
    low, high = gate.doe_bounds[0], gate.doe_bounds[1]
    abs_value = abs(value)

    if value > 0:
        physics_label = (
            labels.get("endothermic", "Endothermic") if isinstance(labels, dict) else "Endothermic"
        )
    elif abs_value < low:
        physics_label = (
            labels.get("weak", "Outside--weak") if isinstance(labels, dict) else "Outside--weak"
        )
    elif abs_value <= high:
        physics_label = (
            labels.get("inside", "Inside (15--25)")
            if isinstance(labels, dict)
            else "Inside (15--25)"
        )
    else:
        physics_label = (
            labels.get("strong", "Outside--strong")
            if isinstance(labels, dict)
            else "Outside--strong"
        )

    screening_verdict = gate.screening_map.get(physics_label, "fail")

    return GateVerdict(
        gate_name=gate.name,
        value=value,
        physics_label=physics_label,
        screening_verdict=screening_verdict,
        threshold_description=f"entropy-corrected DOE [{low}, {high}]",
    )


def _missing_verdict(gate: GateDefinition, raw: object) -> GateVerdict:
    return GateVerdict(
        gate_name=gate.name,
        value=raw,
        physics_label="missing_data",
        screening_verdict="fail",
        threshold_description=f"non-numeric value: {raw!r}",
    )


def _to_float(x: object) -> float | None:
    if isinstance(x, (int, float)):
        return float(x)
    if isinstance(x, str):
        s = x.strip()
        if s == "" or s.lower() in ("none", "nan", "null", ""):
            return None
        try:
            return float(s)
        except ValueError:
            return None
    return None


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------


def _aggregate_verdicts(verdicts: tuple[GateVerdict, ...]) -> str:
    """Combine gate verdicts into an overall classification.

    Default rule: any fail → fail, any flag (no fail) → borderline, all pass → pass.
    """
    has_fail = any(v.screening_verdict == "fail" for v in verdicts)
    has_flag = any(v.screening_verdict == "flag" for v in verdicts)

    if has_fail:
        return "fail"
    if has_flag:
        return "borderline"
    return "pass"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def run_gates(
    config: ScreeningConfig,
    records: Sequence[dict[str, Any]],
) -> tuple[PhysicsProfile, ...]:
    """Apply all gates to all system records, return PhysicsProfiles."""
    profiles: list[PhysicsProfile] = []

    for row in records:
        system = row.get("System", "unknown")
        verdicts: list[GateVerdict] = []

        for gate in config.gates:
            verdict = _execute_gate(gate, row)
            verdicts.append(verdict)

        verdicts_tuple = tuple(verdicts)
        pass_count = sum(1 for v in verdicts_tuple if v.screening_verdict == "pass")
        flag_count = sum(1 for v in verdicts_tuple if v.screening_verdict == "flag")
        fail_count = sum(1 for v in verdicts_tuple if v.screening_verdict == "fail")

        profiles.append(
            PhysicsProfile(
                system=system,
                verdicts=verdicts_tuple,
                overall=_aggregate_verdicts(verdicts_tuple),
                pass_count=pass_count,
                flag_count=flag_count,
                fail_count=fail_count,
            )
        )

    return tuple(profiles)


def load_records_from_csv(
    path: str | Path,
    systems: Sequence[str] | None = None,
) -> list[dict[str, Any]]:
    """Load descriptor records from a CSV file (e.g., labels.csv).

    Returns a list of dicts with numeric values parsed to float where possible.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Data file not found: {path}")

    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    records: list[dict[str, Any]] = []
    for row in rows:
        system = (row.get("System") or "").strip()
        if not system:
            continue
        if systems is not None and system not in systems:
            continue

        normalized: dict[str, Any] = {}
        for k, v in row.items():
            if k == "System":
                normalized[k] = system
                continue
            parsed = _to_float(v)
            normalized[k] = (
                parsed if parsed is not None else (v.strip() if isinstance(v, str) else v)
            )
        records.append(normalized)

    return records
