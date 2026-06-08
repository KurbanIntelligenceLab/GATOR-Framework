"""Structured JSON output schemas and validation for GATOR LLM responses.

Provides:
- ANALYSIS_JSON_SCHEMA: full JSON-Schema dict for the analysis output format
- extract_json_from_response: extract JSON from raw LLM text (direct or fenced)
- validate_analysis_json: validate parsed JSON against schema and expected systems
- render_schema_for_prompt: render the schema as indented JSON for prompt injection
- extract_numerical_claims: extract (system, unit, value) triples from markdown
- verify_claims_against_records: verify numerical claims against source CSV records
- extract_citation_ids: extract [N] citation IDs from markdown
- verify_citation_fidelity: check that cited IDs are within valid range
"""

from __future__ import annotations

import json
import re
from typing import Any

__all__ = [
    "ANALYSIS_JSON_SCHEMA",
    "extract_citation_ids",
    "extract_json_from_response",
    "extract_numerical_claims",
    "render_schema_for_prompt",
    "validate_analysis_json",
    "verify_citation_fidelity",
    "verify_claims_against_records",
]

# ---------------------------------------------------------------------------
# Valid enum values
# ---------------------------------------------------------------------------

_VALID_TIERS = {"pass", "borderline", "fail"}

_SYSTEM_REQUIRED_KEYS = {
    "system",
    "tier",
    "anomaly_flags",
    "sensitivity",
    "dftb_robustness",
    "literature_support",
}

_TOP_LEVEL_REQUIRED_KEYS = {"systems", "ranking", "global_observations"}

# ---------------------------------------------------------------------------
# JSON Schema
# ---------------------------------------------------------------------------

ANALYSIS_JSON_SCHEMA: dict[str, Any] = {
    "type": "object",
    "required": sorted(_TOP_LEVEL_REQUIRED_KEYS),
    "properties": {
        "systems": {
            "type": "array",
            "items": {
                "type": "object",
                "required": sorted(_SYSTEM_REQUIRED_KEYS),
                "properties": {
                    "system": {"type": "string"},
                    "tier": {"type": "string", "enum": sorted(_VALID_TIERS)},
                    "anomaly_flags": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "required": ["type", "descriptors", "description", "severity"],
                            "properties": {
                                "type": {
                                    "type": "string",
                                    "enum": [
                                        "cross_descriptor",
                                        "mode_regime_mismatch",
                                        "t50_doe_mismatch",
                                    ],
                                },
                                "descriptors": {"type": "array", "items": {"type": "string"}},
                                "description": {"type": "string"},
                                "severity": {
                                    "type": "string",
                                    "enum": ["minor", "major", "critical"],
                                },
                            },
                        },
                    },
                    "sensitivity": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "required": [
                                "gate",
                                "current_value",
                                "nearest_threshold",
                                "margin_pct",
                                "flip_consequence",
                            ],
                            "properties": {
                                "gate": {"type": "string"},
                                "current_value": {"type": "number"},
                                "nearest_threshold": {"type": "number"},
                                "margin_pct": {"type": "number"},
                                "flip_consequence": {"type": "string"},
                            },
                        },
                    },
                    "dftb_robustness": {
                        "type": "object",
                        "required": ["confidence", "rationale", "could_flip_verdict"],
                        "properties": {
                            "confidence": {"type": "string", "enum": ["high", "medium", "low"]},
                            "rationale": {"type": "string"},
                            "could_flip_verdict": {"type": "boolean"},
                        },
                    },
                    "literature_support": {
                        "type": "object",
                        "required": ["has_direct_experimental", "citation_ids", "support_strength"],
                        "properties": {
                            "has_direct_experimental": {"type": "boolean"},
                            "citation_ids": {"type": "array", "items": {"type": "integer"}},
                            "support_strength": {
                                "type": "string",
                                "enum": ["direct", "indirect", "none"],
                            },
                        },
                    },
                },
            },
        },
        "ranking": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["rank", "system", "tier", "rationale"],
                "properties": {
                    "rank": {"type": "integer"},
                    "system": {"type": "string"},
                    "tier": {"type": "string"},
                    "rationale": {"type": "string"},
                },
            },
        },
        "global_observations": {
            "type": "array",
            "items": {"type": "string"},
        },
    },
}

# ---------------------------------------------------------------------------
# Extraction helpers
# ---------------------------------------------------------------------------

_FENCED_JSON_RE = re.compile(r"```json\s*\n(.*?)```", re.DOTALL)


def extract_json_from_response(raw: str) -> dict[str, Any] | None:
    """Try to extract a JSON dict from an LLM response string.

    Strategy:
    1. Attempt direct ``json.loads`` on the full string.
    2. Look for a ```json ... ``` fenced block and parse its contents.

    Returns the parsed dict, or ``None`` if extraction fails.
    """
    if not raw:
        return None

    # 1. Direct parse
    try:
        result = json.loads(raw)
        if isinstance(result, dict):
            return result
    except (json.JSONDecodeError, ValueError):
        pass

    # 2. Fenced block
    match = _FENCED_JSON_RE.search(raw)
    if match:
        try:
            result = json.loads(match.group(1))
            if isinstance(result, dict):
                return result
        except (json.JSONDecodeError, ValueError):
            pass

    return None


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def validate_analysis_json(
    data: dict[str, Any],
    expected_systems: list[str],
) -> tuple[bool, list[str]]:
    """Validate a parsed analysis JSON against the schema and expected systems.

    Returns ``(ok, notes)`` where *ok* is ``True`` when all checks pass
    and *notes* is a list of human-readable failure descriptions.
    """
    notes: list[str] = []

    # -- Top-level required keys --
    for key in sorted(_TOP_LEVEL_REQUIRED_KEYS):
        if key not in data:
            notes.append(f"Missing required top-level key: '{key}'")

    # If systems list is absent we can't do further checks
    systems_list = data.get("systems")
    if not isinstance(systems_list, list):
        if not any("systems" in n for n in notes):
            notes.append("'systems' must be an array")
        return (len(notes) == 0, notes)

    # -- Per-system validation --
    present_names: set[str] = set()
    for idx, entry in enumerate(systems_list):
        if not isinstance(entry, dict):
            notes.append(f"systems[{idx}] is not an object")
            continue

        name = entry.get("system", f"<unknown #{idx}>")
        present_names.add(name)

        # Required keys
        for key in sorted(_SYSTEM_REQUIRED_KEYS):
            if key not in entry:
                notes.append(f"System '{name}' (index {idx}) missing required key: '{key}'")

        # Tier enum
        tier = entry.get("tier")
        if tier is not None and tier not in _VALID_TIERS:
            notes.append(
                f"System '{name}' has invalid tier '{tier}'; expected one of {sorted(_VALID_TIERS)}"
            )

    # -- Expected systems coverage --
    for sysname in expected_systems:
        if sysname not in present_names:
            notes.append(f"Expected system '{sysname}' not found in systems array")

    return (len(notes) == 0, notes)


# ---------------------------------------------------------------------------
# Prompt helper
# ---------------------------------------------------------------------------


def render_schema_for_prompt() -> str:
    """Return the analysis JSON schema as pretty-printed JSON for prompt injection."""
    return json.dumps(ANALYSIS_JSON_SCHEMA, indent=2)


# ---------------------------------------------------------------------------
# Numerical-claim extraction & verification (Task 2)
# ---------------------------------------------------------------------------

_UNIT_PATTERNS: dict[str, re.Pattern[str]] = {
    # Capture optional negative sign + digits.  The negative sign is only valid
    # when NOT preceded by a digit (to reject ranges like "300-400 K").
    "kJ/mol": re.compile(r"(?<!\d)([-−]?\d+\.?\d*)\s*kJ\s*/?mol"),
    "eV": re.compile(r"(?<!\d)([-−]?\d+\.?\d*)\s*eV"),
    "K": re.compile(r"(?<!\d)([-−]?\d+\.?\d*)\s*K\b"),
    "Å": re.compile(r"(?<!\d)([-−]?\d+\.?\d*)\s*Å"),
}

_UNIT_COLUMNS: dict[str, list[str]] = {
    "kJ/mol": [
        "E_ads_kJ_mol",
        # ΔH_corr presented to the LLM is the +6 correction (see screening_agent
        # _KEY_DESCRIPTORS); the +25.96 column is not shown, so it is not a
        # validation target. The +25.96 scheme remains in labels.csv for Table 3A.
        "Delta_H_plus6_kJ_mol",
        "Delta_G_std_kJ_mol",
    ],
    "eV": ["E_ads_eV", "E_g_eV", "chi_eV", "eta_eV", "omega_eV"],
    "K": ["T50_at_1bar_K", "T50_at_30bar_K"],
    "Å": ["r_H-H_A", "Delta_r_H-H_A", "r_H-AE_A", "R_O-AE_A"],
}


def extract_numerical_claims(
    markdown: str,
    system_names: list[str],
) -> list[tuple[str, str, float]]:
    """Extract (system_name, unit_str, value) triples from markdown text.

    For each line, checks whether any *system_names* appear.  If so, all
    numbers-with-units matching the known unit patterns are extracted.
    """
    results: list[tuple[str, str, float]] = []
    for line in markdown.splitlines():
        # Identify which system(s) are mentioned on this line
        matched_systems = [s for s in system_names if s in line]
        if not matched_systems:
            continue
        # For each unit pattern, extract all matches on this line
        for unit, pattern in _UNIT_PATTERNS.items():
            for m in pattern.finditer(line):
                value = float(m.group(1).replace("\u2212", "-"))
                for sys_name in matched_systems:
                    results.append((sys_name, unit, value))
    return results


def verify_claims_against_records(
    claims: list[tuple[str, str, float]],
    records: list[dict[str, Any]],
    tolerance_pct: float = 1.0,
) -> list[dict[str, Any]]:
    """Verify extracted claims against source records.

    Returns a list of error dicts for claims that deviate from the
    best-matching column value by more than *tolerance_pct* percent.
    An empty list means all claims are within tolerance.
    """
    # Build {system_name: record} lookup (handle both "System" and "system" keys)
    lookup: dict[str, dict[str, Any]] = {}
    for r in records:
        name = r.get("System") or r.get("system")
        if name:
            lookup[name] = r

    errors: list[dict[str, Any]] = []
    for sys_name, unit, claimed in claims:
        record = lookup.get(sys_name)
        if record is None:
            errors.append(
                {
                    "system": sys_name,
                    "descriptor": f"unknown ({unit})",
                    "claimed": claimed,
                    "actual": None,
                    "deviation_pct": None,
                }
            )
            continue

        candidate_cols = _UNIT_COLUMNS.get(unit, [])
        best_col: str | None = None
        best_dev: float | None = None
        best_actual: float | None = None

        for col in candidate_cols:
            actual = record.get(col)
            # Skip missing / empty / non-numeric descriptors (e.g. R_O-AE_A is
            # blank for AE-free systems such as Pristine): a claim cannot be
            # checked against an absent value, and float("") would raise.
            if actual is None or actual == "":
                continue
            try:
                actual = float(actual)
            except (TypeError, ValueError):
                continue
            # Deviation percentage: |claimed - actual| / |actual| * 100.
            # actual == 0 is treated as a large deviation to guard against division by zero.
            dev = abs(claimed) * 100 if actual == 0 else abs(claimed - actual) / abs(actual) * 100
            if best_dev is None or dev < best_dev:
                best_dev = dev
                best_col = col
                best_actual = actual

        if best_col is None:
            # No matching column found in the record
            errors.append(
                {
                    "system": sys_name,
                    "descriptor": f"no column ({unit})",
                    "claimed": claimed,
                    "actual": None,
                    "deviation_pct": None,
                }
            )
        elif best_dev is not None and best_dev > tolerance_pct:
            errors.append(
                {
                    "system": sys_name,
                    "descriptor": best_col,
                    "claimed": claimed,
                    "actual": best_actual,
                    "deviation_pct": round(best_dev, 4),
                }
            )

    return errors


# ---------------------------------------------------------------------------
# Citation extraction & fidelity verification (Task 2)
# ---------------------------------------------------------------------------

_CITATION_ID_RE = re.compile(r"\[(\d+)\]")


def extract_citation_ids(markdown: str) -> set[int]:
    """Extract all ``[N]`` citation IDs from markdown text.

    Returns a set of integer IDs.
    """
    return {int(m.group(1)) for m in _CITATION_ID_RE.finditer(markdown)}


def verify_citation_fidelity(
    cited_ids: set[int],
    max_ref: int,
) -> list[str]:
    """Check that all cited ``[N]`` IDs are within ``[1]`` to ``[max_ref]``.

    Returns a sorted list of error strings for any fabricated citations.
    """
    errors: list[str] = []
    for cid in cited_ids:
        if cid > max_ref or cid < 1:
            errors.append(f"Fabricated citation [{cid}]: only [1]\u2013[{max_ref}] provided")
    return sorted(errors)
