"""Screening agent: orchestrates the GATOR pipeline.

Pipeline: Load CSV → Gates → (RAG) → LLM Synthesis → Post-hoc Validation.
Central object is ScreeningContext, an immutable dataclass that accumulates
state as it flows through stages.
"""

from __future__ import annotations

import json
import logging
import sys
import uuid
from collections.abc import Sequence
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path
from time import perf_counter
from typing import Any

from .gate_engine import (
    PhysicsProfile,
    ScreeningConfig,
    run_gates,
)
from .llm_providers import call_llm
from .output_schemas import (
    extract_citation_ids,
    extract_json_from_response,
    extract_numerical_claims,
    render_schema_for_prompt,
    validate_analysis_json,
    verify_citation_fidelity,
    verify_claims_against_records,
)

__all__ = [
    "LLMSynthesisResult",
    "ProvenanceRecord",
    "ScreeningContext",
    "StructuredAnalysis",
    "ValidationResult",
    "build_analysis_prompt",
    "build_screening_prompt",
    "build_synthesis_prompt",
    "run_multi_model_benchmark",
    "run_screening_pipeline",
    "validate_llm_output",
    "validate_llm_output_v2",
]


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ProvenanceRecord:
    """Provenance for one pipeline stage."""

    stage: str
    timestamp_utc: str
    config_hash: str
    input_hash: str
    output_hash: str

    def to_dict(self) -> dict[str, str]:
        return {
            "stage": self.stage,
            "timestamp_utc": self.timestamp_utc,
            "config_hash": self.config_hash,
            "input_hash": self.input_hash,
            "output_hash": self.output_hash,
        }


@dataclass(frozen=True)
class LLMSynthesisResult:
    """Parsed LLM output with metadata."""

    raw_output: str
    run_id: str
    model: str
    provider: str
    temperature: float
    elapsed_s: float
    output_hash: str
    call_count: int = 1  # 1 for single-shot, 2 for structured
    analysis_elapsed_s: float = 0.0  # Call 1 time
    synthesis_elapsed_s: float = 0.0  # Call 2 time
    structured_analysis: dict[str, Any] | None = None  # Call 1 JSON (if structured mode)

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "run_id": self.run_id,
            "model": self.model,
            "provider": self.provider,
            "temperature": self.temperature,
            "elapsed_s": self.elapsed_s,
            "output_hash": self.output_hash,
            "raw_output": self.raw_output,
            "call_count": self.call_count,
            "analysis_elapsed_s": self.analysis_elapsed_s,
            "synthesis_elapsed_s": self.synthesis_elapsed_s,
        }
        if self.structured_analysis is not None:
            d["structured_analysis"] = self.structured_analysis
        return d


@dataclass(frozen=True)
class ScreeningContext:
    """Immutable context flowing through the pipeline."""

    config: ScreeningConfig
    records: tuple[dict[str, Any], ...]
    gate_profiles: tuple[PhysicsProfile, ...] | None = None
    rag_chunks: tuple[Any, ...] | None = None  # tuple[ExperimentalContext, ...] at runtime
    llm_result: LLMSynthesisResult | None = None
    validation_passed: bool | None = None
    validation_notes: tuple[str, ...] = ()
    provenance: tuple[ProvenanceRecord, ...] = ()

    def require_gate_profiles(self) -> tuple[PhysicsProfile, ...]:
        """Return gate_profiles, raising if they have not been computed yet."""
        if self.gate_profiles is None:
            raise ValueError("ScreeningContext.gate_profiles is not set; run gates first")
        return self.gate_profiles


@dataclass(frozen=True)
class StructuredAnalysis:
    """Parsed Call 1 JSON output."""

    raw_json: str
    data: dict[str, Any]
    systems: tuple[dict[str, Any], ...]
    ranking: tuple[dict[str, Any], ...]
    global_observations: tuple[str, ...]
    json_hash: str


@dataclass(frozen=True)
class ValidationResult:
    """Enhanced validation result with four layers."""

    passed: bool
    structural_notes: tuple[str, ...]
    ranking_violations: tuple[str, ...]
    numerical_errors: tuple[dict[str, Any], ...]
    citation_errors: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "structural_notes": list(self.structural_notes),
            "ranking_violations": list(self.ranking_violations),
            "numerical_errors": list(self.numerical_errors),
            "citation_errors": list(self.citation_errors),
        }


# ---------------------------------------------------------------------------
# Prompt building
# ---------------------------------------------------------------------------

_VERDICT_SYMBOL = {"pass": "✓", "flag": "⚠", "fail": "✗"}
_GATE_SHORT_NAMES = {
    "adsorption_mode": "Mode",
    "regime": "Regime",
    "doe_raw": "DOE_raw",
    "doe_corrected": "DOE_corr",
    "deliverability": "T₅₀ window",
    "spontaneity": "ΔG°<0",
}


def _format_gate_table(profiles: Sequence[PhysicsProfile]) -> str:
    """Format gate verdicts as a compact markdown table."""
    if not profiles:
        return "(no systems)"

    gate_names = [v.gate_name for v in profiles[0].verdicts]
    short_names = [_GATE_SHORT_NAMES.get(g, g) for g in gate_names]

    header = "| System | " + " | ".join(short_names) + " | Overall |"
    sep = "|---|" + "|".join(["---"] * len(gate_names)) + "|---|"
    rows = []

    for p in profiles:
        cells = []
        for v in p.verdicts:
            sym = _VERDICT_SYMBOL.get(v.screening_verdict, "?")
            # Compact: short label + symbol
            label = v.physics_label.split(" / ")[0]  # Take first part of compound labels
            cells.append(f"{label} {sym}")
        row = f"| {p.system} | " + " | ".join(cells) + f" | {p.overall.upper()} |"
        rows.append(row)

    return "\n".join([header, sep, *rows])


def _format_config_summary(config: ScreeningConfig) -> str:
    """Summarize gate configuration for the LLM."""
    lines = ["### Gate Configuration"]
    for g in config.gates:
        desc = g.description or g.name
        if g.gate_type == "range_multi":
            mag = " (magnitude)" if g.use_magnitude else ""
            lines.append(f"- **{g.name}**: {desc} — thresholds: {g.thresholds}{mag}")
        elif g.gate_type == "threshold":
            lines.append(
                f"- **{g.name}**: {desc} — {g.column} {g.operator_str} {g.threshold_value}"
            )
        elif g.gate_type in ("doe_window", "entropy_corrected_doe"):
            lines.append(f"- **{g.name}**: {desc} — window bounds: {g.doe_bounds} kJ/mol")
    return "\n".join(lines)


_KEY_DESCRIPTORS = [
    ("System", "System"),
    ("E_ads_kJ_mol", "E_ads (kJ/mol)"),
    ("Delta_H_plus6_kJ_mol", "ΔH_corr (kJ/mol)"),
    ("T50_at_1bar_K", "T₅₀ (K)"),
    ("Delta_G_std_kJ_mol", "ΔG° (kJ/mol)"),
    ("r_H-H_A", "r_H-H (Å)"),
    ("percent_elongation", "Elong%"),
]


# Columns needing higher precision (Å-scale bond lengths)
_HIGH_PRECISION_KEYS = {"r_H-H_A", "Delta_r_H-H_A", "r_H-AE_A", "R_O-AE_A"}


def _format_key_descriptors(records: Sequence[dict[str, Any]]) -> str:
    """Format key descriptor values as a compact markdown table."""
    header = "| " + " | ".join(label for _, label in _KEY_DESCRIPTORS) + " |"
    sep = "|" + "|".join(["---"] * len(_KEY_DESCRIPTORS)) + "|"
    rows = []
    for r in records:
        vals = []
        for key, _ in _KEY_DESCRIPTORS:
            v = r.get(key)
            if v is None or v == "":
                vals.append("—")
            elif isinstance(v, float):
                fmt = ".3f" if key in _HIGH_PRECISION_KEYS else ".1f"
                vals.append(f"{v:{fmt}}")
            else:
                vals.append(str(v))
        rows.append("| " + " | ".join(vals) + " |")
    return "\n".join([header, sep, *rows])


def build_screening_prompt(
    profiles: Sequence[PhysicsProfile],
    records: Sequence[dict[str, Any]],
    config: ScreeningConfig,
    rag_context: str | None = None,
) -> str:
    """Build the three-section structured prompt for LLM synthesis.

    Prompt structure:
        1. IMMUTABLE FACTS — gate verdicts (non-overridable, placed first)
        2. SUPPORTING LITERATURE — RAG chunks with citations (if available)
        3. SYNTHESIS INSTRUCTIONS — three output sections + constraints
    """
    sections = []

    # --- Section 1: Immutable Facts ---
    sections.append(
        "# SECTION 1: IMMUTABLE FACTS\n\n"
        "Gate verdicts below are deterministic and FINAL. Your analysis must respect them.\n\n"
        f"{_format_gate_table(profiles)}\n\n"
        f"{_format_key_descriptors(records)}"
    )

    # --- Section 2: Supporting Literature ---
    if rag_context:
        sections.append(
            "# SECTION 2: SUPPORTING EXPERIMENTAL LITERATURE\n\n"
            "The following experimental results were retrieved from published literature. "
            "Use them to contextualize the computed predictions.\n\n"
            f"{rag_context}"
        )
    else:
        sections.append(
            "# SECTION 2: SUPPORTING LITERATURE\n\n"
            "No experimental literature is available for this screening run. "
            "Base your analysis solely on the computed descriptors and gate verdicts."
        )

    # --- Section 3: Synthesis Instructions ---
    cite_note = "Cite [N] for every literature-backed claim. " if rag_context else ""

    sections.append(
        "# SECTION 3: INSTRUCTIONS\n\n"
        "Role: computational chemist, H₂ adsorption on oxide nanoclusters.\n"
        f"Output: Markdown with ## A, ## B, ## C, ## References. Target ≤800 words. "
        f"{cite_note}Do not fabricate references or data.\n\n"
        "## A. Comparative Analysis\n"
        "1. Compare computed E_ads, T₅₀ against literature values [N]\n"
        "2. Flag DFTB+ method uncertainties that could flip a verdict\n"
        "3. Note which systems lack experimental support\n\n"
        "## B. Constrained Ranking\n"
        "1. Rank: BORDERLINE tier above FAIL tier (mandatory gate constraint)\n"
        "2. Justify each position with descriptor values (E_ads, ΔH_corr, ΔG°, T₅₀)\n"
        "3. Cite literature [N] alongside computed values where applicable\n"
        '4. Flag sensitivity: "X would reclassify if threshold Y → Z"\n\n'
        "## C. Experimental Suggestions\n"
        "1. TPD conditions for borderline candidates (T, p, heating rate)\n"
        "2. Higher-level calculations to reduce DFTB+ uncertainty (method, basis)\n"
        "3. Cite literature [N] where it supports a suggestion\n\n"
        "## References\n"
        "Reproduce [1]–[N] from Section 2."
    )

    return "\n\n---\n\n".join(sections)


def build_analysis_prompt(
    profiles: Sequence[PhysicsProfile],
    records: Sequence[dict[str, Any]],
    config: ScreeningConfig,
    rag_context: str | None = None,
) -> str:
    """Build the Call 1 prompt for structured JSON analysis.

    Prompt structure:
        1. IMMUTABLE FACTS — gate verdicts + key descriptors (same as screening)
        2. SUPPORTING LITERATURE — RAG chunks (if available)
        3. STRUCTURED ANALYSIS TASK — instructs LLM to return ONLY a JSON object
    """
    sections = []

    # --- Section 1: Immutable Facts (reused) ---
    sections.append(
        "# SECTION 1: IMMUTABLE FACTS\n\n"
        "Gate verdicts below are deterministic and FINAL. Your analysis must respect them.\n\n"
        f"{_format_gate_table(profiles)}\n\n"
        f"{_format_key_descriptors(records)}"
    )

    # --- Section 2: Supporting Literature (reused) ---
    if rag_context:
        sections.append(
            "# SECTION 2: SUPPORTING EXPERIMENTAL LITERATURE\n\n"
            "The following experimental results were retrieved from published literature. "
            "Use them to contextualize the computed predictions.\n\n"
            f"{rag_context}"
        )
    else:
        sections.append(
            "# SECTION 2: SUPPORTING LITERATURE\n\n"
            "No experimental literature is available for this screening run. "
            "Base your analysis solely on the computed descriptors and gate verdicts."
        )

    # --- Section 3: Structured Analysis Task (new) ---
    sections.append(
        "# SECTION 3: STRUCTURED ANALYSIS TASK\n\n"
        "Role: computational chemist, H₂ adsorption on oxide nanoclusters.\n"
        "Return ONLY a JSON object matching the schema below. No text outside the JSON.\n\n"
        "Tasks:\n"
        "1. CROSS-DESCRIPTOR ANOMALY DETECTION — flag systems where descriptors conflict "
        "(e.g. strong E_ads but weak elongation, or molecular mode with high binding).\n"
        "2. SENSITIVITY ANALYSIS — for each borderline system, identify which gate threshold "
        "shift would flip its verdict.\n"
        "3. DFTB+ ROBUSTNESS — assess whether DFTB+ method uncertainty could change the "
        "screening outcome for each system.\n"
        "4. LITERATURE GROUNDING — compare computed values against experimental data "
        "from Section 2 (if available).\n"
        "5. RANKING — order systems from most to least promising for H₂ storage, "
        "respecting the gate constraint that non-fail systems rank above fail systems.\n\n"
        "## Output Schema\n\n"
        "```json\n"
        f"{render_schema_for_prompt()}\n"
        "```"
    )

    return "\n\n---\n\n".join(sections)


def build_synthesis_prompt(
    analysis_json: dict[str, Any],
    profiles: Sequence[PhysicsProfile],
    records: Sequence[dict[str, Any]],
    config: ScreeningConfig,
) -> str:
    """Build the Call 2 prompt for narrative synthesis from structured analysis.

    Prompt structure:
        1. IMMUTABLE FACTS — gate verdicts + key descriptors (same as screening)
        2. YOUR STRUCTURED ANALYSIS — the Call 1 JSON output for reference
        3. INSTRUCTIONS — A/B/C synthesis sections with constraints
    """
    sections = []

    # --- Section 1: Immutable Facts (reused) ---
    sections.append(
        "# SECTION 1: IMMUTABLE FACTS\n\n"
        "Gate verdicts below are deterministic and FINAL. Your analysis must respect them.\n\n"
        f"{_format_gate_table(profiles)}\n\n"
        f"{_format_key_descriptors(records)}"
    )

    # --- Section 2: Your Structured Analysis ---
    sections.append(
        "# SECTION 2: YOUR STRUCTURED ANALYSIS\n\n"
        "Below is the structured analysis you produced in the previous step. "
        "You MUST incorporate every element of this analysis into your narrative.\n\n"
        "```json\n"
        f"{json.dumps(analysis_json, indent=2)}\n"
        "```"
    )

    # --- Section 3: Instructions ---
    sections.append(
        "# SECTION 3: INSTRUCTIONS\n\n"
        "Role: computational chemist, H₂ adsorption on oxide nanoclusters.\n"
        "Output: Markdown with ## A, ## B, ## C, ## References. Target ≤800 words. "
        "Do not fabricate references or data.\n\n"
        "## A. Comparative Analysis\n"
        "1. Discuss EVERY anomaly flag from your structured analysis\n"
        "2. Compare computed E_ads, T₅₀ against literature values where available\n"
        "3. Flag DFTB+ method uncertainties that could flip a verdict\n"
        "4. Note which systems lack experimental support\n\n"
        "## B. Constrained Ranking\n"
        "1. Reproduce the ranking from your structured analysis exactly\n"
        "2. Justify each position with descriptor values (E_ads, ΔH_corr, ΔG°, T₅₀)\n"
        '3. Flag sensitivity: "X would reclassify if threshold Y → Z"\n\n'
        "## C. Experimental Suggestions\n"
        "1. Prioritize suggestions for systems where dftb_robustness.could_flip_verdict is true\n"
        "2. TPD conditions for borderline candidates (T, p, heating rate)\n"
        "3. Higher-level calculations to reduce DFTB+ uncertainty (method, basis)\n\n"
        "## References\n"
        "Reproduce any cited references from Section 1."
    )

    return "\n\n---\n\n".join(sections)


# ---------------------------------------------------------------------------
# Post-hoc validation
# ---------------------------------------------------------------------------


def validate_llm_output(
    raw_output: str,
    profiles: Sequence[PhysicsProfile],
) -> tuple[bool, list[str]]:
    """Validate that LLM output doesn't contradict gate verdicts.

    Checks:
        - No FAIL system is ranked above a PASS system in the ranking section
        - Basic structural checks (sections A, B, C present)

    Returns (passed, notes) where notes list any violations.
    """
    notes: list[str] = []

    # Check sections exist
    for section in ("## A", "## B", "## C"):
        if section not in raw_output:
            notes.append(f"Missing required section: {section}")

    # Extract system ordering from the ranking section (heuristic)
    # Non-fail = pass or borderline (both should rank above fail)
    non_fail_systems = {p.system for p in profiles if p.overall != "fail"}
    fail_systems = {p.system for p in profiles if p.overall == "fail"}

    # Find the ranking section
    ranking_section = ""
    if "## B" in raw_output:
        start = raw_output.index("## B")
        end = raw_output.index("## C") if "## C" in raw_output else len(raw_output)
        ranking_section = raw_output[start:end]

    # Check if any FAIL system appears before a non-fail system in the ranking
    if ranking_section and non_fail_systems and fail_systems:
        for fail_sys in fail_systems:
            for good_sys in non_fail_systems:
                fail_pos = ranking_section.find(fail_sys)
                good_pos = ranking_section.find(good_sys)
                if fail_pos >= 0 and good_pos >= 0 and fail_pos < good_pos:
                    notes.append(
                        f"VIOLATION: FAIL system '{fail_sys}' appears before "
                        f"non-fail system '{good_sys}' in ranking"
                    )

    passed = len(notes) == 0
    return passed, notes


def validate_llm_output_v2(
    raw_output: str,
    profiles: Sequence[PhysicsProfile],
    records: Sequence[dict[str, Any]],
    max_ref: int = 0,
    tolerance_pct: float = 1.0,
) -> ValidationResult:
    """Enhanced four-layer validation of LLM synthesis output.

    Layers:
        1. Structural — check ## A, ## B, ## C present
        2. Ranking — no FAIL system ranked above non-FAIL
        3. Numerical — verify numerical claims against source records
        4. Citations — verify citation IDs are within valid range

    ``passed`` is True only when structural_notes AND ranking_violations
    are both empty.  Numerical errors and citation errors are informational.
    """
    structural_notes: list[str] = []
    ranking_violations: list[str] = []

    # --- Layer 1: Structural ---
    for section in ("## A", "## B", "## C"):
        if section not in raw_output:
            structural_notes.append(f"Missing required section: {section}")

    # --- Layer 2: Ranking ---
    non_fail_systems = {p.system for p in profiles if p.overall != "fail"}
    fail_systems = {p.system for p in profiles if p.overall == "fail"}

    ranking_section = ""
    if "## B" in raw_output:
        start = raw_output.index("## B")
        end = raw_output.index("## C") if "## C" in raw_output else len(raw_output)
        ranking_section = raw_output[start:end]

    if ranking_section and non_fail_systems and fail_systems:
        for fail_sys in fail_systems:
            for good_sys in non_fail_systems:
                fail_pos = ranking_section.find(fail_sys)
                good_pos = ranking_section.find(good_sys)
                if fail_pos >= 0 and good_pos >= 0 and fail_pos < good_pos:
                    ranking_violations.append(
                        f"VIOLATION: FAIL system '{fail_sys}' appears before "
                        f"non-fail system '{good_sys}' in ranking"
                    )

    # --- Layer 3: Numerical claims ---
    system_names = [r.get("System", "") for r in records]
    claims = extract_numerical_claims(raw_output, system_names)
    numerical_errors = verify_claims_against_records(
        claims,
        list(records),
        tolerance_pct=tolerance_pct,
    )

    # --- Layer 4: Citations ---
    citation_errors: list[str] = []
    if max_ref > 0:
        cited_ids = extract_citation_ids(raw_output)
        citation_errors = verify_citation_fidelity(cited_ids, max_ref)

    passed = len(structural_notes) == 0 and len(ranking_violations) == 0

    return ValidationResult(
        passed=passed,
        structural_notes=tuple(structural_notes),
        ranking_violations=tuple(ranking_violations),
        numerical_errors=tuple(numerical_errors),
        citation_errors=tuple(citation_errors),
    )


# ---------------------------------------------------------------------------
# Provenance helpers
# ---------------------------------------------------------------------------


def _hash_str(s: str) -> str:
    return sha256(s.encode("utf-8")).hexdigest()[:16]


def _hash_dict(d: object) -> str:
    return _hash_str(json.dumps(d, sort_keys=True, default=str))


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


def _make_provenance(
    stage: str, config_data: object, input_data: object, output_data: object
) -> ProvenanceRecord:
    return ProvenanceRecord(
        stage=stage,
        timestamp_utc=_utc_now(),
        config_hash=_hash_dict(config_data),
        input_hash=_hash_dict(input_data),
        output_hash=_hash_dict(output_data),
    )


# ---------------------------------------------------------------------------
# Pipeline orchestration
# ---------------------------------------------------------------------------


def run_screening_pipeline(
    config: ScreeningConfig,
    records: list[dict[str, Any]],
    *,
    llm: str = "ollama",
    model: str | None = None,
    temperature: float = 0.2,
    timeout_s: float = 300.0,
    retries: int = 2,
    trust_remote_code: bool = False,
    runs: int = 1,
    outdir: str = "results/gator_runs",
    tag: str = "",
    use_tqdm: bool = True,
    use_rag: bool = True,
    do_print: bool = True,
    structured: bool = False,
) -> ScreeningContext:
    """Execute the full GATOR screening pipeline."""

    # --- Initialize context ---
    ctx = ScreeningContext(
        config=config,
        records=tuple(records),
    )

    # --- Stage 1: Deterministic Gates ---
    gate_profiles = run_gates(config, records)
    gate_prov = _make_provenance(
        "gates",
        {g.name: g.gate_type for g in config.gates},
        [r.get("System") for r in records],
        [p.to_dict() for p in gate_profiles],
    )
    ctx = replace(ctx, gate_profiles=gate_profiles, provenance=(*ctx.provenance, gate_prov))

    # --- Stage 2: RAG ---
    rag_context_str = None
    if use_rag:
        from .literature_retrieval import format_rag_context, retrieve_literature

        rag_chunks = retrieve_literature(
            gate_profiles,
            records,
            top_k=config.rag_top_k,
            index_dir=config.rag_literature_dir + "/index",
            embedding_model=config.rag_embedding_model,
        )
        if rag_chunks:
            rag_context_str = format_rag_context(rag_chunks)
            rag_prov = _make_provenance(
                "rag_retrieval",
                {"top_k": config.rag_top_k, "model": config.rag_embedding_model},
                [p.system for p in gate_profiles],
                [c.system for c in rag_chunks],
            )
            ctx = replace(ctx, rag_chunks=rag_chunks, provenance=(*ctx.provenance, rag_prov))

    # --- Single-run mode ---
    if runs == 1:
        result = _run_single(
            ctx,
            rag_context_str,
            llm,
            model,
            temperature,
            timeout_s,
            retries,
            trust_remote_code,
            structured=structured,
        )
        if do_print:
            print(result.llm_result.raw_output if result.llm_result else "(no output)")
        return result

    # --- Multi-run batch mode ---
    return _run_batch(
        ctx,
        rag_context_str,
        llm,
        model,
        temperature,
        timeout_s,
        retries,
        trust_remote_code,
        runs,
        outdir,
        tag,
        use_tqdm,
        do_print,
        structured=structured,
    )


def _run_structured(
    ctx: ScreeningContext,
    rag_context: str | None,
    llm: str,
    model: str | None,
    temperature: float,
    timeout_s: float,
    retries: int,
    trust_remote_code: bool,
) -> ScreeningContext:
    """Execute a structured two-call screening run (analysis JSON → synthesis markdown)."""
    log = logging.getLogger("gator.screening_agent")

    # --- Call 1: Structured analysis ---
    analysis_prompt = build_analysis_prompt(
        ctx.require_gate_profiles(),
        ctx.records,
        ctx.config,
        rag_context,
    )

    t0 = perf_counter()
    raw_analysis = call_llm(
        llm,
        analysis_prompt,
        model=model,
        temperature=temperature,
        timeout_s=timeout_s,
        retries=retries,
        trust_remote_code=trust_remote_code,
    )
    analysis_elapsed = perf_counter() - t0

    # Attempt JSON extraction
    parsed = extract_json_from_response(raw_analysis)
    if parsed is None:
        log.warning(
            "Structured mode: failed to parse JSON from Call 1 response; "
            "falling back to single-shot."
        )
        return _run_single(
            ctx,
            rag_context,
            llm,
            model,
            temperature,
            timeout_s,
            retries,
            trust_remote_code,
            structured=False,
        )

    # Validate the analysis JSON
    expected_systems = [r.get("System", "") for r in ctx.records]
    ok, validation_notes = validate_analysis_json(parsed, expected_systems)
    if not ok:
        log.warning(
            "Structured mode: analysis JSON validation failed (%s); falling back to single-shot.",
            validation_notes,
        )
        return _run_single(
            ctx,
            rag_context,
            llm,
            model,
            temperature,
            timeout_s,
            retries,
            trust_remote_code,
            structured=False,
        )

    # --- Call 2: Synthesis from structured analysis ---
    synthesis_prompt = build_synthesis_prompt(
        parsed,
        ctx.require_gate_profiles(),
        ctx.records,
        ctx.config,
    )

    t1 = perf_counter()
    raw_synthesis = call_llm(
        llm,
        synthesis_prompt,
        model=model,
        temperature=temperature,
        timeout_s=timeout_s,
        retries=retries,
        trust_remote_code=trust_remote_code,
    )
    synthesis_elapsed = perf_counter() - t1

    # --- Validation (v2, 4 layers) ---
    max_ref = 0
    if rag_context is not None:
        cited = extract_citation_ids(rag_context)
        max_ref = max(cited) if cited else 0

    val_result = validate_llm_output_v2(
        raw_synthesis,
        ctx.require_gate_profiles(),
        list(ctx.records),
        max_ref=max_ref,
    )

    # --- Build result ---
    model_used = model or "(default)"
    total_elapsed = analysis_elapsed + synthesis_elapsed

    llm_result = LLMSynthesisResult(
        raw_output=raw_synthesis,
        run_id=str(uuid.uuid4()),
        model=model_used,
        provider=llm,
        temperature=temperature,
        elapsed_s=total_elapsed,
        output_hash=sha256(raw_synthesis.encode()).hexdigest(),
        call_count=2,
        analysis_elapsed_s=analysis_elapsed,
        synthesis_elapsed_s=synthesis_elapsed,
        structured_analysis=parsed,
    )

    # Provenance for both stages
    analysis_prov = _make_provenance(
        "structured_analysis",
        {"provider": llm, "model": model_used, "temperature": temperature},
        _hash_str(analysis_prompt),
        _hash_str(raw_analysis),
    )
    synthesis_prov = _make_provenance(
        "structured_synthesis",
        {"provider": llm, "model": model_used, "temperature": temperature},
        _hash_str(synthesis_prompt),
        llm_result.output_hash,
    )

    # Collect validation notes from all layers
    all_notes: list[str] = []
    all_notes.extend(val_result.structural_notes)
    all_notes.extend(val_result.ranking_violations)
    all_notes.extend(str(e) for e in val_result.numerical_errors)
    all_notes.extend(val_result.citation_errors)

    return replace(
        ctx,
        llm_result=llm_result,
        validation_passed=val_result.passed,
        validation_notes=tuple(all_notes),
        provenance=(*ctx.provenance, analysis_prov, synthesis_prov),
    )


def _run_single(
    ctx: ScreeningContext,
    rag_context: str | None,
    llm: str,
    model: str | None,
    temperature: float,
    timeout_s: float,
    retries: int,
    trust_remote_code: bool,
    *,
    structured: bool = False,
) -> ScreeningContext:
    """Execute a single screening run."""
    if structured:
        return _run_structured(
            ctx,
            rag_context,
            llm,
            model,
            temperature,
            timeout_s,
            retries,
            trust_remote_code,
        )

    prompt = build_screening_prompt(
        ctx.require_gate_profiles(),
        ctx.records,
        ctx.config,
        rag_context,
    )

    t0 = perf_counter()
    raw_output = call_llm(
        llm,
        prompt,
        model=model,
        temperature=temperature,
        timeout_s=timeout_s,
        retries=retries,
        trust_remote_code=trust_remote_code,
    )
    elapsed = perf_counter() - t0

    model_used = model or "(default)"
    llm_result = LLMSynthesisResult(
        raw_output=raw_output,
        run_id=str(uuid.uuid4()),
        model=model_used,
        provider=llm,
        temperature=temperature,
        elapsed_s=elapsed,
        output_hash=sha256(raw_output.encode()).hexdigest(),
    )

    llm_prov = _make_provenance(
        "llm_synthesis",
        {"provider": llm, "model": model_used, "temperature": temperature},
        _hash_str(prompt),
        llm_result.output_hash,
    )

    # Post-hoc validation
    passed, notes = validate_llm_output(raw_output, ctx.require_gate_profiles())
    val_prov = _make_provenance("validation", {}, llm_result.output_hash, {"passed": passed})

    return replace(
        ctx,
        llm_result=llm_result,
        validation_passed=passed,
        validation_notes=tuple(notes),
        provenance=(*ctx.provenance, llm_prov, val_prov),
    )


def _run_batch(
    ctx: ScreeningContext,
    rag_context: str | None,
    llm: str,
    model: str | None,
    temperature: float,
    timeout_s: float,
    retries: int,
    trust_remote_code: bool,
    runs: int,
    outdir: str,
    tag: str,
    use_tqdm: bool,
    do_print: bool,
    *,
    structured: bool = False,
) -> ScreeningContext:
    """Execute multiple screening runs with logging."""
    import traceback

    outdir_path = Path(outdir)
    outdir_path.mkdir(parents=True, exist_ok=True)

    ts = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    tag_str = f"_{tag.strip()}" if tag.strip() else ""
    batch_dir = outdir_path / f"{ts}{tag_str}_{llm}"
    batch_dir.mkdir(parents=True, exist_ok=True)

    # Save manifest
    manifest = {
        "framework": "GATOR",
        "created_at_utc": ts,
        "llm": llm,
        "model": model,
        "temperature": temperature,
        "systems": [r.get("System") for r in ctx.records],
        "runs": runs,
        "gate_config": [g.name for g in ctx.config.gates],
    }
    (batch_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    # Save prompt once
    prompt = build_screening_prompt(
        ctx.require_gate_profiles(), ctx.records, ctx.config, rag_context
    )
    (batch_dir / "prompt.md").write_text(prompt, encoding="utf-8")

    # Save gate profiles once
    gate_data = [p.to_dict() for p in ctx.require_gate_profiles()]
    (batch_dir / "gate_profiles.json").write_text(json.dumps(gate_data, indent=2), encoding="utf-8")

    jsonl_path = batch_dir / "runs.jsonl"
    last_ctx = ctx

    # Optional progress bar
    tqdm_iter = None
    if use_tqdm:
        try:
            from tqdm import tqdm

            tqdm_iter = tqdm
        except ImportError:
            tqdm_iter = None

    with jsonl_path.open("w", encoding="utf-8") as f:
        iterator = range(runs)
        if tqdm_iter is not None:
            iterator = tqdm_iter(iterator, total=runs, desc="GATOR runs", unit="run")

        for i in iterator:
            try:
                result = _run_single(
                    ctx,
                    rag_context,
                    llm,
                    model,
                    temperature,
                    timeout_s,
                    retries,
                    trust_remote_code,
                    structured=structured,
                )
                if result.llm_result is None:
                    raise RuntimeError("run produced no llm_result")
                rec = result.llm_result.to_dict()
                rec["validation_passed"] = result.validation_passed
                rec["validation_notes"] = list(result.validation_notes)
                last_ctx = result
            except Exception as e:
                rec = {
                    "run_id": f"error_{i:04d}",
                    "error": {
                        "type": type(e).__name__,
                        "message": str(e),
                        "traceback": traceback.format_exc(),
                    },
                }

            # Write per-run files
            run_stub = f"run_{i:04d}"
            (batch_dir / f"{run_stub}.json").write_text(json.dumps(rec, indent=2), encoding="utf-8")
            (batch_dir / f"{run_stub}.md").write_text(rec.get("raw_output", ""), encoding="utf-8")

            if structured and isinstance(rec.get("structured_analysis"), dict):
                (batch_dir / f"{run_stub}_analysis.json").write_text(
                    json.dumps(rec["structured_analysis"], indent=2),
                    encoding="utf-8",
                )

            f.write(json.dumps(rec, sort_keys=False) + "\n")
            f.flush()

            if tqdm_iter is None:
                print(f"[{i + 1}/{runs}] wrote {run_stub}", file=sys.stderr)

    if do_print and last_ctx.llm_result:
        print(last_ctx.llm_result.raw_output)

    return last_ctx


# ---------------------------------------------------------------------------
# Multi-model benchmark
# ---------------------------------------------------------------------------


def run_multi_model_benchmark(
    config: ScreeningConfig,
    records: list[dict[str, Any]],
    models: list[str],
    *,
    llm: str = "openrouter",
    temperature: float = 0.2,
    timeout_s: float = 300.0,
    retries: int = 2,
    trust_remote_code: bool = False,
    runs: int = 5,
    outdir: str = "results/gator_runs",
    tag: str = "",
    use_tqdm: bool = True,
    use_rag: bool = True,
    do_print: bool = True,
    structured: bool = False,
) -> None:
    """Run the full pipeline with multiple models for comparison.

    Gates and RAG are computed once; each model gets its own batch
    subdirectory with `runs` repeated LLM calls.
    """
    # --- Shared stages: Gates + RAG ---
    ctx = ScreeningContext(config=config, records=tuple(records))

    gate_profiles = run_gates(config, records)
    gate_prov = _make_provenance(
        "gates",
        {g.name: g.gate_type for g in config.gates},
        [r.get("System") for r in records],
        [p.to_dict() for p in gate_profiles],
    )
    ctx = replace(ctx, gate_profiles=gate_profiles, provenance=(*ctx.provenance, gate_prov))

    rag_context_str = None
    if use_rag:
        from .literature_retrieval import format_rag_context, retrieve_literature

        rag_chunks = retrieve_literature(
            gate_profiles,
            records,
            top_k=config.rag_top_k,
            index_dir=config.rag_literature_dir + "/index",
            embedding_model=config.rag_embedding_model,
        )
        if rag_chunks:
            rag_context_str = format_rag_context(rag_chunks)
            rag_prov = _make_provenance(
                "rag_retrieval",
                {"top_k": config.rag_top_k, "model": config.rag_embedding_model},
                [p.system for p in gate_profiles],
                [c.system for c in rag_chunks],
            )
            ctx = replace(ctx, rag_chunks=rag_chunks, provenance=(*ctx.provenance, rag_prov))

    # --- Create top-level benchmark directory ---
    ts = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    tag_str = f"_{tag.strip()}" if tag.strip() else ""
    benchmark_dir = Path(outdir) / f"{ts}{tag_str}"
    benchmark_dir.mkdir(parents=True, exist_ok=True)

    # Save shared artifacts
    top_manifest = {
        "framework": "GATOR",
        "created_at_utc": ts,
        "llm": llm,
        "models": models,
        "temperature": temperature,
        "systems": [r.get("System") for r in records],
        "runs_per_model": runs,
        "gate_config": [g.name for g in config.gates],
    }
    (benchmark_dir / "manifest.json").write_text(
        json.dumps(top_manifest, indent=2),
        encoding="utf-8",
    )

    prompt = build_screening_prompt(
        ctx.require_gate_profiles(), ctx.records, ctx.config, rag_context_str
    )
    (benchmark_dir / "prompt.md").write_text(prompt, encoding="utf-8")

    gate_data = [p.to_dict() for p in ctx.require_gate_profiles()]
    (benchmark_dir / "gate_profiles.json").write_text(
        json.dumps(gate_data, indent=2),
        encoding="utf-8",
    )

    # --- Per-model runs ---
    summary: list[dict[str, Any]] = []

    for model in models:
        model_short = model.split("/")[-1] if "/" in model else model
        model_dir = benchmark_dir / model_short
        model_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n{'=' * 60}", file=sys.stderr)
        print(f"Model: {model} ({runs} runs)", file=sys.stderr)
        print(f"{'=' * 60}", file=sys.stderr)

        model_manifest = {
            "framework": "GATOR",
            "created_at_utc": ts,
            "llm": llm,
            "model": model,
            "temperature": temperature,
            "systems": [r.get("System") for r in records],
            "runs": runs,
            "gate_config": [g.name for g in config.gates],
        }
        (model_dir / "manifest.json").write_text(
            json.dumps(model_manifest, indent=2),
            encoding="utf-8",
        )

        passed_count = 0
        error_count = 0

        jsonl_path = model_dir / "runs.jsonl"
        tqdm_iter = None
        if use_tqdm:
            try:
                from tqdm import tqdm

                tqdm_iter = tqdm
            except ImportError:
                pass

        with jsonl_path.open("w", encoding="utf-8") as f:
            iterator = range(runs)
            if tqdm_iter is not None:
                iterator = tqdm_iter(iterator, total=runs, desc=model_short, unit="run")

            for i in iterator:
                try:
                    result = _run_single(
                        ctx,
                        rag_context_str,
                        llm,
                        model,
                        temperature,
                        timeout_s,
                        retries,
                        trust_remote_code,
                        structured=structured,
                    )
                    if result.llm_result is None:
                        raise RuntimeError("run produced no llm_result")
                    rec = result.llm_result.to_dict()
                    rec["validation_passed"] = result.validation_passed
                    rec["validation_notes"] = list(result.validation_notes)
                    if result.validation_passed:
                        passed_count += 1
                except Exception as e:
                    import traceback

                    rec = {
                        "run_id": f"error_{i:04d}",
                        "error": {
                            "type": type(e).__name__,
                            "message": str(e),
                            "traceback": traceback.format_exc(),
                        },
                    }
                    error_count += 1

                run_stub = f"run_{i:04d}"
                (model_dir / f"{run_stub}.json").write_text(
                    json.dumps(rec, indent=2),
                    encoding="utf-8",
                )
                (model_dir / f"{run_stub}.md").write_text(
                    rec.get("raw_output", ""),
                    encoding="utf-8",
                )

                if structured and isinstance(rec.get("structured_analysis"), dict):
                    (model_dir / f"{run_stub}_analysis.json").write_text(
                        json.dumps(rec["structured_analysis"], indent=2),
                        encoding="utf-8",
                    )

                f.write(json.dumps(rec, sort_keys=False) + "\n")
                f.flush()

                if tqdm_iter is None:
                    print(f"  [{i + 1}/{runs}] wrote {run_stub}", file=sys.stderr)

        completed = runs - error_count
        summary.append(
            {
                "model": model,
                "runs": runs,
                "completed": completed,
                "errors": error_count,
                "validation_pass_rate": f"{passed_count}/{completed}" if completed else "0/0",
            }
        )

    # --- Print summary ---
    print(f"\n{'=' * 60}", file=sys.stderr)
    print("BENCHMARK SUMMARY", file=sys.stderr)
    print(f"{'=' * 60}", file=sys.stderr)
    for s in summary:
        print(
            f"  {s['model']:40s}  completed={s['completed']}/{s['runs']}  "
            f"validation={s['validation_pass_rate']}  errors={s['errors']}",
            file=sys.stderr,
        )
    print(f"\nResults saved to: {benchmark_dir}", file=sys.stderr)

    # Save summary
    (benchmark_dir / "summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
