"""Tests for the GATOR screening agent.

Covers prompt building, post-hoc validation, provenance, and ScreeningContext.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from gator.gate_engine import load_config, load_records_from_csv, run_gates
from gator.screening_agent import (
    LLMSynthesisResult,
    ProvenanceRecord,
    ScreeningContext,
    build_analysis_prompt,
    build_screening_prompt,
    build_synthesis_prompt,
    validate_llm_output,
    validate_llm_output_v2,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = REPO_ROOT / "configs" / "screening_config.yaml"
DATA_PATH = REPO_ROOT / "data" / "labels.csv"


@pytest.fixture
def config():
    return load_config(CONFIG_PATH)


@pytest.fixture
def records():
    return load_records_from_csv(DATA_PATH)


@pytest.fixture
def profiles(config, records):
    return run_gates(config, records)


@pytest.fixture
def prompt(profiles, records, config):
    return build_screening_prompt(profiles, records, config)


# ---------- Prompt structure ----------


class TestPromptBuilding:
    def test_has_three_sections(self, prompt):
        assert "# SECTION 1: IMMUTABLE FACTS" in prompt
        assert "# SECTION 2: SUPPORTING LITERATURE" in prompt
        assert "# SECTION 3: INSTRUCTIONS" in prompt

    def test_contains_all_systems(self, prompt):
        for system in [
            "Pristine",
            "Be-TiO2",
            "Mg-TiO2",
            "Ca-TiO2",
            "Sr-TiO2",
            "Ba-TiO2",
            "Ra-TiO2",
        ]:
            assert system in prompt

    def test_contains_gate_verdicts(self, prompt):
        assert "molecular" in prompt
        assert "activated" in prompt
        assert "✓" in prompt
        assert "✗" in prompt

    def test_contains_key_descriptors(self, prompt):
        assert "E_ads (kJ/mol)" in prompt
        assert "T₅₀ (K)" in prompt

    def test_contains_synthesis_instructions(self, prompt):
        assert "## A. Comparative Analysis" in prompt
        assert "## B. Constrained Ranking" in prompt
        assert "## C. Experimental Suggestions" in prompt

    def test_no_rag_message_without_literature(self, prompt):
        assert "No experimental literature" in prompt

    def test_with_rag_context(self, profiles, records, config):
        rag_text = "### Experimental: TPD on Mg-TiO2 showed desorption at 340 K"
        prompt = build_screening_prompt(profiles, records, config, rag_context=rag_text)
        assert "TPD on Mg-TiO2" in prompt
        assert "No experimental literature" not in prompt

    def test_prompt_is_reasonable_length(self, prompt):
        # Prompt should be substantial but not absurdly long
        assert 2000 < len(prompt) < 50000


# ---------- Post-hoc validation ----------


class TestValidation:
    def test_valid_output_passes(self, profiles):
        """Output with correct section structure and ordering passes."""
        output = (
            "## A. Comparative Analysis\nPristine and Mg-TiO2 agree with experiment.\n"
            "## B. Constrained Ranking\n1. Pristine\n2. Mg-TiO2\n3. Be-TiO2\n"
            "## C. Experimental Design Suggestions\nRun TPD on Mg-TiO2."
        )
        passed, notes = validate_llm_output(output, profiles)
        assert passed is True
        assert len(notes) == 0

    def test_missing_section_flagged(self, profiles):
        """Output missing a required section is flagged."""
        output = "## A. Comparative\n## B. Ranking\nPristine\nMg-TiO2"
        passed, notes = validate_llm_output(output, profiles)
        assert passed is False
        assert any("## C" in n for n in notes)

    def test_fail_system_ranked_above_nonfail_flagged(self, profiles):
        """FAIL system appearing before BORDERLINE system in ranking is caught."""
        # Be-TiO2 is FAIL, Pristine is BORDERLINE
        output = (
            "## A. Analysis\nSome analysis.\n"
            "## B. Constrained Ranking\n"
            "1. Be-TiO2 (best candidate)\n"
            "2. Pristine\n"
            "## C. Suggestions\nDo experiments."
        )
        passed, notes = validate_llm_output(output, profiles)
        assert passed is False
        assert any("VIOLATION" in n for n in notes)
        assert any("Be-TiO2" in n for n in notes)

    def test_multiple_violations(self, profiles):
        """Multiple FAIL systems before non-fail systems all flagged."""
        output = (
            "## A. Analysis\n"
            "## B. Ranking\n"
            "1. Ca-TiO2\n2. Ba-TiO2\n3. Pristine\n4. Mg-TiO2\n"
            "## C. Suggestions"
        )
        passed, notes = validate_llm_output(output, profiles)
        assert passed is False
        # Ca and Ba are both FAIL; Pristine and Mg are borderline
        violations = [n for n in notes if "VIOLATION" in n]
        assert len(violations) >= 1

    def test_empty_output(self, profiles):
        """Empty output should fail validation (missing all sections)."""
        passed, notes = validate_llm_output("", profiles)
        assert passed is False
        assert len(notes) == 3  # Missing ## A, ## B, ## C


# ---------- ScreeningContext immutability ----------


class TestScreeningContext:
    def test_context_is_frozen(self, config, records):
        ctx = ScreeningContext(config=config, records=tuple(records))
        with pytest.raises(AttributeError):
            ctx.config = None

    def test_context_fields(self, config, records):
        ctx = ScreeningContext(config=config, records=tuple(records))
        assert ctx.gate_profiles is None
        assert ctx.rag_chunks is None
        assert ctx.llm_result is None
        assert ctx.validation_passed is None
        assert ctx.provenance == ()


# ---------- ProvenanceRecord ----------


class TestProvenance:
    def test_provenance_is_frozen(self):
        p = ProvenanceRecord(
            stage="test",
            timestamp_utc="2026-03-30T00:00:00Z",
            config_hash="abc",
            input_hash="def",
            output_hash="ghi",
        )
        with pytest.raises(AttributeError):
            p.stage = "hacked"

    def test_provenance_to_dict(self):
        p = ProvenanceRecord(
            stage="gates",
            timestamp_utc="2026-03-30T00:00:00Z",
            config_hash="abc123",
            input_hash="def456",
            output_hash="ghi789",
        )
        d = p.to_dict()
        assert d["stage"] == "gates"
        assert d["config_hash"] == "abc123"


# ---------- LLMSynthesisResult ----------


class TestLLMSynthesisResult:
    def test_result_is_frozen(self):
        r = LLMSynthesisResult(
            raw_output="test",
            run_id="123",
            model="test",
            provider="ollama",
            temperature=0.2,
            elapsed_s=1.0,
            output_hash="abc",
        )
        with pytest.raises(AttributeError):
            r.raw_output = "hacked"

    def test_result_to_dict(self):
        r = LLMSynthesisResult(
            raw_output="report text",
            run_id="run-001",
            model="llama3.2:3b",
            provider="ollama",
            temperature=0.2,
            elapsed_s=5.5,
            output_hash="abc123",
        )
        d = r.to_dict()
        assert d["provider"] == "ollama"
        assert d["model"] == "llama3.2:3b"
        assert d["elapsed_s"] == 5.5


# ---------- Analysis prompt ----------


class TestAnalysisPrompt:
    def test_has_three_sections(self, profiles, records, config):
        prompt = build_analysis_prompt(profiles, records, config)
        assert "SECTION 1: IMMUTABLE FACTS" in prompt
        assert "SECTION 2: SUPPORTING" in prompt
        assert "SECTION 3: STRUCTURED ANALYSIS TASK" in prompt

    def test_contains_json_schema(self, profiles, records, config):
        prompt = build_analysis_prompt(profiles, records, config)
        assert '"systems"' in prompt
        assert '"ranking"' in prompt
        assert '"anomaly_flags"' in prompt

    def test_contains_all_systems(self, profiles, records, config):
        prompt = build_analysis_prompt(profiles, records, config)
        for system in [
            "Pristine",
            "Be-TiO2",
            "Mg-TiO2",
            "Ca-TiO2",
            "Sr-TiO2",
            "Ba-TiO2",
            "Ra-TiO2",
        ]:
            assert system in prompt

    def test_contains_analysis_tasks(self, profiles, records, config):
        prompt = build_analysis_prompt(profiles, records, config)
        for task in [
            "CROSS-DESCRIPTOR ANOMALY DETECTION",
            "SENSITIVITY ANALYSIS",
            "DFTB+ ROBUSTNESS",
            "LITERATURE GROUNDING",
            "RANKING",
        ]:
            assert task in prompt

    def test_contains_gate_verdicts(self, profiles, records, config):
        prompt = build_analysis_prompt(profiles, records, config)
        assert "✓" in prompt or "✗" in prompt

    def test_with_rag_context(self, profiles, records, config):
        rag_text = "### Experimental: TPD on Mg-TiO2 showed desorption at 340 K"
        prompt = build_analysis_prompt(profiles, records, config, rag_context=rag_text)
        assert "TPD on Mg-TiO2" in prompt

    def test_json_only_instruction(self, profiles, records, config):
        prompt = build_analysis_prompt(profiles, records, config)
        assert "Return ONLY a JSON object" in prompt


# ---------- Synthesis prompt ----------


class TestSynthesisPrompt:
    @staticmethod
    def _sample_analysis_json():
        return {
            "systems": [
                {
                    "system": "Pristine",
                    "tier": "borderline",
                    "anomaly_flags": ["large elongation spread"],
                    "sensitivity": "T50 near threshold",
                    "dftb_robustness": "stable",
                    "literature_support": "partial",
                },
            ],
            "ranking": ["Pristine", "Mg-TiO2"],
            "global_observations": ["DFTB+ may overestimate binding for activated systems"],
        }

    def test_has_three_sections(self, profiles, records, config):
        aj = self._sample_analysis_json()
        prompt = build_synthesis_prompt(aj, profiles, records, config)
        assert "SECTION 1: IMMUTABLE FACTS" in prompt
        assert "SECTION 2: YOUR STRUCTURED ANALYSIS" in prompt
        assert "SECTION 3: INSTRUCTIONS" in prompt

    def test_contains_analysis_json(self, profiles, records, config):
        aj = self._sample_analysis_json()
        prompt = build_synthesis_prompt(aj, profiles, records, config)
        assert '"anomaly_flags"' in prompt
        assert "large elongation spread" in prompt

    def test_contains_synthesis_instructions(self, profiles, records, config):
        aj = self._sample_analysis_json()
        prompt = build_synthesis_prompt(aj, profiles, records, config)
        assert "## A. Comparative Analysis" in prompt
        assert "## B. Constrained Ranking" in prompt
        assert "## C. Experimental Suggestions" in prompt

    def test_contains_constraint_language(self, profiles, records, config):
        aj = self._sample_analysis_json()
        prompt = build_synthesis_prompt(aj, profiles, records, config)
        assert "MUST incorporate every element" in prompt
        assert "Reproduce the ranking from your structured analysis exactly" in prompt


# ---------- Validation V2 ----------


class TestValidationV2:
    def test_good_output_passes(self, profiles, records):
        output = (
            "## A. Comparative Analysis\n"
            "Pristine and Mg-TiO2 agree with experiment.\n"
            "## B. Constrained Ranking\n"
            "1. Pristine\n2. Mg-TiO2\n3. Be-TiO2\n"
            "## C. Experimental Suggestions\nRun TPD on Mg-TiO2."
        )
        result = validate_llm_output_v2(output, profiles, records)
        assert result.passed is True
        assert len(result.structural_notes) == 0
        assert len(result.ranking_violations) == 0

    def test_missing_section_flagged(self, profiles, records):
        output = (
            "## A. Comparative Analysis\nSome analysis.\n## B. Constrained Ranking\n1. Pristine\n"
        )
        result = validate_llm_output_v2(output, profiles, records)
        assert result.passed is False
        assert any("## C" in n for n in result.structural_notes)

    def test_fabricated_citation_flagged(self, profiles, records):
        output = (
            "## A. Comparative Analysis\n"
            "Pristine shows E_ads of -47.8 kJ/mol [99].\n"
            "## B. Constrained Ranking\n"
            "1. Pristine\n2. Mg-TiO2\n"
            "## C. Experimental Suggestions\nRun TPD."
        )
        result = validate_llm_output_v2(output, profiles, records, max_ref=5)
        assert any("99" in e for e in result.citation_errors)

    def test_fail_before_nonfail_flagged(self, profiles, records):
        output = (
            "## A. Comparative Analysis\nSome analysis.\n"
            "## B. Constrained Ranking\n"
            "1. Be-TiO2 (best candidate)\n"
            "2. Pristine\n"
            "## C. Experimental Suggestions\nDo experiments."
        )
        result = validate_llm_output_v2(output, profiles, records)
        assert result.passed is False
        assert len(result.ranking_violations) > 0


# ---------- Structured run single ----------


class TestStructuredRunSingle:
    def test_structured_fallback_on_bad_json(self, config, records):
        """If Call 1 returns unparseable JSON, falls back to single-shot."""
        from dataclasses import replace

        from gator.gate_engine import run_gates
        from gator.screening_agent import _run_single

        ctx = ScreeningContext(config=config, records=tuple(records))
        profiles = run_gates(config, records)
        ctx = replace(ctx, gate_profiles=profiles)

        with patch("gator.screening_agent.call_llm") as mock_llm:
            mock_llm.side_effect = [
                "this is not json at all",  # Call 1 fails
                # Fallback single-shot
                "## A. Analysis\nGood.\n## B. Ranking\n1. Pristine\n## C. Suggestions\nTPD.",
            ]
            result = _run_single(ctx, None, "ollama", None, 0.2, 30.0, 0, False, structured=True)
            assert result.llm_result is not None
            assert result.llm_result.call_count == 1  # Fell back
            assert "## A" in result.llm_result.raw_output

    def test_structured_success_two_calls(self, config, records):
        """Successful structured run makes 2 LLM calls."""
        import json as json_mod
        from dataclasses import replace

        from gator.gate_engine import run_gates
        from gator.screening_agent import _run_single

        ctx = ScreeningContext(config=config, records=tuple(records))
        profiles = run_gates(config, records)
        ctx = replace(ctx, gate_profiles=profiles)

        valid_json = json_mod.dumps(
            {
                "systems": [
                    {
                        "system": r.get("System"),
                        "tier": "borderline",
                        "anomaly_flags": [],
                        "sensitivity": [],
                        "dftb_robustness": {
                            "confidence": "high",
                            "rationale": "OK",
                            "could_flip_verdict": False,
                        },
                        "literature_support": {
                            "has_direct_experimental": False,
                            "citation_ids": [],
                            "support_strength": "none",
                        },
                    }
                    for r in records
                ],
                "ranking": [
                    {
                        "rank": i + 1,
                        "system": r.get("System"),
                        "tier": "borderline",
                        "rationale": "test",
                    }
                    for i, r in enumerate(records)
                ],
                "global_observations": ["test note"],
            }
        )

        synthesis_md = "## A. Analysis\nDone.\n## B. Ranking\n1. Pristine\n## C. Suggestions\nTPD."

        with patch("gator.screening_agent.call_llm") as mock_llm:
            mock_llm.side_effect = [valid_json, synthesis_md]
            result = _run_single(ctx, None, "ollama", None, 0.2, 30.0, 0, False, structured=True)
            assert result.llm_result is not None
            assert result.llm_result.call_count == 2
            assert result.llm_result.structured_analysis is not None
            assert "## A" in result.llm_result.raw_output
