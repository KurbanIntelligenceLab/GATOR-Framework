"""Tests for GATOR output_schemas module.

Covers JSON extraction from LLM responses, schema validation
for the structured analysis output format, numerical claim
verification, and citation fidelity checking.
"""

from __future__ import annotations

import pytest

from gator.output_schemas import (
    extract_citation_ids,
    extract_json_from_response,
    extract_numerical_claims,
    validate_analysis_json,
    verify_citation_fidelity,
    verify_claims_against_records,
)

# ---------- JSON extraction ----------


class TestExtractJsonFromResponse:
    def test_direct_json_string(self):
        """Parse a valid JSON string directly."""
        raw = '{"systems": [], "ranking": [], "global_observations": []}'
        result = extract_json_from_response(raw)
        assert result == {"systems": [], "ranking": [], "global_observations": []}

    def test_fenced_json_block(self):
        """Extract JSON from a ```json ... ``` fenced block with surrounding text."""
        raw = (
            "Here is the analysis:\n"
            "```json\n"
            '{"systems": [], "ranking": [], "global_observations": []}\n'
            "```\n"
            "That concludes my report."
        )
        result = extract_json_from_response(raw)
        assert result == {"systems": [], "ranking": [], "global_observations": []}

    def test_returns_none_on_garbage(self):
        """Non-JSON input returns None."""
        assert extract_json_from_response("not json at all") is None

    def test_returns_none_on_empty(self):
        """Empty string returns None."""
        assert extract_json_from_response("") is None

    def test_fenced_block_with_extra_whitespace(self):
        """Fenced block with indented JSON still parses correctly."""
        raw = (
            "```json\n"
            "  {\n"
            '    "systems": [],\n'
            '    "ranking": [],\n'
            '    "global_observations": []\n'
            "  }\n"
            "```"
        )
        result = extract_json_from_response(raw)
        assert result == {"systems": [], "ranking": [], "global_observations": []}


# ---------- Helpers for validation tests ----------


def _minimal_system(name: str = "Pristine") -> dict:
    """Return a minimal valid system entry."""
    return {
        "system": name,
        "tier": "pass",
        "anomaly_flags": [],
        "sensitivity": "low",
        "dftb_robustness": "high",
        "literature_support": "consistent",
    }


def _minimal_valid_json(systems: list[str] | None = None) -> dict:
    """Return a minimal valid analysis JSON with given system names."""
    if systems is None:
        systems = ["Pristine"]
    return {
        "systems": [_minimal_system(s) for s in systems],
        "ranking": systems,
        "global_observations": ["All systems look reasonable."],
    }


# ---------- Schema validation ----------


class TestValidateAnalysisJson:
    def test_valid_json_passes(self):
        """Minimal valid JSON with one system passes validation."""
        data = _minimal_valid_json(["Pristine"])
        ok, notes = validate_analysis_json(data, ["Pristine"])
        assert ok is True
        assert notes == []

    def test_missing_top_level_key(self):
        """Deleting a required top-level key causes failure."""
        data = _minimal_valid_json(["Pristine"])
        del data["ranking"]
        ok, notes = validate_analysis_json(data, ["Pristine"])
        assert ok is False
        assert any("ranking" in n for n in notes)

    def test_missing_system_entry(self):
        """Validating against expected systems when one is missing fails."""
        data = _minimal_valid_json(["Pristine"])
        ok, notes = validate_analysis_json(data, ["Pristine", "Mg-TiO2"])
        assert ok is False
        assert any("Mg-TiO2" in n for n in notes)

    def test_invalid_tier_value(self):
        """Setting tier to an invalid value causes failure."""
        data = _minimal_valid_json(["Pristine"])
        data["systems"][0]["tier"] = "unknown"
        ok, notes = validate_analysis_json(data, ["Pristine"])
        assert ok is False
        assert any("tier" in n for n in notes)

    def test_missing_system_level_key(self):
        """Removing a required key from a system entry causes failure."""
        data = _minimal_valid_json(["Pristine"])
        del data["systems"][0]["dftb_robustness"]
        ok, notes = validate_analysis_json(data, ["Pristine"])
        assert ok is False
        assert any("dftb_robustness" in n for n in notes)


# ---------- Numerical claim verification ----------


class TestNumericalClaims:
    def test_extracts_eads_claim(self):
        """Extract an E_ads claim in kJ/mol from markdown text."""
        md = "Mg-TiO2 has E_ads = -53.2 kJ/mol"
        claims = extract_numerical_claims(md, ["Mg-TiO2"])
        assert len(claims) >= 1
        matched = [c for c in claims if c[2] == pytest.approx(-53.2)]
        assert len(matched) == 1
        assert matched[0][0] == "Mg-TiO2"
        assert matched[0][1] == "kJ/mol"

    def test_extracts_t50_claim(self):
        """Extract a T50 claim in K from markdown text."""
        md = "Pristine TiO\u2082 shows T\u2085\u2080 = 320 K at 1 bar."
        claims = extract_numerical_claims(md, ["Pristine"])
        assert len(claims) >= 1
        matched = [c for c in claims if c[2] == pytest.approx(320.0)]
        assert len(matched) == 1
        assert matched[0][0] == "Pristine"
        assert matched[0][1] == "K"

    def test_no_claims_in_text_without_numbers(self):
        """Text with a system name but no numerical values yields no claims."""
        md = "Mg-TiO2 shows moderate binding."
        claims = extract_numerical_claims(md, ["Mg-TiO2"])
        assert claims == []

    def test_verify_correct_claim(self):
        """A claim within tolerance of the source record produces no errors."""
        claims = [("Mg-TiO2", "kJ/mol", -53.16)]
        records = [{"system": "Mg-TiO2", "E_ads_kJ_mol": -53.1632}]
        errors = verify_claims_against_records(claims, records)
        assert errors == []

    def test_verify_wrong_claim(self):
        """A claim far from the source record produces at least one error."""
        claims = [("Mg-TiO2", "kJ/mol", -60.0)]
        records = [{"system": "Mg-TiO2", "E_ads_kJ_mol": -53.1632}]
        errors = verify_claims_against_records(claims, records)
        assert len(errors) >= 1

    def test_verify_skips_empty_column_value(self):
        """An empty-string descriptor (e.g. R_O-AE_A is blank for AE-free
        Pristine) must be skipped, not crash on float("")."""
        claims = [("Pristine", "Å", 0.773)]
        records = [{"system": "Pristine", "r_H-H_A": 0.773, "R_O-AE_A": ""}]
        errors = verify_claims_against_records(claims, records)
        # Matches r_H-H_A within tolerance; the empty R_O-AE_A is ignored.
        assert errors == []


# ---------- Citation fidelity ----------


class TestCitationFidelity:
    def test_extracts_citation_ids(self):
        """Extract bracketed citation IDs from markdown."""
        md = "As shown in [1] and confirmed by [3]"
        ids = extract_citation_ids(md)
        assert ids == {1, 3}

    def test_no_citations(self):
        """Text without citations yields an empty set."""
        md = "No citations here."
        ids = extract_citation_ids(md)
        assert ids == set()

    def test_fidelity_all_valid(self):
        """All cited IDs within range produce no errors."""
        errors = verify_citation_fidelity({1, 2, 3}, max_ref=5)
        assert errors == []

    def test_fidelity_fabricated_citation(self):
        """A citation ID beyond max_ref produces an error mentioning that ID."""
        errors = verify_citation_fidelity({1, 2, 99}, max_ref=5)
        assert len(errors) == 1
        assert "99" in errors[0]
