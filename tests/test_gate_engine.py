"""Tests for the GATOR gate engine.

Verifies gate classifications against known labels.csv values,
ensuring physics labels match the established vocabulary.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from gator.gate_engine import (
    GateVerdict,
    PhysicsProfile,
    ScreeningConfig,
    load_config,
    load_records_from_csv,
    run_gates,
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


def _get_profile(profiles, system: str) -> PhysicsProfile:
    for p in profiles:
        if p.system == system:
            return p
    raise KeyError(f"System '{system}' not found in profiles")


def _get_verdict(profile: PhysicsProfile, gate_name: str) -> GateVerdict:
    for v in profile.verdicts:
        if v.gate_name == gate_name:
            return v
    raise KeyError(f"Gate '{gate_name}' not found in profile for {profile.system}")


# ---------- Config loading ----------


class TestConfigLoading:
    def test_loads_successfully(self, config):
        assert isinstance(config, ScreeningConfig)
        assert len(config.gates) == 6

    def test_gate_names(self, config):
        names = [g.name for g in config.gates]
        assert names == [
            "adsorption_mode",
            "regime",
            "doe_raw",
            "doe_corrected",
            "deliverability",
            "spontaneity",
        ]

    def test_rejects_missing_file(self):
        with pytest.raises(FileNotFoundError):
            load_config("nonexistent.yaml")

    def test_rag_settings(self, config):
        assert config.rag_top_k == 5
        assert config.rag_embedding_model == "all-MiniLM-L6-v2"


# ---------- CSV loading ----------


class TestCSVLoading:
    def test_loads_all_systems(self, records):
        systems = [r["System"] for r in records]
        assert len(systems) == 7
        assert "Pristine" in systems
        assert "Ba-TiO2" in systems

    def test_numeric_parsing(self, records):
        pristine = next(r for r in records if r["System"] == "Pristine")
        assert isinstance(pristine["E_ads_kJ_mol"], float)
        assert pristine["E_ads_kJ_mol"] == pytest.approx(-47.84, abs=0.1)

    def test_filter_systems(self):
        records = load_records_from_csv(DATA_PATH, systems=["Pristine", "Mg-TiO2"])
        assert len(records) == 2

    def test_string_fields_preserved(self, records):
        pristine = next(r for r in records if r["System"] == "Pristine")
        assert pristine["Mode"] == "molecular"


# ---------- Adsorption mode gate ----------


class TestAdsorptionModeGate:
    """Verify mode classifications against known labels.csv values."""

    def test_pristine_molecular(self, profiles):
        v = _get_verdict(_get_profile(profiles, "Pristine"), "adsorption_mode")
        assert v.physics_label == "molecular"
        assert v.screening_verdict == "pass"

    def test_be_activated(self, profiles):
        v = _get_verdict(_get_profile(profiles, "Be-TiO2"), "adsorption_mode")
        assert v.physics_label == "activated"
        assert v.screening_verdict == "flag"

    def test_mg_molecular(self, profiles):
        v = _get_verdict(_get_profile(profiles, "Mg-TiO2"), "adsorption_mode")
        assert v.physics_label == "molecular"
        assert v.screening_verdict == "pass"

    def test_ca_molecular(self, profiles):
        v = _get_verdict(_get_profile(profiles, "Ca-TiO2"), "adsorption_mode")
        assert v.physics_label == "molecular"

    def test_ra_molecular(self, profiles):
        """Ra-TiO2: r_H-H = 0.751 Å, molecular despite strong E_ads."""
        v = _get_verdict(_get_profile(profiles, "Ra-TiO2"), "adsorption_mode")
        assert v.physics_label == "molecular"


# ---------- Regime gate ----------


class TestRegimeGate:
    """Verify regime classification against known classify_regime() thresholds."""

    def test_ba_weak(self, profiles):
        """Ba-TiO2: |E_ads| = 17.4 < 18 → weak physisorption."""
        v = _get_verdict(_get_profile(profiles, "Ba-TiO2"), "regime")
        assert v.physics_label == "weak physisorption"
        assert v.screening_verdict == "fail"

    def test_pristine_moderate(self, profiles):
        """Pristine: |E_ads| = 47.8, 18 ≤ x < 50 → moderate physisorption."""
        v = _get_verdict(_get_profile(profiles, "Pristine"), "regime")
        assert v.physics_label == "moderate physisorption"
        assert v.screening_verdict == "pass"

    def test_mg_strong(self, profiles):
        """Mg-TiO2: |E_ads| = 53.2, 50 ≤ x < 100 → strong physisorption / activated."""
        v = _get_verdict(_get_profile(profiles, "Mg-TiO2"), "regime")
        assert v.physics_label == "strong physisorption / activated"
        assert v.screening_verdict == "flag"

    def test_be_very_strong(self, profiles):
        """Be-TiO2: |E_ads| = 151.0, ≥ 100 → very strong / likely dissociative."""
        v = _get_verdict(_get_profile(profiles, "Be-TiO2"), "regime")
        assert v.physics_label == "very strong / likely dissociative"
        assert v.screening_verdict == "fail"

    def test_ra_strong(self, profiles):
        """Ra-TiO2: |E_ads| = 66.4, 50 ≤ x < 100 → strong physisorption / activated."""
        v = _get_verdict(_get_profile(profiles, "Ra-TiO2"), "regime")
        assert v.physics_label == "strong physisorption / activated"


# ---------- DOE corrected gate ----------


class TestDOECorrectedGate:
    """Verify entropy-corrected DOE labels match labels.csv DOE_plus6 column.

    The deciding adsorption-enthalpy gate uses the +6 correction
    (config column Delta_H_plus6_kJ_mol); the +25.96 scheme is retained only as
    the Table 3A upper-bound sensitivity.
    """

    def test_pristine_outside_strong(self, profiles):
        """Pristine: ΔH+6 = -41.84, |value| > 25 → Outside--strong."""
        v = _get_verdict(_get_profile(profiles, "Pristine"), "doe_corrected")
        assert v.physics_label == "Outside--strong"
        assert v.screening_verdict == "flag"

    def test_be_outside_strong(self, profiles):
        """Be-TiO2: ΔH+6 = -145.0 → Outside--strong."""
        v = _get_verdict(_get_profile(profiles, "Be-TiO2"), "doe_corrected")
        assert v.physics_label == "Outside--strong"
        assert v.screening_verdict == "flag"

    def test_ca_outside_weak(self, profiles):
        """Ca-TiO2: ΔH+6 = -13.80, |value| < 15 → Outside--weak."""
        v = _get_verdict(_get_profile(profiles, "Ca-TiO2"), "doe_corrected")
        assert v.physics_label == "Outside--weak"
        assert v.screening_verdict == "fail"

    def test_mg_outside_strong(self, profiles):
        """Mg-TiO2: ΔH+6 = -47.16, |value| > 25 → Outside--strong."""
        v = _get_verdict(_get_profile(profiles, "Mg-TiO2"), "doe_corrected")
        assert v.physics_label == "Outside--strong"

    def test_sr_outside_weak(self, profiles):
        """Sr-TiO2: ΔH+6 = -13.62, |value| < 15 → Outside--weak."""
        v = _get_verdict(_get_profile(profiles, "Sr-TiO2"), "doe_corrected")
        assert v.physics_label == "Outside--weak"


# ---------- Deliverability gate ----------


class TestDeliverabilityGate:
    def test_pristine_deliverable(self, profiles):
        """Pristine: T50@1bar = 320 K → Deliverable @ ~298 K."""
        v = _get_verdict(_get_profile(profiles, "Pristine"), "deliverability")
        assert v.physics_label == "Deliverable @ ~298 K"
        assert v.screening_verdict == "pass"

    def test_be_heating_needed(self, profiles):
        """Be-TiO2: T50@1bar = 1109 K → Non-deliverable—heating-needed."""
        v = _get_verdict(_get_profile(profiles, "Be-TiO2"), "deliverability")
        assert v.physics_label == "Non-deliverable—heating-needed"
        assert v.screening_verdict == "fail"

    def test_ba_underload(self, profiles):
        """Ba-TiO2: T50@1bar = 88 K → Non-deliverable—under-load."""
        v = _get_verdict(_get_profile(profiles, "Ba-TiO2"), "deliverability")
        assert v.physics_label == "Non-deliverable—under-load"
        assert v.screening_verdict == "fail"

    def test_mg_deliverable(self, profiles):
        """Mg-TiO2: T50@1bar = 361 K → Deliverable @ ~298 K."""
        v = _get_verdict(_get_profile(profiles, "Mg-TiO2"), "deliverability")
        assert v.physics_label == "Deliverable @ ~298 K"

    def test_ra_heating_needed(self, profiles):
        """Ra-TiO2: T50@1bar = 463 K → Non-deliverable—heating-needed."""
        v = _get_verdict(_get_profile(profiles, "Ra-TiO2"), "deliverability")
        assert v.physics_label == "Non-deliverable—heating-needed"


# ---------- Spontaneity gate ----------


class TestSpontaneityGate:
    def test_pristine_spontaneous(self, profiles):
        """Pristine: ΔG° = -2.9 < 0 → spontaneous."""
        v = _get_verdict(_get_profile(profiles, "Pristine"), "spontaneity")
        assert v.physics_label == "spontaneous"
        assert v.screening_verdict == "pass"

    def test_ca_nonspontaneous(self, profiles):
        """Ca-TiO2: ΔG° = 25.1 > 0 → non-spontaneous."""
        v = _get_verdict(_get_profile(profiles, "Ca-TiO2"), "spontaneity")
        assert v.physics_label == "non-spontaneous"
        assert v.screening_verdict == "fail"


# ---------- Overall aggregation ----------


class TestAggregation:
    def test_pristine_overall(self, profiles):
        """Pristine should be borderline or pass depending on gate results."""
        p = _get_profile(profiles, "Pristine")
        # Pristine: mode=pass, regime=pass, doe_raw=pass(Inside 19.3--57.9),
        # doe_corrected=flag(Outside--strong, +6), deliverability=pass, spontaneity=pass
        # → borderline (one flag from doe_corrected)
        assert p.overall in ("pass", "borderline")

    def test_be_overall_fail(self, profiles):
        """Be-TiO2: multiple fails → overall fail."""
        p = _get_profile(profiles, "Be-TiO2")
        assert p.overall == "fail"

    def test_ca_overall_fail(self, profiles):
        """Ca-TiO2: endothermic + underload → fail."""
        p = _get_profile(profiles, "Ca-TiO2")
        assert p.overall == "fail"

    def test_all_systems_have_profiles(self, profiles):
        assert len(profiles) == 7
        systems = {p.system for p in profiles}
        assert "Pristine" in systems
        assert "Be-TiO2" in systems
        assert "Mg-TiO2" in systems
        assert "Ca-TiO2" in systems
        assert "Sr-TiO2" in systems
        assert "Ba-TiO2" in systems
        assert "Ra-TiO2" in systems


# ---------- Immutability ----------


class TestImmutability:
    def test_profile_is_frozen(self, profiles):
        p = profiles[0]
        with pytest.raises(AttributeError):
            p.overall = "hacked"

    def test_verdict_is_frozen(self, profiles):
        v = profiles[0].verdicts[0]
        with pytest.raises(AttributeError):
            v.screening_verdict = "pass"


# ---------- Profile serialization ----------


class TestSerialization:
    def test_to_dict(self, profiles):
        d = profiles[0].to_dict()
        assert "system" in d
        assert "overall" in d
        assert "verdicts" in d
        assert isinstance(d["verdicts"], list)
        assert len(d["verdicts"]) == 6
