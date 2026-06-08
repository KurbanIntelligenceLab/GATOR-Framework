"""GATOR: Gate-Augmented Thermodynamic Ordering and Retrieval.

A physics-gated screening pipeline with RAG for LLM-assisted
thermodynamic evaluation of hydrogen storage materials.
"""

from __future__ import annotations

__version__ = "0.1.0"

from .adsorption_energies import (
    AdsorptionEnergies,
    calculate_adsorption_energies,
    classify_doe_window,
    classify_regime,
)
from .adsorption_metrics import (
    AdsorptionMetrics,
    calculate_adsorption_metrics,
    classify_adsorption_mode,
    load_energies_from_tab3,
    parse_xyz_file,
)
from .electronic_calculations import (
    ElectronicProperties,
    calculate_metrics_from_homo_lumo,
)
from .gate_engine import (
    GateVerdict,
    PhysicsProfile,
    ScreeningConfig,
    load_config,
    load_records_from_csv,
    run_gates,
)
from .llm_providers import call_llm
from .screening_agent import (
    LLMSynthesisResult,
    ProvenanceRecord,
    ScreeningContext,
    StructuredAnalysis,
    ValidationResult,
    build_analysis_prompt,
    build_screening_prompt,
    build_synthesis_prompt,
    run_screening_pipeline,
    validate_llm_output,
    validate_llm_output_v2,
)
from .thermodynamic_projections import (
    ThermodynamicProjections,
    calculate_delta_g_std,
    calculate_t50,
    calculate_thermodynamic_projections,
    classify_doe_label,
)

__all__ = [
    "AdsorptionEnergies",
    "AdsorptionMetrics",
    "ElectronicProperties",
    "GateVerdict",
    "LLMSynthesisResult",
    "PhysicsProfile",
    "ProvenanceRecord",
    "ScreeningConfig",
    "ScreeningContext",
    "StructuredAnalysis",
    "ThermodynamicProjections",
    "ValidationResult",
    "build_analysis_prompt",
    "build_screening_prompt",
    "build_synthesis_prompt",
    "calculate_adsorption_energies",
    "calculate_adsorption_metrics",
    "calculate_delta_g_std",
    "calculate_metrics_from_homo_lumo",
    "calculate_t50",
    "calculate_thermodynamic_projections",
    "call_llm",
    "classify_adsorption_mode",
    "classify_doe_label",
    "classify_doe_window",
    "classify_regime",
    "load_config",
    "load_energies_from_tab3",
    "load_records_from_csv",
    "parse_xyz_file",
    "run_gates",
    "run_screening_pipeline",
    "validate_llm_output",
    "validate_llm_output_v2",
]
