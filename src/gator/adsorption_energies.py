"""
Adsorption energy calculations with DOE classification.

This module provides classes and functions to calculate adsorption energies
from total electronic energies and classify them according to DOE guidelines.
"""

from __future__ import annotations

from typing import Any

# Conversion factor: 1 eV = 96.485 kJ/mol
EV_TO_KJ_MOL = 96.485

# Broad raw-electronic E_ads screening window (kJ/mol), |E_ads| = 0.2--0.6 eV.
# This is distinct from the DOE/HyMARC entropy-corrected *enthalpy* window
# (15--25 kJ/mol), which applies only to ΔH_corr (see thermodynamic_projections.py).
ELECTRONIC_WINDOW_MIN = 19.3  # = 0.2 eV × 96.485 kJ/mol
ELECTRONIC_WINDOW_MAX = 57.9  # = 0.6 eV × 96.485 kJ/mol


def classify_regime(e_ads_kj_mol: float) -> str:
    """
    Classify adsorption regime based on adsorption energy.

    Parameters
    ----------
    e_ads_kj_mol : float
        Adsorption energy in kJ/mol (negative for exothermic)

    Returns
    -------
    str
        Regime classification
    """
    # Use absolute value for classification
    abs_e_ads = abs(e_ads_kj_mol)

    # Thresholds based on tab3 data:
    # Ba-TiO2: 17.4 kJ/mol -> weak physisorption
    # Ca/Sr-TiO2: ~19.6-19.8 kJ/mol -> moderate physisorption
    # Pristine: 47.8 kJ/mol -> moderate physisorption
    # Mg-TiO2: 53.2 kJ/mol -> strong physisorption / activated
    # Ra-TiO2: 66.4 kJ/mol -> strong physisorption / activated
    # Be-TiO2: 151.0 kJ/mol -> very strong / likely dissociative

    if abs_e_ads < 18:
        return "weak physisorption"
    if abs_e_ads < 50:
        return "moderate physisorption"
    if abs_e_ads < 100:
        return "strong physisorption / activated"
    # abs_e_ads >= 100
    return "very strong / likely dissociative"


def classify_doe_window(e_ads_kj_mol: float) -> str:
    """
    Classify whether the raw electronic adsorption energy falls within the broad
    electronic screening window.

    This is the broad raw-electronic E_ads window, |E_ads| = 0.2--0.6 eV
    (19.3--57.9 kJ/mol). It is distinct from the DOE/HyMARC entropy-corrected
    enthalpy window (15--25 kJ/mol), which applies only to ΔH_corr.

    Parameters
    ----------
    e_ads_kj_mol : float
        Adsorption energy in kJ/mol (negative for exothermic)

    Returns
    -------
    str
        "Inside (19.3--57.9)" if within window, "Outside" otherwise
    """
    abs_e_ads = abs(e_ads_kj_mol)

    if ELECTRONIC_WINDOW_MIN <= abs_e_ads <= ELECTRONIC_WINDOW_MAX:
        return "Inside (19.3--57.9)"
    return "Outside"


class AdsorptionEnergies:
    """
    Calculate adsorption energies and DOE classifications from total energies.

    Calculates adsorption energies from total electronic energies and provides
    classifications according to DOE guidelines for hydrogen storage.

    Parameters
    ----------
    e_np_h2 : float
        Total energy of NP + H₂ system (in eV)
    e_np : float
        Total energy of NP without H₂ (in eV)
    e_h2 : float
        Total energy of isolated H₂ (in eV)
    material_name : str
        Name of the material/system (e.g., "Pristine", "Ba-TiO2")

    Attributes
    ----------
    material_name : str
        Name of the material/system
    e_np_h2 : float
        Total energy of NP + H₂ system (in eV)
    e_np : float
        Total energy of NP without H₂ (in eV)
    e_h2 : float
        Total energy of isolated H₂ (in eV)
    e_ads : float
        Adsorption energy E_ads = E(NP+H₂) - E(NP) - E(H₂) (in eV)
    e_ads_kj_mol : float
        Adsorption energy in kJ/mol
    regime : str
        Qualitative interpretation of adsorption strength
    doe_window : str
        Whether |E_ads| falls inside (19.3--57.9 kJ/mol) or outside the broad
        raw-electronic screening window
    """

    def __init__(self, e_np_h2: float, e_np: float, e_h2: float, material_name: str) -> None:
        """
        Initialize with total energies and material name.

        Parameters
        ----------
        e_np_h2 : float
            Total energy of NP + H₂ system (in eV)
        e_np : float
            Total energy of NP without H₂ (in eV)
        e_h2 : float
            Total energy of isolated H₂ (in eV)
        material_name : str
            Name of the material/system
        """
        self.material_name = str(material_name)
        self.e_np_h2 = float(e_np_h2)
        self.e_np = float(e_np)
        self.e_h2 = float(e_h2)

        # Calculate adsorption energy
        self.e_ads = self.e_np_h2 - self.e_np - self.e_h2

        # Convert to kJ/mol
        self.e_ads_kj_mol = self.e_ads * EV_TO_KJ_MOL

        # Classify regime and DOE window
        self.regime = classify_regime(self.e_ads_kj_mol)
        self.doe_window = classify_doe_window(self.e_ads_kj_mol)

    def to_dict(self) -> dict[str, Any]:
        """
        Convert to dictionary format matching metrics_tab3.csv.

        Returns
        -------
        dict
            Dictionary with keys matching CSV column names
        """
        return {
            "System": self.material_name,
            "E_NP_H2_eV": self.e_np_h2,
            "E_NP_eV": self.e_np,
            "E_H2_eV": self.e_h2,
            "E_ads_eV": self.e_ads,
            "E_ads_kJ_mol": self.e_ads_kj_mol,
            "Regime": self.regime,
            "DOE_window": self.doe_window,
        }

    def __repr__(self) -> str:
        """String representation of the object."""
        return (
            f"AdsorptionEnergies(material_name='{self.material_name}', "
            f"e_ads={self.e_ads:.4f} eV, "
            f"regime='{self.regime}')"
        )

    def __str__(self) -> str:
        """Human-readable string representation."""
        return (
            f"Material: {self.material_name}\n"
            f"Adsorption Energy:\n"
            f"  E(NP+H₂): {self.e_np_h2:.4f} eV\n"
            f"  E(NP): {self.e_np:.4f} eV\n"
            f"  E(H₂): {self.e_h2:.4f} eV\n"
            f"  E_ads: {self.e_ads:.4f} eV ({self.e_ads_kj_mol:.4f} kJ/mol)\n"
            f"  Regime: {self.regime}\n"
            f"  DOE Window: {self.doe_window}"
        )


def calculate_adsorption_energies(
    e_np_h2: float, e_np: float, e_h2: float, material_name: str
) -> dict[str, Any]:
    """
    Convenience function to calculate all adsorption energy metrics.

    Parameters
    ----------
    e_np_h2 : float
        Total energy of NP + H₂ system (in eV)
    e_np : float
        Total energy of NP without H₂ (in eV)
    e_h2 : float
        Total energy of isolated H₂ (in eV)
    material_name : str
        Name of the material/system

    Returns
    -------
    dict
        Dictionary with all calculated metrics
    """
    energies = AdsorptionEnergies(e_np_h2, e_np, e_h2, material_name)
    return energies.to_dict()
