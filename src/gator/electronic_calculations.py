"""
Electronic structure calculations for DFTB+ eigenvalue-based quantities.

This module provides classes and functions to calculate electronic properties
from HOMO and LUMO eigenvalues, including band gaps, ionization potentials,
electron affinities, and conceptual DFT descriptors.
"""

from __future__ import annotations

from typing import Any

import numpy as np


class ElectronicProperties:
    """
    Calculate electronic properties from HOMO and LUMO eigenvalues.

    Based on DFTB+ eigenvalue-based quantities:
    - Band gap: E_g = ε_LUMO - ε_HOMO
    - Koopmans-like estimates: IP_Koop ≈ -ε_HOMO, EA_Koop ≈ -ε_LUMO
    - Conceptual DFT descriptors: χ, η, ω

    Parameters
    ----------
    epsilon_homo : float
        HOMO eigenvalue in eV (typically negative)
    epsilon_lumo : float
        LUMO eigenvalue in eV (typically negative)
    material_name : str
        Name of the material/system (e.g., "Pristine", "Ba-TiO2")

    Attributes
    ----------
    epsilon_homo : float
        HOMO eigenvalue in eV
    epsilon_lumo : float
        LUMO eigenvalue in eV
    material_name : str
        Name of the material/system
    band_gap : float
        Band gap E_g = ε_LUMO - ε_HOMO (in eV)
    ip_koop : float
        Koopmans-like ionization potential IP ≈ -ε_HOMO (in eV)
    ea_koop : float
        Koopmans-like electron affinity EA ≈ -ε_LUMO (in eV)
    chi : float
        Electronegativity χ = (IP + EA)/2 (in eV)
    eta : float
        Chemical hardness η = (IP - EA)/2 (in eV)
    omega : float
        Electrophilicity index ω = χ²/(2η) (in eV)
    """

    def __init__(self, epsilon_homo: float, epsilon_lumo: float, material_name: str) -> None:
        """
        Initialize with HOMO and LUMO eigenvalues.

        Parameters
        ----------
        epsilon_homo : float
            HOMO eigenvalue in eV
        epsilon_lumo : float
            LUMO eigenvalue in eV
        material_name : str
            Name of the material/system (e.g., "Pristine", "Ba-TiO2")
        """
        self.epsilon_homo = float(epsilon_homo)
        self.epsilon_lumo = float(epsilon_lumo)
        self.material_name = str(material_name)

        # Calculate all properties
        self._calculate_properties()

    def _calculate_properties(self) -> None:
        """Calculate all electronic properties from HOMO and LUMO values."""
        # Band gap
        self.band_gap = self.epsilon_lumo - self.epsilon_homo

        # Koopmans-like estimates
        ip_raw = -self.epsilon_homo
        ea_raw = -self.epsilon_lumo

        # Round IP and EA to 3 decimals to match CSV format
        # Use floor for values ending in .5 to match CSV behavior (e.g., 8.1165 -> 8.116)
        self.ip_koop = (
            np.floor(ip_raw * 1000) / 1000 if ip_raw * 1000 % 1 == 0.5 else round(ip_raw, 3)
        )
        self.ea_koop = (
            np.floor(ea_raw * 1000) / 1000 if ea_raw * 1000 % 1 == 0.5 else round(ea_raw, 3)
        )

        # Conceptual DFT descriptors (calculated from rounded IP/EA to match CSV)
        self.chi = (self.ip_koop + self.ea_koop) / 2.0
        self.eta = (self.ip_koop - self.ea_koop) / 2.0

        # Electrophilicity index (avoid division by zero)
        if abs(self.eta) < 1e-10:
            self.omega = np.inf if self.chi != 0 else 0.0
        else:
            self.omega = (self.chi**2) / (2.0 * self.eta)

    def to_dict(self, system_name: str | None = None) -> dict[str, Any]:
        """
        Convert properties to dictionary format matching metrics_tab1.csv.

        Parameters
        ----------
        system_name : str, optional
            Override material name if provided, otherwise uses self.material_name

        Returns
        -------
        dict
            Dictionary with keys matching CSV column names
        """
        return {
            "System": system_name if system_name is not None else self.material_name,
            "epsilon_HOMO_eV": self.epsilon_homo,
            "epsilon_LUMO_eV": self.epsilon_lumo,
            "E_g_eV": self.band_gap,
            "IP_Koop_eV": self.ip_koop,
            "EA_Koop_eV": self.ea_koop,
            "chi_eV": self.chi,
            "eta_eV": self.eta,
            "omega_eV": self.omega,
        }

    def __repr__(self) -> str:
        """String representation of the object."""
        return (
            f"ElectronicProperties(epsilon_homo={self.epsilon_homo:.4f} eV, "
            f"epsilon_lumo={self.epsilon_lumo:.4f} eV, "
            f"material_name='{self.material_name}', "
            f"band_gap={self.band_gap:.4f} eV)"
        )

    def __str__(self) -> str:
        """Human-readable string representation."""
        return (
            f"Material: {self.material_name}\n"
            f"Electronic Properties:\n"
            f"  HOMO: {self.epsilon_homo:.4f} eV\n"
            f"  LUMO: {self.epsilon_lumo:.4f} eV\n"
            f"  Band Gap (E_g): {self.band_gap:.4f} eV\n"
            f"  IP (Koopmans): {self.ip_koop:.4f} eV\n"
            f"  EA (Koopmans): {self.ea_koop:.4f} eV\n"
            f"  Electronegativity (χ): {self.chi:.4f} eV\n"
            f"  Hardness (η): {self.eta:.4f} eV\n"
            f"  Electrophilicity (ω): {self.omega:.4f} eV"
        )


def calculate_metrics_from_homo_lumo(
    epsilon_homo: float, epsilon_lumo: float, material_name: str
) -> dict[str, float]:
    """
    Convenience function to calculate all metrics from HOMO and LUMO values.

    Parameters
    ----------
    epsilon_homo : float
        HOMO eigenvalue in eV
    epsilon_lumo : float
        LUMO eigenvalue in eV
    material_name : str
        Name of the material/system

    Returns
    -------
    dict
        Dictionary with all calculated metrics
    """
    props = ElectronicProperties(epsilon_homo, epsilon_lumo, material_name)
    return props.to_dict()
