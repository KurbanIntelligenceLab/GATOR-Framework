"""
Adsorption metrics calculations for H₂ on TiO₂ nanoclusters.

This module provides classes and functions to calculate bond distances,
adsorption modes, and adsorption energies from optimized xyz geometries.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

# Gas-phase H₂ reference bond length (in Å)
H2_GAS_PHASE_BOND_LENGTH = 0.741

# Mode classification thresholds (in Å)
MODE_MOLECULAR_MAX = 0.80
MODE_ACTIVATED_MIN = 0.80
MODE_ACTIVATED_MAX = 0.95
MODE_DISSOCIATIVE_MIN = 0.95


def parse_xyz_file(filepath: str) -> tuple[list[str], np.ndarray]:
    """
    Parse an xyz file and return element symbols and coordinates.

    Parameters
    ----------
    filepath : str
        Path to the xyz file

    Returns
    -------
    elements : list of str
        List of element symbols
    coordinates : np.ndarray
        Array of shape (n_atoms, 3) with xyz coordinates in Å
    """
    with Path(filepath).open() as f:
        lines = f.readlines()

    n_atoms = int(lines[0].strip())

    elements = []
    coordinates = []

    # Skip first two lines (atom count and comment)
    for i in range(2, 2 + n_atoms):
        parts = lines[i].split()
        elements.append(parts[0])
        coordinates.append([float(parts[1]), float(parts[2]), float(parts[3])])

    return elements, np.array(coordinates)


def calculate_distance(coord1: np.ndarray, coord2: np.ndarray) -> float:
    """
    Calculate Euclidean distance between two 3D points.

    Parameters
    ----------
    coord1 : np.ndarray
        First coordinate (x, y, z)
    coord2 : np.ndarray
        Second coordinate (x, y, z)

    Returns
    -------
    float
        Distance in Å
    """
    return float(np.linalg.norm(coord1 - coord2))


def classify_adsorption_mode(r_hh: float) -> str:
    """
    Classify adsorption mode based on H--H bond distance.

    Parameters
    ----------
    r_hh : float
        H--H bond distance in Å

    Returns
    -------
    str
        Mode classification: 'molecular', 'activated', or 'dissociative'
    """
    if r_hh < MODE_MOLECULAR_MAX:
        return "molecular"
    if MODE_ACTIVATED_MIN <= r_hh <= MODE_ACTIVATED_MAX:
        return "activated"
    # r_hh > MODE_DISSOCIATIVE_MIN
    return "dissociative"


def load_energies_from_tab3(
    csv_filepath: str, system_name: str
) -> tuple[float, float, float] | None:
    """
    Load energy values from metrics_tab3.csv for a given system.

    Parameters
    ----------
    csv_filepath : str
        Path to metrics_tab3.csv file
    system_name : str
        Name of the system to look up

    Returns
    -------
    tuple of (e_np_h2, e_np, e_h2) or None
        Energy values in eV, or None if system not found
    """
    import csv

    try:
        with Path(csv_filepath).open() as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row["System"] == system_name:
                    e_np_h2 = float(row["E_NP_H2_eV"])
                    e_np = float(row["E_NP_eV"])
                    e_h2 = float(row["E_H2_eV"])
                    return (e_np_h2, e_np, e_h2)
        return None
    except FileNotFoundError:
        return None
    except KeyError:
        return None


class AdsorptionMetrics:
    """
    Calculate adsorption metrics from optimized xyz geometries.

    Calculates bond distances, adsorption modes, and adsorption energies
    for H₂ adsorbed on pristine and AE-modified TiO₂ nanoclusters.

    Parameters
    ----------
    xyz_filepath : str
        Path to xyz file of the system with H₂ adsorbed
    material_name : str
        Name of the material/system (e.g., "Pristine", "Ba-TiO2")
    e_np_h2 : float, optional
        Total energy of NP + H₂ system (in eV)
    e_np : float, optional
        Total energy of NP without H₂ (in eV)
    e_h2 : float, optional
        Total energy of isolated H₂ (in eV)

    Attributes
    ----------
    material_name : str
        Name of the material/system
    r_hh : float
        Intramolecular H--H bond length (in Å)
    delta_r_hh : float
        Elongation relative to gas-phase H₂ (in Å)
    percent_elongation : float
        Percentage elongation: 100 × Δr_H-H / 0.741
    mode : str
        Adsorption mode classification: 'molecular', 'activated', or 'dissociative'
    r_h_ae : float, optional
        Shortest H–AE distance (in Å), None if no AE present
    r_o_ae : float, optional
        Nearest surface O–AE distance (in Å), None if no AE present
    e_ads : float, optional
        Adsorption energy (in eV), None if energies not provided
    """

    def __init__(
        self,
        xyz_filepath: str,
        material_name: str,
        e_np_h2: float | None = None,
        e_np: float | None = None,
        e_h2: float | None = None,
        tab3_filepath: str | None = None,
    ) -> None:
        """
        Initialize with xyz file path and optional energies.

        Parameters
        ----------
        xyz_filepath : str
            Path to xyz file of the system with H₂ adsorbed
        material_name : str
            Name of the material/system
        e_np_h2 : float, optional
            Total energy of NP + H₂ system (in eV)
        e_np : float, optional
            Total energy of NP without H₂ (in eV)
        e_h2 : float, optional
            Total energy of isolated H₂ (in eV)
        tab3_filepath : str, optional
            Path to metrics_tab3.csv to automatically load energies
        """
        self.material_name = str(material_name)
        self.xyz_filepath = xyz_filepath

        # Parse xyz file
        self.elements, self.coordinates = parse_xyz_file(xyz_filepath)

        # Calculate all metrics
        self._calculate_metrics()

        # Try to load energies from tab3 if not provided and tab3_filepath given
        if e_np_h2 is None and e_np is None and e_h2 is None and tab3_filepath is not None:
            energies = load_energies_from_tab3(tab3_filepath, self.material_name)
            if energies is not None:
                e_np_h2, e_np, e_h2 = energies

        # Calculate adsorption energy if energies provided
        if e_np_h2 is not None and e_np is not None and e_h2 is not None:
            self.e_ads: float | None = e_np_h2 - e_np - e_h2
        else:
            self.e_ads = None

    def _calculate_metrics(self) -> None:
        """Calculate all adsorption metrics from geometry."""
        # Find H atoms
        h_indices = [i for i, elem in enumerate(self.elements) if elem == "H"]

        if len(h_indices) < 2:
            raise ValueError(f"Expected at least 2 H atoms, found {len(h_indices)}")

        # Calculate H--H bond distance (find the two H atoms closest together)
        # For H₂, there should be exactly 2 H atoms that are bonded
        min_hh_distance = float("inf")
        h1_idx, h2_idx = None, None

        for i in range(len(h_indices)):
            for j in range(i + 1, len(h_indices)):
                dist = calculate_distance(
                    self.coordinates[h_indices[i]], self.coordinates[h_indices[j]]
                )
                if dist < min_hh_distance:
                    min_hh_distance = dist
                    h1_idx = h_indices[i]
                    h2_idx = h_indices[j]

        self.r_hh = min_hh_distance
        self.delta_r_hh = self.r_hh - H2_GAS_PHASE_BOND_LENGTH
        self.percent_elongation = 100.0 * self.delta_r_hh / H2_GAS_PHASE_BOND_LENGTH
        self.mode = classify_adsorption_mode(self.r_hh)

        # Find AE (alkaline earth) atoms: Be, Mg, Ca, Sr, Ba, Ra
        ae_elements = ["Be", "Mg", "Ca", "Sr", "Ba", "Ra"]
        ae_indices = [i for i, elem in enumerate(self.elements) if elem in ae_elements]

        if len(ae_indices) > 0:
            # Calculate shortest H-AE distance
            min_h_ae_distance = float("inf")
            for h_idx in [h1_idx, h2_idx]:
                for ae_idx in ae_indices:
                    dist = calculate_distance(self.coordinates[h_idx], self.coordinates[ae_idx])
                    if dist < min_h_ae_distance:
                        min_h_ae_distance = dist

            self.r_h_ae: float | None = min_h_ae_distance

            # Calculate nearest O-AE distance
            o_indices = [i for i, elem in enumerate(self.elements) if elem == "O"]
            min_o_ae_distance = float("inf")

            for o_idx in o_indices:
                for ae_idx in ae_indices:
                    dist = calculate_distance(self.coordinates[o_idx], self.coordinates[ae_idx])
                    if dist < min_o_ae_distance:
                        min_o_ae_distance = dist

            self.r_o_ae: float | None = min_o_ae_distance
        else:
            # No AE present (pristine system)
            self.r_h_ae = None
            self.r_o_ae = None

    def to_dict(self) -> dict[str, Any]:
        """
        Convert metrics to dictionary format matching metrics_tab2.csv.

        Returns
        -------
        dict
            Dictionary with keys matching CSV column names
        """
        result: dict[str, Any] = {
            "System": self.material_name,
            "r_H-H_A": float(self.r_hh),
            "Delta_r_H-H_A": float(self.delta_r_hh),
            "percent_elongation": float(self.percent_elongation),
            "Mode": self.mode,
        }

        # Add optional fields
        if self.r_h_ae is not None:
            result["r_H-AE_A"] = float(self.r_h_ae)
        else:
            result["r_H-AE_A"] = None

        if self.r_o_ae is not None:
            result["R_O-AE_A"] = float(self.r_o_ae)
        else:
            result["R_O-AE_A"] = None

        if self.e_ads is not None:
            result["E_ads_eV"] = float(self.e_ads)
        else:
            result["E_ads_eV"] = None

        return result

    def __repr__(self) -> str:
        """String representation of the object."""
        energy_str = f", e_ads={self.e_ads:.4f} eV" if self.e_ads is not None else ""
        return (
            f"AdsorptionMetrics(material_name='{self.material_name}', "
            f"r_hh={self.r_hh:.5f} Å, mode='{self.mode}'{energy_str})"
        )

    def __str__(self) -> str:
        """Human-readable string representation."""
        lines = [
            f"Material: {self.material_name}",
            "Adsorption Metrics:",
            f"  H--H bond distance (r_H-H): {self.r_hh:.5f} Å",
            f"  Elongation (Δr_H-H): {self.delta_r_hh:.5f} Å",
            f"  Percent elongation: {self.percent_elongation:.2f}%",
            f"  Adsorption mode: {self.mode}",
        ]

        if self.r_h_ae is not None:
            lines.append(f"  Shortest H-AE distance: {self.r_h_ae:.5f} Å")

        if self.r_o_ae is not None:
            lines.append(f"  Nearest O-AE distance: {self.r_o_ae:.5f} Å")

        if self.e_ads is not None:
            lines.append(f"  Adsorption energy: {self.e_ads:.4f} eV")

        return "\n".join(lines)


def calculate_adsorption_metrics(
    xyz_filepath: str,
    material_name: str,
    e_np_h2: float | None = None,
    e_np: float | None = None,
    e_h2: float | None = None,
    tab3_filepath: str | None = None,
) -> dict[str, Any]:
    """
    Convenience function to calculate all adsorption metrics.

    Parameters
    ----------
    xyz_filepath : str
        Path to xyz file of the system with H₂ adsorbed
    material_name : str
        Name of the material/system
    e_np_h2 : float, optional
        Total energy of NP + H₂ system (in eV)
    e_np : float, optional
        Total energy of NP without H₂ (in eV)
    e_h2 : float, optional
        Total energy of isolated H₂ (in eV)
    tab3_filepath : str, optional
        Path to metrics_tab3.csv to automatically load energies

    Returns
    -------
    dict
        Dictionary with all calculated metrics
    """
    metrics = AdsorptionMetrics(xyz_filepath, material_name, e_np_h2, e_np, e_h2, tab3_filepath)
    return metrics.to_dict()
