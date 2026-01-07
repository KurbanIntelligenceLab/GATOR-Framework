"""
Thermodynamic projections for H₂ adsorption at finite temperature.

This module provides classes and functions to calculate temperature-corrected
adsorption enthalpies, standard free energies, and desorption midpoint
temperatures from electronic adsorption energies.
"""

from typing import Dict, Optional
import numpy as np


# Conversion factor: 1 eV = 96.485 kJ/mol
EV_TO_KJ_MOL = 96.485

# Standard molar entropy of H₂ at 298 K (J/(mol·K))
# From the text: S°(H₂, 298K) ≈ 130.68 J/(mol·K)
S_H2_298_J_MOL_K = 130.68
S_H2_298_KJ_MOL_K = S_H2_298_J_MOL_K / 1000.0  # kJ/(mol·K)

# Entropy loss: ΔS_loss ≈ -S°(H₂, T) per the caption definition
# The negative sign indicates loss of entropy upon adsorption
DELTA_S_LOSS = -S_H2_298_KJ_MOL_K  # kJ/(mol·K) = -0.13068 kJ/(mol·K)

# Gas constant R (kJ/(mol·K))
R_KJ_MOL_K = 0.008314  # 8.314 J/(mol·K) = 0.008314 kJ/(mol·K)

# Standard temperature (K)
T_STD = 298.0

# Entropy corrections at 298 K (kJ/mol)
DELTA_S_T_CORRECTION_MILD = 6.0  # Mild physisorption correction
DELTA_S_T_CORRECTION_LITERATURE = 25.96  # Literature-style upper correction

# DOE window thresholds (kJ/mol)
DOE_WINDOW_MIN = 15.0
DOE_WINDOW_MAX = 25.0


def classify_doe_label(delta_h_corr: float) -> str:
    """
    Classify DOE label based on corrected enthalpy.
    
    Parameters
    ----------
    delta_h_corr : float
        Corrected adsorption enthalpy in kJ/mol
    
    Returns
    -------
    str
        DOE label: "Inside (15--25)", "Outside--weak", "Outside--strong", or "Endothermic"
    """
    abs_delta_h = abs(delta_h_corr)
    
    if delta_h_corr > 0:
        return "Endothermic"
    elif DOE_WINDOW_MIN <= abs_delta_h <= DOE_WINDOW_MAX:
        return "Inside (15--25)"
    elif abs_delta_h < DOE_WINDOW_MIN:
        return "Outside--weak"
    else:  # abs_delta_h > DOE_WINDOW_MAX
        return "Outside--strong"


def calculate_t50(delta_h_corr: float, pressure: float, 
                  delta_s_loss: float = DELTA_S_LOSS) -> float:
    """
    Calculate desorption midpoint temperature T₅₀ at given pressure.
    
    T₅₀(p) = ΔH_corr / (ΔS_loss + R ln p)
    
    Per Equation in the text: T₅₀(p) = ΔH_corr / (ΔS_loss + R ln p)
    where ΔS_loss ≈ S°(H₂, T)
    
    Parameters
    ----------
    delta_h_corr : float
        Corrected adsorption enthalpy in kJ/mol
    pressure : float
        Pressure in bar
    delta_s_loss : float, optional
        Entropy loss in kJ/(mol·K), default is S°(H₂, 298K) = 0.13068 kJ/(mol·K)
    
    Returns
    -------
    float
        Desorption midpoint temperature in K
    """
    if pressure <= 0:
        raise ValueError("Pressure must be positive")
    
    denominator = delta_s_loss + R_KJ_MOL_K * np.log(pressure)
    
    if abs(denominator) < 1e-10:
        raise ValueError("Denominator too small, cannot calculate T₅₀")
    
    return delta_h_corr / denominator


def calculate_delta_g_std(delta_h_corr: float, temperature: float = T_STD,
                          delta_s_loss: float = DELTA_S_LOSS) -> float:
    """
    Calculate standard-state adsorption free energy.
    
    ΔG°(T) = ΔH_corr(T) - T·ΔS_loss
    
    Per Equation in the text: ΔG°(T) = ΔH_corr(T) - T·ΔS_loss
    where ΔS_loss ≈ S°(H₂, T)
    
    Parameters
    ----------
    delta_h_corr : float
        Corrected adsorption enthalpy in kJ/mol
    temperature : float, optional
        Temperature in K, default is 298 K
    delta_s_loss : float, optional
        Entropy loss in kJ/(mol·K), default is S°(H₂, 298K) = 0.13068 kJ/(mol·K)
    
    Returns
    -------
    float
        Standard free energy in kJ/mol
    """
    return delta_h_corr - temperature * delta_s_loss


class ThermodynamicProjections:
    """
    Calculate thermodynamic projections from electronic adsorption energy.
    
    Calculates temperature-corrected enthalpies, standard free energies,
    and desorption midpoint temperatures for H₂ adsorption.
    
    Parameters
    ----------
    e_ads_kj_mol : float
        Electronic adsorption energy in kJ/mol (negative for exothermic)
    material_name : str
        Name of the material/system (e.g., "Pristine", "Ba-TiO2")
    
    Attributes
    ----------
    material_name : str
        Name of the material/system
    e_ads_kj_mol : float
        Electronic adsorption energy in kJ/mol
    delta_h_plus6 : float
        Corrected enthalpy with +6 kJ/mol correction (kJ/mol)
    delta_h_plus25_96 : float
        Corrected enthalpy with +25.96 kJ/mol correction (kJ/mol)
    doe_plus6 : str
        DOE label for +6 kJ/mol correction
    doe_plus25_96 : str
        DOE label for +25.96 kJ/mol correction
    t50_1bar : float
        Desorption midpoint temperature at 1 bar (K)
    t50_30bar : float
        Desorption midpoint temperature at 30 bar (K)
    delta_g_std : float
        Standard free energy at 298 K (kJ/mol)
    """
    
    def __init__(self, e_ads_kj_mol: float, material_name: str):
        """
        Initialize with adsorption energy and material name.
        
        Parameters
        ----------
        e_ads_kj_mol : float
            Electronic adsorption energy in kJ/mol (negative for exothermic)
        material_name : str
            Name of the material/system
        """
        self.material_name = str(material_name)
        self.e_ads_kj_mol = float(e_ads_kj_mol)
        
        # Calculate corrected enthalpies per the definition:
        # ΔH_corr(T) = ΔH + ΔS·T, where ΔH = E_ads
        # For +6 correction: ΔH_corr = E_ads + 6 kJ/mol
        # For +25.96 correction: ΔH_corr = E_ads + 25.96 kJ/mol
        self.delta_h_plus6 = self.e_ads_kj_mol + DELTA_S_T_CORRECTION_MILD
        self.delta_h_plus25_96 = self.e_ads_kj_mol + DELTA_S_T_CORRECTION_LITERATURE
        
        # Classify DOE labels based on corrected enthalpies
        self.doe_plus6 = classify_doe_label(self.delta_h_plus6)
        self.doe_plus25_96 = classify_doe_label(self.delta_h_plus25_96)
        
        # Calculate T₅₀ at different pressures using +25.96 correction
        # T₅₀(p) = ΔH_corr / (ΔS_loss + R ln p)
        # where ΔS_loss ≈ S°(H₂, T) = 0.13068 kJ/(mol·K)
        self.t50_1bar = calculate_t50(self.delta_h_plus25_96, pressure=1.0)
        self.t50_30bar = calculate_t50(self.delta_h_plus25_96, pressure=30.0)
        
        # Calculate standard free energy using +25.96 correction
        # ΔG°(T) = ΔH_corr(T) - T·ΔS_loss
        # where ΔS_loss ≈ S°(H₂, T) = 0.13068 kJ/(mol·K)
        self.delta_g_std = calculate_delta_g_std(self.delta_h_plus25_96)
    
    def to_dict(self) -> Dict:
        """
        Convert to dictionary format matching metrics_tab4.csv.
        
        Returns
        -------
        dict
            Dictionary with keys matching CSV column names
        """
        return {
            'System': self.material_name,
            'Delta_H_plus6_kJ_mol': self.delta_h_plus6,
            'DOE_plus6': self.doe_plus6,
            'Delta_H_plus25_96_kJ_mol': self.delta_h_plus25_96,
            'DOE_plus25_96': self.doe_plus25_96,
            'T50_at_1bar_K': self.t50_1bar,
            'T50_at_30bar_K': self.t50_30bar,
            'Delta_G_std_kJ_mol': self.delta_g_std
        }
    
    def __repr__(self) -> str:
        """String representation of the object."""
        return (f"ThermodynamicProjections(material_name='{self.material_name}', "
                f"e_ads={self.e_ads_kj_mol:.2f} kJ/mol, "
                f"delta_h_+25.96={self.delta_h_plus25_96:.2f} kJ/mol)")
    
    def __str__(self) -> str:
        """Human-readable string representation."""
        return (
            f"Material: {self.material_name}\n"
            f"Thermodynamic Projections (298 K):\n"
            f"  E_ads: {self.e_ads_kj_mol:.2f} kJ/mol\n"
            f"  ΔH_corr (+6): {self.delta_h_plus6:.2f} kJ/mol ({self.doe_plus6})\n"
            f"  ΔH_corr (+25.96): {self.delta_h_plus25_96:.2f} kJ/mol ({self.doe_plus25_96})\n"
            f"  T₅₀ at 1 bar: {self.t50_1bar:.1f} K\n"
            f"  T₅₀ at 30 bar: {self.t50_30bar:.1f} K\n"
            f"  ΔG°(298): {self.delta_g_std:.1f} kJ/mol"
        )


def calculate_thermodynamic_projections(e_ads_kj_mol: float, 
                                       material_name: str) -> Dict:
    """
    Convenience function to calculate all thermodynamic projections.
    
    Parameters
    ----------
    e_ads_kj_mol : float
        Electronic adsorption energy in kJ/mol
    material_name : str
        Name of the material/system
    
    Returns
    -------
    dict
        Dictionary with all calculated metrics
    """
    projections = ThermodynamicProjections(e_ads_kj_mol, material_name)
    return projections.to_dict()
