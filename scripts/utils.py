"""scripts/utils.py

Small utilities shared across scripts.
"""

from __future__ import annotations

import numpy as np
import torch


def sanitize_material_name(name: str) -> str:
    """Sanitize a material name for filesystem paths.

    - lowercase
    - replace '-' with '_'
    - replace whitespace with '_'
    """
    return "_".join(name.strip().lower().replace("-", "_").split())


def set_seed(seed: int) -> None:
    """Set Python/NumPy/PyTorch seeds for reproducibility."""
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
