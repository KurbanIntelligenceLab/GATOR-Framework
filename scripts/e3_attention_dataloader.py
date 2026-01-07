"""
Data loader for Orbital Attention HOMO/LUMO prediction.

This module provides dataset classes for loading molecular data from XYZ files
and matching them with HOMO/LUMO and energy data from CSV files.
Uses .npz caching for faster loading.
"""

import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data, Batch
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple
from tqdm import tqdm


# ============================================================================
# XYZ File Reading
# ============================================================================

def parse_xyz_file(filepath: str) -> Tuple[list, np.ndarray]:
    """Parse an xyz file and return element symbols and coordinates."""
    with open(filepath, 'r') as f:
        lines = f.readlines()
    
    n_atoms = int(lines[0].strip())
    elements = []
    coordinates = []
    
    for i in range(2, 2 + n_atoms):
        parts = lines[i].split()
        elements.append(parts[0])
        coordinates.append([float(parts[1]), float(parts[2]), float(parts[3])])
    
    return elements, np.array(coordinates)


def xyz_to_encoder_input(elements: list, coordinates: np.ndarray, cutoff: float = 5.0):
    """
    Convert XYZ data to encoder input format.
    
    Parameters
    ----------
    elements : list
        List of element symbols
    coordinates : np.ndarray
        Atomic coordinates in Å, shape (n_atoms, 3)
    cutoff : float
        Cutoff radius for edge construction
    
    Returns
    -------
    tuple
        (atomic_numbers, edge_index, edge_diff, edge_vec)
    """
    import ase
    from torch_cluster import radius_graph
    
    # Convert elements to atomic numbers
    atomic_numbers = torch.tensor([ase.data.atomic_numbers[elem] for elem in elements], 
                                  dtype=torch.long)
    
    # Convert coordinates to tensor
    pos = torch.tensor(coordinates, dtype=torch.float)
    
    # Create edges using radius graph
    edge_index = radius_graph(pos, r=cutoff, loop=False)
    
    # Calculate edge distances and vectors
    row, col = edge_index
    edge_vec = pos[row] - pos[col]
    edge_dist = torch.norm(edge_vec, dim=1, keepdim=True)
    
    return atomic_numbers, edge_index, edge_dist, edge_vec


class E3AttentionDataloader(Dataset):
    """
    Data loader for HOMO/LUMO prediction from XYZ files.
    
    Reads HOMO/LUMO from metrics_tab1.csv and energies from metrics_tab3.csv.
    Only uses XYZ files with H2 in the name.
    Uses .npz caching for faster loading.
    
    Parameters
    ----------
    xyz_dir : str
        Directory containing XYZ files
    tab1_file : str
        Path to metrics_tab1.csv with HOMO/LUMO data
    tab3_file : str
        Path to metrics_tab3.csv with energy data
    cutoff : float, optional
        Cutoff radius for graph construction, by default 5.0
    cache_dir : str, optional
        Directory to save/load .npz dataset files, by default "data"
    force_reload : bool, optional
        Force reload from XYZ files even if cache exists, by default False
    
    Attributes
    ----------
    xyz_files : list
        List of matched XYZ file paths
    labels : list
        List of label dictionaries containing homo, lumo, and energy data
    cached_data : list
        List of preprocessed graph data
    normalized_labels : list
        List of normalized label dictionaries
    norm_stats : dict
        Dictionary containing mean and std for each label type
    """

    # Targets supported by the dataset/training scripts.
    #
    # Note: We support predicting E_H2 using the standalone H2 geometry (H2.xyz).
    TARGET_NAMES = ("homo", "lumo", "e_np_h2", "e_np", "e_h2")

    @staticmethod
    def select_target(
        target_name: str,
        *,
        homo,
        lumo,
        e_np_h2,
        e_np,
        e_h2,
    ):
        """Pick a single target tensor/value from a dataloader batch tuple."""
        if target_name == "homo":
            return homo
        if target_name == "lumo":
            return lumo
        if target_name == "e_np_h2":
            return e_np_h2
        if target_name == "e_np":
            return e_np
        if target_name == "e_h2":
            return e_h2
        raise ValueError(f"Unknown target '{target_name}'. Must be one of: {E3AttentionDataloader.TARGET_NAMES}")

    @staticmethod
    def map_system_to_xyz_name(system_name: str) -> str:
        """Map system name from CSV to XYZ filename."""
        if system_name == "H2":
            return "H2.xyz"
        if system_name == "Pristine":
            return "TiO2-H2.xyz"
        return f"{system_name}-H2.xyz"

    def create_train_val_split(self, train_ratio: float = 0.8, seed: int = 42):
        """Create train/validation split from this dataset."""
        from torch.utils.data import random_split

        dataset_size = len(self)

        # For small datasets, use a fixed split
        if dataset_size <= 10:
            val_size = min(2, dataset_size // 4)
            train_size = dataset_size - val_size
        else:
            train_size = int(train_ratio * dataset_size)
            val_size = dataset_size - train_size

        train_dataset, val_dataset = random_split(
            self,
            [train_size, val_size],
            generator=torch.Generator().manual_seed(seed),
        )
        return train_dataset, val_dataset

    def create_k_fold_split(self, n_splits: int = 3, seed: int = 42):
        """Create k-fold splits (list of (train_indices, val_indices))."""
        try:
            from sklearn.model_selection import KFold
        except ImportError:
            # Fallback implementation if sklearn is not available
            import random

            random.seed(seed)
            np.random.seed(seed)

            dataset_size = len(self)
            indices = list(range(dataset_size))
            random.shuffle(indices)

            fold_size = dataset_size // n_splits
            folds = []

            for i in range(n_splits):
                start_idx = i * fold_size
                end_idx = (i + 1) * fold_size if i < n_splits - 1 else dataset_size
                val_idx = indices[start_idx:end_idx]
                train_idx = indices[:start_idx] + indices[end_idx:]
                folds.append((train_idx, val_idx))

            return folds

        dataset_size = len(self)
        kfold = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
        indices = np.arange(dataset_size)
        return [(tr.tolist(), va.tolist()) for tr, va in kfold.split(indices)]
    
    def __init__(
        self, 
        xyz_dir: str, 
        tab1_file: str, 
        tab3_file: str, 
        cutoff: float = 5.0,
        cache_dir: str = "data",
        force_reload: bool = False
    ):
        self.xyz_dir = Path(xyz_dir)
        self.cutoff = cutoff
        self.cache_dir = Path(cache_dir)
        self.force_reload = force_reload
        
        # Generate dataset filename based on input parameters
        dataset_filename = self._generate_cache_filename(tab1_file, tab3_file, cutoff)
        self.cache_path = self.cache_dir / dataset_filename
        
        # Try to load from dataset file
        if not force_reload and self.cache_path.exists():
            print(f"Loading dataset from {self.cache_path}...")
            self._load_from_cache()
        else:
            print("Processing XYZ files and creating dataset...")
            self._process_and_cache(tab1_file, tab3_file)
    
    def _generate_cache_filename(self, tab1_file: str, tab3_file: str, cutoff: float) -> str:
        """Generate dataset filename."""
        tab1 = Path(tab1_file)
        tab3 = Path(tab3_file)
        tab1_mtime = int(tab1.stat().st_mtime) if tab1.exists() else 0
        tab3_mtime = int(tab3.stat().st_mtime) if tab3.exists() else 0
        cutoff_tag = str(float(cutoff)).replace(".", "p")
        return f"orbital_attention_dataset_cutoff{cutoff_tag}_t1{tab1_mtime}_t3{tab3_mtime}.npz"
    
    def _process_and_cache(self, tab1_file: str, tab3_file: str):
        """Process XYZ files and save to cache."""
        # Load CSV files
        print(f"Loading HOMO/LUMO from {tab1_file}...")
        self.tab1_df = pd.read_csv(tab1_file)
        print(f"Loading energies from {tab3_file}...")
        self.tab3_df = pd.read_csv(tab3_file)
        
        # Create mapping from system name to data
        self.system_data = {}
        for _, row in self.tab1_df.iterrows():
            system = row['System']
            self.system_data[system] = {
                'homo': float(row['epsilon_HOMO_eV']),
                'lumo': float(row['epsilon_LUMO_eV'])
            }
        
        # Add energy data from tab3.
        #
        # Updated convention: E_H2 is provided once in a dedicated "H2" row,
        # so we read it as a global constant and fill it for every material.
        self.e_h2_ev = None

        # Read global H2 energy (robust to column placement in the H2 row).
        try:
            h2_rows = self.tab3_df[self.tab3_df["System"].astype(str).str.strip().str.lower() == "h2"]
        except Exception:
            h2_rows = pd.DataFrame()

        if len(h2_rows) > 0:
            h2_row = h2_rows.iloc[0]
            # Prefer E_H2_eV if present, else fall back to E_NP_eV for older/shifted CSVs.
            v = h2_row.get("E_H2_eV", None)
            if v is None or pd.isna(v):
                v = h2_row.get("E_NP_eV", None)
            if v is not None and not pd.isna(v):
                self.e_h2_ev = float(v)
        # If there is no dedicated H2 row, fall back to the first finite value
        # found in the E_H2_eV column (often constant across rows).
        if self.e_h2_ev is None and "E_H2_eV" in self.tab3_df.columns:
            for v in self.tab3_df["E_H2_eV"].tolist():
                if v is not None and not pd.isna(v):
                    self.e_h2_ev = float(v)
                    break

        # Per-system energies (skip NaNs and fill E_H2 from global constant).
        for _, row in self.tab3_df.iterrows():
            system = row.get("System", None)
            if system not in self.system_data:
                continue

            v_np_h2 = row.get("E_NP_H2_eV", None)
            v_np = row.get("E_NP_eV", None)
            v_h2 = row.get("E_H2_eV", None)

            if v_np_h2 is not None and not pd.isna(v_np_h2):
                self.system_data[system]["e_np_h2"] = float(v_np_h2)
            if v_np is not None and not pd.isna(v_np):
                self.system_data[system]["e_np"] = float(v_np)

            # If E_H2_eV is missing per-material, fill from the global H2 row.
            if v_h2 is not None and not pd.isna(v_h2):
                self.system_data[system]["e_h2"] = float(v_h2)
            elif self.e_h2_ev is not None:
                self.system_data[system]["e_h2"] = float(self.e_h2_ev)

        # Add a standalone H2 sample if we have an H2 energy available.
        # HOMO/LUMO and nanoparticle energies are not defined for H2 in this dataset,
        # but we fill them with NaNs and handle them safely during normalization.
        if self.e_h2_ev is not None and "H2" not in self.system_data:
            self.system_data["H2"] = {
                "homo": float("nan"),
                "lumo": float("nan"),
                "e_np_h2": float("nan"),
                "e_np": float("nan"),
                "e_h2": float(self.e_h2_ev),
            }
        
        # Get all XYZ files and match with CSV data.
        # Includes adsorption geometries (*-H2.xyz, TiO2-H2.xyz) plus standalone H2.xyz.
        all_xyz_files = (
            list(self.xyz_dir.glob("*-H2.xyz"))
            + list(self.xyz_dir.glob("TiO2-H2.xyz"))
            + list(self.xyz_dir.glob("H2.xyz"))
        )
        # Remove duplicates
        all_xyz_files = list(set(all_xyz_files))
        
        self.xyz_files = []
        self.labels = []
        
        print("\nMatching XYZ files with CSV data...")
        for xyz_path in all_xyz_files:
            xyz_name = xyz_path.name
            
            # Find matching system in CSV
            matched_system = None
            for system in self.system_data.keys():
                expected_xyz = self.map_system_to_xyz_name(system)
                if xyz_name == expected_xyz:
                    matched_system = system
                    break
            
            if matched_system:
                self.xyz_files.append(xyz_path)
                self.labels.append(self.system_data[matched_system])
                print(f"  ✓ Matched: {xyz_name} -> {matched_system}")
            else:
                print(f"  ✗ No match for: {xyz_name}")
        
        print(f"\nTotal matched files: {len(self.xyz_files)}")
        
        if len(self.xyz_files) == 0:
            raise ValueError("No matching XYZ files found! Check your data directory and CSV files.")
        
        # Process all XYZ files and prepare data for caching
        print("\nProcessing XYZ files...")
        self.cached_data = []
        
        for idx, xyz_path in enumerate(tqdm(self.xyz_files, desc="Processing")):
            # Parse XYZ file
            elements, coordinates = parse_xyz_file(str(xyz_path))
            
            # Convert to encoder input format
            atomic_numbers, edge_index, edge_diff, edge_vec = xyz_to_encoder_input(
                elements, coordinates, cutoff=self.cutoff
            )
            
            # Store data in numpy-compatible format
            data_dict = {
                'z': atomic_numbers.numpy(),
                'pos': np.array(coordinates, dtype=np.float32),
                'edge_index': edge_index.cpu().numpy(),
            }
            
            self.cached_data.append(data_dict)
        
        # Compute normalization statistics for labels
        print("\nComputing label normalization statistics...")
        self._compute_normalization_stats()
        
        # Normalize labels
        print("Normalizing labels...")
        self._normalize_labels()
        
        # Save to dataset file
        print(f"\nSaving dataset to {self.cache_path}...")
        self._save_to_cache()
        print("✓ Dataset saved successfully!")
    
    def _compute_normalization_stats(self):
        """Compute mean and std for label normalization."""
        def finite_vals(key: str) -> list:
            vals = []
            for label in self.labels:
                v = label.get(key, None)
                if v is None:
                    continue
                try:
                    fv = float(v)
                except Exception:
                    continue
                if np.isfinite(fv):
                    vals.append(fv)
            return vals

        homo_vals = finite_vals("homo")
        lumo_vals = finite_vals("lumo")
        e_np_h2_vals = finite_vals("e_np_h2")
        e_np_vals = finite_vals("e_np")
        e_h2_vals = finite_vals("e_h2")
        
        # Compute statistics
        self.norm_stats = {
            'homo': {'mean': np.mean(homo_vals) if homo_vals else 0.0, 'std': np.std(homo_vals) if homo_vals else 1.0},
            'lumo': {'mean': np.mean(lumo_vals) if lumo_vals else 0.0, 'std': np.std(lumo_vals) if lumo_vals else 1.0},
            'e_np_h2': {'mean': np.mean(e_np_h2_vals) if e_np_h2_vals else 0.0, 'std': np.std(e_np_h2_vals) if e_np_h2_vals else 1.0},
            'e_np': {'mean': np.mean(e_np_vals) if e_np_vals else 0.0, 'std': np.std(e_np_vals) if e_np_vals else 1.0},
            'e_h2': {'mean': np.mean(e_h2_vals) if e_h2_vals else 0.0, 'std': np.std(e_h2_vals) if e_h2_vals else 1.0},
        }
        
        # Avoid division by zero
        for key in self.norm_stats:
            if self.norm_stats[key]['std'] < 1e-10:
                self.norm_stats[key]['std'] = 1.0
        
        print(f"  HOMO: mean={self.norm_stats['homo']['mean']:.4f}, std={self.norm_stats['homo']['std']:.4f}")
        print(f"  LUMO: mean={self.norm_stats['lumo']['mean']:.4f}, std={self.norm_stats['lumo']['std']:.4f}")
        print(f"  E_NP_H2: mean={self.norm_stats['e_np_h2']['mean']:.4f}, std={self.norm_stats['e_np_h2']['std']:.4f}")
        print(f"  E_NP: mean={self.norm_stats['e_np']['mean']:.4f}, std={self.norm_stats['e_np']['std']:.4f}")
        print(f"  E_H2: mean={self.norm_stats['e_h2']['mean']:.4f}, std={self.norm_stats['e_h2']['std']:.4f}")
    
    def _normalize_labels(self):
        """Normalize all labels using computed statistics."""
        self.normalized_labels = []
        for label in self.labels:
            # For systems where a label is undefined (e.g., standalone H2 for HOMO/LUMO),
            # fall back to the mean so the normalized value is 0.0 and does not affect
            # training for targets that exclude that system.
            def get_or_mean(key: str) -> float:
                v = label.get(key, None)
                try:
                    fv = float(v)
                except Exception:
                    fv = float("nan")
                if not np.isfinite(fv):
                    fv = float(self.norm_stats[key]["mean"])
                return fv

            normalized = {
                'homo': (get_or_mean("homo") - self.norm_stats['homo']['mean']) / self.norm_stats['homo']['std'],
                'lumo': (get_or_mean("lumo") - self.norm_stats['lumo']['mean']) / self.norm_stats['lumo']['std'],
                'e_np_h2': (get_or_mean("e_np_h2") - self.norm_stats['e_np_h2']['mean']) / self.norm_stats['e_np_h2']['std'],
                'e_np': (get_or_mean("e_np") - self.norm_stats['e_np']['mean']) / self.norm_stats['e_np']['std'],
                'e_h2': (get_or_mean("e_h2") - self.norm_stats['e_h2']['mean']) / self.norm_stats['e_h2']['std'],
            }
            self.normalized_labels.append(normalized)
    
    def _save_to_cache(self):
        """Save processed data to .npz file."""
        # Ensure cache directory exists
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Prepare data for saving
        save_dict = {
            'cutoff': np.array([self.cutoff]),
            'n_samples': np.array([len(self.cached_data)]),
        }
        
        # Save normalization statistics
        save_dict['homo_mean'] = np.array([self.norm_stats['homo']['mean']])
        save_dict['homo_std'] = np.array([self.norm_stats['homo']['std']])
        save_dict['lumo_mean'] = np.array([self.norm_stats['lumo']['mean']])
        save_dict['lumo_std'] = np.array([self.norm_stats['lumo']['std']])
        save_dict['e_np_h2_mean'] = np.array([self.norm_stats['e_np_h2']['mean']])
        save_dict['e_np_h2_std'] = np.array([self.norm_stats['e_np_h2']['std']])
        save_dict['e_np_mean'] = np.array([self.norm_stats['e_np']['mean']])
        save_dict['e_np_std'] = np.array([self.norm_stats['e_np']['std']])
        save_dict['e_h2_mean'] = np.array([self.norm_stats['e_h2']['mean']])
        save_dict['e_h2_std'] = np.array([self.norm_stats['e_h2']['std']])
        
        # Save graph data
        for idx, data_dict in enumerate(self.cached_data):
            save_dict[f'z_{idx}'] = data_dict['z']
            save_dict[f'pos_{idx}'] = data_dict['pos']
            save_dict[f'edge_index_{idx}'] = data_dict['edge_index']
        
        # Save normalized labels
        for idx, label in enumerate(self.normalized_labels):
            save_dict[f'homo_{idx}'] = np.array([label['homo']])
            save_dict[f'lumo_{idx}'] = np.array([label['lumo']])
            save_dict[f'e_np_h2_{idx}'] = np.array([label['e_np_h2']])
            save_dict[f'e_np_{idx}'] = np.array([label['e_np']])
            save_dict[f'e_h2_{idx}'] = np.array([label['e_h2']])
        
        # Save system names for reference
        self.system_names = []
        for xyz_path in self.xyz_files:
            xyz_name = xyz_path.name
            matched_system = None
            for system in self.system_data.keys():
                expected_xyz = self.map_system_to_xyz_name(system)
                if xyz_name == expected_xyz:
                    matched_system = system
                    break
            self.system_names.append(matched_system if matched_system else "Unknown")
        
        save_dict['system_names'] = np.array(self.system_names, dtype=object)
        
        np.savez_compressed(self.cache_path, **save_dict)
    
    def _load_from_cache(self):
        """Load processed data from .npz file."""
        cache_data = np.load(self.cache_path, allow_pickle=True)
        
        self.cutoff = float(cache_data['cutoff'][0])
        n_samples = int(cache_data['n_samples'][0])
        
        # Load normalization statistics
        self.norm_stats = {
            'homo': {
                'mean': float(cache_data['homo_mean'][0]),
                'std': float(cache_data['homo_std'][0])
            },
            'lumo': {
                'mean': float(cache_data['lumo_mean'][0]),
                'std': float(cache_data['lumo_std'][0])
            },
            'e_np_h2': {
                'mean': float(cache_data['e_np_h2_mean'][0]),
                'std': float(cache_data['e_np_h2_std'][0])
            },
            'e_np': {
                'mean': float(cache_data['e_np_mean'][0]),
                'std': float(cache_data['e_np_std'][0])
            },
            'e_h2': {
                'mean': float(cache_data['e_h2_mean'][0]),
                'std': float(cache_data['e_h2_std'][0])
            },
        }
        
        # Load graph data
        self.cached_data = []
        for idx in range(n_samples):
            data_dict = {
                'z': cache_data[f'z_{idx}'],
                'pos': cache_data[f'pos_{idx}'],
                'edge_index': cache_data[f'edge_index_{idx}'],
            }
            self.cached_data.append(data_dict)
        
        # Load normalized labels
        self.normalized_labels = []
        for idx in range(n_samples):
            label = {
                'homo': float(cache_data[f'homo_{idx}'][0]),
                'lumo': float(cache_data[f'lumo_{idx}'][0]),
                'e_np_h2': float(cache_data[f'e_np_h2_{idx}'][0]),
                'e_np': float(cache_data[f'e_np_{idx}'][0]),
                'e_h2': float(cache_data[f'e_h2_{idx}'][0]),
            }
            self.normalized_labels.append(label)
        
        # Load system names
        if 'system_names' in cache_data:
            self.system_names = cache_data['system_names'].tolist()
        else:
            self.system_names = [f"System_{i}" for i in range(n_samples)]
        
        print(f"✓ Loaded {n_samples} samples from dataset file")
        print(f"  Cutoff: {self.cutoff} Å")
        print("  Labels are normalized (mean=0, std=1)")
        
        # Create dummy xyz_files list (not needed when using dataset file)
        self.xyz_files = [None] * n_samples
    
    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.cached_data)
    
    def __getitem__(self, idx: int) -> Tuple[Data, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get a single sample from the dataset.
        
        Parameters
        ----------
        idx : int
            Index of the sample
        
        Returns
        -------
        tuple
            (data, homo, lumo, e_np_h2, e_np, e_h2) where:
            - data: PyTorch Geometric Data object
            - homo: Normalized HOMO energy
            - lumo: Normalized LUMO energy
            - e_np_h2: Normalized total energy of NP + H2
            - e_np: Normalized total energy of NP
            - e_h2: Normalized total energy of H2
        """
        label_data = self.normalized_labels[idx]
        cached_data = self.cached_data[idx]
        
        # Reconstruct PyG data from cached data
        data = Data(
            z=torch.from_numpy(cached_data['z']).long(),
            pos=torch.from_numpy(cached_data['pos']).float(),
            edge_index=torch.from_numpy(cached_data['edge_index']).long(),
        )
        
        # Get normalized labels
        homo = torch.tensor(label_data['homo'], dtype=torch.float)
        lumo = torch.tensor(label_data['lumo'], dtype=torch.float)
        e_np_h2 = torch.tensor(label_data['e_np_h2'], dtype=torch.float)
        e_np = torch.tensor(label_data['e_np'], dtype=torch.float)
        e_h2 = torch.tensor(label_data['e_h2'], dtype=torch.float)
        
        return data, homo, lumo, e_np_h2, e_np, e_h2

    @staticmethod
    def collate_fn(batch):
        """
        Custom collate function for PyG Data objects.

        Args:
            batch: List of tuples (data, homo, lumo, e_np_h2, e_np, e_h2)

        Returns:
            Tuple of (batched_data, homo_tensor, lumo_tensor, e_np_h2_tensor, e_np_tensor, e_h2_tensor)
        """
        data_list, homo_list, lumo_list, e_np_h2_list, e_np_list, e_h2_list = zip(*batch)

        batched_data = Batch.from_data_list(data_list)

        def to_tensor(x):
            if isinstance(x, torch.Tensor):
                return x
            if isinstance(x, np.ndarray):
                return torch.from_numpy(x)
            return torch.tensor(x)

        homo = torch.stack([to_tensor(x) for x in homo_list])
        lumo = torch.stack([to_tensor(x) for x in lumo_list])
        e_np_h2 = torch.stack([to_tensor(x) for x in e_np_h2_list])
        e_np = torch.stack([to_tensor(x) for x in e_np_list])
        e_h2 = torch.stack([to_tensor(x) for x in e_h2_list])

        return batched_data, homo, lumo, e_np_h2, e_np, e_h2
    
    def denormalize(self, homo_norm: float, lumo_norm: float, 
                    e_np_h2_norm: float = None, e_np_norm: float = None, e_h2_norm: float = None):
        """
        Denormalize labels back to original scale.
        
        Parameters
        ----------
        homo_norm : float
            Normalized HOMO value
        lumo_norm : float
            Normalized LUMO value
        e_np_h2_norm : float, optional
            Normalized E_NP_H2 value
        e_np_norm : float, optional
            Normalized E_NP value
        e_h2_norm : float, optional
            Normalized E_H2 value
        
        Returns
        -------
        dict
            Dictionary with denormalized values
        """
        result = {
            'homo': homo_norm * self.norm_stats['homo']['std'] + self.norm_stats['homo']['mean'],
            'lumo': lumo_norm * self.norm_stats['lumo']['std'] + self.norm_stats['lumo']['mean'],
        }
        
        if e_np_h2_norm is not None:
            result['e_np_h2'] = e_np_h2_norm * self.norm_stats['e_np_h2']['std'] + self.norm_stats['e_np_h2']['mean']
        if e_np_norm is not None:
            result['e_np'] = e_np_norm * self.norm_stats['e_np']['std'] + self.norm_stats['e_np']['mean']
        if e_h2_norm is not None:
            result['e_h2'] = e_h2_norm * self.norm_stats['e_h2']['std'] + self.norm_stats['e_h2']['mean']
        
        return result

    def denormalize_value(self, target_name: str, value_norm: float) -> float:
        """Denormalize one normalized scalar for a specific target name."""
        if target_name == "homo":
            return float(self.denormalize(value_norm, 0.0)["homo"])
        if target_name == "lumo":
            return float(self.denormalize(0.0, value_norm)["lumo"])
        if target_name == "e_np_h2":
            return float(self.denormalize(0.0, 0.0, e_np_h2_norm=value_norm)["e_np_h2"])
        if target_name == "e_np":
            return float(self.denormalize(0.0, 0.0, e_np_norm=value_norm)["e_np"])
        if target_name == "e_h2":
            return float(self.denormalize(0.0, 0.0, e_h2_norm=value_norm)["e_h2"])
        raise ValueError(f"Unknown target '{target_name}'. Must be one of: {self.TARGET_NAMES}")
    
    def get_system_name(self, idx: int) -> str:
        """
        Get the system name for a given index.
        
        Parameters
        ----------
        idx : int
            Index of the sample
        
        Returns
        -------
        str
            System name
        """
        if hasattr(self, 'system_names') and idx < len(self.system_names):
            return self.system_names[idx]
        elif self.xyz_files[idx] is not None:
            xyz_name = self.xyz_files[idx].name
            # Find matching system
            if hasattr(self, 'system_data'):
                for system in self.system_data.keys():
                    expected_xyz = self.map_system_to_xyz_name(system)
                    if xyz_name == expected_xyz:
                        return system
        return f"System_{idx}"
    
    def get_statistics(self) -> dict:
        """
        Get statistics about the dataset.
        
        Returns
        -------
        dict
            Dictionary with statistics including normalization parameters
        """
        if not hasattr(self, 'normalized_labels') or len(self.normalized_labels) == 0:
            return {}
        
        # Get normalized values
        homo_norm_values = [d['homo'] for d in self.normalized_labels]
        lumo_norm_values = [d['lumo'] for d in self.normalized_labels]
        
        # Denormalize to get original ranges
        homo_original = [self.denormalize(h, 0.0)['homo'] for h in homo_norm_values]
        lumo_original = [self.denormalize(0.0, lumo_val)["lumo"] for lumo_val in lumo_norm_values]
        
        return {
            'n_samples': len(self.normalized_labels),
            'homo_min': min(homo_original),
            'homo_max': max(homo_original),
            'homo_mean': self.norm_stats['homo']['mean'],
            'homo_std': self.norm_stats['homo']['std'],
            'lumo_min': min(lumo_original),
            'lumo_max': max(lumo_original),
            'lumo_mean': self.norm_stats['lumo']['mean'],
            'lumo_std': self.norm_stats['lumo']['std'],
            'e_np_h2_mean': self.norm_stats['e_np_h2']['mean'],
            'e_np_h2_std': self.norm_stats['e_np_h2']['std'],
            'e_np_mean': self.norm_stats['e_np']['mean'],
            'e_np_std': self.norm_stats['e_np']['std'],
            'e_h2_mean': self.norm_stats['e_h2']['mean'],
            'e_h2_std': self.norm_stats['e_h2']['std'],
        }


def create_train_val_split(dataset: E3AttentionDataloader, train_ratio: float = 0.8, seed: int = 42):
    """Backward-compatible wrapper (prefer dataset.create_train_val_split)."""
    return dataset.create_train_val_split(train_ratio=train_ratio, seed=seed)


def create_k_fold_split(dataset: E3AttentionDataloader, n_splits: int = 3, seed: int = 42):
    """Backward-compatible wrapper (prefer dataset.create_k_fold_split)."""
    return dataset.create_k_fold_split(n_splits=n_splits, seed=seed)


# Backward-compatible wrappers (prefer using the class methods instead)
def map_system_to_xyz_name(system_name: str) -> str:
    return E3AttentionDataloader.map_system_to_xyz_name(system_name)


def select_target(target_name: str, *, homo, lumo, e_np_h2, e_np, e_h2):
    return E3AttentionDataloader.select_target(
        target_name, homo=homo, lumo=lumo, e_np_h2=e_np_h2, e_np=e_np, e_h2=e_h2
    )


def collate_fn(batch):
    return E3AttentionDataloader.collate_fn(batch)
