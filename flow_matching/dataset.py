"""
Flow Matching Dataset - Phase 3

加载 latent vectors + conditions
"""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path


class LatentConditionDataset(Dataset):
    """
    Dataset for Conditional Flow Matching
    
    Returns:
        z: latent vector (64,)
        c: conditions [charge, hydrophobicity, length] normalized to [0, 1]
    """
    
    def __init__(self, latents_path, metadata_path, indices, 
                 charge_range=(-20, 30), hydro_range=(-4.5, 4.2), length_range=(2, 100)):
        """
        Args:
            latents_path: path to *_latents.npz
            metadata_path: path to metadata_amp_only.csv
            indices: which indices to use (from splits)
            charge_range: (min, max) for normalization
            hydro_range: (min, max) for normalization
            length_range: (min, max) for normalization
        """
        # Load latents
        data = np.load(latents_path)
        self.latents = data["latents"]  # (N, 64)
        
        # Load metadata
        self.metadata = pd.read_csv(metadata_path)
        
        # Filter by indices
        self.indices = indices
        
        # Normalization ranges
        self.charge_range = charge_range
        self.hydro_range = hydro_range
        self.length_range = length_range
        
        print(f"Loaded {len(self.indices)} samples")
    
    def normalize(self, value, min_val, max_val):
        """Normalize to [0, 1]"""
        return (value - min_val) / (max_val - min_val + 1e-8)
    
    def denormalize(self, value, min_val, max_val):
        """Denormalize from [0, 1] to original range"""
        return value * (max_val - min_val) + min_val
    
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        """
        Returns:
            z: (64,) latent vector
            c: (3,) conditions [charge_norm, hydro_norm, length_norm]
        """
        # Important: latents are already filtered, so idx is direct index
        # But we need to map back to metadata using self.indices
        metadata_idx = self.indices[idx]
        
        # Get latent (use idx directly, since latents are already filtered)
        z = self.latents[idx].astype(np.float32)
        
        # Get conditions (use metadata_idx to get correct row from metadata)
        row = self.metadata.iloc[metadata_idx]
        charge = row["charge"]
        hydro = row["hydrophobicity"]
        length = row["length"]
        
        # Normalize conditions to [0, 1]
        charge_norm = self.normalize(charge, *self.charge_range)
        hydro_norm = self.normalize(hydro, *self.hydro_range)
        length_norm = self.normalize(length, *self.length_range)
        
        c = np.array([charge_norm, hydro_norm, length_norm], dtype=np.float32)
        
        return torch.from_numpy(z), torch.from_numpy(c)


def get_dataloaders(latents_dir, metadata_path, splits_dir, batch_size=512, num_workers=4):
    """
    Create train/val/test dataloaders
    
    Args:
        latents_dir: path to outputs/rae_amp_only_eval/
        metadata_path: path to data/embeddings/metadata_amp_only.csv
        splits_dir: path to data/splits_amp_only/
        batch_size: batch size
        num_workers: num workers
    """
    latents_dir = Path(latents_dir)
    splits_dir = Path(splits_dir)
    
    # Load indices
    train_indices = np.load(splits_dir / "train_indices.npy")
    val_indices = np.load(splits_dir / "val_indices.npy")
    test_indices = np.load(splits_dir / "test_indices.npy")
    
    # Create datasets
    train_dataset = LatentConditionDataset(
        latents_dir / "train_latents.npz",
        metadata_path,
        train_indices
    )
    
    val_dataset = LatentConditionDataset(
        latents_dir / "val_latents.npz",
        metadata_path,
        val_indices
    )
    
    test_dataset = LatentConditionDataset(
        latents_dir / "test_latents.npz",
        metadata_path,
        test_indices
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    
    return train_loader, val_loader, test_loader


def test_dataset():
    """Test dataset"""
    train_loader, val_loader, test_loader = get_dataloaders(
        latents_dir="outputs/rae_eval",
        metadata_path="data/embeddings/metadata.csv",
        splits_dir="data/splits",
        batch_size=32,
        num_workers=0,
    )
    
    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    print(f"Test batches: {len(test_loader)}")
    
    # Test one batch
    z, c = next(iter(train_loader))
    print(f"\nBatch shapes:")
    print(f"  z: {z.shape}, dtype: {z.dtype}")
    print(f"  c: {c.shape}, dtype: {c.dtype}")
    print(f"\nCondition stats:")
    print(f"  charge (norm):        min={c[:, 0].min():.3f}, max={c[:, 0].max():.3f}")
    print(f"  hydrophobicity (norm): min={c[:, 1].min():.3f}, max={c[:, 1].max():.3f}")
    print(f"  length (norm):        min={c[:, 2].min():.3f}, max={c[:, 2].max():.3f}")


if __name__ == "__main__":
    test_dataset()

