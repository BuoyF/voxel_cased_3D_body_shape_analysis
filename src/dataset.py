"""
Dataset class for voxel-based body composition estimation
"""

import os
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset


class VoxelCompositionDataset(Dataset):
    """
    Dataset for loading voxel maps and corresponding body composition targets
    
    Args:
        voxel_dir: Directory containing .npy voxel files (512×512×3)
        composition_csv: Path to CSV with 66 composition targets
        demographics_csv: Path to CSV with demographics (Height, Weight, Sex, Ethnicity)
        normalize: Whether to z-score normalize composition targets
    """
    
    def __init__(
        self,
        voxel_dir,
        composition_csv,
        demographics_csv,
        normalize=True
    ):
        self.voxel_dir = voxel_dir
        self.normalize = normalize
        
        # Load composition data
        self.comp_data = pd.read_csv(composition_csv)
        self.demo_data = pd.read_csv(demographics_csv)
        
        # Extract composition target columns (66 targets)
        self.comp_cols = [col for col in self.comp_data.columns 
                         if col not in ['Subject_ID', 'Sample_ID']]
        assert len(self.comp_cols) == 66, f"Expected 66 targets, got {len(self.comp_cols)}"
        
        # Compute normalization statistics (on training data)
        if normalize:
            comp_values = self.comp_data[self.comp_cols].values.astype(np.float32)
            self.comp_mean = comp_values.mean(axis=0)
            self.comp_std = comp_values.std(axis=0) + 1e-6
        
        # Merge datasets on Subject_ID
        self.data = pd.merge(
            self.comp_data,
            self.demo_data,
            on='Subject_ID',
            how='inner'
        )
        
        # Filter samples with existing voxel files
        self.samples = []
        for _, row in self.data.iterrows():
            sample_id = row['Sample_ID']
            voxel_path = os.path.join(voxel_dir, f"{sample_id}.npy")
            if os.path.exists(voxel_path):
                self.samples.append((sample_id, voxel_path, row))
        
        print(f"✅ Loaded {len(self.samples)} samples with voxel maps")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample_id, voxel_path, row = self.samples[idx]
        
        # Load voxel map: [H, W, 3] -> transpose to [3, H, W]
        voxel = np.load(voxel_path).astype(np.float32)
        voxel = torch.tensor(voxel.transpose(2, 0, 1), dtype=torch.float32)
        
        # Extract demographics: [height, weight, sex, ethnicity]
        demographics = torch.tensor([
            row['Height'],
            row['Weight'],
            row['Sex'],  # 0=Female, 1=Male
            row['Ethnicity']  # Categorical encoded
        ], dtype=torch.float32)
        
        # Extract composition targets
        comp_values = row[self.comp_cols].values.astype(np.float32)
        if self.normalize:
            comp_values = (comp_values - self.comp_mean) / self.comp_std
        composition = torch.tensor(comp_values, dtype=torch.float32)
        
        return {
            'voxel': voxel,
            'demographics': demographics,
            'composition': composition,
            'sample_id': sample_id
        }
    
    def denormalize(self, normalized_comp):
        """Convert z-normalized predictions back to original scale"""
        if isinstance(normalized_comp, torch.Tensor):
            normalized_comp = normalized_comp.cpu().numpy()
        return normalized_comp * self.comp_std + self.comp_mean
