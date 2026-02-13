"""
Training script for voxel-based body composition estimation
"""

import os
import argparse
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from model import VoxelDemographicNet
from dataset import VoxelCompositionDataset
from utils.metrics import compute_metrics


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    
    for batch in tqdm(loader, desc="Training"):
        voxel = batch['voxel'].to(device)
        demo = batch['demographics'].to(device)
        target = batch['composition'].to(device)
        
        # Forward pass
        pred = model(voxel, demo)
        loss = criterion(pred, target)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(loader)


def validate(model, loader, criterion, device, dataset):
    model.eval()
    all_preds, all_targets = [], []
    
    with torch.no_grad():
        for batch in loader:
            voxel = batch['voxel'].to(device)
            demo = batch['demographics'].to(device)
            target = batch['composition'].to(device)
            
            pred = model(voxel, demo)
            
            all_preds.append(pred.cpu().numpy())
            all_targets.append(target.cpu().numpy())
    
    # Concatenate all predictions
    preds = np.vstack(all_preds)
    targets = np.vstack(all_targets)
    
    # Denormalize to original units
    preds_real = dataset.denormalize(preds)
    targets_real = dataset.denormalize(targets)
    
    # Compute metrics
    metrics = compute_metrics(targets_real, preds_real, dataset.comp_cols)
    
    return metrics


def main(args):
    # Device
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Dataset
    dataset = VoxelCompositionDataset(
        voxel_dir=args.voxel_dir,
        composition_csv=args.csv_path,
        demographics_csv=args.demo_path,
        normalize=True
    )
    
    # Train/val split (90/10)
    indices = list(range(len(dataset)))
    train_idx, val_idx = train_test_split(
        indices, test_size=0.1, random_state=42
    )
    
    train_loader = DataLoader(
        Subset(dataset, train_idx),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4
    )
    val_loader = DataLoader(
        Subset(dataset, val_idx),
        batch_size=args.batch_size,
        num_workers=4
    )
    
    print(f"Train samples: {len(train_idx)}, Val samples: {len(val_idx)}")
    
    # Model
    model = VoxelDemographicNet(
        num_outputs=66,
        use_demographics=True,
        use_residual=True,
        use_se=True
    ).to(device)
    
    # Optimizer & Scheduler
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=1e-4
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=50, eta_min=1e-6
    )
    criterion = nn.MSELoss()
    
    # Training loop
    best_val_loss = float('inf')
    os.makedirs(args.save_dir, exist_ok=True)
    
    for epoch in range(1, args.epochs + 1):
        print(f"\n{'='*50}")
        print(f"Epoch {epoch}/{args.epochs}")
        print(f"{'='*50}")
        
        # Train
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        scheduler.step()
        
        # Validate
        val_metrics = validate(model, val_loader, criterion, device, dataset)
        
        print(f"Train Loss: {train_loss:.4f}")
        print(f"Val R² (Lean): {val_metrics['Total_Lean']['R2']:.4f}")
        print(f"Val R² (Fat): {val_metrics['Total_Fat']['R2']:.4f}")
        print(f"Val R² (BMC): {val_metrics['Total_BMC']['R2']:.4f}")
        
        # Save best model
        if train_loss < best_val_loss:
            best_val_loss = train_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': train_loss,
            }, os.path.join(args.save_dir, 'best_model.pth'))
            print("✅ Saved best model!")
    
    print(f"Training complete! Best val loss: {best_val_loss:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--voxel_dir', type=str, required=True)
    parser.add_argument('--csv_path', type=str, required=True)
    parser.add_argument('--demo_path', type=str, required=True)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--save_dir', type=str, default='checkpoints')
    
    args = parser.parse_args()
    main(args)
