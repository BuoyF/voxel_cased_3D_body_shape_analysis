
import os
import torch
import pandas as pd
import numpy as np
from torch import nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import pearsonr
from tqdm import tqdm
import torch.nn.functional as F
import time
import platform

# ----------------------------- Voxel Map Dataset ----------------------------- #
class VoxelCompositionDataset(Dataset):
    def __init__(self, csv_path, voxel_dir="human_voxelmaps"):
        self.data = pd.read_csv(csv_path).fillna(0.0)
        self.comp_names = [col for col in self.data.columns if col.endswith(")")]
        self.voxel_dir = voxel_dir
        self.means = self.data[self.comp_names].mean().values.astype(np.float32)
        self.stds = self.data[self.comp_names].std().values.astype(np.float32) + 1e-6

        self.samples = []
        for _, row in self.data.iterrows():
            sample_id = row["Sample_ID"]
            voxel_path = os.path.join(voxel_dir, f"{sample_id}.npy")
            if os.path.exists(voxel_path):
                self.samples.append((sample_id, voxel_path))
        print(f"✅ Loaded {len(self.samples)} voxel samples.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample_id, voxel_path = self.samples[idx]
        voxel = np.load(voxel_path).astype(np.float32).transpose(2, 0, 1)  # [3, H, W]
        voxel_tensor = torch.tensor(voxel, dtype=torch.float32)

        row = self.data[self.data['Sample_ID'] == sample_id].iloc[0]
        demo_tensor = torch.tensor([
            float(row['Height']),
            float(row['Age']),
            float(row['Weight']),
            float(row['Sex']),
            float(row['Ethnicity'])
        ], dtype=torch.float32)

        comp_values = row[self.comp_names].values.astype(np.float32)
        comp_tensor = torch.tensor((comp_values - self.means) / self.stds, dtype=torch.float32)

        return voxel_tensor, demo_tensor, comp_tensor, sample_id

# ----------------------------- Improved Model ----------------------------- #
class EnhancedVoxelNet(nn.Module):
    def __init__(self, out_dim):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1), nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1), nn.BatchNorm2d(256), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(256, 256, kernel_size=3, padding=1), nn.BatchNorm2d(256), nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4))
        )
        self.demo_proj = nn.Sequential(
            nn.Linear(5, 64), nn.ReLU(), nn.Linear(64, 64)
        )
        self.classifier = nn.Sequential(
            nn.Linear(256*4*4 + 64, 256), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(256, out_dim)
        )

    def forward(self, voxel, demo):
        feat = self.encoder(voxel).flatten(1)
        demo_feat = self.demo_proj(demo)
        x = torch.cat([feat, demo_feat], dim=1)
        return self.classifier(x)

# ----------------------------- Training Loop ----------------------------- #
def train(csv_path, voxel_dir):
    # Record start time
    start_time = time.time()
    
    # Print hardware information
    print("\n" + "="*50)
    print("🖥️  HARDWARE INFORMATION")
    print("="*50)
    print(f"OS: {platform.system()} {platform.release()}")
    print(f"Python: {platform.python_version()}")
    print(f"PyTorch: {torch.__version__}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA Version: {torch.version.cuda}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    else:
        print(f"Device: CPU")
    print("="*50 + "\n")
    
    dataset = VoxelCompositionDataset(csv_path, voxel_dir)
    train_ids, val_ids = train_test_split(range(len(dataset)), test_size=0.1, random_state=42)
    train_loader = DataLoader(torch.utils.data.Subset(dataset, train_ids), batch_size=16, shuffle=True)
    val_loader = DataLoader(torch.utils.data.Subset(dataset, val_ids), batch_size=16)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = EnhancedVoxelNet(out_dim=len(dataset.comp_names)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)
    criterion = nn.MSELoss()

    best_val_loss = float('inf')
    for epoch in range(1, 101):
        model.train()
        for vox, demo, target, _ in train_loader:
            vox, demo, target = vox.to(device), demo.to(device), target.to(device)
            optimizer.zero_grad()
            pred = model(vox, demo)
            loss = criterion(pred, target)
            loss.backward()
            optimizer.step()
        scheduler.step()

        model.eval()
        val_losses, all_preds, all_targets = [], [], []
        with torch.no_grad():
            for vox, demo, target, _ in val_loader:
                vox, demo, target = vox.to(device), demo.to(device), target.to(device)
                pred = model(vox, demo)
                val_loss = criterion(pred, target).item()
                val_losses.append(val_loss)
                all_preds.append(pred.cpu().numpy())
                all_targets.append(target.cpu().numpy())

        avg_val_loss = np.mean(val_losses)
        print(f"Epoch {epoch:03d} - Val Loss: {avg_val_loss:.4f}")
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), "best_voxel_model.pth")

    # Evaluation
    preds = np.vstack(all_preds) * dataset.stds + dataset.means
    targets = np.vstack(all_targets) * dataset.stds + dataset.means

    results = {
        name: {
            "MSE": mean_squared_error(targets[:, i], preds[:, i]),
            "MAE": mean_absolute_error(targets[:, i], preds[:, i]),
            "RMSE": np.sqrt(mean_squared_error(targets[:, i], preds[:, i])),
            "R2": r2_score(targets[:, i], preds[:, i]),
            "PearsonR": np.nan if np.std(targets[:, i]) == 0 or np.std(preds[:, i]) == 0 else pearsonr(targets[:, i], preds[:, i])[0],
            "Unit": "kg" if "(kg)" in name else "g" if "(g)" in name else "%"
        }
        for i, name in enumerate(dataset.comp_names)
    }

    df_result = pd.DataFrame({**{f"GT_{name}": targets[:, i] for i, name in enumerate(dataset.comp_names)},
                              **{f"Pred_{name}": preds[:, i] for i, name in enumerate(dataset.comp_names)}})
    df_metrics = pd.DataFrame(results).T

    with pd.ExcelWriter("body_comp_results_final.xlsx") as writer:
        df_result.to_excel(writer, sheet_name="Predictions", index=False)
        df_metrics.to_excel(writer, sheet_name="Metrics")

    print(" Final results saved to body_comp_results_final.xlsx")
    
    # Print training time
    elapsed_time = time.time() - start_time
    hours = int(elapsed_time // 3600)
    minutes = int((elapsed_time % 3600) // 60)
    seconds = int(elapsed_time % 60)
    print("\n" + "="*50)
    print("TRAINING TIME")
    print("="*50)
    print(f"Total Training Time: {hours}h {minutes}m {seconds}s ({elapsed_time:.2f} seconds)")
    print("="*50 + "\n")

if __name__ == "__main__":
    train("all_compositions_with_mesh_path.csv", "voxelmap/human_voxelmaps")
