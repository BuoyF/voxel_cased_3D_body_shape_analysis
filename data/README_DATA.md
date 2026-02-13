# Dataset Specification

## Overview

The dataset consists of paired 3D body scans and DXA (Dual-Energy X-ray Absorptiometry) measurements from 287 participants across three cohorts:

- **Bariatric (BA)**: n=88, BMI > 35 kg/m²
- **Normal BMI (NB)**: n=161, BMI 18.5-24.9 kg/m²
- **Geriatric (GE)**: n=38, Age 65+

## File Structure
```
data/
├── voxel_maps/
│   ├── BA_001.npy
│   ├── NB_001.npy
│   ├── GE_001.npy
│   └── ...
├── compositions.csv
└── demographics.csv
```

## Voxel Map Format

**File**: `{cohort}_{id}.npy`  
**Shape**: `[512, 512, 3]`  
**Dtype**: `float32`

### Channel Definitions:

1. **Channel 0** - Front curvature (vertices with z > 0)
2. **Channel 1** - Back curvature (vertices with z ≤ 0)
3. **Channel 2** - Depth (absolute z-coordinate)

### Preprocessing:
- Meshes normalized: Y-axis rotation (180°), centered, original scale preserved
- Orthographic projection onto 512×512 grid
- Mean curvature computed using local neighborhood topology

## compositions.csv

| Column | Type | Description | Unit |
|--------|------|-------------|------|
| Sample_ID | str | Unique identifier (e.g., "BA_001") | - |
| Total_Lean | float | Total lean mass | kg |
| Total_Fat | float | Total fat mass | kg |
| Total_BMC | float | Total bone mineral content | g |
| Trunk_Lean | float | Trunk lean mass | kg |
| ... | ... | ... | ... |
| *66 columns total* | | | |

## demographics.csv

| Column | Type | Description | Unit |
|--------|------|-------------|------|
| Sample_ID | str | Unique identifier | - |
| Height | float | Height | cm |
| Weight | float | Weight | kg |
| Sex | int | 0=Female, 1=Male | - |
| Ethnicity | int | Categorical encoding | - |

## Data Access

Due to IRB restrictions, this dataset is **not publicly available**. To request access:

**Contact**: Boyuan Feng (fby@gwu.edu)

**Required documentation**:
1. Institutional affiliation
2. Intended research use
3. Ethics board approval

## Expected Performance

With this dataset, the model should achieve:
- **Lean Mass**: R²=0.91, RMSE=3.92 kg
- **Fat Mass**: R²=0.90, RMSE=3.05 kg
- **BMC**: R²=0.84, RMSE=86.2 g
