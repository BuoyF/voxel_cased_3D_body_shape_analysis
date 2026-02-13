# voxel_cased_3D_body_shape_analysis
# Voxel-based Deep Regression for Enhanced Body Composition Estimation

[![Paper](https://img.shields.io/badge/Paper-SN%20Computer%20Science-blue)](link-to-paper)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

Official implementation of **"Voxel-based Deep Regression for Enhanced Body Composition Estimation from 3D Body Scans"**

---

## Overview

This repository provides an end-to-end deep learning framework that predicts body composition (lean mass, fat mass, bone mineral content) from 3D body scans without manual feature engineering. Our voxel-based approach achieves:

- **R² = 0.91** for lean mass prediction
- **R² = 0.90** for fat mass prediction  
- **R² = 0.84** for bone mineral content prediction

**Key Features:**
- End-to-end voxel-based learning (512×512 resolution)
- Demographic feature fusion via learned embeddings
- SE (Squeeze-and-Excitation) attention mechanisms
- Residual connections for gradient flow
- Comprehensive ablation study framework

---

## Installation

### Requirements
- Python 3.8+
- PyTorch 1.12+
- CUDA 11.3+ (for GPU acceleration)
- 16GB+ RAM (24GB+ recommended for training)
---

## Dataset

Due to privacy constraints and IRB restrictions, the clinical 3D body scan dataset is **not publicly available**. 

**To request access to the dataset**, please contact:
- **Boyuan Feng**: fby@gwu.edu
- Include: Your institution, intended use, and ethics approval documentation

### Expected Data Format

The code expects data organized as follows:
```
data/
├── voxel_maps/
│   ├── subject_001.npy  # Shape: [512, 512, 3] (float32)
│   ├── subject_002.npy
│   └── ...
├── compositions.csv     # DXA measurements (66 targets)
└── demographics.csv     # Height, Weight, Sex, Ethnicity
```

**Voxel Map Channels (3 channels):**
1. **Channel 0**: Mean curvature (front-facing vertices, z > 0)
2. **Channel 1**: Mean curvature (back-facing vertices, z ≤ 0)  
3. **Channel 2**: Depth (absolute z-coordinate)

See `data/README_DATA.md` for detailed specifications.

---

## Quick Start

### 1. Training (Full Model)
```bash
python src/train.py \
    --voxel_dir data/voxel_maps \
    --csv_path data/compositions.csv \
    --demo_path data/demographics.csv \
    --batch_size 16 \
    --epochs 100 \
    --lr 1e-4 \
    --device cuda
```

### 2. Ablation Study

Run all 8 ablation configurations (D×R×S):
```bash
bash scripts/run_ablation.sh
```

This generates results matching **Figure 3** in the paper:
- D0_R0_S0, D0_R0_S1, D0_R1_S0, D0_R1_S1
- D1_R0_S0, D1_R0_S1, D1_R1_S0, D1_R1_S1

### 3. Evaluation
```bash
python src/evaluate.py \
    --checkpoint checkpoints/best_model.pth \
    --test_csv data/test_compositions.csv \
    --output_dir results/
```

---

## Model Architecture
```
Input: 3×512×512 Voxel Map + 4D Demographics

Stem Block:
├─ Conv2D(3→64, k=3, s=1, p=1)
├─ BatchNorm2D(64)
├─ ReLU
└─ MaxPool2D(2×2)

Encoder Stage 1 (64 channels):
├─ ResidualBlock (optional, ablation)
├─ SEBlock (r=16, optional, ablation)
└─ MaxPool2D → 64×128×128

Encoder Stage 2 (128 channels):
├─ Conv2D(64→128, k=3)
├─ ResidualBlock (optional)
├─ SEBlock (r=16, optional)
└─ MaxPool2D → 128×64×64

Encoder Stage 3 (256 channels):
├─ Conv2D(128→256, k=3)
├─ ResidualBlock (optional)
├─ SEBlock (r=16, optional)
└─ AdaptiveAvgPool2D(4×4) → 256×4×4

Demographic Embedding:
├─ Linear(4→64)
├─ ReLU
└─ Linear(64→64)

Fusion & Classification:
├─ Flatten(256×4×4) = 4096D
├─ Concat[4096D voxel + 64D demo] = 4160D
├─ Linear(4160→256), ReLU, Dropout(0.3)
└─ Linear(256→66) → Body Composition Outputs
```

**Training Configuration:**
- Optimizer: AdamW (lr=1e-4, weight_decay=1e-4)
- Scheduler: CosineAnnealingLR (T_max=50)
- Loss: MSE on z-normalized targets
- Batch size: 16
- Epochs: 100 with early stopping

---

## 66 Body Composition Targets

Targets are organized by anatomical region:

| Region | Lean Mass (kg) | Fat Mass (kg) | BMC (g) |
|--------|----------------|---------------|---------|
| Total Body | ✓ | ✓ | ✓ |
| Trunk | ✓ | ✓ | ✓ |
| Arms (L/R) | ✓ | ✓ | ✓ |
| Legs (L/R) | ✓ | ✓ | ✓ |
| Android | ✓ | ✓ | - |
| Gynoid | ✓ | ✓ | - |

See `docs/composition_targets.md` for complete list.

---

## Results Reproduction

Our paper reports the following metrics (Table in Section 5.2):

| Metric | Lean Mass | Fat Mass | BMC |
|--------|-----------|----------|-----|
| **R²** | 0.9087 | 0.8968 | 0.8378 |
| **RMSE (kg)** | 0.0392 | 0.0305 | 0.0862 |
| **Pearson r** | 0.9541 | 0.9528 | 0.9182 |

**Sample sizes:**
- Total participants: 287 (BA: 88, NB: 161, GE: 38)
- Training samples: ~258 (90% split)
- Validation samples: ~29 (10% split)
- Evaluation predictions: n=9,700 (287 subjects × 66 targets × validation)

---

## Hardware Requirements

**Minimum (Inference):**
- GPU: NVIDIA RTX 3060 (12GB VRAM)
- RAM: 16GB
- Storage: 10GB

**Recommended (Training):**
- GPU: NVIDIA RTX 4090 (24GB VRAM)
- RAM: 32GB
- Storage: 50GB (for checkpoints + logs)

**Inference Speed:**
- GPU (RTX 4090): ~0.3 seconds per scan
- CPU (Intel i9): ~4.5 seconds per scan

---

## Citation

If you use this code in your research, please cite:
```bibtex
@article{feng2025voxel,
  title={Voxel-based Deep Regression for Enhanced Body Composition Estimation from 3D Body Scans},
  author={Feng, Boyuan and Cheng, Ruting and Zheng, Yijiang and Feng, Shuya and Vaziri, Khashayar and Hahn, James},
  journal={SN Computer Science},
  year={2025},
  publisher={Springer Nature}
}
```

---

## License

This project is licensed under the MIT License - see [LICENSE](LICENSE) file for details.

---

## Acknowledgments

This work was supported by:
- National Institute of Diabetes and Digestive and Kidney Diseases (NIDDK): R01DK129809
- National Institute On Aging (NIA): R56AG089080

---

## Contact

- **Boyuan Feng** - fby@gwu.edu
- **Project Lab**: [Institute for Innovation in Health Computing, GWU](https://iihc.cs.gwu.edu/)

For questions about:
- **Code/Implementation**: Open a GitHub issue
- **Dataset Access**: Email fby@gwu.edu with IRB approval
- **Collaboration**: Contact Dr. James Hahn (hahn@gwu.edu)
