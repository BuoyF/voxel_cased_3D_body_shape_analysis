#!/bin/bash

# Run all 8 ablation configurations for the paper

echo "========================================="
echo "Running Ablation Study (8 configurations)"
echo "========================================="

# D0_R0_S0
python src/train_ablation.py \
    --use_demo 0 --use_residual 0 --use_se 0 \
    --save_dir results/D0_R0_S0

# D0_R0_S1
python src/train_ablation.py \
    --use_demo 0 --use_residual 0 --use_se 1 \
    --save_dir results/D0_R0_S1

# D0_R1_S0
python src/train_ablation.py \
    --use_demo 0 --use_residual 1 --use_se 0 \
    --save_dir results/D0_R1_S0

# D0_R1_S1
python src/train_ablation.py \
    --use_demo 0 --use_residual 1 --use_se 1 \
    --save_dir results/D0_R1_S1

# D1_R0_S0
python src/train_ablation.py \
    --use_demo 1 --use_residual 0 --use_se 0 \
    --save_dir results/D1_R0_S0

# D1_R0_S1
python src/train_ablation.py \
    --use_demo 1 --use_residual 0 --use_se 1 \
    --save_dir results/D1_R0_S1

# D1_R1_S0
python src/train_ablation.py \
    --use_demo 1 --use_residual 1 --use_se 0 \
    --save_dir results/D1_R1_S0

# D1_R1_S1 (FULL MODEL)
python src/train_ablation.py \
    --use_demo 1 --use_residual 1 --use_se 1 \
    --save_dir results/D1_R1_S1

echo "✅ Ablation study complete!"
