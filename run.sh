#!/usr/bin/bash

#SBATCH -J bash
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem=32G
#SBATCH -p batch_eebme_ugrad
#SBATCH -w moana-r5
#SBATCH -t 1-0
#SBATCH -o logs/slurm-%A.out
#SBATCH --signal=B:SIGTERM@60

# 1단계: Feature 빌드 
# python scripts/01_build_features.py --config configs/default.yaml

# 2단계: 모델 성능 비교 (6개 모델, E_L_F)
# python scripts/02_run_model_comparison.py --config configs/default.yaml

# # 3단계: Ablation (7가지 feature 조합)
# python scripts/03_run_ablation.py --config configs/default.yaml

# # 4단계: Horizon 조기 예측 (text_only, h1, h3, h6, h12)
# python scripts/04_run_horizon_experiment.py --config configs/default.yaml

# # 5단계: Feature Importance (gain + permutation + group + SHAP)
# python scripts/05_run_feature_importance.py --config configs/default.yaml

# 6단계: Teacher-Student Distillation
python scripts/06_run_distillation.py --config configs/default.yaml