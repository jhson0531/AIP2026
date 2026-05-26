"""
Feature importance analysis.

Usage:
    python scripts/05_run_feature_importance.py --config configs/default.yaml

Outputs:
    outputs/importance/lgbm_gain_importance.csv
    outputs/importance/permutation_importance.csv
    outputs/importance/group_permutation_importance.csv
    outputs/importance/shap_values.npy
    outputs/importance/shap_importance.csv
    outputs/figures/lgbm_gain_importance_top30.png
    outputs/figures/permutation_importance_top30.png
    outputs/figures/group_permutation_importance.png
    outputs/figures/shap_summary.png
    outputs/figures/shap_bar.png

Requires: 01_build_features.py must have been run first.

Model: LightGBM, feature_set=E_L_F, horizon=12
Analyses: built-in gain, permutation, group permutation, SHAP
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import load_config, resolve_paths
from src.experiments.feature_importance import run_feature_importance

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--out-dir", default="outputs")
    parser.add_argument("--feat-dir", default="data/features")
    args = parser.parse_args()

    base_dir = str(Path(__file__).parent.parent)
    cfg = load_config(args.config)
    cfg = resolve_paths(cfg, base_dir)

    run_feature_importance(cfg, out_dir=args.out_dir, feat_dir=args.feat_dir)
