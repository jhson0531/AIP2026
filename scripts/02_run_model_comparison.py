"""
Model comparison experiment.

Usage:
    python scripts/02_run_model_comparison.py --config configs/default.yaml

Outputs:
    outputs/results/model_comparison.csv
    outputs/figures/model_comparison_auroc.png
    outputs/models/best_full_model.pkl

Requires: 01_build_features.py must have been run first.

Models:
    Logistic Regression, Random Forest, LightGBM,
    XGBoost, CatBoost, MLP
Feature set: E_L_F (horizon=12)
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import load_config, resolve_paths
from src.experiments.model_comparison import run_model_comparison

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--out-dir", default="outputs")
    parser.add_argument("--feat-dir", default="data/features")
    args = parser.parse_args()

    base_dir = str(Path(__file__).parent.parent)
    cfg = load_config(args.config)
    cfg = resolve_paths(cfg, base_dir)

    df = run_model_comparison(cfg, out_dir=args.out_dir, feat_dir=args.feat_dir)
    print("\n=== Model Comparison Results ===")
    print(df[["model_name", "auroc", "auprc", "f1", "balanced_accuracy"]].to_string(index=False))
