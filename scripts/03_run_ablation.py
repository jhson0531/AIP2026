"""
Ablation study across feature groups.

Usage:
    python scripts/03_run_ablation.py --config configs/default.yaml

Outputs:
    outputs/results/ablation_results.csv
    outputs/figures/ablation_auroc.png
    outputs/figures/ablation_auprc.png
    outputs/figures/ablation_f1.png

Requires: 01_build_features.py must have been run first.

Feature sets tested:
    E, L, F, E_L, E_F, L_F, E_L_F

Key comparisons:
    F vs E_L_F : does text help beyond frequency?
    E vs E_L   : does hand-crafted linguistic help?
    E_L vs F   : text-only vs frequency-only
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import load_config, resolve_paths
from src.experiments.ablation import run_ablation

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--out-dir", default="outputs")
    parser.add_argument("--feat-dir", default="data/features")
    parser.add_argument("--horizon", type=int, default=12)
    args = parser.parse_args()

    base_dir = str(Path(__file__).parent.parent)
    cfg = load_config(args.config)
    cfg = resolve_paths(cfg, base_dir)

    df = run_ablation(cfg, out_dir=args.out_dir, feat_dir=args.feat_dir, horizon=args.horizon)
    print("\n=== Ablation Results ===")
    print(df[["feature_set", "n_samples", "n_features", "auroc", "f1"]].to_string(index=False))

    # Print key comparisons
    def get_auroc(fs):
        row = df[df["feature_set"] == fs]
        return row["auroc"].values[0] if len(row) else None

    print("\n--- Key Comparisons ---")
    print(f"F vs E_L_F : {get_auroc('F'):.4f} vs {get_auroc('E_L_F'):.4f}")
    print(f"E vs E_L   : {get_auroc('E'):.4f} vs {get_auroc('E_L'):.4f}")
    print(f"E_L vs F   : {get_auroc('E_L'):.4f} vs {get_auroc('F'):.4f}")
