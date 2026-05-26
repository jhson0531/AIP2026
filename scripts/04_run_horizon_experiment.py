"""
Horizon (early prediction) experiment.

Usage:
    python scripts/04_run_horizon_experiment.py --config configs/default.yaml

Outputs:
    outputs/results/horizon_results.csv
    outputs/figures/horizon_auroc.png
    outputs/figures/horizon_auprc.png
    outputs/figures/horizon_f1.png

Requires: 01_build_features.py must have been run first.

Experiments:
    text_only : E+L only (all samples)
    h1        : E+L+F using m1       (observed >= 1)
    h3        : E+L+F using m1~m3    (observed >= 3)
    h6        : E+L+F using m1~m6    (observed >= 6)
    h12       : E+L+F using m1~m12   (observed == 12)

Key question: how many months of frequency data are needed?
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import load_config, resolve_paths
from src.experiments.horizon_experiment import run_horizon_experiment

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--out-dir", default="outputs")
    parser.add_argument("--feat-dir", default="data/features")
    args = parser.parse_args()

    base_dir = str(Path(__file__).parent.parent)
    cfg = load_config(args.config)
    cfg = resolve_paths(cfg, base_dir)

    df = run_horizon_experiment(cfg, out_dir=args.out_dir, feat_dir=args.feat_dir)
    print("\n=== Horizon Experiment Results ===")
    print(df[["experiment_name", "n_samples", "auroc", "f1"]].to_string(index=False))
