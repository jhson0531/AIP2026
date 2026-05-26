"""
Teacher-Student Distillation experiment.

Usage:
    python scripts/06_run_distillation.py --config configs/default.yaml

Outputs:
    outputs/results/distillation_results.csv
    outputs/models/best_full_model.pkl
    outputs/models/best_text_only_model.pkl
    outputs/models/best_distilled_student.pt

Requires: 01_build_features.py must have been run first.

Comparison:
    full_teacher      : E+L+F → hard label
    text_only_baseline: E+L   → hard label
    distilled_student : E+L   → hard label + teacher soft (alpha sweep)

Loss = BCE(y, p_student) + alpha * MSE(p_teacher, p_student)
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import load_config, resolve_paths
from src.experiments.text_only_distillation import run_distillation

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--out-dir", default="outputs")
    parser.add_argument("--feat-dir", default="data/features")
    args = parser.parse_args()

    base_dir = str(Path(__file__).parent.parent)
    cfg = load_config(args.config)
    cfg = resolve_paths(cfg, base_dir)

    df = run_distillation(cfg, out_dir=args.out_dir, feat_dir=args.feat_dir)
    print("\n=== Distillation Results ===")
    print(df[["model", "alpha", "auroc", "f1"]].to_string(index=False))
