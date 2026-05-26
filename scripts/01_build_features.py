"""
Feature building script.

Usage:
    python scripts/01_build_features.py --config configs/default.yaml

Outputs:
    data/features/features_E.csv
    data/features/features_L.csv
    data/features/features_F_h1.csv
    data/features/features_F_h3.csv
    data/features/features_F_h6.csv
    data/features/features_F_h12.csv
    data/features/meta.csv
    data/features/final_feature_table.csv
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import load_config, resolve_paths
from src.features.feature_builder import build_all_features

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--feat-dir", default="data/features")
    args = parser.parse_args()

    base_dir = str(Path(__file__).parent.parent)
    cfg = load_config(args.config)
    cfg = resolve_paths(cfg, base_dir)

    build_all_features(cfg, out_dir=args.feat_dir)
