import numpy as np
import pandas as pd
from pathlib import Path

from src.data_loader import load_dataset
from src.features.linguistic_features import extract_linguistic_features
from src.features.frequency_features import compute_freq_features
from src.features.embedding_features import extract_embedding_features
from src.utils import save_csv, ensure_dir


def build_all_features(cfg: dict, out_dir: str = "data/features"):
    ensure_dir(out_dir)
    df, freq_cols, freq_mat, freq_mask = load_dataset(cfg)
    texts = df[cfg["data"]["text_col"]]
    label = df[cfg["data"]["label_col"]]

    print("=== Extracting Linguistic Features ===")
    feat_L = extract_linguistic_features(texts, cfg)
    feat_L.index = df.index
    save_csv(feat_L, f"{out_dir}/features_L.csv")

    print("=== Extracting Embedding Features ===")
    feat_E = extract_embedding_features(texts, cfg)
    feat_E.index = df.index
    save_csv(feat_E, f"{out_dir}/features_E.csv")

    print("=== Extracting Frequency Features ===")
    for h in cfg["frequency"]["horizons"]:
        print(f"  horizon={h}")
        feat_F = compute_freq_features(
            freq_mat, h, freq_mask,
            active_threshold=cfg["frequency"]["active_threshold"]
        )
        feat_F.index = df.index
        save_csv(feat_F, f"{out_dir}/features_F_h{h}.csv")

    # Save meta info
    meta = pd.DataFrame({
        "text": texts.values,
        "label": label.values,
        "observed_months": df["observed_months"].values,
        "year": df[cfg["data"]["year_col"]].values,
        "month": df[cfg["data"]["month_col"]].values,
    })
    save_csv(meta, f"{out_dir}/meta.csv")

    # Build final feature table (E + L + F_h12, full horizon)
    feat_F12 = pd.read_csv(f"{out_dir}/features_F_h12.csv")
    feat_F12.index = df.index

    final = pd.concat([feat_E, feat_L, feat_F12, label.reset_index(drop=True)], axis=1)
    save_csv(final, f"{out_dir}/final_feature_table.csv")
    print("=== Feature building complete ===")


def load_features(cfg: dict, feature_set: str, horizon: int = 12,
                  out_dir: str = "data/features"):
    """Load features by feature_set string: 'E', 'L', 'F', 'E_L', 'E_F', 'L_F', 'E_L_F'."""
    parts = []
    col_groups = {}

    if "E" in feature_set:
        df_e = pd.read_csv(f"{out_dir}/features_E.csv")
        col_groups["embedding"] = list(df_e.columns)
        parts.append(df_e)

    if "L" in feature_set:
        df_l = pd.read_csv(f"{out_dir}/features_L.csv")
        col_groups["linguistic"] = list(df_l.columns)
        parts.append(df_l)

    if "F" in feature_set:
        df_f = pd.read_csv(f"{out_dir}/features_F_h{horizon}.csv")
        col_groups["frequency"] = list(df_f.columns)
        parts.append(df_f)

    meta = pd.read_csv(f"{out_dir}/meta.csv")
    label = meta["label"]
    observed = meta["observed_months"]

    X = pd.concat(parts, axis=1)

    # For F features: exclude samples with NaN (not enough observed months)
    if "F" in feature_set:
        f_cols = col_groups["frequency"]
        valid_mask = ~X[f_cols].isna().any(axis=1)
    else:
        valid_mask = pd.Series([True] * len(X))

    return X[valid_mask].reset_index(drop=True), label[valid_mask].reset_index(drop=True), col_groups, valid_mask
