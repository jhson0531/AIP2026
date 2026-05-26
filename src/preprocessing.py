import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def validate_labels(df: pd.DataFrame, label_col: str, valid_values=(0, 1)):
    vals = set(df[label_col].dropna().unique())
    assert vals.issubset(set(valid_values)), f"Unexpected label values: {vals}"


def get_horizon_mask(observed_months: pd.Series, horizon: int) -> np.ndarray:
    """Boolean array: True for samples with observed_months >= horizon."""
    return (observed_months >= horizon).values


def split_dataset(df: pd.DataFrame, cfg: dict):
    sc = cfg["split"]
    label_col = cfg["data"]["label_col"]
    rs = sc["random_state"]
    stratify = df[label_col] if sc.get("stratify", True) else None

    if sc["split_type"] == "time":
        year_col = cfg["data"].get("year_col", "year")
        sorted_idx = df[year_col].argsort().values
        n = len(df)
        n_test = int(n * sc["test_size"])
        n_valid = int(n * sc["valid_size"])
        test_idx = sorted_idx[-n_test:]
        valid_idx = sorted_idx[-(n_test + n_valid):-n_test]
        train_idx = sorted_idx[:-(n_test + n_valid)]
        return (
            df.iloc[train_idx].index.tolist(),
            df.iloc[valid_idx].index.tolist(),
            df.iloc[test_idx].index.tolist(),
        )
    else:
        idx = df.index.tolist()
        labels = df[label_col].tolist()
        train_val_idx, test_idx, train_val_lab, _ = train_test_split(
            idx, labels, test_size=sc["test_size"], random_state=rs, stratify=labels
        )
        val_ratio = sc["valid_size"] / (1 - sc["test_size"])
        train_idx, valid_idx = train_test_split(
            train_val_idx, test_size=val_ratio, random_state=rs, stratify=train_val_lab
        )
        return train_idx, valid_idx, test_idx
