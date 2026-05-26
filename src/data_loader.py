import pandas as pd
import numpy as np
from pathlib import Path


def load_raw(cfg: dict) -> pd.DataFrame:
    dc = cfg["data"]
    df = pd.read_csv(dc["input_csv"], encoding="utf-8-sig")

    # Drop unnamed/empty columns
    df = df.loc[:, ~df.columns.str.match(r"^Unnamed")]
    df = df.loc[:, df.columns != ""]

    # Normalize text column name (BOM may have been stripped by utf-8-sig)
    if dc["text_col"] not in df.columns:
        # Try stripping BOM manually
        df.columns = [c.lstrip("﻿") for c in df.columns]

    return df


def detect_freq_cols(df: pd.DataFrame, cfg: dict) -> list:
    prefix = cfg["data"]["freq_col_prefix"]
    count = cfg["data"]["freq_col_count"]
    cols = []
    for i in range(1, count + 1):
        name = f"{prefix}{i:02d}"
        if name in df.columns:
            cols.append(name)
    if not cols:
        # fallback: any column starting with prefix
        cols = sorted([c for c in df.columns if c.startswith(prefix)])
    return cols


def compute_observed_months(df: pd.DataFrame, cfg: dict) -> pd.Series:
    dc = cfg["data"]
    cy = dc["collection_year"]
    cm = dc["collection_month"]
    year = df[dc["year_col"]].astype(int)
    month = df[dc["month_col"]].astype(int)
    observed = (cy - year) * 12 + (cm - month + 1)
    return observed.clip(lower=1, upper=12)


def build_frequency_matrix(df: pd.DataFrame, freq_cols: list) -> np.ndarray:
    """Returns (N, 12) float array. Unobserved future months remain NaN."""
    mat = df[freq_cols].values.astype(float)
    return mat


def load_dataset(cfg: dict):
    df = load_raw(cfg)

    # month==0 samples are excluded: their first-appearance month is unknown
    month_col = cfg["data"]["month_col"]
    before = len(df)
    df = df[df[month_col].astype(int) != 0].reset_index(drop=True)
    print(f"Dropped {before - len(df)} samples with month==0 ({len(df)} remaining)")

    freq_cols = detect_freq_cols(df, cfg)
    df["observed_months"] = compute_observed_months(df, cfg)
    freq_mat = build_frequency_matrix(df, freq_cols)
    # Build mask: True where value is observed (not NaN)
    freq_mask = ~np.isnan(freq_mat)
    return df, freq_cols, freq_mat, freq_mask
