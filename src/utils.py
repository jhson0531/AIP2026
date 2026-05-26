import os
import random
import numpy as np
import pandas as pd
from pathlib import Path


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
    except ImportError:
        pass


def ensure_dir(path: str):
    Path(path).mkdir(parents=True, exist_ok=True)


def save_csv(df: pd.DataFrame, path: str):
    ensure_dir(str(Path(path).parent))
    df.to_csv(path, index=False, encoding="utf-8-sig")
    print(f"Saved: {path}")


def append_result(result: dict, path: str):
    ensure_dir(str(Path(path).parent))
    row = pd.DataFrame([result])
    if Path(path).exists():
        row.to_csv(path, mode="a", header=False, index=False, encoding="utf-8-sig")
    else:
        row.to_csv(path, index=False, encoding="utf-8-sig")
