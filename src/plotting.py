import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from pathlib import Path


def _save(fig, path: str):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"Saved figure: {path}")


def plot_bar_comparison(df: pd.DataFrame, x_col: str, y_col: str, title: str, path: str):
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(df[x_col].astype(str), df[y_col])
    ax.set_title(title)
    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    plt.xticks(rotation=30, ha="right")
    _save(fig, path)


def plot_importance_bar(importances: pd.Series, title: str, path: str, top_n: int = 30):
    top = importances.nlargest(top_n)
    fig, ax = plt.subplots(figsize=(10, max(6, top_n * 0.3)))
    top.sort_values().plot.barh(ax=ax)
    ax.set_title(title)
    _save(fig, path)


def plot_group_importance(group_scores: dict, path: str):
    fig, ax = plt.subplots(figsize=(7, 4))
    names = list(group_scores.keys())
    vals = [group_scores[n] for n in names]
    ax.bar(names, vals)
    ax.set_title("Group Permutation Importance")
    ax.set_ylabel("Importance Score")
    _save(fig, path)
