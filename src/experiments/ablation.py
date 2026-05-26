from pathlib import Path

import pandas as pd

from src.features.feature_builder import load_features
from src.preprocessing import split_dataset
from src.evaluation import evaluate_model
from src.model_factory import get_model, fit_model, compute_spw
from src.utils import append_result, ensure_dir, set_seed
from src.plotting import plot_bar_comparison

FEATURE_SETS = ["E", "L", "F", "E_L", "E_F", "L_F", "E_L_F"]


def run_ablation(cfg: dict, out_dir: str = "outputs", feat_dir: str = "data/features",
                 horizon: int = 12):
    set_seed(cfg["training"]["random_state"])
    ensure_dir(f"{out_dir}/results")
    ensure_dir(f"{out_dir}/figures")

    result_path = f"{out_dir}/results/ablation_results.csv"
    Path(result_path).unlink(missing_ok=True)

    results = []

    for fs in FEATURE_SETS:
        print(f"\n[Ablation] feature_set={fs}")
        X, y, col_groups, valid_mask = load_features(cfg, fs, horizon=horizon, out_dir=feat_dir)
        spw = compute_spw(y)
        df_label = pd.DataFrame({"label": y})
        train_idx, valid_idx, test_idx = split_dataset(df_label, cfg)

        X_train = X.iloc[train_idx].reset_index(drop=True)
        y_train = y.iloc[train_idx].values
        X_val = X.iloc[valid_idx].reset_index(drop=True)
        y_val = y.iloc[valid_idx].values
        X_test = X.iloc[test_idx].reset_index(drop=True)
        y_test = y.iloc[test_idx].values

        model_name = "lightgbm"
        model = get_model(model_name, cfg, spw)
        model = fit_model(model_name, model, X_train, y_train, X_val, y_val, cfg)
        res = evaluate_model(model, X_val, y_val, X_test, y_test)
        m = res["metrics_optimal"]

        row = {
            "experiment_name": "ablation",
            "model_name": model_name,
            "feature_set": fs,
            "n_samples": len(y),
            "n_features": X.shape[1],
            "horizon": horizon,
            **m,
        }
        results.append(row)
        append_result(row, result_path)
        print(f"  n={len(y)} feat={X.shape[1]} AUROC={m['auroc']:.4f} F1={m['f1']:.4f}")

    df_res = pd.DataFrame(results)

    for metric in ["auroc", "auprc", "f1"]:
        plot_bar_comparison(
            df_res, "feature_set", metric,
            f"Ablation Study — {metric.upper()}",
            f"{out_dir}/figures/ablation_{metric}.png",
        )

    return df_res
