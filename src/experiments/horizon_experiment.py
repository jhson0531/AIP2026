from pathlib import Path

import pandas as pd

from src.features.feature_builder import load_features
from src.preprocessing import split_dataset
from src.evaluation import evaluate_model
from src.model_factory import get_model, fit_model, compute_spw
from src.utils import append_result, ensure_dir, set_seed
from src.plotting import plot_bar_comparison


def run_horizon_experiment(cfg: dict, out_dir: str = "outputs", feat_dir: str = "data/features"):
    set_seed(cfg["training"]["random_state"])
    ensure_dir(f"{out_dir}/results")
    ensure_dir(f"{out_dir}/figures")

    result_path = f"{out_dir}/results/horizon_results.csv"
    Path(result_path).unlink(missing_ok=True)

    experiments = [("text_only", "E_L", None)]
    for h in cfg["frequency"]["horizons"]:
        experiments.append((f"h{h}", "E_L_F", h))

    results = []

    for exp_name, fs, horizon in experiments:
        h = horizon if horizon is not None else 12
        print(f"\n[Horizon] experiment={exp_name}  feature_set={fs}  horizon={h}")
        X, y, col_groups, valid_mask = load_features(cfg, fs, horizon=h, out_dir=feat_dir)
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
            "experiment_name": exp_name,
            "feature_set": fs,
            "horizon": horizon,
            "n_samples": len(y),
            "n_features": X.shape[1],
            **m,
        }
        results.append(row)
        append_result(row, result_path)
        print(f"  n={len(y)} AUROC={m['auroc']:.4f} F1={m['f1']:.4f}")

    df_res = pd.DataFrame(results)

    for metric in ["auroc", "auprc", "f1"]:
        plot_bar_comparison(
            df_res, "experiment_name", metric,
            f"Horizon Experiment — {metric.upper()}",
            f"{out_dir}/figures/horizon_{metric}.png",
        )

    return df_res
