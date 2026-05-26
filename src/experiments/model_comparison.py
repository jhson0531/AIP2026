import pickle
from pathlib import Path

import pandas as pd

from src.features.feature_builder import load_features
from src.preprocessing import split_dataset
from src.evaluation import evaluate_model
from src.model_factory import get_model, fit_model, compute_spw
from src.utils import append_result, ensure_dir, set_seed
from src.plotting import plot_bar_comparison


def run_model_comparison(cfg: dict, out_dir: str = "outputs", feat_dir: str = "data/features"):
    set_seed(cfg["training"]["random_state"])
    ensure_dir(f"{out_dir}/results")
    ensure_dir(f"{out_dir}/figures")
    ensure_dir(f"{out_dir}/models")

    result_path = f"{out_dir}/results/model_comparison.csv"
    Path(result_path).unlink(missing_ok=True)

    X, y, col_groups, valid_mask = load_features(cfg, "E_L_F", horizon=12, out_dir=feat_dir)
    spw = compute_spw(y)
    df_label = pd.DataFrame({"label": y})
    train_idx, valid_idx, test_idx = split_dataset(df_label, cfg)

    X_train = X.iloc[train_idx].reset_index(drop=True)
    y_train = y.iloc[train_idx].values
    X_val = X.iloc[valid_idx].reset_index(drop=True)
    y_val = y.iloc[valid_idx].values
    X_test = X.iloc[test_idx].reset_index(drop=True)
    y_test = y.iloc[test_idx].values

    trained_models = {}
    results = []

    for model_name in cfg["models"]["run_models"]:
        print(f"\n[Model Comparison] Training: {model_name}")
        model = get_model(model_name, cfg, spw, input_dim=X_train.shape[1])
        model = fit_model(model_name, model, X_train, y_train, X_val, y_val, cfg)
        res = evaluate_model(model, X_val, y_val, X_test, y_test)
        m = res["metrics_optimal"]
        row = {
            "experiment_name": "model_comparison",
            "model_name": model_name,
            "feature_set": "E_L_F",
            "horizon": 12,
            "split_type": cfg["split"]["split_type"],
            **m,
        }
        results.append(row)
        append_result(row, result_path)
        trained_models[model_name] = model
        print(f"  AUROC={m['auroc']:.4f}  F1={m['f1']:.4f}")

    best = max(results, key=lambda r: r["auroc"])
    with open(f"{out_dir}/models/best_full_model.pkl", "wb") as f:
        pickle.dump(trained_models[best["model_name"]], f)
    print(f"\nBest model: {best['model_name']} (AUROC={best['auroc']:.4f})")

    df_res = pd.DataFrame(results)
    plot_bar_comparison(
        df_res, "model_name", "auroc",
        "Model Comparison — AUROC (E_L_F, horizon=12)",
        f"{out_dir}/figures/model_comparison_auroc.png",
    )
    return df_res
