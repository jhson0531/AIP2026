import numpy as np
import pandas as pd
from pathlib import Path

from src.features.feature_builder import load_features
from src.preprocessing import split_dataset
from src.evaluation import evaluate_model
from src.model_factory import get_model, fit_model, compute_spw
from src.utils import ensure_dir, save_csv, set_seed
from src.plotting import plot_importance_bar, plot_group_importance


def run_feature_importance(cfg: dict, out_dir: str = "outputs", feat_dir: str = "data/features"):
    set_seed(cfg["training"]["random_state"])
    ensure_dir(f"{out_dir}/importance")
    ensure_dir(f"{out_dir}/figures")

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
    feature_names = list(X.columns)

    print("[Feature Importance] Training LightGBM (E_L_F)")
    model = get_model("lightgbm", cfg, spw)
    model = fit_model("lightgbm", model, X_train, y_train, X_val, y_val, cfg)

    # 1. Built-in gain importance
    print("  1. Built-in gain importance")
    gain_imp = model.booster_.feature_importance(importance_type="gain")
    df_gain = pd.DataFrame({"feature": feature_names, "importance": gain_imp})
    df_gain = df_gain.sort_values("importance", ascending=False)
    save_csv(df_gain, f"{out_dir}/importance/lgbm_gain_importance.csv")
    plot_importance_bar(
        df_gain.set_index("feature")["importance"],
        "LightGBM Gain Importance (top 30)",
        f"{out_dir}/figures/lgbm_gain_importance_top30.png",
    )

    # 2. Permutation importance
    print("  2. Permutation importance")
    from sklearn.inspection import permutation_importance
    pi = permutation_importance(model, X_test, y_test, n_repeats=10,
                                random_state=cfg["training"]["random_state"],
                                scoring="roc_auc")
    df_perm = pd.DataFrame({
        "feature": feature_names,
        "importance_mean": pi.importances_mean,
        "importance_std": pi.importances_std,
    }).sort_values("importance_mean", ascending=False)
    save_csv(df_perm, f"{out_dir}/importance/permutation_importance.csv")
    plot_importance_bar(
        df_perm.set_index("feature")["importance_mean"],
        "Permutation Importance (top 30)",
        f"{out_dir}/figures/permutation_importance_top30.png",
    )

    # 3. Group permutation importance
    print("  3. Group permutation importance")
    from sklearn.metrics import roc_auc_score
    base_score = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
    group_scores = {}
    X_test_np = X_test.values  # numpy for efficient column shuffling
    for group_name, group_cols in col_groups.items():
        col_idx = [feature_names.index(c) for c in group_cols if c in feature_names]
        scores = []
        for _ in range(10):
            X_perm = X_test_np.copy()
            idx = np.random.permutation(len(X_perm))
            X_perm[:, col_idx] = X_perm[idx][:, col_idx]
            X_perm_df = pd.DataFrame(X_perm, columns=feature_names)
            score = roc_auc_score(y_test, model.predict_proba(X_perm_df)[:, 1])
            scores.append(base_score - score)
        group_scores[group_name] = float(np.mean(scores))

    df_group = pd.DataFrame([
        {"group": g, "importance": v} for g, v in group_scores.items()
    ])
    save_csv(df_group, f"{out_dir}/importance/group_permutation_importance.csv")
    plot_group_importance(group_scores, f"{out_dir}/figures/group_permutation_importance.png")

    # 4. SHAP
    print("  4. SHAP analysis")
    try:
        import shap
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_test)
        if isinstance(shap_values, list):
            shap_values = shap_values[1]
        np.save(f"{out_dir}/importance/shap_values.npy", shap_values)

        mean_abs_shap = np.abs(shap_values).mean(axis=0)
        df_shap = pd.DataFrame({
            "feature": feature_names,
            "shap_importance": mean_abs_shap,
        }).sort_values("shap_importance", ascending=False)
        save_csv(df_shap, f"{out_dir}/importance/shap_importance.csv")

        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        plt.figure(figsize=(10, 8))
        shap.summary_plot(shap_values, X_test, feature_names=feature_names,
                          show=False, max_display=20)
        plt.savefig(f"{out_dir}/figures/shap_summary.png", bbox_inches="tight", dpi=150)
        plt.close()

        plt.figure(figsize=(10, 6))
        shap.summary_plot(shap_values, X_test, feature_names=feature_names,
                          plot_type="bar", show=False, max_display=20)
        plt.savefig(f"{out_dir}/figures/shap_bar.png", bbox_inches="tight", dpi=150)
        plt.close()
        print(f"  SHAP saved.")
    except Exception as e:
        print(f"  SHAP skipped: {e}")

    print("[Feature Importance] Done.")
