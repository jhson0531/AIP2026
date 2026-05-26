import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from src.features.feature_builder import load_features
from src.preprocessing import split_dataset
from src.evaluation import evaluate_model
from src.model_factory import get_model, fit_model, compute_spw
from src.models.distillation import DistilledMLPClassifier
from src.utils import append_result, ensure_dir, set_seed
from src.plotting import plot_bar_comparison


def run_distillation(cfg: dict, out_dir: str = "outputs", feat_dir: str = "data/features"):
    set_seed(cfg["training"]["random_state"])
    ensure_dir(f"{out_dir}/results")
    ensure_dir(f"{out_dir}/figures")
    ensure_dir(f"{out_dir}/models")

    result_path = f"{out_dir}/results/distillation_results.csv"
    Path(result_path).unlink(missing_ok=True)

    dc = cfg["distillation"]

    # Load full-feature data for teacher (only samples with observed_months >= 12)
    X_full, y_full, _, valid_mask_full = load_features(cfg, "E_L_F", horizon=12, out_dir=feat_dir)
    # Load text-only data (no F → no filter applied), then apply same mask as E_L_F
    X_text_all, _, _, _ = load_features(cfg, "E_L", horizon=12, out_dir=feat_dir)
    X_text = X_text_all[valid_mask_full.values].reset_index(drop=True)

    assert len(X_full) == len(X_text), f"Sample count mismatch: {len(X_full)} vs {len(X_text)}"
    y = y_full.reset_index(drop=True)

    spw_full = compute_spw(y)
    df_label = pd.DataFrame({"label": y})
    train_idx, valid_idx, test_idx = split_dataset(df_label, cfg)

    # Full feature splits
    Xf_train = X_full.iloc[train_idx].reset_index(drop=True); yf_train = y.iloc[train_idx].values
    Xf_val = X_full.iloc[valid_idx].reset_index(drop=True);   yf_val = y.iloc[valid_idx].values
    Xf_test = X_full.iloc[test_idx].reset_index(drop=True);   yf_test = y.iloc[test_idx].values

    # Text-only splits
    Xt_train = X_text.iloc[train_idx].reset_index(drop=True)
    Xt_val = X_text.iloc[valid_idx].reset_index(drop=True)
    Xt_test = X_text.iloc[test_idx].reset_index(drop=True)

    results = []

    # 1. Full teacher (LightGBM, E_L_F)
    teacher_name = dc.get("teacher_model", "lightgbm")
    print(f"\n[Distillation] Training teacher: {teacher_name}")
    teacher = get_model(teacher_name, cfg, spw_full)
    teacher = fit_model(teacher_name, teacher, Xf_train, yf_train, Xf_val, yf_val, cfg)
    res = evaluate_model(teacher, Xf_val, yf_val, Xf_test, yf_test)
    m = res["metrics_optimal"]
    results.append({"model": "full_teacher", "feature_set": "E_L_F", "alpha": None, **m})
    append_result(results[-1], result_path)
    print(f"  Teacher AUROC={m['auroc']:.4f}")

    # Get teacher soft predictions for train set
    teacher_train_probs = teacher.predict_proba(Xf_train)[:, 1]

    # Save best full model
    with open(f"{out_dir}/models/best_full_model.pkl", "wb") as f:
        pickle.dump(teacher, f)

    # 2. Text-only baseline (no distillation)
    print(f"\n[Distillation] Text-only baseline (E_L)")
    baseline = get_model("mlp", cfg, spw_full, input_dim=Xt_train.shape[1])
    baseline = fit_model("mlp", baseline, Xt_train, yf_train, Xt_val, yf_val, cfg)
    res = evaluate_model(baseline, Xt_val, yf_val, Xt_test, yf_test)
    m = res["metrics_optimal"]
    results.append({"model": "text_only_baseline", "feature_set": "E_L", "alpha": None, **m})
    append_result(results[-1], result_path)
    print(f"  Baseline AUROC={m['auroc']:.4f}")

    # Save text-only model
    with open(f"{out_dir}/models/best_text_only_model.pkl", "wb") as f:
        pickle.dump(baseline, f)

    # 3. Distilled students (multiple alphas)
    best_student = None
    best_auroc = -1

    for alpha in dc["alpha_values"]:
        print(f"\n[Distillation] Distilled student alpha={alpha}")
        student = DistilledMLPClassifier(
            input_dim=Xt_train.shape[1],
            epochs=dc["epochs"], batch_size=dc["batch_size"],
            lr=dc["learning_rate"], alpha=alpha,
            weight_pos=spw_full,
            seed=cfg["training"]["random_state"],
        )
        student.fit(Xt_train, yf_train, teacher_train_probs, Xt_val, yf_val)
        res = evaluate_model(student, Xt_val, yf_val, Xt_test, yf_test)
        m = res["metrics_optimal"]
        results.append({"model": "distilled_student", "feature_set": "E_L", "alpha": alpha, **m})
        append_result(results[-1], result_path)
        print(f"  alpha={alpha} AUROC={m['auroc']:.4f}")

        if m["auroc"] > best_auroc:
            best_auroc = m["auroc"]
            best_student = student

    if best_student is not None:
        best_student.save(f"{out_dir}/models/best_distilled_student.pt")

    df_res = pd.DataFrame(results)
    print("\n[Distillation] Summary:")
    print(df_res[["model", "alpha", "auroc", "f1"]].to_string(index=False))
    df_res.to_csv(result_path, index=False, encoding="utf-8-sig")
    return df_res
