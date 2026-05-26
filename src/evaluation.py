import numpy as np
from sklearn.metrics import (
    roc_auc_score, average_precision_score, f1_score,
    accuracy_score, balanced_accuracy_score, precision_score, recall_score,
    roc_curve, precision_recall_curve,
)


def find_optimal_threshold(y_true, y_prob):
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    j = tpr - fpr
    return float(thresholds[np.argmax(j)])


def compute_metrics(y_true, y_prob, threshold=0.5) -> dict:
    y_pred = (y_prob >= threshold).astype(int)
    auroc = roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else float("nan")
    auprc = average_precision_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else float("nan")
    return {
        "auroc": auroc,
        "auprc": auprc,
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "threshold": threshold,
    }


def evaluate_model(model, X_val, y_val, X_test, y_test):
    val_prob = model.predict_proba(X_val)[:, 1]
    opt_threshold = find_optimal_threshold(y_val, val_prob)

    test_prob = model.predict_proba(X_test)[:, 1]
    metrics_default = compute_metrics(y_test, test_prob, threshold=0.5)
    metrics_optimal = compute_metrics(y_test, test_prob, threshold=opt_threshold)

    return {
        "test_prob": test_prob,
        "metrics_default": metrics_default,
        "metrics_optimal": metrics_optimal,
        "optimal_threshold": opt_threshold,
    }
