"""Shared model creation and training utilities."""
import numpy as np


def get_model(name: str, cfg: dict, spw: float = 1.0, input_dim: int = None):
    if name == "logistic_regression":
        from src.models.train_sklearn import get_logistic_regression
        return get_logistic_regression(cfg)
    elif name == "random_forest":
        from src.models.train_sklearn import get_random_forest
        return get_random_forest(cfg)
    elif name == "lightgbm":
        from src.models.train_lgbm import get_lgbm
        return get_lgbm(cfg, spw)
    elif name == "xgboost":
        from src.models.train_xgb import get_xgb
        return get_xgb(cfg, spw)
    elif name == "catboost":
        from src.models.train_catboost import get_catboost
        return get_catboost(cfg, spw)
    elif name == "mlp":
        from src.models.train_mlp import MLPClassifier
        dc = cfg["distillation"]
        return MLPClassifier(
            input_dim=input_dim,
            epochs=dc["epochs"], batch_size=dc["batch_size"],
            lr=dc["learning_rate"], weight_pos=spw,
            seed=cfg["training"]["random_state"],
        )
    else:
        raise ValueError(f"Unknown model: {name}")


def fit_model(name: str, model, X_train, y_train, X_val, y_val, cfg: dict):
    if name == "lightgbm":
        from src.models.train_lgbm import train_lgbm
        return train_lgbm(model, X_train, y_train, X_val, y_val, cfg)
    elif name == "xgboost":
        from src.models.train_xgb import train_xgb
        return train_xgb(model, X_train, y_train, X_val, y_val, cfg)
    elif name == "catboost":
        from src.models.train_catboost import train_catboost
        return train_catboost(model, X_train, y_train, X_val, y_val, cfg)
    elif name == "mlp":
        model.fit(X_train, y_train, X_val, y_val)
        return model
    else:
        model.fit(X_train, y_train)
        return model


def compute_spw(y) -> float:
    y = np.asarray(y)
    pos = y.sum()
    neg = len(y) - pos
    return float(neg / max(pos, 1))
