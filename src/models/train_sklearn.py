import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline


def get_logistic_regression(cfg: dict):
    rs = cfg["training"]["random_state"]
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(
            class_weight="balanced",
            max_iter=1000,
            random_state=rs,
            n_jobs=cfg["training"]["n_jobs"],
        )),
    ])


def get_random_forest(cfg: dict):
    rs = cfg["training"]["random_state"]
    return RandomForestClassifier(
        n_estimators=300,
        class_weight="balanced",
        random_state=rs,
        n_jobs=cfg["training"]["n_jobs"],
    )
