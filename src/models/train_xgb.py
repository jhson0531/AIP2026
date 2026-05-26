import xgboost as xgb


def get_xgb(cfg: dict, scale_pos_weight: float = 1.0):
    rs = cfg["training"]["random_state"]
    early = cfg["training"]["early_stopping_rounds"]
    return xgb.XGBClassifier(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=6,
        scale_pos_weight=scale_pos_weight,
        early_stopping_rounds=early,
        random_state=rs,
        n_jobs=cfg["training"]["n_jobs"],
        eval_metric="logloss",
        verbosity=0,
    )


def train_xgb(model, X_train, y_train, X_val, y_val, cfg: dict):
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=False,
    )
    return model
