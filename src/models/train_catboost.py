from catboost import CatBoostClassifier


def get_catboost(cfg: dict, scale_pos_weight: float = 1.0):
    rs = cfg["training"]["random_state"]
    return CatBoostClassifier(
        iterations=500,
        learning_rate=0.05,
        depth=6,
        scale_pos_weight=scale_pos_weight,
        random_seed=rs,
        verbose=0,
    )


def train_catboost(model, X_train, y_train, X_val, y_val, cfg: dict):
    early = cfg["training"]["early_stopping_rounds"]
    model.fit(
        X_train, y_train,
        eval_set=(X_val, y_val),
        early_stopping_rounds=early,
        verbose=False,
    )
    return model
